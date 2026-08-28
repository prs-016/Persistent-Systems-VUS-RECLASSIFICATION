import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from catboost import CatBoostClassifier
from lifelines import CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.utils import concordance_index
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from feature_pipeline import (  # noqa: E402
    load_data, BASE_FEATURE_COLS, fit_gene_target_encoding, oof_gene_encoding, fit_lgbm_xgb_rf_stack,
)

RANDOM_STATE = 42

if __name__ == "__main__":
    df, y = load_data()
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    global_rate = y.loc[train_idx].mean()
    print(f"split: train={len(train_idx)}, test={len(test_idx)}, global_rate={global_rate:.4f}")
    print(f"has_dbsnp_id coverage, full population: {df['has_dbsnp_id'].mean()*100:.2f}%, "
          f"VUS watchlist population: {df.loc[df['bucket_t1']=='VUS','has_dbsnp_id'].mean()*100:.2f}%")

    # gene-level target encodings, shared by every model below
    oof_encoding = oof_gene_encoding(df, train_idx, "resolved", global_rate)
    df.loc[train_idx, "gene_resolved_rate_te"] = oof_encoding.reindex(df.loc[train_idx].index).values
    full_train_encoding = fit_gene_target_encoding(df, train_idx, "resolved", global_rate)
    df.loc[test_idx, "gene_resolved_rate_te"] = df.loc[test_idx, "GeneSymbol"].map(full_train_encoding["rate"]).fillna(global_rate)
    # a non-OOF version too, for Cox/AFT which score the whole watchlist, not just train
    df["gene_resolved_rate_te_full"] = df["GeneSymbol"].map(full_train_encoding["rate"]).fillna(global_rate)
    gene_avg_submitters = df.loc[train_idx].groupby("GeneSymbol")["n_submitters_t0"].mean()
    global_avg_submitters = df.loc[train_idx, "n_submitters_t0"].mean()
    df["gene_avg_submitters_te"] = df["GeneSymbol"].map(gene_avg_submitters).fillna(global_avg_submitters)

    full_train_encoding[["rate"]].rename(columns={"rate": "gene_resolved_rate_te"}).join(
        gene_avg_submitters.rename("gene_avg_submitters_te")
    ).to_csv("data/stage2/gene_target_encoding_lookup_v13.csv")
    with open("data/stage2/gene_target_encoding_global_rate_v13.txt", "w") as f:
        f.write(f"{global_rate}\n{global_avg_submitters}\n")

    # production reclass-probability model: CatBoost with GeneSymbol as a
    # native categorical, now with has_dbsnp_id added in
    reclass_feature_cols = [c for c in BASE_FEATURE_COLS if not c.startswith("chr_")] + ["GeneSymbol"]
    X_train, X_test = df.loc[train_idx, reclass_feature_cols], df.loc[test_idx, reclass_feature_cols]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    gene_col_index = [reclass_feature_cols.index("GeneSymbol")]
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    reclass_model = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.03, l2_leaf_reg=5,
                                        scale_pos_weight=scale_pos_weight, random_seed=RANDOM_STATE, verbose=False,
                                        cat_features=gene_col_index, eval_metric="PRAUC")
    reclass_model.fit(X_train, y_train)
    test_proba = reclass_model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, test_proba)
    roc_auc = roc_auc_score(y_test, test_proba)
    print(f"\n[reclass model] CatBoost + dbSNP: PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")
    print(f"  delta vs pre-dbSNP model (PR-AUC 0.4127): {pr_auc - 0.4127:+.4f}")

    reclass_model.save_model("data/stage2/best_model_v13.cbm")
    with open("data/stage2/best_model_features_v13.txt", "w") as f:
        f.write(",".join(reclass_feature_cols))
    with open("data/stage2/v13_production_metrics.txt", "w") as f:
        f.write(f"pr_auc={pr_auc:.4f}\nroc_auc={roc_auc:.4f}\n")

    # direction classifier (Pathogenic vs Benign), resolved rows only
    resolved_mask = df["resolved_direction"].isin(["Pathogenic", "Benign"])
    direction_df = df[resolved_mask].copy()
    direction_y = (direction_df["resolved_direction"] == "Pathogenic").astype(int)
    direction_train_idx = train_idx.intersection(direction_df.index)
    direction_test_idx = test_idx.intersection(direction_df.index)
    direction_global_rate = direction_y.loc[direction_train_idx].mean()
    print(f"\n[direction model] population: {len(direction_df)} rows "
          f"({(direction_y==1).sum()} Pathogenic / {(direction_y==0).sum()} Benign)")

    direction_df["resolved_direction_pathogenic"] = direction_y
    direction_gene_encoding = fit_gene_target_encoding(direction_df, direction_train_idx, "resolved_direction_pathogenic", direction_global_rate)
    direction_df["gene_pathogenic_rate_te"] = direction_df["GeneSymbol"].map(direction_gene_encoding["rate"]).fillna(direction_global_rate)
    direction_feature_cols = BASE_FEATURE_COLS + ["gene_pathogenic_rate_te"]

    Xd_train, Xd_test = direction_df.loc[direction_train_idx, direction_feature_cols], direction_df.loc[direction_test_idx, direction_feature_cols]
    yd_train, yd_test = direction_y.loc[direction_train_idx], direction_y.loc[direction_test_idx]
    direction_models, direction_comparison = fit_lgbm_xgb_rf_stack(Xd_train, yd_train, Xd_test, yd_test,
                                                                     "direction model (Pathogenic vs Benign) + dbSNP")
    direction_roc_auc = direction_comparison.loc["SimpleAverage", "roc_auc"]
    print(f"  delta vs pre-dbSNP direction model (ROC-AUC 0.9462): {direction_roc_auc - 0.9462:+.4f}")

    joblib.dump(direction_models, "data/stage2/direction_model_v2.joblib")
    with open("data/stage2/direction_model_features_v2.txt", "w") as f:
        f.write(",".join(direction_feature_cols))
    direction_gene_encoding[["rate"]].rename(columns={"rate": "gene_pathogenic_rate_te"}).to_csv(
        "data/stage2/gene_pathogenic_rate_lookup_v2.csv"
    )
    with open("data/stage2/gene_pathogenic_rate_global_v2.txt", "w") as f:
        f.write(str(direction_global_rate))
    direction_comparison.to_csv("data/stage2/step17_direction_model_comparison.csv")

    # Cox proportional-hazards survival model (short-horizon plus the
    # year-by-year "by when" breakdown), also with has_dbsnp_id added
    survival_feature_cols = (
        ["gnomad_af", "gnomad_af_popmax", "low_tissue_expression_flag", "n_submitters_t0", "has_mave_coverage",
         "submission_velocity_t0", "submitter_multiple_flag", "pubmed_count_t0",
         "litvar2_pmids_count", "litvar2_queried", "review_status_stars_t0",
         "stars_x_submitters", "pubmed_x_litvar2", "gene_resolved_rate_te_full", "gene_avg_submitters_te",
         "has_dbsnp_id"]
        + [c for c in BASE_FEATURE_COLS if c.startswith("ef_")]
    )

    dates_t0 = pd.to_datetime(df["last_evaluated_t0"], errors="coerce")
    dates_t1 = pd.to_datetime(df["last_evaluated_t1"], errors="coerce")
    snapshot_date = dates_t1.max()
    event = y.astype(int)
    duration_days = np.where(event == 1, (dates_t1 - dates_t0).dt.days, (snapshot_date - dates_t0).dt.days)
    df["duration_years"] = duration_days / 365.25
    df["event"] = event

    survival_df = df[survival_feature_cols + ["duration_years", "event"]].copy()
    survival_df = survival_df[survival_df["duration_years"] > 0]
    constant_cols = [c for c in survival_feature_cols if survival_df[c].nunique() <= 1]
    if constant_cols:
        print(f"\n[survival] dropping constant columns: {constant_cols}")
    survival_feature_cols = [c for c in survival_feature_cols if c not in constant_cols]
    survival_train_idx = train_idx.intersection(survival_df.index)
    survival_test_idx = test_idx.intersection(survival_df.index)

    cox_model = CoxPHFitter(penalizer=1.0)
    cox_model.fit(survival_df.loc[survival_train_idx, survival_feature_cols + ["duration_years", "event"]],
                   duration_col="duration_years", event_col="event")
    cox_c_index = cox_model.score(
        survival_df.loc[survival_test_idx, survival_feature_cols + ["duration_years", "event"]],
        scoring_method="concordance_index",
    )
    print(f"\n[Cox PH] held-out c-index + dbSNP: {cox_c_index:.4f} (pre-dbSNP was 0.7468, "
          f"delta {cox_c_index - 0.7468:+.4f})")
    joblib.dump(cox_model, "data/stage2/cox_ph_model_v13.joblib")
    with open("data/stage2/cox_ph_cindex_v13.txt", "w") as f:
        f.write(f"{cox_c_index:.4f}\n")

    # AFT models (secondary reference: the extrapolation caveat on
    # predictions beyond the observed duration range still applies)
    best_aft = None
    for name, aft_class in [("Weibull", WeibullAFTFitter), ("LogNormal", LogNormalAFTFitter),
                             ("LogLogistic", LogLogisticAFTFitter)]:
        aft_model = aft_class(penalizer=0.5)
        aft_model.fit(survival_df.loc[survival_train_idx, survival_feature_cols + ["duration_years", "event"]],
                       duration_col="duration_years", event_col="event")
        median_pred = aft_model.predict_median(survival_df.loc[survival_test_idx, survival_feature_cols])
        c_index = concordance_index(
            survival_df.loc[survival_test_idx, "duration_years"], median_pred, survival_df.loc[survival_test_idx, "event"]
        )
        print(f"[AFT] {name} + dbSNP: c-index={c_index:.4f}")
        if best_aft is None or c_index > best_aft[1]:
            best_aft = (name, c_index, aft_model)
    aft_name, aft_c_index, aft_model = best_aft
    print(f"[AFT] best: {aft_name} (c-index={aft_c_index:.4f}, pre-dbSNP LogNormal was 0.7475, "
          f"delta {aft_c_index - 0.7475:+.4f})")
    joblib.dump(aft_model, "data/stage2/aft_model_v2.joblib")

    # consolidated watchlist: reclass probability, direction, Cox short-horizon
    # plus the full year-by-year (1-10y, then 10+y) breakdown
    watchlist = df[df["bucket_t1"] == "VUS"].copy()
    print(f"\n[watchlist] population: {len(watchlist)} rows")

    watchlist["reclass_probability_v13"] = reclass_model.predict_proba(watchlist[reclass_feature_cols])[:, 1]

    def ensemble_average(models, X):
        return np.column_stack([
            models["lgbm"].predict_proba(X)[:, 1],
            models["xgb"].predict_proba(X)[:, 1],
            models["rf"].predict_proba(X)[:, 1],
        ]).mean(axis=1)

    watchlist["direction_pathogenic_probability_if_resolved"] = ensemble_average(
        direction_models, watchlist[direction_feature_cols]
    )

    watchlist_survival_features = watchlist[survival_feature_cols]
    partial_hazard = cox_model.predict_partial_hazard(watchlist_survival_features).values
    baseline_times = cox_model.baseline_survival_.index.values
    baseline_survival = cox_model.baseline_survival_.iloc[:, 0].values
    print(f"Cox baseline observed range: {baseline_times.min():.3f} - {baseline_times.max():.3f} years "
          f"(years 1-10 fall safely inside this range)")

    def survival_at(t):
        i = np.searchsorted(baseline_times, t, side="right") - 1
        i = np.clip(i, 0, len(baseline_survival) - 1)
        return baseline_survival[i]

    watchlist["p_resolved_12mo"] = 1 - (survival_at(1.0) ** partial_hazard)
    watchlist["p_resolved_24mo"] = 1 - (survival_at(2.0) ** partial_hazard)
    year_cols = []
    for yr in range(1, 11):
        col = f"p_resolved_by_{yr}y"
        watchlist[col] = 1 - (survival_at(float(yr)) ** partial_hazard)
        year_cols.append(col)
    watchlist["p_unresolved_after_10y"] = 1 - watchlist["p_resolved_by_10y"]

    diffs = watchlist[year_cols].diff(axis=1).iloc[:, 1:]
    n_violations = (diffs < -1e-9).any(axis=1).sum()
    print(f"monotonicity check (should be 0): {n_violations}")

    watchlist_aft_features = watchlist[survival_feature_cols]
    watchlist["median_years_to_reclass_aft"] = aft_model.predict_median(watchlist_aft_features).values
    max_observed_years = survival_df["duration_years"].max()
    watchlist["extrapolated_beyond_observed_range"] = watchlist["median_years_to_reclass_aft"] > max_observed_years

    output_cols = [
        "VariationID", "GeneSymbol", "Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele",
        "dbsnp_rsid", "has_dbsnp_id",
        "n_submitters_t0", "review_status_stars_t0", "submission_velocity_t0", "pubmed_count_t0",
        "litvar2_pmids_count", "has_mave_coverage", "gnomad_af", "gnomad_af_popmax",
        "gene_resolved_rate_te_full", "annovar_exonic_func",
        "reclass_probability_v13", "direction_pathogenic_probability_if_resolved",
        "p_resolved_12mo", "p_resolved_24mo",
    ] + year_cols + [
        "p_unresolved_after_10y", "median_years_to_reclass_aft", "extrapolated_beyond_observed_range",
    ]
    watchlist_out = watchlist[output_cols].rename(columns={"gene_resolved_rate_te_full": "gene_resolved_rate_te"})
    watchlist_out = watchlist_out.sort_values("reclass_probability_v13", ascending=False)
    watchlist_out.to_csv("data/stage2/vus_global_watchlist_v15.csv", index=False)
    watchlist_out.to_csv("data/stage2/vus_global_watchlist_v15.csv.gz", index=False, compression="gzip")
    print(f"\nwrote data/stage2/vus_global_watchlist_v15.csv(.gz), shape={watchlist_out.shape}")
    print(watchlist_out.head(10)[["VariationID", "GeneSymbol", "has_dbsnp_id", "reclass_probability_v13",
                                   "direction_pathogenic_probability_if_resolved", "p_resolved_by_10y",
                                   "p_unresolved_after_10y"]].round(4).to_string())

    # expected reclassification count per year, for the summary bar chart
    year_probs = watchlist_out[year_cols].values
    yearly_increments = np.diff(year_probs, axis=1, prepend=0)
    expected_per_year = yearly_increments.sum(axis=0)
    expected_10plus = watchlist_out["p_unresolved_after_10y"].sum()
    print("\nexpected reclassification count per year (dbSNP-updated):")
    for yr, count in zip(range(1, 11), expected_per_year):
        print(f"  year {yr}: {count:.1f}")
    print(f"  10+ years: {expected_10plus:.1f} ({expected_10plus/len(watchlist_out)*100:.1f}%)")
    print(f"  sum check: {expected_per_year.sum() + expected_10plus:.1f} vs watchlist size {len(watchlist_out)}")

    with open("data/stage2/step17_summary.txt", "w") as f:
        f.write(f"reclass PR-AUC (CatBoost+dbSNP): {pr_auc:.4f} (pre-dbSNP 0.4127, delta {pr_auc-0.4127:+.4f})\n")
        f.write(f"reclass ROC-AUC (CatBoost+dbSNP): {roc_auc:.4f}\n")
        f.write(f"direction ROC-AUC (+dbSNP): {direction_roc_auc:.4f} (pre-dbSNP 0.9462, delta {direction_roc_auc-0.9462:+.4f})\n")
        f.write(f"Cox c-index (+dbSNP): {cox_c_index:.4f} (pre-dbSNP 0.7468, delta {cox_c_index-0.7468:+.4f})\n")
        f.write(f"AFT best {aft_name} c-index (+dbSNP): {aft_c_index:.4f} (pre-dbSNP LogNormal 0.7475, delta {aft_c_index-0.7475:+.4f})\n")
        f.write(f"expected_per_year: {expected_per_year.tolist()}\n")
        f.write(f"expected_10plus: {expected_10plus}\n")
    print("\nwrote data/stage2/step17_summary.txt")
