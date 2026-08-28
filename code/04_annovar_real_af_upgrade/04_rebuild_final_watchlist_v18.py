import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from feature_pipeline import load_data, BASE_FEATURE_COLS, fit_gene_target_encoding  # noqa: E402

RANDOM_STATE = 42

if __name__ == "__main__":
    df, y = load_data()
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    global_rate = y.loc[train_idx].mean()

    full_train_encoding = fit_gene_target_encoding(df, train_idx, "resolved", global_rate)
    df["gene_resolved_rate_te_full"] = df["GeneSymbol"].map(full_train_encoding["rate"]).fillna(global_rate)
    gene_avg_submitters = df.loc[train_idx].groupby("GeneSymbol")["n_submitters_t0"].mean()
    global_avg_submitters = df.loc[train_idx, "n_submitters_t0"].mean()
    df["gene_avg_submitters_te"] = df["GeneSymbol"].map(gene_avg_submitters).fillna(global_avg_submitters)

    direction_gene_lookup = pd.read_csv("data/stage2/gene_pathogenic_rate_lookup_v2.csv", index_col=0)
    with open("data/stage2/gene_pathogenic_rate_global_v2.txt") as f:
        direction_global_rate = float(f.read().strip())
    df["gene_pathogenic_rate_te"] = df["GeneSymbol"].map(direction_gene_lookup["gene_pathogenic_rate_te"]).fillna(direction_global_rate)

    # reload the already-trained artifacts (direction, Cox unchanged; reclass model = v15_realaf)
    reclass_model = CatBoostClassifier()
    reclass_model.load_model("data/stage2/best_model_v15_realaf.cbm")
    with open("data/stage2/best_model_features_v15_realaf.txt") as f:
        reclass_feature_cols = f.read().split(",")

    direction_models = joblib.load("data/stage2/direction_model_v2.joblib")
    with open("data/stage2/direction_model_features_v2.txt") as f:
        direction_feature_cols = f.read().split(",")

    cox_model = joblib.load("data/stage2/cox_ph_model_v13.joblib")
    survival_feature_cols = list(cox_model.params_.index)

    dates_t0 = pd.to_datetime(df["last_evaluated_t0"], errors="coerce")
    dates_t1 = pd.to_datetime(df["last_evaluated_t1"], errors="coerce")
    snapshot_date = dates_t1.max()
    event = y.astype(int)
    duration_days = np.where(event == 1, (dates_t1 - dates_t0).dt.days, (snapshot_date - dates_t0).dt.days)
    df["duration_years"] = duration_days / 365.25

    watchlist = df[df["bucket_t1"] == "VUS"].copy()
    print(f"core watchlist population: {len(watchlist)} rows")

    watchlist["reclass_probability"] = reclass_model.predict_proba(watchlist[reclass_feature_cols])[:, 1]
    watchlist["scoring_model"] = "v15_realaf"

    def ensemble_average(models, X):
        return np.column_stack([
            models["lgbm"].predict_proba(X)[:, 1],
            models["xgb"].predict_proba(X)[:, 1],
            models["rf"].predict_proba(X)[:, 1],
        ]).mean(axis=1)

    watchlist["direction_pathogenic_probability_if_resolved"] = ensemble_average(direction_models, watchlist[direction_feature_cols])

    watchlist_survival_features = watchlist[survival_feature_cols]
    partial_hazard = cox_model.predict_partial_hazard(watchlist_survival_features).values
    baseline_times = cox_model.baseline_survival_.index.values
    baseline_survival = cox_model.baseline_survival_.iloc[:, 0].values

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

    aft_model = joblib.load("data/stage2/aft_model_v2.joblib")
    watchlist_aft_features = watchlist[survival_feature_cols]
    watchlist["median_years_to_reclass_aft"] = aft_model.predict_median(watchlist_aft_features).values
    max_observed_years = df.loc[df["duration_years"] > 0, "duration_years"].max()
    watchlist["extrapolated_beyond_observed_range"] = watchlist["median_years_to_reclass_aft"] > max_observed_years

    watchlist["cohort"] = "core_2019_tracked"
    watchlist["feature_completeness"] = "full"

    output_cols = [
        "VariationID", "GeneSymbol", "Chromosome", "Start", "ReferenceAllele", "AlternateAllele",
        "dbsnp_rsid", "has_dbsnp_id",
        "n_submitters_t0", "review_status_stars_t0", "submission_velocity_t0", "pubmed_count_t0",
        "litvar2_pmids_count", "has_mave_coverage", "gnomad_af", "gnomad_af_popmax",
        "gene_resolved_rate_te_full", "annovar_exonic_func",
        "reclass_probability", "scoring_model", "direction_pathogenic_probability_if_resolved",
        "p_resolved_12mo", "p_resolved_24mo",
    ] + year_cols + [
        "p_unresolved_after_10y", "median_years_to_reclass_aft", "extrapolated_beyond_observed_range",
        "cohort", "feature_completeness",
    ]
    core_out = watchlist[output_cols].rename(columns={"gene_resolved_rate_te_full": "gene_resolved_rate_te",
                                                        "VariationID": "ClinvarID"})
    core_out.to_csv("data/stage2/vus_global_watchlist_core_v18.csv", index=False)
    print(f"\nwrote data/stage2/vus_global_watchlist_core_v18.csv, shape={core_out.shape}")
    print(f"reclass_probability mean: {core_out['reclass_probability'].mean():.4f}")

    # union with the real-AF extended population into the final v18 watchlist
    extended_out = pd.read_csv("data/stage2/vus_watchlist_extended_new_since_2019_v15_realaf.csv.gz", low_memory=False)
    print(f"\nextended (v15_realaf): {len(extended_out)} rows")
    assert set(core_out.columns) == set(extended_out.columns), \
        f"column mismatch: core-only={set(core_out.columns)-set(extended_out.columns)}, ext-only={set(extended_out.columns)-set(core_out.columns)}"
    extended_out = extended_out[core_out.columns]

    combined = pd.concat([core_out, extended_out], ignore_index=True)
    print(f"combined: {len(combined)} rows (expected {len(core_out)+len(extended_out)})")
    assert len(combined) == len(core_out) + len(extended_out)
    assert combined["ClinvarID"].duplicated().sum() == 0, "duplicate ClinvarIDs across core+extended, overlap bug"
    print(combined["scoring_model"].value_counts())
    print(combined["cohort"].value_counts())
    print(f"reclass_probability mean: core={core_out['reclass_probability'].mean():.4f}, "
          f"extended={extended_out['reclass_probability'].mean():.4f}, combined={combined['reclass_probability'].mean():.4f}")

    output_path = "data/stage2/vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz"
    combined.to_csv(output_path, index=False, compression="gzip")
    print(f"\nwrote {output_path}")
