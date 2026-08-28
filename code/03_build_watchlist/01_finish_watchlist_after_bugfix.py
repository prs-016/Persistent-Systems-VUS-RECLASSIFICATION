import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from feature_pipeline import load_data, BASE_FEATURE_COLS, fit_gene_target_encoding, oof_gene_encoding  # noqa: E402

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

    # this is the merge that was missing before, applied to the FULL df (including
    # non-resolved VUS rows), using the direction model's saved train-only lookup
    direction_gene_lookup = pd.read_csv("data/stage2/gene_pathogenic_rate_lookup_v2.csv", index_col=0)
    with open("data/stage2/gene_pathogenic_rate_global_v2.txt") as f:
        direction_global_rate = float(f.read().strip())
    df["gene_pathogenic_rate_te"] = df["GeneSymbol"].map(direction_gene_lookup["gene_pathogenic_rate_te"]).fillna(direction_global_rate)

    # reload the already-trained models
    reclass_model = CatBoostClassifier()
    reclass_model.load_model("data/stage2/best_model_v13.cbm")
    with open("data/stage2/best_model_features_v13.txt") as f:
        reclass_feature_cols = f.read().split(",")

    direction_models = joblib.load("data/stage2/direction_model_v2.joblib")
    with open("data/stage2/direction_model_features_v2.txt") as f:
        direction_feature_cols = f.read().split(",")

    cox_model = joblib.load("data/stage2/cox_ph_model_v13.joblib")
    survival_feature_cols = list(cox_model.params_.index)

    aft_model = joblib.load("data/stage2/aft_model_v2.joblib")

    dates_t0 = pd.to_datetime(df["last_evaluated_t0"], errors="coerce")
    dates_t1 = pd.to_datetime(df["last_evaluated_t1"], errors="coerce")
    snapshot_date = dates_t1.max()
    event = y.astype(int)
    duration_days = np.where(event == 1, (dates_t1 - dates_t0).dt.days, (snapshot_date - dates_t0).dt.days)
    df["duration_years"] = duration_days / 365.25
    max_observed_years = df.loc[df["duration_years"] > 0, "duration_years"].max()

    watchlist = df[df["bucket_t1"] == "VUS"].copy()
    print(f"watchlist population: {len(watchlist)} rows")

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
    print(f"Cox baseline observed range: {baseline_times.min():.3f} - {baseline_times.max():.3f} years")

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

    year_probs = watchlist_out[year_cols].values
    yearly_increments = np.diff(year_probs, axis=1, prepend=0)
    expected_per_year = yearly_increments.sum(axis=0)
    expected_10plus = watchlist_out["p_unresolved_after_10y"].sum()
    print("\nexpected reclassification count per year (dbSNP-updated):")
    for yr, count in zip(range(1, 11), expected_per_year):
        print(f"  year {yr}: {count:.1f}")
    print(f"  10+ years: {expected_10plus:.1f} ({expected_10plus/len(watchlist_out)*100:.1f}%)")
    print(f"  sum check: {expected_per_year.sum() + expected_10plus:.1f} vs watchlist size {len(watchlist_out)}")

    with open("data/stage2/step17_yearly_counts.txt", "w") as f:
        f.write(f"expected_per_year: {expected_per_year.tolist()}\n")
        f.write(f"expected_10plus: {expected_10plus}\n")
    print("\nwrote data/stage2/step17_yearly_counts.txt")
