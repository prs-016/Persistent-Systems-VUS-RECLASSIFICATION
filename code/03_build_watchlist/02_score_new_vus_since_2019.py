import gc
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import joblib

CHUNK_SIZE = 200_000

REVIEW_STAR_MAP = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, single submitter": 1,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "no assertion criteria provided": 0,
    "no interpretation for the single variant": 0,
    "no classification for the single variant": 0,
    "no classification provided": 0,
    "no assertion provided": 0,
}
EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]
CHROM_LIST = [str(i) for i in range(1, 23)] + ["X"]
YEAR_COLS = [f"p_resolved_by_{yr}y" for yr in range(1, 11)]

OUTPUT_COLS = [
    "VariationID", "GeneSymbol", "Chromosome", "Start", "ReferenceAllele", "AlternateAllele",
    "dbsnp_rsid", "has_dbsnp_id",
    "n_submitters_t0", "review_status_stars_t0", "submission_velocity_t0", "pubmed_count_t0",
    "litvar2_pmids_count", "has_mave_coverage", "gnomad_af", "gnomad_af_popmax",
    "gene_resolved_rate_te", "annovar_exonic_func",
    "reclass_probability_v13", "direction_pathogenic_probability_if_resolved",
    "p_resolved_12mo", "p_resolved_24mo",
] + YEAR_COLS + [
    "p_unresolved_after_10y", "median_years_to_reclass_aft", "extrapolated_beyond_observed_range",
    "cohort", "feature_completeness",
]


def build_gene_lookups():
    core_features = pd.read_csv("data/stage2/vus_features_v9.csv", low_memory=False)
    gene_level_lookup = core_features.groupby("GeneSymbol").agg(
        pubmed_count_t0=("pubmed_count_t0", "median"),
        low_tissue_expression_flag=("low_tissue_expression_flag", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else 0),
        litvar2_pmids_count=("litvar2_pmids_count", "median"),
        litvar2_queried=("litvar2_queried", "max"),
    )
    defaults = {
        "pubmed_count_t0": core_features["pubmed_count_t0"].median(),
        "low_tissue_expression_flag": 0,
        "litvar2_pmids_count": core_features["litvar2_pmids_count"].median(),
        "litvar2_queried": 0,
    }
    del core_features
    gc.collect()

    mave_coverage = pd.read_csv("data/stage2/mavedb_gene_coverage_v2.csv").set_index("gene")
    gene_target_encoding = pd.read_csv("data/stage2/gene_target_encoding_lookup_v13.csv", index_col=0)
    with open("data/stage2/gene_target_encoding_global_rate_v13.txt") as f:
        lines = f.read().split("\n")
        global_resolved_rate, global_avg_submitters = float(lines[0]), float(lines[1])
    direction_gene_lookup = pd.read_csv("data/stage2/gene_pathogenic_rate_lookup_v2.csv", index_col=0)
    with open("data/stage2/gene_pathogenic_rate_global_v2.txt") as f:
        direction_global_rate = float(f.read().strip())
    return {
        "gene_level_lookup": gene_level_lookup, "defaults": defaults, "mave_coverage": mave_coverage,
        "gene_target_encoding": gene_target_encoding, "global_resolved_rate": global_resolved_rate,
        "global_avg_submitters": global_avg_submitters,
        "direction_gene_lookup": direction_gene_lookup, "direction_global_rate": direction_global_rate,
    }


def score_chunk(chunk, lookups, rsid_lookup, reclass_model, reclass_feature_cols,
                 direction_models, direction_feature_cols, cox_model, survival_feature_cols):
    features = chunk.copy()
    features["Start"] = pd.to_numeric(features["PositionVCF"], errors="coerce")
    features["ReferenceAllele"] = features["ReferenceAlleleVCF"]
    features["AlternateAllele"] = features["AlternateAlleleVCF"]
    features["n_submitters_t0"] = pd.to_numeric(features["NumberSubmitters"], errors="coerce").fillna(0).astype(int)
    features["n_submitters_2018_12"] = 0.0
    features["submission_velocity_t0"] = 0.0
    features["submitter_multiple_flag"] = (features["n_submitters_t0"] > 1).astype(int)
    features["review_status_stars_t0"] = features["ReviewStatus"].map(REVIEW_STAR_MAP).fillna(0).astype(int)
    features["stars_x_submitters"] = features["review_status_stars_t0"] * features["n_submitters_t0"]

    features["gnomad_af"] = 0.0
    features["gnomad_af_popmax"] = 0.0
    features["gnomad_af_log"] = 0.0
    features["gnomad_af_popmax_log"] = 0.0
    for col in EXONIC_FUNC_DUMMIES:
        features[col] = 0
    features["annovar_exonic_func"] = "unknown"

    chrom = features["Chromosome"].astype(str)
    for c in CHROM_LIST:
        features[f"chr_{c}"] = (chrom == c).astype(int)
    features["chr_other"] = (~chrom.isin(CHROM_LIST)).astype(int)

    gene_level_lookup, defaults = lookups["gene_level_lookup"], lookups["defaults"]
    for col in ["pubmed_count_t0", "low_tissue_expression_flag", "litvar2_pmids_count", "litvar2_queried"]:
        features[col] = features["GeneSymbol"].map(gene_level_lookup[col]).fillna(defaults.get(col, 0))
    features["pubmed_queried"] = 0
    features["pubmed_x_litvar2"] = np.log1p(features["pubmed_count_t0"].clip(lower=0)) * features["litvar2_queried"]

    mave_coverage = lookups["mave_coverage"]
    features["has_mave_coverage"] = features["GeneSymbol"].map(mave_coverage["has_mave_coverage"]).fillna(False).astype(int)
    features["mave_num_variants"] = features["GeneSymbol"].map(mave_coverage["mave_num_variants"]).fillna(0)
    features["mave_num_variants_log"] = np.log1p(features["mave_num_variants"])

    gene_target_encoding = lookups["gene_target_encoding"]
    features["gene_resolved_rate_te"] = features["GeneSymbol"].map(gene_target_encoding["gene_resolved_rate_te"]).fillna(lookups["global_resolved_rate"])
    features["gene_resolved_rate_te_full"] = features["gene_resolved_rate_te"]
    features["gene_avg_submitters_te"] = features["GeneSymbol"].map(gene_target_encoding["gene_avg_submitters_te"]).fillna(lookups["global_avg_submitters"])
    features["gene_pathogenic_rate_te"] = features["GeneSymbol"].map(lookups["direction_gene_lookup"]["gene_pathogenic_rate_te"]).fillna(lookups["direction_global_rate"])

    features = features.merge(rsid_lookup, on="VariationID", how="left")
    features["has_dbsnp_id"] = features["has_dbsnp_id"].fillna(0).astype(int)
    features["dbsnp_rsid"] = features["dbsnp_rsid"].fillna(-1).astype(int)

    features["reclass_probability_v13"] = reclass_model.predict_proba(features[reclass_feature_cols])[:, 1]

    def ensemble_average(models, X):
        return np.column_stack([
            models["lgbm"].predict_proba(X)[:, 1],
            models["xgb"].predict_proba(X)[:, 1],
            models["rf"].predict_proba(X)[:, 1],
        ]).mean(axis=1)

    features["direction_pathogenic_probability_if_resolved"] = ensemble_average(direction_models, features[direction_feature_cols])

    partial_hazard = cox_model.predict_partial_hazard(features[survival_feature_cols]).values
    baseline_times = cox_model.baseline_survival_.index.values
    baseline_survival = cox_model.baseline_survival_.iloc[:, 0].values

    def survival_at(t):
        i = np.searchsorted(baseline_times, t, side="right") - 1
        i = np.clip(i, 0, len(baseline_survival) - 1)
        return baseline_survival[i]

    features["p_resolved_12mo"] = 1 - (survival_at(1.0) ** partial_hazard)
    features["p_resolved_24mo"] = 1 - (survival_at(2.0) ** partial_hazard)
    for yr in range(1, 11):
        features[f"p_resolved_by_{yr}y"] = 1 - (survival_at(float(yr)) ** partial_hazard)
    features["p_unresolved_after_10y"] = 1 - features["p_resolved_by_10y"]

    features["median_years_to_reclass_aft"] = np.nan
    features["extrapolated_beyond_observed_range"] = np.nan
    features["cohort"] = "extended_new_since_2019"
    features["feature_completeness"] = "partial_no_af_no_exonic_func"

    return features[OUTPUT_COLS]


if __name__ == "__main__":
    tracked = pd.read_csv("data/stage2/vus_features_v9.csv", usecols=["VariationID"], low_memory=False)
    tracked_ids = set(tracked["VariationID"])
    del tracked
    gc.collect()

    current_vus = pd.read_csv("data/stage2/current_vus_full.tsv", sep="\t", dtype={"VariationID": "int64"})
    current_vus = current_vus.drop_duplicates(subset=["VariationID"])
    current_vus = current_vus[
        (current_vus["PositionVCF"] != "na") & (current_vus["ReferenceAlleleVCF"] != "na")
        & (current_vus["AlternateAlleleVCF"] != "na")
    ]
    print(f"current real 'Uncertain significance' VariationIDs (GRCh38, usable coords): {len(current_vus)}")

    new_vus = current_vus[~current_vus["VariationID"].isin(tracked_ids)].reset_index(drop=True)
    print(f"NEW since 2019 baseline: {len(new_vus)} ({len(new_vus)/len(current_vus)*100:.1f}% of current VUS)")
    del current_vus
    gc.collect()

    rsid_lookup = pd.read_csv("data/stage2/dbsnp_rsid_lookup.csv")
    lookups = build_gene_lookups()

    reclass_model = CatBoostClassifier()
    reclass_model.load_model("data/stage2/best_model_v13.cbm")
    with open("data/stage2/best_model_features_v13.txt") as f:
        reclass_feature_cols = f.read().split(",")
    direction_models = joblib.load("data/stage2/direction_model_v2.joblib")
    with open("data/stage2/direction_model_features_v2.txt") as f:
        direction_feature_cols = f.read().split(",")
    cox_model = joblib.load("data/stage2/cox_ph_model_v13.joblib")
    survival_feature_cols = list(cox_model.params_.index)

    output_path = "data/stage2/vus_watchlist_extended_new_since_2019.csv.gz"
    n_chunks = (len(new_vus) + CHUNK_SIZE - 1) // CHUNK_SIZE
    first_chunk = True
    for i in range(n_chunks):
        chunk = new_vus.iloc[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        scored = score_chunk(chunk, lookups, rsid_lookup, reclass_model, reclass_feature_cols,
                              direction_models, direction_feature_cols, cox_model, survival_feature_cols)
        scored.to_csv(output_path, mode=("w" if first_chunk else "a"), header=first_chunk, index=False, compression="gzip")
        first_chunk = False
        print(f"chunk {i+1}/{n_chunks} scored + written ({len(scored)} rows)")
        del chunk, scored
        gc.collect()

    print(f"\nwrote {output_path}")
