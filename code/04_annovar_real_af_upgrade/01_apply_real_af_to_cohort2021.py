import pandas as pd
import numpy as np

EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]
EXONIC_FUNC_TO_DUMMY = {
    "frameshift deletion": "ef_frameshift_deletion",
    "frameshift substitution": "ef_frameshift_substitution",
    "frameshift insertion": "ef_frameshift_deletion",  # no dedicated insertion dummy in the v9 schema; grouped with frameshift indel
    "nonframeshift deletion": "ef_nonframeshift_deletion",
    "nonframeshift substitution": "ef_nonframeshift_substitution",
    "nonframeshift insertion": "ef_nonframeshift_deletion",  # same grouping rationale
    "nonsynonymous SNV": "ef_nonsynonymous_snv",
    "startloss": "ef_startloss",
    "stopgain": "ef_stopgain",
    "stoploss": "ef_stoploss",
    "synonymous SNV": "ef_synonymous_snv",
}

if __name__ == "__main__":
    cohort2021 = pd.read_csv("data/stage2/cohort_2021_features.csv", low_memory=False)
    print(f"cohort2021 before: {len(cohort2021)} rows")
    af_lookup = pd.read_csv("data/stage2/annovar_real_af_exonic_func_lookup.csv")
    print(f"real AF/exonic-func lookup: {len(af_lookup)} rows")

    matched_before = cohort2021["VariationID"].isin(af_lookup["VariationID"]).sum()
    print(f"cohort2021 rows with a real ANNOVAR match: {matched_before}/{len(cohort2021)} "
          f"({matched_before/len(cohort2021)*100:.1f}%)")

    cohort2021 = cohort2021.merge(af_lookup, on="VariationID", how="left")

    cohort2021["gnomad_af"] = cohort2021["gnomad_af_real"].fillna(0.0)
    cohort2021["gnomad_af_popmax"] = cohort2021["gnomad_af_popmax_real"].fillna(0.0)
    cohort2021["gnomad_af_log"] = np.log1p(cohort2021["gnomad_af"])
    cohort2021["gnomad_af_popmax_log"] = np.log1p(cohort2021["gnomad_af_popmax"])
    cohort2021["annovar_exonic_func"] = cohort2021["annovar_exonic_func_real"].fillna("unknown")

    for col in EXONIC_FUNC_DUMMIES:
        cohort2021[col] = 0
    for raw_value, dummy_col in EXONIC_FUNC_TO_DUMMY.items():
        cohort2021.loc[cohort2021["annovar_exonic_func"] == raw_value, dummy_col] = 1

    cohort2021["has_real_af"] = 1
    cohort2021["feature_source"] = "real_annovar_gnomad211_exome"
    cohort2021 = cohort2021.drop(columns=["gnomad_af_real", "gnomad_af_popmax_real", "annovar_exonic_func_real", "annovar_func_region_real"])

    cohort2021.to_csv("data/stage2/cohort_2021_features_v2_real_af.csv", index=False)
    print(f"\nwrote data/stage2/cohort_2021_features_v2_real_af.csv, shape={cohort2021.shape}")
    print(f"nonzero gnomad_af: {(cohort2021['gnomad_af']>0).mean()*100:.1f}%")
    print(cohort2021["annovar_exonic_func"].value_counts())
