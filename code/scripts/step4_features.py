"""Step 4 — compute the core classification feature table."""
import pandas as pd


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vaf"] = out["t_alt_count"] / (out["t_alt_count"] + out["t_ref_count"])
    # gnomad_af_model: the Step 6 classifier was trained on ANNOVAR's
    # gnomad211_exome AF (annovar_gnomad_af), not VEP's live gnomAD lookup —
    # ANNOVAR was used for training because running VEP REST across the full
    # ~9,934-row training set was infeasible (see step5b_annotate_training.py),
    # while VEP was fine for this patient's 48 variants. To keep train/serve
    # features from the same source, the MODEL FEATURE uses annovar_gnomad_af
    # here. VEP's gnomad_af is kept as its own column too (unused by the
    # model, but retained for the final report / human cross-checking).
    out["gnomad_af_model"] = pd.to_numeric(out["annovar_gnomad_af"], errors="coerce").fillna(0.0)
    out["gnomad_af"] = out["gnomad_af"].fillna(0.0)
    out["cosmic_hotspot"] = out["cosmic_hotspot"].astype(bool)
    out["low_tissue_expression_flag"] = out["low_tissue_expression_flag"].astype(bool)
    return out[[
        "chrom", "pos", "end", "ref", "alt", "gene",
        "vaf", "gnomad_af", "gnomad_af_model", "cosmic_hotspot", "low_tissue_expression_flag",
        "dbsnp_id", "clnsig", "hpa_expression_level", "hpa_reliability",
    ]]


if __name__ == "__main__":
    df = pd.read_csv("data/patient/step3_hpa.csv")
    features = compute_features(df)
    print(features.head(10).to_string())
    print("\nvaf stats:")
    print(features["vaf"].describe())
    features.to_csv("data/patient/step4_features.csv", index=False)
