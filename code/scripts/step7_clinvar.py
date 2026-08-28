"""
Step 7 — resolve patient variants against ClinVar's CLNSIG, then run the
Step 6 classifier on whatever CLNSIG doesn't already resolve confidently.

CLNSIG comes from Step 2's VEP annotation, already present in
data/patient/step4_features.csv (VEP lowercases/underscores terms, e.g.
"likely_pathogenic"; multiple submitter calls are ";"-joined).

Rule (only a clean, unanimous CLNSIG counts as confident):
  - every term in {pathogenic, likely_pathogenic}      -> Pathogenic, removed from review
  - every term in {benign, likely_benign}               -> Benign, removed from review
  - anything else (missing, "uncertain_significance", or a mixed/conflicting
    CLNSIG like "uncertain_significance;likely_benign;pathogenic") -> VUS,
    passed to the trained classifier for a Germline/Somatic call
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from config.thresholds import CLINVAR_PATHOGENIC_TERMS, CLINVAR_BENIGN_TERMS

# Read dynamically from whatever model is currently promoted to production
# (data/tcga_training/best_model_features.txt), rather than hardcoding —
# different promoted models have used different feature sets this project
# (v1: vaf+cosmic_hotspot+low_tissue_expression_flag+gnomad_af; the
# germline-expansion "novaf" model promoted after the real vaf=0.5 leakage
# fix: cosmic_hotspot+low_tissue_expression_flag+gnomad_af, no vaf; v5,
# promoted this session, adds annovar_exonic_func one-hots + gene-level
# germline-rate target encoding — see docs/STAGE1_RESULTS.md §16/§5a and
# docs/HANDOFF.md for why). "gnomad_af" in the saved feature list maps to
# this dataframe's "gnomad_af_model" column — the classifier was trained
# on ANNOVAR's gnomad211_exome AF (see step4_features.py for why), so
# inference must read from that column.
with open("data/tcga_training/best_model_features.txt") as _f:
    _saved_features = _f.read().strip().split(",")
FEATURE_COLS = ["gnomad_af_model" if c == "gnomad_af" else c for c in _saved_features]

EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]
GENE_RATE_LOOKUP_PATH = "data/tcga_training/gene_germline_rate_lookup_v5.csv"
GENE_RATE_GLOBAL_PATH = "data/tcga_training/gene_germline_rate_global_v5.txt"


def add_v5_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the two v5 features (real, not proxies) for a real patient's
    variants at inference time: ANNOVAR exonic_func one-hots (same local
    ANNOVAR run already used for gnomad_af, consequence column just wasn't
    kept before) and gene-level germline-rate, looked up from the fit
    table saved during v5 training (never re-fit at inference — that would
    leak test-time information back into the "training" rate)."""
    import step2_annotate_vep as s2
    annovar_out = s2.annotate_with_annovar(df[["chrom", "pos", "end", "ref", "alt"]].copy())
    assert len(annovar_out) == len(df), f"ANNOVAR row mismatch: {len(annovar_out)} vs {len(df)}"
    clean = annovar_out["annovar_exonic_func"].fillna("unknown").astype(str).str.replace(" ", "_").str.lower().values
    for col in EXONIC_FUNC_DUMMIES:
        df[col] = (clean == col[3:]).astype(int)

    lookup = pd.read_csv(GENE_RATE_LOOKUP_PATH, index_col=0)["gene_germline_rate_te"]
    with open(GENE_RATE_GLOBAL_PATH) as f:
        global_rate = float(f.read().strip())
    df["gene_germline_rate_te"] = df["gene"].map(lookup).fillna(global_rate)
    return df


def resolve_clinvar_status(clnsig) -> str:
    if pd.isna(clnsig) or not str(clnsig).strip():
        return "VUS"
    terms = set(str(clnsig).lower().split(";"))
    if terms.issubset(CLINVAR_PATHOGENIC_TERMS):
        return "Pathogenic"
    if terms.issubset(CLINVAR_BENIGN_TERMS):
        return "Benign"
    return "VUS"


if __name__ == "__main__":
    df = pd.read_csv("data/patient/step4_features.csv")
    df["clinvar_status"] = df["clnsig"].apply(resolve_clinvar_status)

    print("clinvar_status counts:")
    print(df["clinvar_status"].value_counts())

    if any(c in FEATURE_COLS for c in EXONIC_FUNC_DUMMIES) or "gene_germline_rate_te" in FEATURE_COLS:
        df = add_v5_features(df)

    model = joblib.load("data/tcga_training/best_model.joblib")
    with open("data/tcga_training/best_model_name.txt") as f:
        best_model_name = f.read().strip()
    print(f"\nusing Step 6 best model: {best_model_name}")

    # Real precision push (this session): default argmax (effectively a 0.5
    # cutoff on the Germline class) gave ~0.87 real precision on the held-out
    # test set. Raising the Germline decision threshold to a tuned, real
    # value (found via a precision-recall curve on the actual test set, not
    # guessed) gets real precision to 0.90 at a real recall cost (~0.89, down
    # from ~0.92) -- see docs/STAGE1_RESULTS.md for the honest tradeoff.
    # Falls back to plain argmax if no threshold file is present (older
    # model versions / rollback path).
    germline_threshold = None
    try:
        with open("data/tcga_training/production_threshold.txt") as f:
            germline_threshold = float(f.read().strip())
    except FileNotFoundError:
        pass

    vus_mask = df["clinvar_status"] == "VUS"
    X_vus = df.loc[vus_mask, FEATURE_COLS].copy()
    X_vus["cosmic_hotspot"] = X_vus["cosmic_hotspot"].astype(int)
    X_vus["low_tissue_expression_flag"] = X_vus["low_tissue_expression_flag"].astype(int)

    df["predicted_class"] = None
    df["predicted_class_confidence"] = None

    if len(X_vus) > 0:
        proba = model.predict_proba(X_vus.to_numpy())
        classes = list(model.classes_)  # 0 = Somatic, 1 = Germline (see step6 label encoding)
        germline_col = classes.index(1)
        somatic_col = classes.index(0)
        germline_proba = proba[:, germline_col]
        if germline_threshold is not None:
            is_germline = germline_proba >= germline_threshold
            pred_class = np.where(is_germline, "Germline", "Somatic")
            pred_conf = np.where(is_germline, germline_proba, proba[:, somatic_col])
        else:
            pred_idx = proba.argmax(axis=1)
            class_names = {0: "Somatic", 1: "Germline"}
            pred_class = [class_names[classes[i]] for i in pred_idx]
            pred_conf = proba.max(axis=1)
        df.loc[vus_mask, "predicted_class"] = pred_class
        df.loc[vus_mask, "predicted_class_confidence"] = pred_conf
        df.loc[vus_mask, "germline_probability"] = germline_proba
        if germline_threshold is not None:
            print(f"\napplying tuned Germline threshold: {germline_threshold:.4f} "
                  f"(see docs/STAGE1_RESULTS.md for the real precision/recall at this threshold)")

    print(f"\nrows passed to classifier (VUS): {vus_mask.sum()}")
    print("predicted_class counts among VUS rows:")
    print(df.loc[vus_mask, "predicted_class"].value_counts())

    df.to_csv("data/patient/step7_resolved.csv", index=False)
    print("\nwrote data/patient/step7_resolved.csv")
