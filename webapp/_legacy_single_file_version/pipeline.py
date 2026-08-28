"""
VUS Classifier — inference pipeline glue.

Wraps the project's own Stage 1 (Germline/Somatic/VUS) and Stage 2 (VUS
reclassification-likelihood) scripts into one callable function that takes
an uploaded patient variant file (MAF or VCF/"TCF") and returns a full,
per-variant classification + reclassification-review table.

This module does NOT reimplement the modeling logic — it imports and calls
the real, already-tested scripts in stage1/scripts/ and loads the real
trained model artifacts in stage1/data/. If those files are missing (e.g.
this webapp folder was copied somewhere else), it fails loudly rather than
silently guessing.
"""
from __future__ import annotations

import gzip
import io
import os
import sys
import tempfile
import traceback

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring: this file lives at stage1/webapp/pipeline.py. Everything else
# (scripts/, config/, data/, annovar/) is a sibling of webapp/'s parent.
# ---------------------------------------------------------------------------
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE1_DIR = os.path.dirname(WEBAPP_DIR)
SCRIPTS_DIR = os.path.join(STAGE1_DIR, "scripts")
DATA_DIR = os.path.join(STAGE1_DIR, "data")
TCGA_DIR = os.path.join(DATA_DIR, "tcga_training")
STAGE2_DIR = os.path.join(DATA_DIR, "stage2")
HPA_PATH = os.path.join(DATA_DIR, "hpa", "proteinatlas.tsv")

for p in (STAGE1_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# cd so the imported scripts' relative "config.thresholds" import resolves
_orig_cwd = os.getcwd()
os.chdir(STAGE1_DIR)
try:
    import step2_annotate_vep as s2  # noqa: E402
    from config.thresholds import (  # noqa: E402
        CLINVAR_PATHOGENIC_TERMS,
        CLINVAR_BENIGN_TERMS,
        HPA_LOW_EXPRESSION_CATEGORIES,
        HPA_RELIABLE_CATEGORIES,
    )
finally:
    os.chdir(_orig_cwd)

EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]

CHROM_ALLOWED = {f"chr{i}" for i in range(1, 23)} | {"chr1", "chrX", "chrY", "chrM"}


class PipelineError(Exception):
    pass


# ---------------------------------------------------------------------------
# Step 1 — parse the uploaded file (MAF, gzipped MAF, or VCF/"TCF") into the
# working table: chrom, pos, end, ref, alt, gene(optional), t_ref_count,
# t_alt_count, t_depth.
# ---------------------------------------------------------------------------
def _read_text(raw_bytes: bytes) -> str:
    if raw_bytes[:2] == b"\x1f\x8b":
        return gzip.decompress(raw_bytes).decode("utf-8", errors="replace")
    return raw_bytes.decode("utf-8", errors="replace")


def parse_uploaded_variants(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    text = _read_text(raw_bytes)
    lower_name = filename.lower()

    if "vcf" in lower_name or text.lstrip().startswith("##fileformat=VCF"):
        return _parse_vcf(text)
    return _parse_maf(text)


def _parse_maf(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    body = [l for l in lines if l.strip() and not l.startswith("#")]
    if not body:
        raise PipelineError("No data rows found in the uploaded MAF file.")
    header = body[0].split("\t")
    rows = [l.split("\t") for l in body[1:]]
    raw = pd.DataFrame(rows, columns=header)

    def col(*names, default=None):
        for n in names:
            if n in raw.columns:
                return raw[n]
        if default is not None:
            return pd.Series([default] * len(raw))
        raise PipelineError(f"Uploaded MAF is missing required column(s): {names}")

    df = pd.DataFrame({
        "chrom": col("Chromosome"),
        "pos": pd.to_numeric(col("Start_Position"), errors="coerce"),
        "end": pd.to_numeric(col("End_Position", default=None), errors="coerce"),
        "ref": col("Reference_Allele"),
        "alt": col("Tumor_Seq_Allele2", "Tumor_Seq_Allele1"),
        "gene": col("Hugo_Symbol", default="Unknown"),
        "t_depth": pd.to_numeric(col("t_depth", default=0), errors="coerce").fillna(0).astype(int),
        "t_ref_count": pd.to_numeric(col("t_ref_count", default=0), errors="coerce").fillna(0).astype(int),
        "t_alt_count": pd.to_numeric(col("t_alt_count", default=0), errors="coerce").fillna(0).astype(int),
    })
    df["end"] = df["end"].fillna(df["pos"])
    df = df.dropna(subset=["pos", "ref", "alt"])
    df["pos"] = df["pos"].astype(int)
    df["end"] = df["end"].astype(int)
    df["chrom"] = df["chrom"].astype(str).apply(lambda c: c if c.startswith("chr") else f"chr{c}")
    return df.reset_index(drop=True)


def _parse_vcf(text: str) -> pd.DataFrame:
    """Best-effort VCF/"TCF" parser: pulls chrom/pos/ref/alt plus AD (allele
    depth) from the first sample's FORMAT field when present, for VAF."""
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        chrom, pos, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
        alt_first = alt.split(",")[0]
        ref_count = alt_count = 0
        if len(parts) >= 10:
            fmt_keys = parts[8].split(":")
            sample_vals = parts[9].split(":")
            fmt = dict(zip(fmt_keys, sample_vals))
            if "AD" in fmt:
                ad_parts = fmt["AD"].split(",")
                if len(ad_parts) >= 2:
                    try:
                        ref_count, alt_count = int(ad_parts[0]), int(ad_parts[1])
                    except ValueError:
                        pass
        rows.append({
            "chrom": chrom if str(chrom).startswith("chr") else f"chr{chrom}",
            "pos": int(pos),
            "end": int(pos) + max(len(ref) - 1, 0),
            "ref": ref,
            "alt": alt_first,
            "gene": "Unknown",
            "t_depth": ref_count + alt_count,
            "t_ref_count": ref_count,
            "t_alt_count": alt_count,
        })
    if not rows:
        raise PipelineError("No variant records found in the uploaded VCF/TCF file.")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Steps 2-4 — VEP + ANNOVAR annotation, HPA lookup, feature computation.
# Reuses the project's own scripts so results match the documented pipeline.
# ---------------------------------------------------------------------------
def annotate_and_featurize(df: pd.DataFrame, tissue: str, progress=None) -> pd.DataFrame:
    def report(msg):
        if progress:
            progress(msg)

    report("Querying Ensembl VEP (ClinVar / gnomAD / COSMIC / dbSNP)...")
    vep_df = s2.annotate(df)

    report("Running local ANNOVAR (gene consequence, ClinVar, gnomAD)...")
    annovar_df = s2.annotate_with_annovar(df)
    annotated = pd.concat([vep_df.reset_index(drop=True), annovar_df.reset_index(drop=True)], axis=1)

    report("Looking up tissue expression breadth (Human Protein Atlas)...")
    hpa_lookup = _load_hpa_lookup()
    levels, reliabilities = [], []
    for gene in annotated["gene"]:
        hit = hpa_lookup.get(gene, {"hpa_expression_level": "Unknown", "hpa_reliability": "Unknown"})
        levels.append(hit["hpa_expression_level"])
        reliabilities.append(hit["hpa_reliability"])
    annotated["hpa_expression_level"] = levels
    annotated["hpa_reliability"] = reliabilities
    annotated["hpa_query_tissue"] = tissue
    annotated["low_tissue_expression_flag"] = (
        annotated["hpa_expression_level"].isin(HPA_LOW_EXPRESSION_CATEGORIES)
        & annotated["hpa_reliability"].isin(HPA_RELIABLE_CATEGORIES)
    )

    report("Computing classification features...")
    annotated["vaf"] = annotated["t_alt_count"] / (annotated["t_alt_count"] + annotated["t_ref_count"]).replace(0, np.nan)
    annotated["vaf"] = annotated["vaf"].fillna(0.0)
    annovar_af = annotated["annovar_gnomad_af"] if "annovar_gnomad_af" in annotated.columns else pd.Series([None] * len(annotated))
    vep_af = annotated["gnomad_af"] if "gnomad_af" in annotated.columns else pd.Series([None] * len(annotated))
    annotated["gnomad_af_model"] = pd.to_numeric(annovar_af, errors="coerce").fillna(0.0)
    annotated["gnomad_af"] = pd.to_numeric(vep_af, errors="coerce").fillna(0.0)
    annotated["cosmic_hotspot"] = annotated["cosmic_hotspot"].astype(bool)
    annotated["low_tissue_expression_flag"] = annotated["low_tissue_expression_flag"].astype(bool)
    return annotated


_hpa_cache = None


def _load_hpa_lookup() -> dict:
    global _hpa_cache
    if _hpa_cache is not None:
        return _hpa_cache
    if not os.path.exists(HPA_PATH):
        raise PipelineError(f"HPA reference file not found at {HPA_PATH}")
    lookup = {}
    with open(HPA_PATH, newline="", encoding="utf-8") as f:
        import csv
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lookup[row["Gene"]] = {
                "hpa_expression_level": row["RNA tissue distribution"] or "Unknown",
                "hpa_reliability": row["Reliability (IH)"] or "Unknown",
            }
    _hpa_cache = lookup
    return lookup


# ---------------------------------------------------------------------------
# Step 7 — ClinVar resolution, then the Stage 1 XGBoost model on whatever's
# left as VUS (Germline vs Somatic origin, production threshold 0.9218).
# ---------------------------------------------------------------------------
_stage1_model = None
_stage1_feature_cols = None
_gene_germline_lookup = None
_gene_germline_global = None
_germline_threshold = None


def _load_stage1_artifacts():
    global _stage1_model, _stage1_feature_cols, _gene_germline_lookup, _gene_germline_global, _germline_threshold
    if _stage1_model is not None:
        return
    with open(os.path.join(TCGA_DIR, "best_model_features.txt")) as f:
        saved = f.read().strip().split(",")
    _stage1_feature_cols = ["gnomad_af_model" if c == "gnomad_af" else c for c in saved]
    _stage1_model = joblib.load(os.path.join(TCGA_DIR, "best_model.joblib"))
    lookup_path = os.path.join(TCGA_DIR, "gene_germline_rate_lookup_v5.csv")
    _gene_germline_lookup = pd.read_csv(lookup_path, index_col=0)["gene_germline_rate_te"]
    with open(os.path.join(TCGA_DIR, "gene_germline_rate_global_v5.txt")) as f:
        _gene_germline_global = float(f.read().strip())
    try:
        with open(os.path.join(TCGA_DIR, "production_threshold.txt")) as f:
            _germline_threshold = float(f.read().strip())
    except FileNotFoundError:
        _germline_threshold = None


def resolve_clinvar_status(clnsig) -> str:
    if pd.isna(clnsig) or not str(clnsig).strip():
        return "VUS"
    terms = set(str(clnsig).lower().split(";"))
    if terms.issubset(CLINVAR_PATHOGENIC_TERMS):
        return "Pathogenic"
    if terms.issubset(CLINVAR_BENIGN_TERMS):
        return "Benign"
    return "VUS"


def classify_stage1(df: pd.DataFrame, progress=None) -> pd.DataFrame:
    def report(msg):
        if progress:
            progress(msg)

    _load_stage1_artifacts()
    df = df.copy()
    df["clinvar_status"] = df["clnsig"].apply(resolve_clinvar_status)

    report("Deriving exonic-consequence features from ANNOVAR output...")
    clean = (
        df.get("annovar_exonic_func", pd.Series(["unknown"] * len(df)))
        .fillna("unknown").astype(str).str.replace(" ", "_").str.lower().values
    )
    for col in EXONIC_FUNC_DUMMIES:
        df[col] = (clean == col[3:]).astype(int)
    df["gene_germline_rate_te"] = df["gene"].map(_gene_germline_lookup).fillna(_gene_germline_global)

    report("Scoring VUS with the Stage 1 Germline/Somatic classifier (XGBoost, threshold 0.9218)...")
    vus_mask = df["clinvar_status"] == "VUS"
    df["predicted_class"] = None
    df["predicted_class_confidence"] = np.nan
    df["germline_probability"] = np.nan

    if vus_mask.any():
        X_vus = df.loc[vus_mask, _stage1_feature_cols].copy()
        X_vus["cosmic_hotspot"] = X_vus["cosmic_hotspot"].astype(int)
        X_vus["low_tissue_expression_flag"] = X_vus["low_tissue_expression_flag"].astype(int)
        proba = _stage1_model.predict_proba(X_vus.to_numpy())
        classes = list(_stage1_model.classes_)
        germline_col = classes.index(1)
        somatic_col = classes.index(0)
        germline_proba = proba[:, germline_col]
        if _germline_threshold is not None:
            is_germline = germline_proba >= _germline_threshold
            pred_class = np.where(is_germline, "Germline", "Somatic")
            pred_conf = np.where(is_germline, germline_proba, proba[:, somatic_col])
        else:
            pred_idx = proba.argmax(axis=1)
            names = {0: "Somatic", 1: "Germline"}
            pred_class = [names[classes[i]] for i in pred_idx]
            pred_conf = proba.max(axis=1)
        df.loc[vus_mask, "predicted_class"] = pred_class
        df.loc[vus_mask, "predicted_class_confidence"] = pred_conf
        df.loc[vus_mask, "germline_probability"] = germline_proba

    return df


# ---------------------------------------------------------------------------
# Stage 2 — "generalizable" v2 model: scores VUS with no ClinVar submission
# history (i.e. every variant coming out of a fresh upload) for how similar
# they are, feature-wise, to VUS that have historically been reclassified.
# ---------------------------------------------------------------------------
_stage2_model = None
_stage2_feature_cols = None
_gene_resolved_lookup = None
_gene_resolved_global = None
_mave_lookup = None

WATCH_CLOSELY = 0.60
MODEST_SIGNAL_FLOOR = 0.478418  # the v2 generalizable model's flat baseline score


def _load_stage2_artifacts():
    global _stage2_model, _stage2_feature_cols, _gene_resolved_lookup, _gene_resolved_global, _mave_lookup
    if _stage2_model is not None:
        return
    with open(os.path.join(STAGE2_DIR, "generalizable_model_features_v2.txt")) as f:
        _stage2_feature_cols = f.read().strip().split(",")
    _stage2_model = joblib.load(os.path.join(STAGE2_DIR, "generalizable_model_v2.joblib"))
    lookup_path = os.path.join(STAGE2_DIR, "gene_target_encoding_lookup_generalizable.csv")
    _gene_resolved_lookup = pd.read_csv(lookup_path, index_col=0)["gene_resolved_rate_te"]
    with open(os.path.join(STAGE2_DIR, "gene_target_encoding_global_rate_generalizable.txt")) as f:
        _gene_resolved_global = float(f.read().strip())
    mave_path = os.path.join(STAGE2_DIR, "mavedb_gene_coverage_v2.csv")
    if not os.path.exists(mave_path):
        mave_path = os.path.join(STAGE2_DIR, "mavedb_gene_coverage.csv")
    _mave_lookup = pd.read_csv(mave_path)[["gene", "has_mave_coverage"]]


def score_stage2(df: pd.DataFrame, progress=None) -> pd.DataFrame:
    def report(msg):
        if progress:
            progress(msg)

    _load_stage2_artifacts()
    df = df.copy()
    df["stage2_score"] = np.nan
    df["stage2_band"] = None
    df["has_mave_coverage"] = 0
    df["reclassification_flag"] = False

    vus_mask = df["clinvar_status"] == "VUS"
    if not vus_mask.any():
        return df

    report("Scoring VUS for reclassification likelihood (Stage 2 generalizable model)...")
    sub = df.loc[vus_mask, ["gene", "gnomad_af", "low_tissue_expression_flag"]].copy()
    sub = sub.merge(_mave_lookup, on="gene", how="left")
    sub["has_mave_coverage"] = sub["has_mave_coverage"].fillna(0).astype(int)
    sub["gnomad_af"] = pd.to_numeric(sub["gnomad_af"], errors="coerce").fillna(0.0)
    sub["low_tissue_expression_flag"] = sub["low_tissue_expression_flag"].astype(bool).astype(int)
    sub["gene_resolved_rate_te"] = sub["gene"].map(_gene_resolved_lookup).fillna(_gene_resolved_global)
    sub.index = df.index[vus_mask]

    X = sub[_stage2_feature_cols]
    scores = _stage2_model.predict_proba(X)[:, 1]
    df.loc[vus_mask, "stage2_score"] = scores
    df.loc[vus_mask, "has_mave_coverage"] = sub["has_mave_coverage"].values

    def band(score):
        if pd.isna(score):
            return None
        if score >= WATCH_CLOSELY:
            return "Watch closely"
        if score >= MODEST_SIGNAL_FLOOR:
            return "Modest signal"
        if abs(score - MODEST_SIGNAL_FLOOR) < 1e-6:
            return "No distinguishing signal"
        return "Below baseline"

    df.loc[vus_mask, "stage2_band"] = df.loc[vus_mask, "stage2_score"].apply(band)
    df.loc[vus_mask, "reclassification_flag"] = df.loc[vus_mask, "stage2_score"] >= WATCH_CLOSELY
    return df


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
FINAL_COLS = [
    "chrom", "pos", "ref", "alt", "gene", "vaf", "gnomad_af", "cosmic_hotspot",
    "dbsnp_id", "hpa_expression_level", "low_tissue_expression_flag",
    "clinvar_status", "predicted_class", "predicted_class_confidence",
    "germline_probability", "has_mave_coverage", "stage2_score", "stage2_band",
    "reclassification_flag",
]


def run_pipeline(filename: str, raw_bytes: bytes, tissue: str = "breast", progress=None) -> dict:
    def report(msg):
        if progress:
            progress(msg)

    report("Parsing uploaded file...")
    df = parse_uploaded_variants(filename, raw_bytes)
    if len(df) == 0:
        raise PipelineError("No variants could be parsed from the uploaded file.")

    annotated = annotate_and_featurize(df, tissue, progress=progress)
    resolved = classify_stage1(annotated, progress=progress)
    scored = score_stage2(resolved, progress=progress)

    out = scored[[c for c in FINAL_COLS if c in scored.columns]].copy()
    out = out.sort_values(
        by=["reclassification_flag", "stage2_score"], ascending=[False, False], na_position="last"
    )

    n_total = len(out)
    n_pathogenic = int((out["clinvar_status"] == "Pathogenic").sum())
    n_benign = int((out["clinvar_status"] == "Benign").sum())
    n_vus = int((out["clinvar_status"] == "VUS").sum())
    vus_rows = out[out["clinvar_status"] == "VUS"]
    n_germline = int((vus_rows["predicted_class"] == "Germline").sum())
    n_somatic = int((vus_rows["predicted_class"] == "Somatic").sum())
    n_flagged = int(out["reclassification_flag"].sum())

    summary = {
        "total_variants": n_total,
        "resolved_pathogenic": n_pathogenic,
        "resolved_benign": n_benign,
        "vus_count": n_vus,
        "vus_predicted_germline": n_germline,
        "vus_predicted_somatic": n_somatic,
        "flagged_for_reclassification_review": n_flagged,
    }

    records = out.replace({np.nan: None}).to_dict(orient="records")
    return {"summary": summary, "variants": records}
