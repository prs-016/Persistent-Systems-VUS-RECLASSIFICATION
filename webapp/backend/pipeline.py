"""
VUS Classifier — inference pipeline glue.

Wraps the project's own Stage 1 (Germline/Somatic/VUS) and Stage 2 (VUS
reclassification-likelihood) scripts into one callable function that takes
an uploaded patient variant file (MAF or VCF/"TCF") and returns a full,
per-variant classification + reclassification-review table.

This module does NOT reimplement the modeling logic — it imports and calls
the real, already-tested scripts in code/scripts/ and loads the real
trained model artifacts in code/data/. If those files are missing (e.g.
this webapp folder was copied somewhere else), it fails loudly rather than
silently guessing.
"""
from __future__ import annotations

import glob
import gzip
import io
import os
import shutil
import sys
import tempfile
import traceback

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path wiring: this file lives at webapp/backend/pipeline.py, with webapp/
# sitting at the top level of the project, next to code/ (not inside it).
# scripts/, config/, data/, annovar/ all live under code/.
# ---------------------------------------------------------------------------
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
WEBAPP_DIR = os.path.dirname(BACKEND_DIR)
PROJECT_ROOT = os.path.dirname(WEBAPP_DIR)
CODE_DIR = os.path.join(PROJECT_ROOT, "code")
SCRIPTS_DIR = os.path.join(CODE_DIR, "scripts")
DATA_DIR = os.path.join(CODE_DIR, "data")
TCGA_DIR = os.path.join(DATA_DIR, "tcga_training")
STAGE2_DIR = os.path.join(DATA_DIR, "stage2")
HPA_PATH = os.path.join(DATA_DIR, "hpa", "proteinatlas.tsv")

for p in (CODE_DIR, SCRIPTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# cd so the imported scripts' relative "config.thresholds" import resolves
_orig_cwd = os.getcwd()
os.chdir(CODE_DIR)
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

# ---------------------------------------------------------------------------
# Variant identity: readable variant type + full HGVS-style name.
#
# ANNOVAR's refGeneWithVer annotation already carries this, per the real
# gene model, not guessed from the raw ref/alt alone: ExonicFunc.refGeneWithVer
# for coding consequence (missense, nonsense, frameshift, ...) and
# AAChange.refGeneWithVer for the transcript-level c./p. notation
# (e.g. "NM_007294.4:exon11:c.5095C>T:p.Arg1699Trp", comma-separated when
# more than one transcript overlaps the position). For a non-exonic variant
# ExonicFunc is empty, so Func.refGeneWithVer (intronic, UTR, splicing, ...)
# is used as the readable type instead.
# ---------------------------------------------------------------------------
EXONIC_FUNC_LABELS = {
    "nonsynonymous_snv": "Missense",
    "synonymous_snv": "Synonymous",
    "stopgain": "Nonsense (stop-gain)",
    "stoploss": "Stop-loss",
    "startloss": "Start-loss",
    "frameshift_deletion": "Frameshift deletion",
    "frameshift_insertion": "Frameshift insertion",
    "frameshift_substitution": "Frameshift substitution",
    "nonframeshift_deletion": "In-frame deletion",
    "nonframeshift_insertion": "In-frame insertion",
    "nonframeshift_substitution": "In-frame substitution",
}

FUNC_REGION_LABELS = {
    "intronic": "Intronic",
    "utr3": "3' UTR",
    "utr5": "5' UTR",
    "splicing": "Splice site",
    "ncrna_exonic": "Non-coding RNA, exonic",
    "ncrna_intronic": "Non-coding RNA, intronic",
    "ncrna_splicing": "Non-coding RNA, splice site",
    "upstream": "Upstream of gene",
    "downstream": "Downstream of gene",
    "intergenic": "Intergenic",
}


def _label_from_map(raw, mapping: dict) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    key = str(raw).strip().lower().replace(" ", "_").replace(";", "_")
    return mapping.get(key)


def _parse_aa_change(raw) -> dict:
    """Take the first transcript's entry out of ANNOVAR's comma-separated
    AAChange.refGeneWithVer string and split it into transcript / coding
    change / protein change. Empty/'.'/NaN (non-exonic variants) yield all
    None rather than raising."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)) or str(raw).strip() in ("", "."):
        return {"transcript": None, "hgvs_c": None, "hgvs_p": None}
    first = str(raw).split(",")[0]
    parts = first.split(":")
    transcript = parts[0] if parts and parts[0] else None
    hgvs_c = next((p for p in parts if p.startswith("c.")), None)
    hgvs_p = next((p for p in parts if p.startswith("p.")), None)
    return {"transcript": transcript, "hgvs_c": hgvs_c, "hgvs_p": hgvs_p}


def add_variant_identity(df: pd.DataFrame) -> pd.DataFrame:
    """Adds variant_type (human-readable consequence), transcript, hgvs_c,
    hgvs_p, and variant_label (a single readable full variant name, e.g.
    "BRCA1 c.5095C>T (p.Arg1699Trp)", falling back to "GENE chr17:41246481
    G>A" when there's no coding change to report, e.g. an intronic variant)
    onto an already-annotated dataframe. Needs annovar_exonic_func,
    annovar_func, and annovar_aa_change from annotate_with_annovar."""
    df = df.copy()

    exonic_type = df.get("annovar_exonic_func", pd.Series([None] * len(df), index=df.index)).apply(
        lambda v: _label_from_map(v, EXONIC_FUNC_LABELS)
    )
    region_type = df.get("annovar_func", pd.Series([None] * len(df), index=df.index)).apply(
        lambda v: _label_from_map(v, FUNC_REGION_LABELS)
    )
    df["variant_type"] = exonic_type.where(exonic_type.notna(), region_type)

    parsed = df.get("annovar_aa_change", pd.Series([None] * len(df), index=df.index)).apply(_parse_aa_change)
    df["transcript"] = parsed.apply(lambda d: d["transcript"])
    df["hgvs_c"] = parsed.apply(lambda d: d["hgvs_c"])
    df["hgvs_p"] = parsed.apply(lambda d: d["hgvs_p"])

    def _label(row) -> str:
        # NaN is truthy in a plain `if x:` check, so pd.notna() guards are
        # needed here rather than `row.get(...) or default` (that would
        # print the literal string "nan" for missing gene/hgvs values).
        gene_val = row.get("gene")
        gene = gene_val if pd.notna(gene_val) and gene_val else "Unknown gene"
        base = f"{row['chrom']}:{row['pos']} {row['ref']}>{row['alt']}"
        hgvs_c_val = row.get("hgvs_c")
        if pd.notna(hgvs_c_val):
            core = hgvs_c_val
            hgvs_p_val = row.get("hgvs_p")
            if pd.notna(hgvs_p_val):
                core = f"{core} ({hgvs_p_val})"
            return f"{gene} {core}"
        return f"{gene} {base}"

    df["variant_label"] = df.apply(_label, axis=1)
    return df


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

    report("Deriving variant type and full variant name from ANNOVAR output...")
    annotated = add_variant_identity(annotated)
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
# Step 7 — ClinVar resolution, then the Stage 1 XGBoost model on every
# variant in the file (Germline vs Somatic origin, production threshold
# 0.9218). ClinVar's Pathogenic/Benign call and Stage 1's origin call are
# independent, a resolved variant gets both.
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

    report("Scoring Stage 1 Germline/Somatic classifier (XGBoost, threshold 0.9218)...")
    # Stage 1's features (see best_model_features.txt) are annotation-derived
    # only, cosmic_hotspot, tissue expression, gnomAD AF, gene germline rate,
    # exonic consequence, none of them reference ClinVar's own call. So the
    # model has an origin opinion on every variant, not just the ones ClinVar
    # left unresolved. Run it on the whole file: for ClinVar-resolved rows
    # this is a second, independent signal alongside ClinVar's own
    # Pathogenic/Benign call, not a replacement for it.
    df["predicted_class"] = None
    df["predicted_class_confidence"] = np.nan
    df["germline_probability"] = np.nan

    if len(df):
        X_all = df[_stage1_feature_cols].copy()
        X_all["cosmic_hotspot"] = X_all["cosmic_hotspot"].astype(int)
        X_all["low_tissue_expression_flag"] = X_all["low_tissue_expression_flag"].astype(int)
        proba = _stage1_model.predict_proba(X_all.to_numpy())
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
        df["predicted_class"] = pred_class
        df["predicted_class_confidence"] = pred_conf
        df["germline_probability"] = germline_proba

    return df


# ---------------------------------------------------------------------------
# Stage 2, path A — an uploaded variant's exact chrom/pos/ref/alt already
# exists in the project's own global ClinVar VUS watchlist. When that
# happens, use those real numbers directly — the reclassification
# probability, the full year-by-year timing profile, and
# direction_pathogenic_probability_if_resolved — instead of the weaker
# generalizable estimate below. Requires both files to be on the same
# genome build (GRCh38, same as Stage 1 throughout).
#
# v18 (vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz) is the current
# watchlist: every VUS in ClinVar as of the last rebuild, ~2.3M rows,
# scored with best_model_v15_realaf across the whole population (the
# earlier v14 file, 121,736 rows, only covered a "core" tracked subset).
# A variant not in even this larger watchlist, e.g. one added to ClinVar
# after the last rebuild, still gets scored by the generalizable model
# below rather than failing, so new/future VUS are always covered by one
# tier or the other. v14 is kept as a fallback candidate in case v18
# hasn't been built yet on a given machine.
#
# v18's build script renamed two columns versus v14 (reclass_probability
# instead of reclass_probability_v12, ClinvarID instead of VariationID);
# _load_watchlist_index() below accepts either name so this keeps working
# across future rebuilds too, not just this one.
# ---------------------------------------------------------------------------
WATCHLIST_CANDIDATES = [
    "vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz",
    "vus_global_watchlist_v14.csv.gz",
]
_watchlist_index = None  # dict[(chrom, pos, ref, alt)] -> row dict, lazy-loaded


def _assemble_watchlist_from_parts(target_path: str) -> bool:
    """A watchlist this large sometimes has to travel to this machine as
    `split`-produced parts (target_path + '.part_aa', '.part_ab', ... —
    concatenated gzip streams decompress fine as a single file). If
    target_path is missing but its parts are present alongside it,
    concatenate them into target_path once, so every later run just finds
    the assembled file and skips this. Returns True if target_path exists,
    or now does, after this call."""
    if os.path.exists(target_path):
        return True
    parts = sorted(glob.glob(target_path + ".part_*"))
    if not parts:
        return False
    tmp_path = target_path + ".assembling"
    with open(tmp_path, "wb") as out_f:
        for part in parts:
            with open(part, "rb") as in_f:
                shutil.copyfileobj(in_f, out_f, length=8 * 1024 * 1024)
    os.replace(tmp_path, target_path)
    return True


def _resolve_watchlist_path() -> str | None:
    for name in WATCHLIST_CANDIDATES:
        candidate = os.path.join(STAGE2_DIR, name)
        if _assemble_watchlist_from_parts(candidate):
            return candidate
    return None


def _load_watchlist_index():
    global _watchlist_index
    if _watchlist_index is not None:
        return _watchlist_index
    path = _resolve_watchlist_path()
    if path is None:
        _watchlist_index = {}
        return _watchlist_index

    # 2.3M rows for v18 versus 121,736 for v14 — noticeably slower than the
    # older file to read and index, but this only happens once per server
    # process (cached in _watchlist_index above), not per request.
    wl = pd.read_csv(path, low_memory=False)
    wl["chrom"] = "chr" + wl["Chromosome"].astype(str)
    wl["pos"] = pd.to_numeric(wl["Start"], errors="coerce")

    rename_map = {}
    if "reclass_probability_v12" not in wl.columns and "reclass_probability" in wl.columns:
        rename_map["reclass_probability"] = "reclass_probability_v12"
    if "VariationID" not in wl.columns and "ClinvarID" in wl.columns:
        rename_map["ClinvarID"] = "VariationID"
    if rename_map:
        wl = wl.rename(columns=rename_map)

    wl = wl.dropna(subset=["pos", "chrom", "ReferenceAllele", "AlternateAllele"])
    wl["pos"] = wl["pos"].astype(int)

    keep = [
        "reclass_probability_v12", "direction_pathogenic_probability_if_resolved",
        "p_resolved_by_1y", "p_resolved_by_2y", "p_resolved_by_3y", "p_resolved_by_4y",
        "p_resolved_by_5y", "p_resolved_by_6y", "p_resolved_by_7y", "p_resolved_by_8y",
        "p_resolved_by_9y", "p_resolved_by_10y", "p_unresolved_after_10y", "VariationID",
        "has_mave_coverage",
    ]
    keep_present = [c for c in keep if c in wl.columns]
    records = wl[keep_present].to_dict("records")
    keys = zip(wl["chrom"], wl["pos"], wl["ReferenceAllele"], wl["AlternateAllele"])
    index = dict(zip(keys, records))
    _watchlist_index = index
    return _watchlist_index


# ---------------------------------------------------------------------------
# Stage 2, path B — the project's "generalizable" models, for variants with
# no ClinVar record at all (the common case for a freshly uploaded patient
# file): a resolution-likelihood score (v2, gene-target-encoding), plus —
# if the corresponding training script has been run — a generalizable
# timing model (Cox, cox_generalizable_model.joblib) and a generalizable
# direction model (generalizable_direction_model.joblib). Both of those are
# newer additions on top of the original v2 score and degrade gracefully:
# if their artifact files aren't present yet (the training scripts haven't
# been run), timing/direction are simply omitted for generalizable-source
# rows rather than the whole pipeline failing.
# ---------------------------------------------------------------------------
_stage2_model = None
_stage2_feature_cols = None
_gene_resolved_lookup = None
_gene_resolved_global = None
_mave_lookup = None

_cox_model = None
_cox_features = None
_cox_observed_max = None

_direction_model = None
_direction_features = None
_gene_pathogenic_lookup = None
_gene_pathogenic_global = None

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


def _load_generalizable_timing():
    """Optional: only available once stage2_step17_generalizable_timing.py
    has been run AND lifelines is installed (needed to unpickle the fitted
    CoxPHFitter). Returns False (and leaves timing columns unset) if either
    is missing, rather than failing the whole run."""
    global _cox_model, _cox_features, _cox_observed_max
    if _cox_model is not None:
        return True
    model_path = os.path.join(STAGE2_DIR, "cox_generalizable_model.joblib")
    if not os.path.exists(model_path):
        return False
    try:
        _cox_model = joblib.load(model_path)
    except ModuleNotFoundError:
        return False
    with open(os.path.join(STAGE2_DIR, "cox_generalizable_features.txt")) as f:
        _cox_features = f.read().strip().split(",")
    with open(os.path.join(STAGE2_DIR, "cox_generalizable_observed_range.txt")) as f:
        _, max_observed = f.read().strip().split(",")
        _cox_observed_max = float(max_observed)
    return True


def _load_generalizable_direction():
    """Optional: only available once stage2_step18_generalizable_direction.py
    has been run. Returns False (and leaves direction unset) if not."""
    global _direction_model, _direction_features, _gene_pathogenic_lookup, _gene_pathogenic_global
    if _direction_model is not None:
        return True
    model_path = os.path.join(STAGE2_DIR, "generalizable_direction_model.joblib")
    if not os.path.exists(model_path):
        return False
    _direction_model = joblib.load(model_path)
    with open(os.path.join(STAGE2_DIR, "generalizable_direction_features.txt")) as f:
        _direction_features = f.read().strip().split(",")
    lookup_path = os.path.join(STAGE2_DIR, "gene_pathogenic_rate_lookup_v1.csv")
    _gene_pathogenic_lookup = pd.read_csv(lookup_path, index_col=0)["gene_pathogenic_rate_te"]
    with open(os.path.join(STAGE2_DIR, "gene_pathogenic_rate_global_v1.txt")) as f:
        _gene_pathogenic_global = float(f.read().strip())
    return True


# Real, already-validated probability-value tiers from this project's own
# global-watchlist work (stage2_step15/16): P(resolved within 10y) >= 0.15
# is "high_priority", 0.10-0.15 is "elevated". A watchlist-matched variant's
# stage2_score (reclass_probability_v12, "will this ever resolve") lives on
# a different scale than the generalizable model's score, so it gets its
# own band logic keyed on the real 10-year timing probability instead of
# reusing the generalizable model's 0.60/baseline cutoffs.
WATCHLIST_HIGH_PRIORITY = 0.15
WATCHLIST_ELEVATED = 0.10


def _band(score, source):
    if pd.isna(score):
        return None
    if source == "clinvar_v12":
        return None  # banded via _watchlist_band on p_resolved_by_10y instead
    if score >= WATCH_CLOSELY:
        return "Watch closely"
    if abs(score - MODEST_SIGNAL_FLOOR) < 1e-6:
        return "No distinguishing signal"
    if score >= MODEST_SIGNAL_FLOOR:
        return "Modest signal"
    return "Below baseline"


def _watchlist_band(p10y):
    if pd.isna(p10y):
        return None
    if p10y >= WATCHLIST_HIGH_PRIORITY:
        return "Watch closely"
    if p10y >= WATCHLIST_ELEVATED:
        return "Modest signal"
    return "Below baseline"


def score_stage2(df: pd.DataFrame, progress=None) -> pd.DataFrame:
    def report(msg):
        if progress:
            progress(msg)

    _load_stage2_artifacts()
    watchlist = _load_watchlist_index()
    have_cox = _load_generalizable_timing()
    have_direction = _load_generalizable_direction()

    df = df.copy()
    df["stage2_score"] = np.nan
    df["stage2_source"] = None
    df["stage2_band"] = None
    df["has_mave_coverage"] = 0
    df["direction_pathogenic_probability"] = np.nan
    df["p_resolved_by_10y"] = np.nan
    df["reclassification_flag"] = False

    vus_mask = df["clinvar_status"] == "VUS"
    if not vus_mask.any():
        return df

    report("Checking each VUS against the global ClinVar watchlist...")
    n_matched = 0
    for idx in df.index[vus_mask]:
        key = (df.at[idx, "chrom"], int(df.at[idx, "pos"]), df.at[idx, "ref"], df.at[idx, "alt"])
        hit = watchlist.get(key)
        if hit is None:
            continue
        n_matched += 1
        df.at[idx, "stage2_score"] = hit["reclass_probability_v12"]
        df.at[idx, "stage2_source"] = "clinvar_v12"
        df.at[idx, "direction_pathogenic_probability"] = hit["direction_pathogenic_probability_if_resolved"]
        df.at[idx, "p_resolved_by_10y"] = hit["p_resolved_by_10y"]
        df.at[idx, "has_mave_coverage"] = hit["has_mave_coverage"]
    if n_matched:
        report(f"{n_matched} variant(s) matched the ClinVar watchlist and use the stronger v12 model directly.")

    remaining_mask = vus_mask & df["stage2_source"].isna()
    if remaining_mask.any():
        report("Scoring remaining VUS for reclassification likelihood (Stage 2 generalizable model)...")
        sub = df.loc[remaining_mask, ["gene", "gnomad_af", "low_tissue_expression_flag"]].copy()
        sub = sub.merge(_mave_lookup, on="gene", how="left")
        sub["has_mave_coverage"] = sub["has_mave_coverage"].fillna(0).astype(int)
        sub["gnomad_af"] = pd.to_numeric(sub["gnomad_af"], errors="coerce").fillna(0.0)
        sub["low_tissue_expression_flag"] = sub["low_tissue_expression_flag"].astype(bool).astype(int)
        sub["gene_resolved_rate_te"] = sub["gene"].map(_gene_resolved_lookup).fillna(_gene_resolved_global)
        sub.index = df.index[remaining_mask]

        X = sub[_stage2_feature_cols]
        scores = _stage2_model.predict_proba(X)[:, 1]
        df.loc[remaining_mask, "stage2_score"] = scores
        df.loc[remaining_mask, "stage2_source"] = "generalizable"
        df.loc[remaining_mask, "has_mave_coverage"] = sub["has_mave_coverage"].values

        if have_direction:
            report("Scoring reclassification direction (generalizable model)...")
            sub["gene_pathogenic_rate_te"] = sub["gene"].map(_gene_pathogenic_lookup).fillna(_gene_pathogenic_global)
            Xd = sub[_direction_features]
            df.loc[remaining_mask, "direction_pathogenic_probability"] = _direction_model.predict_proba(Xd)[:, 1]

        if have_cox:
            report("Estimating time to reclassification (generalizable timing model)...")
            sub["gene_resolved_rate_te"] = sub["gene_resolved_rate_te"]  # already computed above
            Xc = sub[_cox_features]
            ph = _cox_model.predict_partial_hazard(Xc).values
            base_times = _cox_model.baseline_survival_.index.values
            base_s0 = _cox_model.baseline_survival_.iloc[:, 0].values
            i = np.searchsorted(base_times, 10.0, side="right") - 1
            i = np.clip(i, 0, len(base_s0) - 1)
            s0_10 = base_s0[i]
            df.loc[remaining_mask, "p_resolved_by_10y"] = 1 - (s0_10 ** ph)
    else:
        # every VUS matched the watchlist directly; nothing left to score generalizably
        pass

    df.loc[vus_mask, "has_mave_coverage"] = df.loc[vus_mask, "has_mave_coverage"].fillna(0)

    watchlist_rows = vus_mask & (df["stage2_source"] == "clinvar_v12")
    generalizable_rows = vus_mask & (df["stage2_source"] == "generalizable")
    df.loc[watchlist_rows, "stage2_band"] = df.loc[watchlist_rows, "p_resolved_by_10y"].apply(_watchlist_band)
    df.loc[generalizable_rows, "stage2_band"] = df.loc[generalizable_rows, "stage2_score"].apply(
        lambda s: _band(s, "generalizable")
    )

    df.loc[watchlist_rows, "reclassification_flag"] = (
        df.loc[watchlist_rows, "p_resolved_by_10y"] >= WATCHLIST_HIGH_PRIORITY
    )
    df.loc[generalizable_rows, "reclassification_flag"] = (
        df.loc[generalizable_rows, "stage2_score"] >= WATCH_CLOSELY
    )
    return df


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------
FINAL_COLS = [
    "chrom", "pos", "ref", "alt", "gene", "vaf", "gnomad_af", "cosmic_hotspot",
    "dbsnp_id", "hpa_expression_level", "low_tissue_expression_flag",
    "clinvar_status", "predicted_class", "predicted_class_confidence",
    "germline_probability", "has_mave_coverage", "stage2_score", "stage2_source",
    "stage2_band", "direction_pathogenic_probability", "p_resolved_by_10y",
    "reclassification_flag", "variant_type", "transcript", "hgvs_c", "hgvs_p",
    "variant_label",
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
    n_watchlist_matches = int((vus_rows.get("stage2_source") == "clinvar_v12").sum()) if "stage2_source" in vus_rows else 0
    # Stage 1 now scores every row, not just VUS (see classify_stage1), so
    # these totals cover the whole file: ClinVar-resolved variants get an
    # independent origin call alongside ClinVar's own Pathogenic/Benign one.
    n_germline_total = int((out["predicted_class"] == "Germline").sum())
    n_somatic_total = int((out["predicted_class"] == "Somatic").sum())

    summary = {
        "total_variants": n_total,
        "resolved_pathogenic": n_pathogenic,
        "resolved_benign": n_benign,
        "vus_count": n_vus,
        "vus_predicted_germline": n_germline,
        "vus_predicted_somatic": n_somatic,
        "predicted_germline_total": n_germline_total,
        "predicted_somatic_total": n_somatic_total,
        "flagged_for_reclassification_review": n_flagged,
        "vus_matched_clinvar_watchlist": n_watchlist_matches,
    }

    records = out.replace({np.nan: None}).to_dict(orient="records")
    return {"summary": summary, "variants": records}
