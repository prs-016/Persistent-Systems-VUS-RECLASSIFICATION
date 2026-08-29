"""
VUS Classifier — inference pipeline glue, memory-lean variant for hosted
deployment (Render, AWS free-tier EC2, anywhere RAM is the scarce resource).

This is a sibling of pipeline.py, not a patch to it. pipeline.py stays
exactly as it is for local runs, where a laptop's 16-64GB of RAM makes
"just load the whole 2.3M-row watchlist into a Python dict at startup"
a completely reasonable thing to do. On a 1GB box it isn't: that dict
alone is comfortably the single biggest thing living in this process's
memory, easily several hundred MB once pandas and Python's own per-object
overhead are counted, and it's held for the lifetime of the server even
though any one upload only ever touches a few hundred rows of it.

The fix here isn't a new idea, it's the same one ANNOVAR itself already
uses for the 2.2GB gnomAD file: don't hold the whole table in memory,
build a small on-disk index once and look rows up by key as you need
them. SQLite is the obvious tool for that job, it ships with Python,
needs no server process of its own, and a `chrom, pos, ref, alt` lookup
against an indexed table of a couple million rows comes back in well
under a millisecond.

Two tables get this treatment: the ClinVar watchlist (the big one, ~2.3M
rows) and the HPA tissue-expression lookup (small by comparison, ~20k
genes, but there's no reason to keep it fully resident either when the
same trick works just as well). Everything else in this file, parsing,
VEP/ANNOVAR annotation, the Stage 1 classifier, the Stage 2 generalizable
models, is identical in behavior to pipeline.py; only how the two lookup
tables are stored changed.

Same contract as pipeline.py: this module doesn't reimplement any modeling
logic, it imports and calls the real scripts in code/scripts/ and loads
the real trained artifacts in code/data/. If those files are missing, it
fails loudly instead of guessing.
"""
from __future__ import annotations

import glob
import gzip
import io
import os

# Cap every BLAS/thread pool numpy, xgboost, and scikit-learn might spin up to
# a single thread each. On Render's free tier we get 0.1 CPU, so multithreading
# buys nothing but contention anyway, and each extra worker thread pins its own
# stack + scratch buffers, which adds up fast on a 512MB box. Has to happen
# before numpy/pandas/xgboost get imported below, since that's when the
# underlying BLAS libraries read these and lock in their thread count.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# The gnomad211_exome ANNOVAR database is a 2.2GB text file, and ANNOVAR's
# filter-based annotation loads it into a Perl hash regardless of how few
# variants are being checked against it -- on a 512MB box that alone is
# enough to get the whole container OOM-killed, independent of input file
# size. We already get gnomAD allele frequency from VEP's REST API in the
# same run (see the gnomad_af fallback in annotate_and_featurize below), so
# skip asking ANNOVAR to redo it. Only step2_annotate_vep.py's *defaults*
# get overridden here (it reads these via os.environ.get with the original
# values as fallback), so pipeline.py running locally is unaffected.
# clinvar_20221231 (363MB) also dropped: annovar_clnsig (the column it
# produces) is never actually read anywhere downstream -- the pipeline's
# real ClinVar significance field, clnsig, comes from VEP's REST call in
# step2_annotate_vep.py's annotate(), not from ANNOVAR. So this filter step
# was pure memory cost with zero functional payoff on the deployed path.
# Only the refGeneWithVer gene-based operation remains, since
# annovar_exonic_func (used in feature engineering) comes from it.
os.environ.setdefault("ANNOVAR_PROTOCOL", "refGeneWithVer")
os.environ.setdefault("ANNOVAR_OPERATION", "g")
# Point ANNOVAR at a pruned humandb copy (see step2_annotate_vep.py) instead
# of the full genome-wide one: measured 433MB -> 154MB peak RSS for the same
# gene-based annotation, which combined with baseline Python/uvicorn usage
# is the difference between OOM-killing this 512MB container and not.
os.environ.setdefault("ANNOVAR_HUMANDB_SUBDIR", "humandb_lite")
import resource
import sqlite3


def _log_mem(tag: str) -> None:
    """Prints this process's peak resident memory so far to stdout, which
    Render (and basically every other host) shows in its plain log tail for
    free, no paid metrics dashboard needed. ru_maxrss is already a running
    high-water mark, not a snapshot, so each of these lines is "the worst
    it's been up to this point," which is exactly what matters for figuring
    out whether a run is going to get OOM-killed. Linux reports this in KB;
    macOS reports bytes, hence the platform check. Temporary instrumentation
    for sizing the deployment, safe to strip out once we know the number.
    """
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        peak_kb //= 1024
    print(f"[mem] {tag}: peak RSS so far = {peak_kb / 1024:.1f} MB", flush=True)


import sys
import tempfile
import traceback

_log_mem("module import start")

import joblib
import numpy as np
import pandas as pd

_log_mem("after joblib/numpy/pandas import")

# ---------------------------------------------------------------------------
# Path wiring — same layout pipeline.py uses. This file also lives at
# webapp/backend/, with webapp/ sitting next to code/ (not inside it).
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
FINAL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "final_output_csv")

# Where the two SQLite indexes get built and cached. Kept out of code/data/
# and code/data/stage2 on purpose, those are "real" project data (or
# git-ignored raw downloads); this folder holds nothing but derived,
# regenerate-any-time cache files, so it's easy to tell at a glance what's
# safe to delete if disk space is ever tight.
CACHE_DIR = os.path.join(DATA_DIR, "runtime_cache")
WATCHLIST_DB_PATH = os.path.join(CACHE_DIR, "watchlist_index.sqlite3")
HPA_DB_PATH = os.path.join(CACHE_DIR, "hpa_index.sqlite3")

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
# Unchanged from pipeline.py — this bit was never a memory concern.
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
    """Adds variant_type, transcript, hgvs_c, hgvs_p, and variant_label onto
    an already-annotated dataframe. Same logic as pipeline.py, verbatim."""
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
# Step 1 — parse the uploaded file. Identical to pipeline.py.
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
# HPA tissue-expression lookup, SQLite-backed.
#
# proteinatlas.tsv is only ~20k gene rows, small enough that pipeline.py's
# plain-dict version was never really the memory problem. It gets the same
# treatment here anyway: it's cheap to do, it's one less thing resident for
# the life of the process, and it means this file doesn't quietly regress
# back to "just load it all" the next time someone's copying patterns from
# it. Built once, on first use, into a tiny SQLite file next to the other
# cache.
# ---------------------------------------------------------------------------
def _ensure_hpa_db() -> None:
    if os.path.exists(HPA_DB_PATH):
        return
    if not os.path.exists(HPA_PATH):
        raise PipelineError(f"HPA reference file not found at {HPA_PATH}")

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = HPA_DB_PATH + f".building-{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute(
            "CREATE TABLE hpa (gene TEXT PRIMARY KEY, expression_level TEXT, reliability TEXT)"
        )
        import csv
        with open(HPA_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = (
                (row["Gene"], row["RNA tissue distribution"] or "Unknown", row["Reliability (IH)"] or "Unknown")
                for row in reader
            )
            conn.executemany("INSERT OR REPLACE INTO hpa VALUES (?, ?, ?)", rows)
        conn.commit()
    finally:
        conn.close()

    # Atomic rename so a request that crashes mid-build never leaves a
    # half-written file behind for the next one to trip over.
    os.replace(tmp_path, HPA_DB_PATH)


_hpa_conn = None


def _hpa_connection() -> sqlite3.Connection:
    global _hpa_conn
    if _hpa_conn is None:
        _ensure_hpa_db()
        _hpa_conn = sqlite3.connect(HPA_DB_PATH, check_same_thread=False)
    return _hpa_conn


def _hpa_lookup_many(genes: list[str]) -> dict[str, dict]:
    """Look up a batch of gene symbols in one query instead of one round
    trip per gene — an uploaded file might have a few hundred variants, and
    there's no reason to pay SQLite's call overhead that many times when a
    single `IN (...)` does it in one shot."""
    conn = _hpa_connection()
    unique = sorted(set(g for g in genes if g))
    result: dict[str, dict] = {}
    if not unique:
        return result
    # SQLite's default parameter-count ceiling is 999; chunk to stay under it.
    for i in range(0, len(unique), 900):
        chunk = unique[i:i + 900]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT gene, expression_level, reliability FROM hpa WHERE gene IN ({placeholders})",
            chunk,
        ).fetchall()
        for gene, level, reliability in rows:
            result[gene] = {"hpa_expression_level": level, "hpa_reliability": reliability}
    return result


# ---------------------------------------------------------------------------
# Steps 2-4 — VEP + ANNOVAR annotation, HPA lookup, feature computation.
# Same shape as pipeline.py's version, just calling the batched SQLite
# lookup above instead of indexing into a fully-loaded dict.
# ---------------------------------------------------------------------------
def annotate_and_featurize(df: pd.DataFrame, tissue: str, progress=None) -> pd.DataFrame:
    def report(msg):
        if progress:
            progress(msg)

    report("Querying Ensembl VEP (ClinVar / gnomAD / COSMIC / dbSNP)...")
    _log_mem("before VEP REST calls")
    vep_df = s2.annotate(df)
    _log_mem("after VEP REST calls")

    report("Running local ANNOVAR (gene consequence, ClinVar, gnomAD)...")
    # ANNOVAR runs as a separate Perl subprocess, so its memory doesn't show
    # up in this process's own RSS at all -- if the container dies between
    # the "before" and "after" lines below with no "after" ever printing,
    # that's ANNOVAR itself blowing the memory ceiling, not anything in
    # this Python process.
    _log_mem("before ANNOVAR subprocess")
    annovar_df = s2.annotate_with_annovar(df)
    _log_mem("after ANNOVAR subprocess")
    # RUSAGE_CHILDREN only accounts for a child process once it's been
    # waited on, which subprocess.run() already did inside
    # annotate_with_annovar -- so if we get here at all (i.e. ANNOVAR
    # didn't itself get OOM-killed), this is the real number for how much
    # memory the table_annovar.pl process actually used.
    annovar_peak_kb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform == "darwin":
        annovar_peak_kb //= 1024
    print(f"[mem] ANNOVAR subprocess itself peaked at {annovar_peak_kb / 1024:.1f} MB", flush=True)
    annotated = pd.concat([vep_df.reset_index(drop=True), annovar_df.reset_index(drop=True)], axis=1)

    report("Looking up tissue expression breadth (Human Protein Atlas)...")
    hpa_hits = _hpa_lookup_many(annotated["gene"].tolist())
    levels, reliabilities = [], []
    for gene in annotated["gene"]:
        hit = hpa_hits.get(gene, {"hpa_expression_level": "Unknown", "hpa_reliability": "Unknown"})
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
    annotated["gnomad_af"] = pd.to_numeric(vep_af, errors="coerce").fillna(0.0)
    # ANNOVAR's gnomad211_exome filter is skipped in this deployment (see the
    # ANNOVAR_PROTOCOL override above), so annovar_gnomad_af won't exist here
    # -- fall back to VEP's already-parsed gnomad_af rather than defaulting
    # to 0.0, which would tell the model every variant is ultra-rare.
    annotated["gnomad_af_model"] = pd.to_numeric(annovar_af, errors="coerce")
    annotated["gnomad_af_model"] = annotated["gnomad_af_model"].fillna(annotated["gnomad_af"])
    annotated["cosmic_hotspot"] = annotated["cosmic_hotspot"].astype(bool)
    annotated["low_tissue_expression_flag"] = annotated["low_tissue_expression_flag"].astype(bool)

    report("Deriving variant type and full variant name from ANNOVAR output...")
    annotated = add_variant_identity(annotated)
    return annotated


# ---------------------------------------------------------------------------
# Step 7 — ClinVar resolution, then the Stage 1 XGBoost model. Identical to
# pipeline.py: the model file itself is only ~300KB-1MB, never the issue.
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
# Stage 2, path A — the ClinVar watchlist match, SQLite-backed.
#
# This is the one that actually matters for memory. pipeline.py reads the
# whole ~2.3M-row watchlist into a pandas DataFrame, converts it into a
# dict of dicts, and holds that for the life of the process. Here it's
# built once into an indexed SQLite table instead, and a request only ever
# pulls the handful of rows it actually needs — a lookup keyed on
# (chrom, pos, ref, alt), same identity the dict version used, just backed
# by disk (with the OS page cache doing most of the real work after the
# first query) instead of RAM.
#
# The watchlist currently ships as 8 gzip parts in final_output_csv/
# (vus_global_watchlist_v18_300k_part{1..8}_of_8.csv.gz — "300k" refers to
# the rows-per-part chunk size, not the total; the 8 parts together are the
# same ~2.3M-row v18 watchlist). Older layouts are still checked as a
# fallback (code/data/stage2/, the single-file name, the older byte-split
# .part_aa style) so this keeps working if the data ever moves back or a
# different machine still has last year's file layout on disk.
# ---------------------------------------------------------------------------
WATCHLIST_SEARCH = [
    # (directory, glob pattern) — checked in order, first non-empty match wins.
    (FINAL_OUTPUT_DIR, "vus_global_watchlist_v18_300k_part*_of_8.csv.gz"),
    (STAGE2_DIR, "vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz"),
    (STAGE2_DIR, "vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz.part_*"),
    (STAGE2_DIR, "vus_global_watchlist_v14.csv.gz"),
]

WATCHLIST_KEEP_COLS = [
    "reclass_probability_v12", "direction_pathogenic_probability_if_resolved",
    "p_resolved_by_1y", "p_resolved_by_2y", "p_resolved_by_3y", "p_resolved_by_4y",
    "p_resolved_by_5y", "p_resolved_by_6y", "p_resolved_by_7y", "p_resolved_by_8y",
    "p_resolved_by_9y", "p_resolved_by_10y", "p_unresolved_after_10y", "variation_id",
    "has_mave_coverage",
]


def _find_watchlist_files() -> list[str]:
    for directory, pattern in WATCHLIST_SEARCH:
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        if matches:
            return matches
    return []


def _ensure_watchlist_db(progress=None) -> None:
    def report(msg):
        if progress:
            progress(msg)

    if os.path.exists(WATCHLIST_DB_PATH):
        return

    files = _find_watchlist_files()
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = WATCHLIST_DB_PATH + f".building-{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    conn = sqlite3.connect(tmp_path)
    try:
        conn.execute(
            """
            CREATE TABLE watchlist (
                chrom TEXT NOT NULL,
                pos INTEGER NOT NULL,
                ref TEXT NOT NULL,
                alt TEXT NOT NULL,
                reclass_probability_v12 REAL,
                direction_pathogenic_probability_if_resolved REAL,
                p_resolved_by_1y REAL, p_resolved_by_2y REAL, p_resolved_by_3y REAL,
                p_resolved_by_4y REAL, p_resolved_by_5y REAL, p_resolved_by_6y REAL,
                p_resolved_by_7y REAL, p_resolved_by_8y REAL, p_resolved_by_9y REAL,
                p_resolved_by_10y REAL, p_unresolved_after_10y REAL,
                variation_id TEXT,
                has_mave_coverage INTEGER
            )
            """
        )

        # Created up front (on the still-empty table) rather than after all
        # rows are in: the source watchlist CSVs have duplicate
        # (chrom, pos, ref, alt) keys (multiple ClinVar submissions/
        # transcripts landing on the same genomic position are common), so
        # building a UNIQUE index after the fact fails the moment it hits
        # one. Creating it first and using INSERT OR IGNORE below instead
        # of INSERT means the first row seen for a given key wins and later
        # duplicates are silently skipped, which is exactly what "the
        # watchlist for this variant" means — one entry per key.
        conn.execute("CREATE UNIQUE INDEX idx_watchlist_key ON watchlist(chrom, pos, ref, alt)")

        if not files:
            # No watchlist on this machine at all — same graceful fallback
            # pipeline.py has: an empty table means every VUS just falls
            # through to the generalizable model below, nothing errors.
            conn.commit()
            os.replace(tmp_path, WATCHLIST_DB_PATH)
            return

        report(f"Building the watchlist lookup index from {len(files)} file(s) (first run only)...")

        # Read and insert in chunks so this build step never holds more
        # than one chunk of the ~2.3M rows in memory at a time either —
        # the whole point of this file is not doing that.
        placeholders = ", ".join(["?"] * (4 + len(WATCHLIST_KEEP_COLS)))
        insert_sql = f"INSERT OR IGNORE INTO watchlist VALUES ({placeholders})"

        total_rows = 0
        for file_path in files:
            for chunk in pd.read_csv(file_path, chunksize=50_000, low_memory=False):
                chunk["chrom"] = "chr" + chunk["Chromosome"].astype(str)
                chunk["pos"] = pd.to_numeric(chunk["Start"], errors="coerce")

                rename_map = {}
                if "reclass_probability_v12" not in chunk.columns and "reclass_probability" in chunk.columns:
                    rename_map["reclass_probability"] = "reclass_probability_v12"
                if "variation_id" not in chunk.columns and "VariationID" in chunk.columns:
                    rename_map["VariationID"] = "variation_id"
                if "variation_id" not in chunk.columns and "ClinvarID" in chunk.columns:
                    rename_map["ClinvarID"] = "variation_id"
                if rename_map:
                    chunk = chunk.rename(columns=rename_map)

                chunk = chunk.dropna(subset=["pos", "chrom", "ReferenceAllele", "AlternateAllele"])
                if chunk.empty:
                    continue
                chunk["pos"] = chunk["pos"].astype(int)

                for col in WATCHLIST_KEEP_COLS:
                    if col not in chunk.columns:
                        chunk[col] = None

                rows = list(
                    zip(
                        chunk["chrom"], chunk["pos"], chunk["ReferenceAllele"], chunk["AlternateAllele"],
                        *[chunk[c] for c in WATCHLIST_KEEP_COLS],
                    )
                )
                conn.executemany(insert_sql, rows)
                total_rows += len(rows)

        conn.commit()
        actual_rows = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()[0]
        report(f"Watchlist index built: {actual_rows:,} variants, cached for future runs.")
    finally:
        conn.close()

    os.replace(tmp_path, WATCHLIST_DB_PATH)


_watchlist_conn = None


def _watchlist_connection() -> sqlite3.Connection:
    global _watchlist_conn
    if _watchlist_conn is None:
        _ensure_watchlist_db()
        _watchlist_conn = sqlite3.connect(WATCHLIST_DB_PATH, check_same_thread=False)
    return _watchlist_conn


def _watchlist_lookup(chrom: str, pos: int, ref: str, alt: str) -> dict | None:
    conn = _watchlist_connection()
    cols = ", ".join(WATCHLIST_KEEP_COLS)
    row = conn.execute(
        f"SELECT {cols} FROM watchlist WHERE chrom = ? AND pos = ? AND ref = ? AND alt = ?",
        (chrom, pos, ref, alt),
    ).fetchone()
    if row is None:
        return None
    return dict(zip(WATCHLIST_KEEP_COLS, row))


# ---------------------------------------------------------------------------
# Stage 2, path B — the generalizable models. Unchanged from pipeline.py;
# these model files are small (the biggest is the Cox model at ~18MB) and
# were never part of the memory problem.
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
    has been run AND lifelines is installed. Returns False (timing columns
    stay blank) if either is missing, rather than failing the whole run."""
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
    has been run. Returns False (direction stays unset) if not."""
    global _direction_model, _direction_features, _gene_pathogenic_lookup, _gene_pathogenic_global
    if _direction_model is not None:
        return True
    model_path = os.path.join(STAGE2_DIR, "generalizable_direction_model.joblib")
    if not os.path.exists(model_path):
        return False
    _direction_model = joblib.load(model_path)
    with open(os.path.join(STAGE2_DIR, "generalizable_direction_features.txt")) as f:
        _direction_features = f.read().strip().split(",")
    lookup_path = os.path.join(STAGE2_DIR, "gene_pathogenic_rate_lookup_v2.csv")
    _gene_pathogenic_lookup = pd.read_csv(lookup_path, index_col=0)["gene_pathogenic_rate_te"]
    with open(os.path.join(STAGE2_DIR, "gene_pathogenic_rate_global_v1.txt")) as f:
        _gene_pathogenic_global = float(f.read().strip())
    return True


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
        chrom = df.at[idx, "chrom"]
        pos = int(df.at[idx, "pos"])
        ref = df.at[idx, "ref"]
        alt = df.at[idx, "alt"]
        hit = _watchlist_lookup(chrom, pos, ref, alt)
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
        n_before_merge = len(sub)
        sub = sub.merge(_mave_lookup, on="gene", how="left")
        if len(sub) != n_before_merge:
            # A duplicate gene symbol in mavedb_gene_coverage_v2.csv would
            # turn this left-merge into a one-to-many join, silently
            # multiplying rows here and breaking the reindex onto
            # df.index[remaining_mask] a few lines down -- fail loudly
            # instead of scoring misaligned variants.
            raise PipelineError(
                f"Internal error: MaveDB gene lookup merge changed row count "
                f"({n_before_merge} -> {len(sub)}); the mavedb_gene_coverage "
                f"lookup table likely has a duplicate gene symbol."
            )
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
            Xc = sub[_cox_features]
            ph = _cox_model.predict_partial_hazard(Xc).values
            base_times = _cox_model.baseline_survival_.index.values
            base_s0 = _cox_model.baseline_survival_.iloc[:, 0].values
            i = np.searchsorted(base_times, 10.0, side="right") - 1
            i = np.clip(i, 0, len(base_s0) - 1)
            s0_10 = base_s0[i]
            df.loc[remaining_mask, "p_resolved_by_10y"] = 1 - (s0_10 ** ph)

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
# Top-level orchestration — identical to pipeline.py.
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

    _log_mem("run_pipeline start")
    report("Parsing uploaded file...")
    df = parse_uploaded_variants(filename, raw_bytes)
    if len(df) == 0:
        raise PipelineError("No variants could be parsed from the uploaded file.")
    _log_mem("after parsing")

    annotated = annotate_and_featurize(df, tissue, progress=progress)
    _log_mem("after annotation (VEP + ANNOVAR)")
    resolved = classify_stage1(annotated, progress=progress)
    _log_mem("after Stage 1 (XGBoost load + score)")
    scored = score_stage2(resolved, progress=progress)
    _log_mem("after Stage 2 (watchlist + generalizable models)")

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


# ---------------------------------------------------------------------------
# Lets you pre-build both SQLite caches ahead of time, e.g. as a Docker
# build step, so the first real request doesn't have to eat the ~2.3M-row
# build cost. Not required, _ensure_watchlist_db()/_ensure_hpa_db() both
# run lazily on first use anyway, but a cold start on a free-tier box with
# a request timeout is exactly the situation where paying that cost ahead
# of time instead of on someone's first upload is worth the extra step.
#
#   python pipeline_app.py --warm-cache
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--warm-cache" in sys.argv:
        print("Building watchlist index...")
        _ensure_watchlist_db(progress=print)
        print("Building HPA index...")
        _ensure_hpa_db()
        print(f"Done. Cache files written to {CACHE_DIR}")
    else:
        print(__doc__)
        print("Run with --warm-cache to pre-build the SQLite lookup caches.")
