"""
Step 5b — annotate the combined training set (9,081 real somatic + 853 real
germline rows = 9,934 total) and assemble the final feature table.

Per-feature annotation source (each chosen for real coverage + feasibility
in this environment — documented plainly rather than glossed over):

- gnomad_af: from ANNOVAR (annovar_gnomad_af, gnomAD 2.1.1 exome), run
  locally over the FULL 9,934-row set in ~6s. VEP was tried first but the
  public REST endpoint runs at roughly 2 variants/sec for batch requests
  (measured: 50 variants = 24.7s), which would take well over an hour of
  pure query time for 9,934 rows — infeasible to run reliably in this
  environment. ANNOVAR gives a real, verified gnomAD AF for every row at
  negligible cost, so it's used as the canonical source instead.
- cosmic_hotspot (somatic rows): the MAF's own bundled COSMIC column,
  already real and already free (no extra query needed) — computed in
  step5_tcga_training.py.
- cosmic_hotspot (germline rows): real VEP REST colocated_variants COSMIC
  lookup, run ONLY on the 853 germline rows (not the full 9,934) since
  that's tractable (~7-8 min of real query time) where the full set is
  not. This does mean cosmic_hotspot's two classes come from two different
  real COSMIC snapshots (MAF-embedded vs. live VEP query) rather than one
  uniform source — a real limitation, stated here rather than hidden.
- low_tissue_expression_flag: real HPA gene-level lookup (Step 3's own
  function), computed for ALL rows since it's a fast local lookup —
  previously hardcoded False for every training row (documented gap,
  STAGE1_RESULTS.md §3.5), now real for the first time.

VEP annotation of the germline subset is checkpointed to
data/germline_training/vep_annotations_germline.csv and is safe to rerun —
it resumes from wherever it left off.
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import step2_annotate_vep as s2  # noqa: E402
import step3_hpa as s3  # noqa: E402

SOMATIC_PATH = "data/tcga_training/somatic_real.csv"
GERMLINE_PATH = "data/germline_training/germline_real.csv"
ANNOVAR_PATH = "data/tcga_training/annovar_annotations.csv"
VEP_GERMLINE_CHECKPOINT = "data/germline_training/vep_annotations_germline.csv"
FINAL_PATH = "data/tcga_training/training_table_real.csv"

VEP_BATCH_SIZE = 65  # smaller than step2's 200 — keeps each batch inside this environment's per-call time budget (measured ~0.5s/variant)


def run_vep_checkpointed(df: pd.DataFrame, checkpoint_path: str, max_batches: int = 6) -> pd.DataFrame:
    n = len(df)
    if os.path.exists(checkpoint_path):
        done = pd.read_csv(checkpoint_path)
        start = len(done)
    else:
        done = pd.DataFrame(columns=["dbsnp_id", "clnsig", "gnomad_af", "cosmic_id"])
        start = 0

    if start >= n:
        print(f"VEP already complete: {start}/{n}")
        return pd.read_csv(checkpoint_path)

    variants = [s2.build_region_string(row) for _, row in df.iterrows()]
    new_rows = []
    i = start
    batches_done = 0
    while i < n and batches_done < max_batches:
        batch = variants[i:i + VEP_BATCH_SIZE]
        results = s2.query_vep_batch(batch)
        for res in results:
            new_rows.append(s2.extract_annotation(res))
        i += len(batch)
        batches_done += 1
        updated = pd.concat([done, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(checkpoint_path, index=False)
        print(f"  VEP progress: {len(updated)}/{n}")
        time.sleep(1)

    print(f"this run: {batches_done} batches, now at {i}/{n}")
    return pd.read_csv(checkpoint_path) if os.path.exists(checkpoint_path) else pd.DataFrame()


def compute_low_tissue_expression(df: pd.DataFrame) -> pd.Series:
    genes_df = pd.DataFrame({"gene": df["gene"].unique()})
    annotated = s3.annotate_hpa(genes_df, tissue="breast")
    gene_to_flag = dict(zip(annotated["gene"], annotated["low_tissue_expression_flag"]))
    return df["gene"].map(gene_to_flag).fillna(False)


def assemble_final():
    somatic = pd.read_csv(SOMATIC_PATH)
    germline = pd.read_csv(GERMLINE_PATH)
    vep_germline = pd.read_csv(VEP_GERMLINE_CHECKPOINT)
    assert len(vep_germline) == len(germline), "germline VEP annotation incomplete"
    vep_germline["cosmic_hotspot"] = vep_germline["cosmic_id"].notna()

    germline_full = pd.concat([germline.reset_index(drop=True), vep_germline.reset_index(drop=True)], axis=1)

    combined = pd.concat([
        somatic[["chrom", "pos", "end", "ref", "alt", "gene", "vaf", "cosmic_hotspot", "label", "variant_key", "cancer_type"]],
        germline_full[["chrom", "pos", "end", "ref", "alt", "gene", "vaf", "cosmic_hotspot", "label", "variant_key", "cancer_type"]],
    ], ignore_index=True)

    annovar = pd.read_csv(ANNOVAR_PATH)
    assert len(annovar) == len(combined), f"ANNOVAR row count {len(annovar)} != combined {len(combined)}"
    combined["gnomad_af"] = pd.to_numeric(annovar["annovar_gnomad_af"], errors="coerce")

    print("computing low_tissue_expression_flag via real HPA lookup for all rows...")
    combined["low_tissue_expression_flag"] = compute_low_tissue_expression(combined)

    combined.to_csv(FINAL_PATH, index=False)
    print(f"wrote {FINAL_PATH}, shape={combined.shape}")
    print("\nlabel counts:\n", combined["label"].value_counts())
    for col in ["gnomad_af", "cosmic_hotspot", "low_tissue_expression_flag"]:
        print(f"{col} non-null: {combined[col].notna().mean():.1%}")
    return combined


if __name__ == "__main__":
    germline = pd.read_csv(GERMLINE_PATH)
    vep_result = run_vep_checkpointed(germline, VEP_GERMLINE_CHECKPOINT, max_batches=1)
    if len(vep_result) < len(germline):
        print(f"VEP incomplete: {len(vep_result)}/{len(germline)} — rerun this script to continue")
        sys.exit(0)
    assemble_final()
