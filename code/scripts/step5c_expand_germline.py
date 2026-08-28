"""
Step 5c — expand the germline training pool beyond Huang et al. 2018's 853
real TCGA calls, per user feedback ("not enough germline... use multiple
databases for various sources").

Adds 10,000 real, randomly-sampled (random_state=42) ClinVar germline
Pathogenic/Likely-Pathogenic variants (real coordinates, real gene, real
VariationID — data/germline_training/clinvar_germline_sample.csv, drawn
from 317,181 real matches found in the current ClinVar variant_summary
snapshot: OriginSimple=="germline", ClinicalSignificance in
{Pathogenic, Likely pathogenic}, GRCh38, usable VCF-normalized coords).

TWO REAL, DISCLOSED LIMITATIONS of the added rows (not hidden):

1. **No measured VAF.** ClinVar is a clinical-significance archive, not a
   sequencing dataset — it does not carry allele-fraction/read-count data
   the way Huang et al.'s real TCGA calls do. The added rows get
   `vaf=0.5` (the standard heterozygous-inheritance assumption for a
   germline variant), NOT a measured value. This is flagged via a
   `vaf_source` column (`measured_tcga_huang2018` vs
   `assumed_heterozygous_clinvar`) so downstream analysis/ablation can
   separate the two, and is the same category of caveat this project
   already applies to `gnomad_af` (see docs/STATE.md's curation-bias
   section) — disclosed, not silently baked in.
2. **No cosmic_hotspot.** The original 853 Huang rows got real
   cosmic_hotspot values from a checkpointed VEP REST query (~7-8 min of
   real query time for 853 rows). Doing the same for 10,000 more rows
   would take ~90+ minutes of real query time — not feasible in one
   session. The added ClinVar rows get `cosmic_hotspot=False` as an
   UNQUERIED default, not a verified negative — flagged via
   `cosmic_hotspot_source` so this is auditable rather than silently
   indistinguishable from a real negative.

Produces data/tcga_training/training_table_v2.csv — a SEPARATE file from
the v1 training_table_real.csv (which stays as the "measured-VAF-only"
baseline for ablation comparison), same pattern as Stage 1's earlier
gnomad_af with/without ablation.
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import step2_annotate_vep as s2  # noqa: E402
import step3_hpa as s3  # noqa: E402

SOMATIC_PATH = "data/tcga_training/somatic_real.csv"
GERMLINE_COMBINED_PATH = "data/germline_training/germline_combined.csv"
OUT_PATH = "data/tcga_training/training_table_v2.csv"


def compute_low_tissue_expression(df: pd.DataFrame) -> pd.Series:
    genes_df = pd.DataFrame({"gene": df["gene"].dropna().unique()})
    annotated = s3.annotate_hpa(genes_df, tissue="breast")
    gene_to_flag = dict(zip(annotated["gene"], annotated["low_tissue_expression_flag"]))
    return df["gene"].map(gene_to_flag).fillna(False)


if __name__ == "__main__":
    somatic = pd.read_csv(SOMATIC_PATH)
    germline = pd.read_csv(GERMLINE_COMBINED_PATH)

    # real cosmic_hotspot: somatic keeps its MAF-embedded COSMIC column;
    # the original 853 Huang germline rows reuse the existing real VEP
    # checkpoint from step5b; the new 10,000 ClinVar rows get the
    # disclosed unqueried default described above.
    vep_germline = pd.read_csv("data/germline_training/vep_annotations_germline.csv")
    huang_mask = germline["vaf_source"] == "measured_tcga_huang2018"
    assert huang_mask.sum() == len(vep_germline), (huang_mask.sum(), len(vep_germline))
    germline.loc[huang_mask, "cosmic_hotspot"] = vep_germline["cosmic_id"].notna().values
    germline.loc[huang_mask, "cosmic_hotspot_source"] = "vep_rest_queried"
    germline.loc[~huang_mask, "cosmic_hotspot"] = False
    germline.loc[~huang_mask, "cosmic_hotspot_source"] = "unqueried_default"

    somatic["cosmic_hotspot_source"] = "maf_embedded"
    somatic["vaf_source"] = "measured_tcga_maf"
    somatic["variation_id"] = ""

    keep_cols = ["chrom", "pos", "end", "ref", "alt", "gene", "vaf", "cosmic_hotspot",
                 "label", "variant_key", "cancer_type", "vaf_source", "cosmic_hotspot_source", "variation_id"]
    combined = pd.concat([somatic[keep_cols], germline[keep_cols]], ignore_index=True)
    print("combined v2 rows:", len(combined))
    print(combined["label"].value_counts())

    print("\nrunning real ANNOVAR gnomAD AF annotation over the full v2 set...")
    annovar = s2.annotate_with_annovar(combined[["chrom", "pos", "end", "ref", "alt"]])
    assert len(annovar) == len(combined)
    combined["gnomad_af"] = pd.to_numeric(annovar["annovar_gnomad_af"], errors="coerce")

    print("computing low_tissue_expression_flag via real HPA lookup...")
    combined["low_tissue_expression_flag"] = compute_low_tissue_expression(combined)

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}, shape={combined.shape}")
    print("label counts:\n", combined["label"].value_counts())
    print("class ratio (Somatic:Germline):", f"{(combined['label']=='Somatic').sum() / (combined['label']=='Germline').sum():.2f}:1")
    for col in ["gnomad_af", "cosmic_hotspot", "low_tissue_expression_flag"]:
        print(f"{col} non-null: {combined[col].notna().mean():.1%}")
