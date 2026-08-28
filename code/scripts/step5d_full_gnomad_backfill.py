"""
Step 5d — real fix for a big coverage gap found in training_table_v3.csv:
73.7% of all 45,706 rows had gnomad_af = NaN (never actually annotated),
because the original ANNOVAR gnomAD run (step5b) only ever covered the
original 9,934-row set (853 germline + 9,081 BRCA-only somatic) -- the
pan-cancer expansion (90 more MAF files, 6 more cancer types) and the
germline ClinVar expansion (10,000 more rows) were both added to the
training table LATER and never got a real gnomAD annotation pass.
Confirmed real, not a false alarm: null rate is ~100% for every
non-original cancer-type/source group and only partially populated even
for the original BRCA/Huang rows (a real regression from an earlier
merge, not by design).

ANNOVAR's gnomad211_exome database is already downloaded locally
(annovar/humandb/hg38_gnomad211_exome.txt) -- this re-annotates gnomad_af
for the FULL 45,706-row table locally (no network calls, fast), fixing
the coverage gap directly rather than leaving it as fillna(0.0) noise.
"""
import sys
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import step2_annotate_vep as s2  # noqa: E402

IN_PATH = "data/tcga_training/training_table_v3.csv"
OUT_PATH = "data/tcga_training/training_table_v4.csv"

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"loaded {len(df)} rows, pre-fix gnomad_af null rate: {df['gnomad_af'].isna().mean():.1%}")

    annovar_in = df[["chrom", "pos", "end", "ref", "alt"]].copy()
    annovar_out = s2.annotate_with_annovar(annovar_in)
    assert len(annovar_out) == len(df), f"row mismatch: {len(annovar_out)} vs {len(df)}"

    df["gnomad_af_v4"] = pd.to_numeric(annovar_out["annovar_gnomad_af"].values, errors="coerce")
    df["gnomad_af_popmax_v4"] = pd.to_numeric(annovar_out["annovar_gnomad_af_popmax"].values, errors="coerce")

    print(f"post-fix gnomad_af_v4 null rate: {df['gnomad_af_v4'].isna().mean():.1%}")
    print("\nby label:")
    print(df.groupby("label")["gnomad_af_v4"].describe())

    df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")
