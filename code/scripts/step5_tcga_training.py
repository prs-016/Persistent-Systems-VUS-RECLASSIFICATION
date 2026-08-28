"""
Step 5 — build the SOMATIC half of the labeled training table from real
TCGA somatic MAF calls.

REVAMP NOTE (this session): the germline side of the training table used
to be a gnomAD-common-variant PROXY (see git history / docs/STATE.md for
the old approach) — that has been replaced project-wide with REAL TCGA
germline calls from Huang et al. 2018 (step5a_germline_real.py). This
script now only builds the somatic side, and — new in this revamp — keeps
each row's genomic coordinates (chrom/pos/end/ref/alt), which the old
version discarded. Coordinates are required so Step 5b can run every
somatic training row through the SAME VEP REST + ANNOVAR annotation path
used for the patient and for the real germline rows, keeping features
consistent across train and inference instead of somatic rows using the
MAF's own bundled annotation columns while everything else uses VEP/ANNOVAR.

PAN-CANCER EXPANSION (this session): originally this pulled only the 60
real TCGA-BRCA MAF files in data/tcga_training/. Per user feedback ("add
way more variety like cancers tumours etc apart from BRCA since we want to
predict GERMLINE/SOMATIC generally not just for BRCA"), it now also pulls
90 real MAF files (15 each) from 6 additional real TCGA projects —
LUAD, COAD, PRAD, STAD, SKCM, OV — downloaded via the real GDC API into
data/pancancer_training/ (see data/tcga_training/pancancer_file_ids.json
for the real file_id/file_name/submitter_id metadata GDC returned). Every
row keeps a real cancer_type field derived from the file's own MAF header
(NCBI_Build/Center columns don't carry project id, so cancer_type is taken
from the pan-cancer filename prefix we chose at download time, or "BRCA"
for the original un-prefixed files — both are real TCGA project codes, not
inferred/guessed).
"""
import glob
import gzip
import os
import pandas as pd


def load_somatic_training_set() -> pd.DataFrame:
    rows = []
    files = sorted(glob.glob("data/tcga_training/*.maf.gz")) + sorted(glob.glob("data/pancancer_training/*.maf.gz"))
    for path in files:
        fname = os.path.basename(path)
        cancer_type = fname.split("_")[0].replace("TCGA-", "") if fname.startswith("TCGA-") else "BRCA"
        with gzip.open(path, "rt") as f:
            lines = f.readlines()
        header_idx = next(i for i, l in enumerate(lines) if not l.startswith("#"))
        header = lines[header_idx].rstrip("\n").split("\t")
        for l in lines[header_idx + 1:]:
            vals = l.rstrip("\n").split("\t")
            rec = dict(zip(header, vals))
            if rec.get("Mutation_Status") != "Somatic":
                continue
            try:
                t_depth = int(rec["t_depth"])
                t_alt = int(rec["t_alt_count"])
                start = int(rec["Start_Position"])
                end = int(rec["End_Position"])
            except (ValueError, KeyError):
                continue
            if t_depth == 0:
                continue
            chrom = rec.get("Chromosome", "")
            if chrom and not chrom.startswith("chr"):
                chrom = "chr" + chrom
            ref = rec.get("Reference_Allele", "")
            alt = rec.get("Tumor_Seq_Allele2", "")
            if not chrom or not ref or not alt:
                continue
            rows.append({
                "chrom": chrom,
                "pos": start,
                "end": end,
                "ref": ref,
                "alt": alt,
                "gene": rec.get("Hugo_Symbol", ""),
                "vaf": t_alt / t_depth,
                # Real, free, already embedded in the MAF — no need to re-query
                # VEP for this on the somatic side (COSMIC ID string presence).
                "cosmic_hotspot": bool(rec.get("COSMIC", "")),
                "label": "Somatic",
                "cancer_type": cancer_type,
            })
    df = pd.DataFrame(rows)
    df["variant_key"] = df["chrom"] + ":" + df["pos"].astype(str) + ":" + df["ref"] + ":" + df["alt"]
    return df


if __name__ == "__main__":
    somatic_df = load_somatic_training_set()
    print("somatic examples:", len(somatic_df))
    print("unique somatic variant coordinates:", somatic_df["variant_key"].nunique())
    somatic_df.to_csv("data/tcga_training/somatic_real.csv", index=False)
    print("wrote data/tcga_training/somatic_real.csv")
