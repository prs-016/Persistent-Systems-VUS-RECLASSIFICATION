"""
Step 1 — parse the patient's variant calls into a working DataFrame.

No real patient VCF exists on this machine, so this uses one real TCGA-BRCA
patient MAF file as the stand-in "patient" input (stated explicitly, per
docs/STATE.md Constraints). Only the columns a real VCF would also provide
(chrom/pos/ref/alt + read counts) are kept here — annotation columns already
present in the MAF (gnomAD_AF, CLIN_SIG, COSMIC, hotspot) are intentionally
NOT carried over yet; Step 2 re-derives them via the VEP REST API so the
pipeline behaves the same way it would on a real, unannotated patient VCF.
"""
import gzip
import pandas as pd

PATIENT_MAF_PATH = "data/patient/patient.maf.gz"


def parse_patient_maf(path: str) -> pd.DataFrame:
    with gzip.open(path, "rt") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if not l.startswith("#"))
    header = lines[header_idx].rstrip("\n").split("\t")
    rows = [l.rstrip("\n").split("\t") for l in lines[header_idx + 1:]]
    raw = pd.DataFrame(rows, columns=header)

    df = pd.DataFrame({
        "chrom": raw["Chromosome"],
        "pos": raw["Start_Position"].astype(int),
        "end": raw["End_Position"].astype(int),
        "ref": raw["Reference_Allele"],
        "alt": raw["Tumor_Seq_Allele2"],
        "gene": raw["Hugo_Symbol"],
        "t_depth": raw["t_depth"].astype(int),
        "t_ref_count": raw["t_ref_count"].astype(int),
        "t_alt_count": raw["t_alt_count"].astype(int),
    })
    return df


if __name__ == "__main__":
    df = parse_patient_maf(PATIENT_MAF_PATH)
    print("row count:", len(df))
    print(df.head(5).to_string())
    df.to_csv("data/patient/step1_parsed.csv", index=False)
