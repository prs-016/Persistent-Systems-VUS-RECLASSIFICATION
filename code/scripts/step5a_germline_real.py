"""
Step 5a — build the REAL germline training set from Huang et al. 2018 Cell
("Pathogenic Germline Variants in 10,389 Adult Cancers", PMC5949147),
Table S2, sheet S2A ("Pathogenic_variants").

This REPLACES the previous gnomAD-common-variant germline PROXY
(docs/STAGE1_RESULTS.md §3.1) with real per-patient TCGA germline calls:
853 rows classified Pathogenic or Likely Pathogenic by the paper's CharGer
pipeline, each with real measured normal-tissue VAF (normalVAF, computed
from real normalRefCnt/normalAltCnt/normalDepth read counts from actual
TCGA germline WES) — not a population-frequency-derived expectation.

Source file: data/germline_training/Huang2018_TableS2_mmc2.xlsx
Downloaded from the paper's open-access Elsevier CDN attachment
(https://ars.els-cdn.com/content/image/1-s2.0-S0092867418303635-mmc2.xlsx,
NONAUTHATTACH — no login required; this is NOT the GDC-hosted version of
this data, which is Controlled Access and requires dbGaP authorization —
see docs/STATE.md for that distinction).

IMPORTANT CAVEAT (documented per project convention of never hiding
limitations): Table S2A is a CURATED PATHOGENIC/LIKELY-PATHOGENIC SUBSET,
not a random/representative sample of "real germline variants." The
ACMG/CharGer criteria used to classify these 853 rows include PM2 (low or
absent population allele frequency — used in 853/853 = 100% of rows) and
PVS1 (loss-of-function variant — used in 644/853 = 75.5% of rows, and
83% of all 853 rows are stop-gained/frameshift/splice-disrupting).
This means population-frequency and functional-severity features are
correlated with the Germline label not purely because of biological
germline-vs-somatic origin, but partly because those same signals were
used to CURATE this set as "pathogenic" in the first place. This is why
`gnomad_af` is included in Step 6 only as an ablation-tested feature (with
the pipeline's existing PR-AUC>0.98 leakage tripwire watched closely), and
why exonic-function/consequence-severity features are excluded entirely
from FEATURE_COLS in Step 6 despite being available in the annotation
output — see docs/STAGE1_RESULTS.md for the full discussion.

De-duplication note: the 853 rows resolve to only 586 unique variant
coordinates (267 rows are the same variant recurring in a different
patient — e.g. chr11:108183151 G>T appears 8 times). All 853 real rows
are kept (each is a real, distinct patient observation with its own real
VAF), but a `variant_key` column is written so Step 6 can do a
variant-level (grouped) train/test split instead of a naive row-level
split, preventing the same variant from appearing in both train and test.
"""
import openpyxl
import pandas as pd

SOURCE_XLSX = "data/germline_training/Huang2018_TableS2_mmc2.xlsx"
SHEET = "S2A.Pathogenic_variants"
OUT_CSV = "data/germline_training/germline_real.csv"


def load_real_germline() -> pd.DataFrame:
    wb = openpyxl.load_workbook(SOURCE_XLSX, read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    idx = {h: i for i, h in enumerate(header)}
    data = [r for r in rows[1:] if r[idx["Overall_Classification"]] in ("Pathogenic", "Likely Pathogenic")]

    out = pd.DataFrame({
        "chrom": ["chr" + str(r[idx["Chromosome"]]) for r in data],
        "pos": [int(r[idx["Start"]]) for r in data],
        "end": [int(r[idx["Stop"]]) for r in data],
        "ref": [r[idx["Reference"]] for r in data],
        "alt": [r[idx["Alternate"]] for r in data],
        "gene": [r[idx["HUGO_Symbol"]] for r in data],
        "vaf": [float(r[idx["normalVAF"]]) for r in data],
        "cancer_type": [r[idx["cancer"]] for r in data],
        "overall_classification": [r[idx["Overall_Classification"]] for r in data],
        "label": "Germline",
    })
    out["variant_key"] = out["chrom"] + ":" + out["pos"].astype(str) + ":" + out["ref"] + ":" + out["alt"]
    return out


if __name__ == "__main__":
    df = load_real_germline()
    print("real germline rows:", len(df))
    print("unique variant coordinates:", df["variant_key"].nunique())
    print("normalVAF stats:\n", df["vaf"].describe())
    print("cancer type counts:\n", df["cancer_type"].value_counts())
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")
