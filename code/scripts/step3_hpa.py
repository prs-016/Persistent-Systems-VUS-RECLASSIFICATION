"""
Step 3 — Human Protein Atlas tissue-expression plausibility check.

Substitution stated explicitly: the classic per-tissue Gene x Tissue x
Level x Reliability table (normal_tissue.tsv) is no longer distributed by
HPA — confirmed 404 on normal_tissue.tsv.zip, rna_tissue_consensus.tsv.zip,
and rna_tissue_hpa.tsv.zip; only the wide-format proteinatlas.tsv remains
(see docs/STATE.md). That file reports each gene's single most
tissue-specific/enriched value, not a value per arbitrary queried tissue.
So `hpa_expression_level` here is HPA's "RNA tissue distribution" column —
a gene-level breadth-of-expression signal (Detected in all/many/some/single
tissues, or Not detected anywhere assayed) — not a lookup restricted to the
--tissue argument. The --tissue argument is still accepted and recorded for
provenance, since a real per-tissue table would use it directly.
"""
import argparse
import csv
import pandas as pd

import sys
sys.path.insert(0, ".")
from config.thresholds import HPA_LOW_EXPRESSION_CATEGORIES, HPA_RELIABLE_CATEGORIES

HPA_TSV_PATH = "data/hpa/proteinatlas.tsv"


def load_hpa_lookup(path: str) -> dict:
    lookup = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            lookup[row["Gene"]] = {
                "hpa_expression_level": row["RNA tissue distribution"] or "Unknown",
                "hpa_reliability": row["Reliability (IH)"] or "Unknown",
            }
    return lookup


def annotate_hpa(df: pd.DataFrame, tissue: str) -> pd.DataFrame:
    lookup = load_hpa_lookup(HPA_TSV_PATH)
    levels, reliabilities = [], []
    for gene in df["gene"]:
        hit = lookup.get(gene, {"hpa_expression_level": "Unknown", "hpa_reliability": "Unknown"})
        levels.append(hit["hpa_expression_level"])
        reliabilities.append(hit["hpa_reliability"])
    out = df.copy()
    out["hpa_expression_level"] = levels
    out["hpa_reliability"] = reliabilities
    out["hpa_query_tissue"] = tissue
    out["low_tissue_expression_flag"] = (
        out["hpa_expression_level"].isin(HPA_LOW_EXPRESSION_CATEGORIES)
        & out["hpa_reliability"].isin(HPA_RELIABLE_CATEGORIES)
    )
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tissue", default="breast")
    args = parser.parse_args()

    df = pd.read_csv("data/patient/step2_annotated.csv")
    out = annotate_hpa(df, args.tissue)

    print(f"threshold: low_tissue_expression_flag = True when "
          f"hpa_expression_level in {HPA_LOW_EXPRESSION_CATEGORIES} "
          f"AND hpa_reliability in {HPA_RELIABLE_CATEGORIES}")
    print(f"reasoning: 'Not detected' with a reliable (non-Uncertain) call is "
          f"the strongest available signal that a somatic call sits in a gene "
          f"with no real expression anywhere HPA assayed it")
    print("\nhpa_expression_level distribution over this patient's genes:")
    print(out["hpa_expression_level"].value_counts())
    print("\nlow_tissue_expression_flag counts:")
    print(out["low_tissue_expression_flag"].value_counts())
    out.to_csv("data/patient/step3_hpa.csv", index=False)
