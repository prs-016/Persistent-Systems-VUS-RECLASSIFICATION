import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

FEATURES_PATH = "data/stage2/vus_features.csv"
CHECKPOINT_PATH = "data/stage2/mavedb_gene_coverage.csv"
GENES_PER_RUN = 250
MAX_WORKERS = 16  # MaveDB didn't rate-limit us up to 16 concurrent requests in testing


def query_gene(gene_symbol: str) -> dict:
    try:
        response = requests.get(f"https://api.mavedb.org/api/v1/genes/{gene_symbol}", timeout=8)
        if response.status_code != 200:
            return {"gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0, "mave_num_variants": 0, "queried_ok": False}
        data = response.json()
        score_sets = data.get("scoreSets", [])
        return {
            "gene": gene_symbol,
            "has_mave_coverage": len(score_sets) > 0,
            "mave_num_score_sets": len(score_sets),
            "mave_num_variants": sum(s.get("numVariants", 0) or 0 for s in score_sets),
            "queried_ok": True,
        }
    except Exception:
        return {"gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0, "mave_num_variants": 0, "queried_ok": False}


if __name__ == "__main__":
    features = pd.read_csv(FEATURES_PATH)
    all_genes = sorted(features["GeneSymbol"].dropna().unique())
    print(f"total unique genes needing MaveDB coverage: {len(all_genes)}")

    if os.path.exists(CHECKPOINT_PATH):
        already_done = pd.read_csv(CHECKPOINT_PATH)
        done_genes = set(already_done["gene"])
    else:
        already_done = pd.DataFrame(columns=["gene", "has_mave_coverage", "mave_num_score_sets", "mave_num_variants", "queried_ok"])
        done_genes = set()

    remaining_genes = [g for g in all_genes if g not in done_genes]
    print(f"already done: {len(done_genes)}, remaining: {len(remaining_genes)}")

    if not remaining_genes:
        print("all genes queried.")
    else:
        this_batch = remaining_genes[:GENES_PER_RUN]
        new_rows = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(query_gene, g): g for g in this_batch}
            for future in as_completed(futures):
                new_rows.append(future.result())

        updated = pd.concat([already_done, pd.DataFrame(new_rows)], ignore_index=True)
        updated.to_csv(CHECKPOINT_PATH, index=False)
        print(f"this run: queried {len(this_batch)} genes, now at {len(updated)}/{len(all_genes)}")
        print(f"coverage found this batch: {sum(r['has_mave_coverage'] for r in new_rows)}/{len(this_batch)}")
