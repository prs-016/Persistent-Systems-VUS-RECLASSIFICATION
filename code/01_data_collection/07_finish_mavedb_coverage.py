import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

FEATURES_PATH = "data/stage2/vus_features_v9.csv"
OUTPUT_PATH = "data/stage2/mavedb_gene_coverage_v2.csv"
MAX_WORKERS = 16
MAX_RETRIES = 3


def query_gene(gene_symbol: str) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"https://api.mavedb.org/api/v1/genes/{gene_symbol}", timeout=10)
            if response.status_code == 200:
                data = response.json()
                score_sets = data.get("scoreSets", [])
                return {
                    "gene": gene_symbol,
                    "has_mave_coverage": len(score_sets) > 0,
                    "mave_num_score_sets": len(score_sets),
                    "mave_num_variants": sum(s.get("numVariants", 0) or 0 for s in score_sets),
                    "queried_ok": True,
                    "http_status": 200,
                    "attempts": attempt + 1,
                }
            elif response.status_code == 404:
                # confirmed negative: MaveDB has no record of this gene symbol at all
                return {
                    "gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0,
                    "mave_num_variants": 0, "queried_ok": True, "http_status": 404, "attempts": attempt + 1,
                }
            else:
                last_error = f"http_{response.status_code}"
        except Exception as e:
            last_error = str(e)[:120]
        time.sleep(0.5 * (attempt + 1))
    return {
        "gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0,
        "mave_num_variants": 0, "queried_ok": False, "http_status": last_error, "attempts": MAX_RETRIES,
    }


if __name__ == "__main__":
    features = pd.read_csv(FEATURES_PATH, low_memory=False)
    all_genes = sorted(features["GeneSymbol"].dropna().unique())
    print(f"total unique genes: {len(all_genes)}")

    rows = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(query_gene, g): g for g in all_genes}
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(all_genes)} done ({time.time()-start_time:.0f}s)")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\ntotal time: {time.time()-start_time:.0f}s")
    print(f"queried_ok: {result['queried_ok'].sum()}/{len(result)}")
    print(f"has_mave_coverage: {result['has_mave_coverage'].sum()}/{len(result)}")

    still_failed = result[~result["queried_ok"]]
    print(f"still failed after {MAX_RETRIES} retries: {len(still_failed)}")
    if len(still_failed):
        print(still_failed[["gene", "http_status"]].to_string())
