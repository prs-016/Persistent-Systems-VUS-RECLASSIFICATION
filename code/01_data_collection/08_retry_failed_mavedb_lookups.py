import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

MAX_WORKERS = 3
MAX_RETRIES = 5


def query_gene(gene_symbol: str) -> dict:
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"https://api.mavedb.org/api/v1/genes/{gene_symbol}", timeout=15)
            if response.status_code == 200:
                data = response.json()
                score_sets = data.get("scoreSets", [])
                return {
                    "gene": gene_symbol, "has_mave_coverage": len(score_sets) > 0,
                    "mave_num_score_sets": len(score_sets),
                    "mave_num_variants": sum(s.get("numVariants", 0) or 0 for s in score_sets),
                    "queried_ok": True, "http_status": 200, "attempts": attempt + 1,
                }
            elif response.status_code == 404:
                return {
                    "gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0,
                    "mave_num_variants": 0, "queried_ok": True, "http_status": 404, "attempts": attempt + 1,
                }
            else:
                last_error = f"http_{response.status_code}"
        except Exception as e:
            last_error = str(e)[:120]
        time.sleep(2.0 * (attempt + 1))
    return {
        "gene": gene_symbol, "has_mave_coverage": False, "mave_num_score_sets": 0,
        "mave_num_variants": 0, "queried_ok": False, "http_status": last_error, "attempts": MAX_RETRIES,
    }


if __name__ == "__main__":
    coverage = pd.read_csv("data/stage2/mavedb_gene_coverage_v2.csv")
    failed_genes = coverage[~coverage["queried_ok"]]["gene"].tolist()
    print(f"retrying {len(failed_genes)} genes at low concurrency (3 workers, up to 5 attempts, 2-10s backoff)")

    rows = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(query_gene, g): g for g in failed_genes}
        done = 0
        for future in as_completed(futures):
            rows.append(future.result())
            done += 1
            if done % 30 == 0:
                print(f"  {done}/{len(failed_genes)} ({time.time()-start_time:.0f}s)")

    retry_result = pd.DataFrame(rows)
    print(f"\nretry done in {time.time()-start_time:.0f}s")
    print(f"now queried_ok: {retry_result['queried_ok'].sum()}/{len(retry_result)}")
    print(f"now has_mave_coverage: {retry_result['has_mave_coverage'].sum()}")

    still_failed = retry_result[~retry_result["queried_ok"]]
    print(f"still failed: {len(still_failed)}")
    if len(still_failed):
        print(still_failed[["gene", "http_status"]].to_string())

    # fold the retry results back into the main coverage table, overwriting
    # only the genes we just retried
    coverage = coverage.set_index("gene")
    retry_result = retry_result.set_index("gene")
    coverage.update(retry_result)
    coverage = coverage.reset_index()
    coverage.to_csv("data/stage2/mavedb_gene_coverage_v2.csv", index=False)
    print(f"\nfinal merged: queried_ok {coverage['queried_ok'].sum()}/{len(coverage)}, "
          f"has_mave_coverage {coverage['has_mave_coverage'].sum()}")
