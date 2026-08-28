import time
import threading
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_PATH = "data/stage2/pubmed_literature_volume_full.csv"
GENES_SOURCE_PATH = "data/stage2/vus_features_v4.csv"
MAX_DATE = "2019/06/30"
SECONDS_BETWEEN_REQUESTS = 0.35  # ~3 req/sec, NCBI's un-keyed limit

_rate_limit_lock = threading.Lock()
_last_request_time = [0.0]


def wait_for_rate_limit():
    with _rate_limit_lock:
        now = time.time()
        time_to_wait = SECONDS_BETWEEN_REQUESTS - (now - _last_request_time[0])
        if time_to_wait > 0:
            time.sleep(time_to_wait)
        _last_request_time[0] = time.time()


def get_pubmed_count(gene: str) -> int | None:
    search_term = f'"{gene}"[Title/Abstract] AND ("variant"[Title/Abstract] OR "mutation"[Title/Abstract])'
    for attempt in range(3):
        try:
            wait_for_rate_limit()
            response = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": search_term, "retmax": 0, "datetype": "pdat",
                        "maxdate": MAX_DATE, "mindate": "1900/01/01", "retmode": "json"},
                timeout=15,
            )
            if response.status_code == 200:
                return int(response.json()["esearchresult"]["count"])
            elif response.status_code == 429:
                time.sleep(2)
                continue
        except Exception:
            time.sleep(1)
            continue
    return None


if __name__ == "__main__":
    all_genes = sorted(pd.read_csv(GENES_SOURCE_PATH, low_memory=False)["GeneSymbol"].dropna().unique().tolist())
    print(f"{len(all_genes)} unique genes to query")

    try:
        already_done = pd.read_csv(OUTPUT_PATH)
        done_genes = set(already_done["gene"])
        results = already_done.to_dict("records")
        print(f"resuming: {len(done_genes)} already done")
    except FileNotFoundError:
        done_genes = set()
        results = []

    remaining_genes = [g for g in all_genes if g not in done_genes]
    print(f"{len(remaining_genes)} remaining")

    completed = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(get_pubmed_count, g): g for g in remaining_genes}
        for future in as_completed(futures):
            gene = futures[future]
            count = future.result()
            results.append({"gene": gene, "pubmed_count_t0": count if count is not None else -1,
                             "pubmed_queried": 1 if count is not None else 0})
            completed += 1
            if completed % 25 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
                print(f"{completed}/{len(remaining_genes)} done, checkpointed")

    final = pd.DataFrame(results)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nfinal: {len(final)} genes, {final['pubmed_queried'].sum()} successfully queried "
          f"({100*final['pubmed_queried'].sum()/len(final):.1f}%)")
