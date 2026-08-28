import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

OUTPUT_PATH = "data/stage2/litvar2_variant_pmids.csv"
NEGATIVE_SAMPLE_SIZE = 10000
RANDOM_STATE = 42


def get_mention_count(rsid: str):
    for attempt in range(3):
        try:
            response = requests.get(
                "https://www.ncbi.nlm.nih.gov/research/litvar2-api/variant/autocomplete/",
                params={"query": rsid}, timeout=15,
            )
            if response.status_code == 200:
                data = response.json()
                match = next((d for d in data if d.get("rsid") == rsid), None)
                return match.get("pmids_count") if match else 0
        except Exception:
            continue
    return None


if __name__ == "__main__":
    rsid_map = pd.read_csv("data/stage2/vus_rsid_map.csv", low_memory=False)
    labels = pd.read_csv("data/stage2/vus_features_v4.csv", low_memory=False, usecols=["VariationID", "resolved"])
    rsid_map = rsid_map.merge(labels, on="VariationID")
    rsid_map = rsid_map[rsid_map["rs_number"].notna() & (rsid_map["rs_number"] != "-1")]

    positive_rsids = set(rsid_map.loc[rsid_map["resolved"], "rs_number"].unique())
    negative_rsids_all = rsid_map.loc[~rsid_map["resolved"], "rs_number"].unique()
    negative_sample = pd.Series(negative_rsids_all).sample(
        n=min(NEGATIVE_SAMPLE_SIZE, len(negative_rsids_all)), random_state=RANDOM_STATE
    )
    target_rsids = positive_rsids | set(negative_sample)
    print(f"target: {len(positive_rsids)} positive-class rsIDs + {len(negative_sample)} sampled negative-class rsIDs = {len(target_rsids)} total")

    try:
        already_done = pd.read_csv(OUTPUT_PATH)
        done_rsids = set(already_done["rs_number"])
        results = already_done.to_dict("records")
        print(f"resuming: {len(done_rsids)} already done")
    except FileNotFoundError:
        done_rsids = set()
        results = []

    remaining_rsids = list(target_rsids - done_rsids)
    print(f"{len(remaining_rsids)} remaining")

    completed = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = {pool.submit(get_mention_count, rs): rs for rs in remaining_rsids}
        for future in as_completed(futures):
            rsid = futures[future]
            count = future.result()
            results.append({"rs_number": rsid, "litvar2_pmids_count": count if count is not None else -1,
                             "litvar2_queried": 1 if count is not None else 0})
            completed += 1
            if completed % 500 == 0:
                pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
                print(f"{completed}/{len(remaining_rsids)} done, checkpointed")

    final = pd.DataFrame(results)
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nfinal: {len(final)} rsIDs, {final['litvar2_queried'].sum()} successfully queried")
