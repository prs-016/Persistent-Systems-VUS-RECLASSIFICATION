import pandas as pd

OLD_SNAPSHOT_PATH = "data/stage2/old_slim.tsv"
CURRENT_SNAPSHOT_PATH = "data/stage2/current_slim.tsv"
OUTPUT_PATH = "data/stage2/reclassification_labels.csv"

PATHOGENIC_TERMS = {"pathogenic", "likely pathogenic", "pathogenic/likely pathogenic"}
BENIGN_TERMS = {"benign", "likely benign", "benign/likely benign"}


def classify_significance(value) -> str:
    if pd.isna(value):
        return "other"
    text = str(value).lower()
    if text in PATHOGENIC_TERMS:
        return "Pathogenic"
    if text in BENIGN_TERMS:
        return "Benign"
    if text == "uncertain significance":
        return "VUS"
    return "other"


def load_snapshot(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    return df.drop_duplicates(subset="VariationID")


def build_labels() -> pd.DataFrame:
    old_snapshot = load_snapshot(OLD_SNAPSHOT_PATH)
    current_snapshot = load_snapshot(CURRENT_SNAPSHOT_PATH)
    old_snapshot["bucket_t0"] = old_snapshot["ClinicalSignificance"].apply(classify_significance)
    current_snapshot["bucket_t1"] = current_snapshot["ClinicalSignificance"].apply(classify_significance)

    vus_at_start = old_snapshot[old_snapshot["bucket_t0"] == "VUS"][
        ["VariationID", "GeneSymbol", "NumberSubmitters", "ReviewStatus", "LastEvaluated"]
    ].rename(columns={
        "NumberSubmitters": "n_submitters_t0",
        "ReviewStatus": "review_status_t0",
        "LastEvaluated": "last_evaluated_t0",
    })

    labels = vus_at_start.merge(
        current_snapshot[["VariationID", "bucket_t1", "ClinicalSignificance", "NumberSubmitters", "LastEvaluated"]].rename(
            columns={"ClinicalSignificance": "clinsig_t1", "NumberSubmitters": "n_submitters_t1", "LastEvaluated": "last_evaluated_t1"}
        ),
        on="VariationID", how="left",
    )
    labels["tracked_in_current"] = labels["bucket_t1"].notna()
    labels["resolved"] = labels["bucket_t1"].isin(["Pathogenic", "Benign"])
    labels["resolved_direction"] = labels["bucket_t1"].where(labels["resolved"])
    return labels


if __name__ == "__main__":
    labels = build_labels()
    print("VUS as of the old snapshot:", len(labels))

    still_tracked = labels[labels["tracked_in_current"]]
    print("still tracked today:", len(still_tracked), f"({len(still_tracked)/len(labels):.1%})")

    print("\ncurrent-status breakdown:")
    print(still_tracked["bucket_t1"].value_counts())

    resolved = still_tracked[still_tracked["resolved"]]
    print(f"\nresolved (VUS -> Pathogenic/Benign): {len(resolved)} / {len(still_tracked)} = {len(resolved)/len(still_tracked):.2%}")
    print(resolved["resolved_direction"].value_counts())

    labels.to_csv(OUTPUT_PATH, index=False)
    print(f"\nwrote {OUTPUT_PATH}, shape={labels.shape}")
