import sys
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import step2_annotate_vep as annovar_helper  # noqa: E402
import step3_hpa as hpa_helper  # noqa: E402

OLD_SNAPSHOT_FULL_PATH = "data/stage2/old_full.tsv"
LABELS_PATH = "data/stage2/reclassification_labels.csv"
OUTPUT_PATH = "data/stage2/vus_features.csv"


def attach_coordinates() -> pd.DataFrame:
    old_snapshot = pd.read_csv(OLD_SNAPSHOT_FULL_PATH, sep="\t", dtype=str)
    old_snapshot_grch38 = old_snapshot[old_snapshot["Assembly"] == "GRCh38"].drop_duplicates(subset="VariationID")

    labels = pd.read_csv(LABELS_PATH, dtype=str)
    with_coords = labels.merge(
        old_snapshot_grch38[["VariationID", "Chromosome", "Start", "Stop", "ReferenceAllele", "AlternateAllele"]],
        on="VariationID", how="left",
    )

    usable = with_coords[
        with_coords["Chromosome"].notna()
        & with_coords["ReferenceAllele"].notna() & (with_coords["ReferenceAllele"] != "na")
        & with_coords["AlternateAllele"].notna() & (with_coords["AlternateAllele"] != "na")
    ].copy()

    usable["chrom"] = "chr" + usable["Chromosome"].astype(str)
    usable["pos"] = pd.to_numeric(usable["Start"], errors="coerce")
    usable["end"] = pd.to_numeric(usable["Stop"], errors="coerce")
    usable["ref"] = usable["ReferenceAllele"]
    usable["alt"] = usable["AlternateAllele"]
    usable = usable.dropna(subset=["pos", "end"])
    usable["pos"] = usable["pos"].astype(int)
    usable["end"] = usable["end"].astype(int)
    return usable.reset_index(drop=True)


def build_features(usable: pd.DataFrame) -> pd.DataFrame:
    annovar_result = annovar_helper.annotate_with_annovar(usable[["chrom", "pos", "end", "ref", "alt"]])
    assert len(annovar_result) == len(usable), f"ANNOVAR row mismatch: {len(annovar_result)} vs {len(usable)}"

    usable = usable.reset_index(drop=True)
    usable["gnomad_af"] = pd.to_numeric(annovar_result["annovar_gnomad_af"], errors="coerce")

    # Cache the full ANNOVAR output (not just gnomad_af) so we can pull in
    # exonic_func / consequence severity later without re-running ANNOVAR
    # over all 195,127 rows again.
    annovar_cache = usable[["VariationID"]].copy()
    annovar_cache["annovar_func"] = annovar_result["annovar_func"].values
    annovar_cache["annovar_exonic_func"] = annovar_result["annovar_exonic_func"].values
    annovar_cache["annovar_gnomad_af_popmax"] = pd.to_numeric(annovar_result["annovar_gnomad_af_popmax"], errors="coerce").values
    annovar_cache.to_csv("data/stage2/annovar_full_cache.csv", index=False)

    genes = pd.DataFrame({"gene": usable["GeneSymbol"].dropna().unique()})
    hpa_result = hpa_helper.annotate_hpa(genes, tissue="breast")
    gene_to_low_expression = dict(zip(hpa_result["gene"], hpa_result["low_tissue_expression_flag"]))
    usable["low_tissue_expression_flag"] = usable["GeneSymbol"].map(gene_to_low_expression).fillna(False)

    usable["n_submitters_t0"] = pd.to_numeric(usable["n_submitters_t0"], errors="coerce").fillna(0).astype(int)
    return usable


if __name__ == "__main__":
    usable_rows = attach_coordinates()
    print(f"VUS with usable GRCh38 coordinates: {len(usable_rows)} / 212782")

    features = build_features(usable_rows)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH}, shape={features.shape}")
    print("\ngnomad_af non-null:", f"{features['gnomad_af'].notna().mean():.1%}")
    print("resolved counts:\n", features["resolved"].value_counts())
