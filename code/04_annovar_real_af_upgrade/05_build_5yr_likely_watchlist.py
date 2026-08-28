import pandas as pd

CUTOFF = 0.05

if __name__ == "__main__":
    watchlist = pd.read_csv("data/stage2/vus_global_watchlist_v18_ALL_CURRENT_VUS.csv.gz", low_memory=False)
    print(f"global watchlist (all current VUS): {len(watchlist)} rows")
    print(f"p_resolved_by_5y: min={watchlist['p_resolved_by_5y'].min():.4f}, "
          f"mean={watchlist['p_resolved_by_5y'].mean():.4f}, median={watchlist['p_resolved_by_5y'].median():.4f}, "
          f"p99={watchlist['p_resolved_by_5y'].quantile(0.99):.4f}, max={watchlist['p_resolved_by_5y'].max():.4f}")

    likely_5y = watchlist[watchlist["p_resolved_by_5y"] >= CUTOFF].copy()
    likely_5y = likely_5y.sort_values("p_resolved_by_5y", ascending=False).reset_index(drop=True)
    print(f"\n'likely within 5 years' tier (p_resolved_by_5y >= {CUTOFF}): "
          f"{len(likely_5y)} rows ({len(likely_5y)/len(watchlist)*100:.2f}% of the global watchlist)")
    print(f"  cohort breakdown:\n{likely_5y['cohort'].value_counts()}")
    print(f"  reclass_probability in this tier: mean={likely_5y['reclass_probability'].mean():.4f}")

    likely_5y.to_csv("data/stage2/vus_watchlist_likely_within_5y.csv", index=False)
    likely_5y.to_csv("data/stage2/vus_watchlist_likely_within_5y.csv.gz", index=False, compression="gzip")
    print(f"\nwrote data/stage2/vus_watchlist_likely_within_5y.csv(.gz), shape={likely_5y.shape}")

    print("\ntop 15 by p_resolved_by_5y:")
    print(likely_5y.head(15)[["ClinvarID", "GeneSymbol", "reclass_probability", "p_resolved_by_5y",
                               "direction_pathogenic_probability_if_resolved", "cohort"]].to_string())
