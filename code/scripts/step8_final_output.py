"""
Step 8 — final Stage 1 output: one CSV with the full resolved call per
patient variant, and a second CSV filtered to VUS-only rows for Stage 2.
"""
import pandas as pd

FINAL_COLS = [
    "chrom", "pos", "ref", "alt", "gene", "vaf", "gnomad_af", "cosmic_hotspot",
    "dbsnp_id", "hpa_expression_level", "hpa_reliability",
    "low_tissue_expression_flag", "clinvar_status", "predicted_class",
    "predicted_class_confidence",
]

if __name__ == "__main__":
    df = pd.read_csv("data/patient/step7_resolved.csv")
    final = df[FINAL_COLS].copy()
    final.to_csv("data/patient/stage1_final.csv", index=False)
    print(f"wrote data/patient/stage1_final.csv ({len(final)} rows)")

    vus_only = final[final["clinvar_status"] == "VUS"].copy()
    vus_only.to_csv("data/patient/stage1_vus_only.csv", index=False)
    print(f"wrote data/patient/stage1_vus_only.csv ({len(vus_only)} rows)")

    print("\nfinal clinvar_status counts:")
    print(final["clinvar_status"].value_counts())
    print("\nfinal predicted_class counts (VUS rows only):")
    print(vus_only["predicted_class"].value_counts())
