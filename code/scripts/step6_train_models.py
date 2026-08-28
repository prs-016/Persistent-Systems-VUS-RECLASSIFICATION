"""
Step 6 — train and compare Logistic Regression, Random Forest, XGBoost, and
TabPFN on the REAL labeled training table (Step 5/5a/5b), with a held-out
variant-level test split.

REVAMP NOTE (this session): this replaces the gnomAD-proxy-based training
run. Key changes, each made for a documented reason (see
docs/STAGE1_RESULTS.md for the full writeup):

1. Real labels only: 9,081 real TCGA somatic MAF calls + 853 real TCGA
   germline Pathogenic/Likely-Pathogenic calls (Huang et al. 2018) — no
   gnomAD-common-variant proxy. Natural class ratio (~91:9) is kept as-is
   (not artificially rebalanced by duplication or discarding real data);
   class_weight="balanced"/scale_pos_weight handles the imbalance, same
   mechanism as before.
2. Variant-level (grouped) train/test split: 279 of the 9,934 rows are
   duplicate variant coordinates (same chrom:pos:ref:alt seen in a
   different patient — mostly on the germline side, 267/853). A plain
   row-level stratified split would let the identical variant appear in
   both train and test. GroupShuffleSplit on `variant_key` prevents that.
3. gnomad_af is now a feature (ablation-tested, run both with and without)
   — safe to include now that the germline label isn't gnomAD-threshold-
   defined, but NOT fully clean: 853/853 germline-positive rows used PM2
   (population rarity) as one of their original pathogenicity-classification
   criteria, so gnomad_af is still correlated with the label through how
   Table S2A was curated, not only through biological origin. Watch the
   PR-AUC>0.98 leakage tripwire below.
4. exonic_func / consequence-severity and clnsig are deliberately NOT
   included as features: clnsig would cause train/serve skew (Step 7 only
   ever routes clnsig-ambiguous variants to this model, so it would never
   see a confident clnsig value at inference), and exonic_func has the same
   curation-bias problem as gnomad_af but worse (PVS1/loss-of-function was
   used in 75.5% of germline-positive rows' classification — 83% of the
   853 rows are stop-gained/frameshift/splice variants, not a natural
   somatic-vs-germline functional distribution).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, average_precision_score
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
POSITIVE_LABEL = "Germline"  # minority class; precision/recall/PR-AUC reported w.r.t. this class

BASE_FEATURE_COLS = ["vaf", "cosmic_hotspot", "low_tissue_expression_flag"]
GNOMAD_COL = "gnomad_af"

# Optional CLI override so this script can be reused for the v2 (expanded
# germline) ablation without duplicating the file — see step5c_expand_germline.py.
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/tcga_training/training_table_real.csv"
OUT_SUFFIX = "_v2" if "v2" in DATA_PATH else ""

# DISCOVERED THIS SESSION: v2's added 10,000 ClinVar germline rows all
# carry an ASSUMED vaf=0.5 (no measured VAF exists for them — see
# step5c_expand_germline.py). 92.5% of v2's Germline rows land exactly at
# vaf==0.5 vs. 0.58% of Somatic rows — this is circular by construction
# (I assigned that constant), not biological signal, and produced a
# suspicious PR-AUC jump to ~0.98 that tripped the leakage warning below.
# --novaf drops `vaf` from BASE_FEATURE_COLS entirely, for a fair,
# non-circular read on whether the expanded germline pool helps using
# only cosmic_hotspot/low_tissue_expression_flag/gnomad_af.
if "--novaf" in sys.argv:
    BASE_FEATURE_COLS = ["cosmic_hotspot", "low_tissue_expression_flag"]
    OUT_SUFFIX += "_novaf"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["cosmic_hotspot"] = df["cosmic_hotspot"].astype(int)
    df["low_tissue_expression_flag"] = df["low_tissue_expression_flag"].astype(int)
    # gnomad_af: "." / NaN from ANNOVAR means "not found in gnomAD", not a
    # measured AF of 0 — filled with 0 here as a documented approximation
    # (XGBoost could take NaN natively, but LogisticRegression/RandomForest
    # cannot, so 0-fill is used uniformly across all four models for a fair
    # comparison).
    df[GNOMAD_COL] = pd.to_numeric(df[GNOMAD_COL], errors="coerce").fillna(0.0)
    y = (df["label"] == POSITIVE_LABEL).astype(int)
    return df, y


def group_split(df, y, feature_cols):
    X = df[feature_cols].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=df["variant_key"]))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def evaluate(name, y_true, y_pred, y_score, results):
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
    })


def run_experiment(df, y, feature_cols, label):
    print(f"\n{'='*70}\nEXPERIMENT: {label}\nfeatures: {feature_cols}\n{'='*70}")
    X_train, X_test, y_train, y_test = group_split(df, y, feature_cols)
    print(f"train: {X_train.shape}, test: {X_test.shape}")
    print(f"train class counts: {y_train.value_counts().to_dict()}")
    print(f"test class counts: {y_test.value_counts().to_dict()}")
    # verify no variant_key leaks across the split
    train_keys = set(df.loc[X_train.index, "variant_key"])
    test_keys = set(df.loc[X_test.index, "variant_key"])
    overlap = train_keys & test_keys
    print(f"variant_key overlap between train/test: {len(overlap)} (must be 0)")
    assert len(overlap) == 0

    results = []

    logreg = LogisticRegression(class_weight="balanced", max_iter=1000)
    logreg.fit(X_train, y_train)
    evaluate("LogisticRegression", y_test, logreg.predict(X_test), logreg.predict_proba(X_test)[:, 1], results)

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)
    evaluate("RandomForest", y_test, rf.predict(X_test), rf.predict_proba(X_test)[:, 1], results)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE)
    xgb.fit(X_train, y_train)
    evaluate("XGBoost", y_test, xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1], results)

    try:
        from tabpfn import TabPFNClassifier
        tabpfn = TabPFNClassifier(random_state=RANDOM_STATE, device="cpu", ignore_pretraining_limits=True)
        tabpfn.fit(X_train.to_numpy(), y_train.to_numpy())
        tabpfn_proba = tabpfn.predict_proba(X_test.to_numpy())[:, 1]
        tabpfn_pred = (tabpfn_proba >= 0.5).astype(int)
        evaluate("TabPFN", y_test, tabpfn_pred, tabpfn_proba, results)
    except Exception as e:
        print(f"TabPFN skipped: {e}")

    results_df = pd.DataFrame(results).set_index("model")
    print(f"\ncomparison table ({label}):")
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))

    for _, row in results_df.iterrows():
        if row["pr_auc"] > 0.98:
            print(f"\nWARNING: {row.name} PR-AUC = {row['pr_auc']:.4f} is suspiciously close to 1.0 — check for leakage")

    return results_df, (logreg, rf, xgb), (X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    df, y = load_data()
    print("full dataset:", len(df), "rows, class counts:", y.value_counts().to_dict())
    print("unique variant_key groups:", df["variant_key"].nunique())

    # Ablation A: without gnomad_af (matches the old feature set, but now on real labels)
    results_no_gnomad, _, _ = run_experiment(df, y, BASE_FEATURE_COLS, "WITHOUT gnomad_af")
    results_no_gnomad.to_csv(f"data/tcga_training/step6_model_comparison_no_gnomad{OUT_SUFFIX}.csv")

    # Ablation B: with gnomad_af
    feature_cols_with_gnomad = BASE_FEATURE_COLS + [GNOMAD_COL]
    results_with_gnomad, models, split = run_experiment(df, y, feature_cols_with_gnomad, "WITH gnomad_af")
    results_with_gnomad.to_csv(f"data/tcga_training/step6_model_comparison_with_gnomad{OUT_SUFFIX}.csv")

    print(f"\n{'='*70}\nSUMMARY: PR-AUC with vs without gnomad_af\n{'='*70}")
    summary = pd.DataFrame({
        "pr_auc_without_gnomad": results_no_gnomad["pr_auc"],
        "pr_auc_with_gnomad": results_with_gnomad["pr_auc"],
    })
    summary["delta"] = summary["pr_auc_with_gnomad"] - summary["pr_auc_without_gnomad"]
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(f"data/tcga_training/step6_gnomad_ablation_summary{OUT_SUFFIX}.csv")

    print("\nselecting best model by PR-AUC (WITH gnomad_af) for Step 7...")
    best_name = results_with_gnomad["pr_auc"].idxmax()
    print("best model:", best_name)

    import joblib
    best_models = dict(zip(["LogisticRegression", "RandomForest", "XGBoost"], models))
    if best_name in best_models:
        # NOTE: v2 (expanded-germline) results are saved with a separate
        # suffix, NOT overwriting the v1 best_model.joblib that Step 7
        # currently loads — promote manually once v1-vs-v2 is compared.
        joblib.dump(best_models[best_name], f"data/tcga_training/best_model{OUT_SUFFIX}.joblib")
        with open(f"data/tcga_training/best_model_name{OUT_SUFFIX}.txt", "w") as f:
            f.write(best_name)
        with open(f"data/tcga_training/best_model_features{OUT_SUFFIX}.txt", "w") as f:
            f.write(",".join(feature_cols_with_gnomad))
        print(f"saved {best_name} (features: {feature_cols_with_gnomad})")
    else:
        print(f"{best_name} was TabPFN — not re-saved (not persisted via joblib in this pipeline)")
