import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "common"))
from feature_pipeline import load_data, BASE_FEATURE_COLS  # noqa: E402

RANDOM_STATE = 42

if __name__ == "__main__":
    core_df, core_y = load_data()
    core_train_idx, core_test_idx = train_test_split(
        core_df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=core_y
    )
    print(f"core: {len(core_df)} rows, train={len(core_train_idx)}, test={len(core_test_idx)}")

    cohort2021 = pd.read_csv("data/stage2/cohort_2021_features_v2_real_af.csv", low_memory=False)
    cohort2021["resolved"] = cohort2021["resolved"].astype(int)
    print(f"cohort2021 (real AF, new-only, not in core): {len(cohort2021)} rows, "
          f"resolved rate {cohort2021['resolved'].mean()*100:.2f}%")

    reclass_feature_cols = [c for c in BASE_FEATURE_COLS if not c.startswith("chr_")] + ["GeneSymbol"]
    gene_col_index = [reclass_feature_cols.index("GeneSymbol")]

    # baseline: core-only, exactly reproducing the existing v13 result
    y_train_core = core_y.loc[core_train_idx]
    y_test_core = core_y.loc[core_test_idx]
    X_train_core = core_df.loc[core_train_idx, reclass_feature_cols]
    X_test_core = core_df.loc[core_test_idx, reclass_feature_cols]
    scale_pos_weight_core = (y_train_core == 0).sum() / (y_train_core == 1).sum()

    baseline_model = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.03, l2_leaf_reg=5,
                                         scale_pos_weight=scale_pos_weight_core, random_seed=RANDOM_STATE,
                                         verbose=False, cat_features=gene_col_index, eval_metric="PRAUC")
    baseline_model.fit(X_train_core, y_train_core)
    baseline_test_proba = baseline_model.predict_proba(X_test_core)[:, 1]
    pr_auc_baseline = average_precision_score(y_test_core, baseline_test_proba)
    roc_auc_baseline = roc_auc_score(y_test_core, baseline_test_proba)
    print(f"\n[baseline, core-only] PR-AUC={pr_auc_baseline:.4f}, ROC-AUC={roc_auc_baseline:.4f}")

    # expanded: core train + cohort2021-real-AF (held-out core test set untouched)
    cohort2021_train_idx, cohort2021_test_idx = train_test_split(
        cohort2021.index, test_size=0.2, random_state=RANDOM_STATE, stratify=cohort2021["resolved"]
    )
    X_train_expanded = pd.concat([
        core_df.loc[core_train_idx, reclass_feature_cols],
        cohort2021.loc[cohort2021_train_idx, reclass_feature_cols],
    ], ignore_index=True)
    y_train_expanded = pd.concat([
        y_train_core.reset_index(drop=True),
        cohort2021.loc[cohort2021_train_idx, "resolved"].reset_index(drop=True),
    ], ignore_index=True)
    print(f"\n[expanded, real AF] train size: {len(X_train_expanded)} (core {len(core_train_idx)} + "
          f"cohort2021-train {len(cohort2021_train_idx)}), positive rate {y_train_expanded.mean()*100:.2f}%")

    scale_pos_weight_expanded = (y_train_expanded == 0).sum() / (y_train_expanded == 1).sum()
    expanded_model = CatBoostClassifier(iterations=1500, depth=8, learning_rate=0.03, l2_leaf_reg=5,
                                         scale_pos_weight=scale_pos_weight_expanded, random_seed=RANDOM_STATE,
                                         verbose=False, cat_features=gene_col_index, eval_metric="PRAUC")
    expanded_model.fit(X_train_expanded, y_train_expanded)

    expanded_on_core_proba = expanded_model.predict_proba(X_test_core)[:, 1]
    pr_auc_expanded_core = average_precision_score(y_test_core, expanded_on_core_proba)
    roc_auc_expanded_core = roc_auc_score(y_test_core, expanded_on_core_proba)
    print(f"\n[expanded model, evaluated on core test set] PR-AUC={pr_auc_expanded_core:.4f}, "
          f"ROC-AUC={roc_auc_expanded_core:.4f}")
    print(f"  delta vs baseline core-only: {pr_auc_expanded_core - pr_auc_baseline:+.4f} PR-AUC")

    X_cohort2021_test = cohort2021.loc[cohort2021_test_idx, reclass_feature_cols]
    y_cohort2021_test = cohort2021.loc[cohort2021_test_idx, "resolved"]
    baseline_on_cohort2021_proba = baseline_model.predict_proba(X_cohort2021_test)[:, 1]
    expanded_on_cohort2021_proba = expanded_model.predict_proba(X_cohort2021_test)[:, 1]
    pr_auc_baseline_cohort2021 = average_precision_score(y_cohort2021_test, baseline_on_cohort2021_proba)
    pr_auc_expanded_cohort2021 = average_precision_score(y_cohort2021_test, expanded_on_cohort2021_proba)
    print(f"\n[held-out cohort2021 (real AF) test, {len(cohort2021_test_idx)} rows, "
          f"resolved rate {y_cohort2021_test.mean()*100:.2f}%]")
    print(f"  baseline (core-only) model: PR-AUC={pr_auc_baseline_cohort2021:.4f}")
    print(f"  expanded (core+2021, real AF) model: PR-AUC={pr_auc_expanded_cohort2021:.4f}")

    comparison = pd.DataFrame([
        {"model": "baseline_core_only", "eval_set": "core_test", "pr_auc": pr_auc_baseline, "roc_auc": roc_auc_baseline},
        {"model": "expanded_core+2021_realAF", "eval_set": "core_test", "pr_auc": pr_auc_expanded_core, "roc_auc": roc_auc_expanded_core},
        {"model": "baseline_core_only", "eval_set": "cohort2021_test", "pr_auc": pr_auc_baseline_cohort2021, "roc_auc": np.nan},
        {"model": "expanded_core+2021_realAF", "eval_set": "cohort2021_test", "pr_auc": pr_auc_expanded_cohort2021, "roc_auc": np.nan},
    ])
    comparison.to_csv("data/stage2/step23_real_af_comparison.csv", index=False)
    print("\nfull comparison:")
    print(comparison.round(4).to_string())

    decision = "expanded" if pr_auc_expanded_core >= pr_auc_baseline else "baseline"
    print(f"\nDECISION (by measured PR-AUC on the untouched core test set): {decision}")
    with open("data/stage2/step23_decision.txt", "w") as f:
        f.write(f"decision={decision}\n")
        f.write(f"pr_auc_baseline_core_only={pr_auc_baseline:.4f}\n")
        f.write(f"pr_auc_expanded_realAF_on_core_test={pr_auc_expanded_core:.4f}\n")
        f.write(f"pr_auc_baseline_on_cohort2021_realAF_test={pr_auc_baseline_cohort2021:.4f}\n")
        f.write(f"pr_auc_expanded_on_cohort2021_realAF_test={pr_auc_expanded_cohort2021:.4f}\n")

    # persist the expanded model regardless of `decision`, it's worth keeping either way
    expanded_model.save_model("data/stage2/best_model_v15_realaf.cbm")
    with open("data/stage2/best_model_features_v15_realaf.txt", "w") as f:
        f.write(",".join(reclass_feature_cols))
    print(f"\nsaved expanded (real-AF) model as best_model_v15_realaf.cbm, decision={decision}")
