import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 42
TARGET_ENCODING_SMOOTHING = 20
EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]
CHROM_LIST = [str(i) for i in range(1, 23)] + ["X"]  # Y/MT collapsed into chr_other
CHROM_DUMMIES = [f"chr_{c}" for c in CHROM_LIST] + ["chr_other"]


def load_data():
    df = pd.read_csv("data/stage2/vus_features_v9.csv", low_memory=False)

    # MaveDB coverage was completed later (see 01_data_collection/07 and 08),
    # so overwrite the has_mave_coverage/mave_num_variants columns with that
    # complete, fully-retried result instead of trusting whatever shipped
    # with vus_features_v9.csv.
    mave_coverage = pd.read_csv("data/stage2/mavedb_gene_coverage_v2.csv")
    df = df.drop(columns=["has_mave_coverage", "mave_num_variants"]).merge(
        mave_coverage[["gene", "has_mave_coverage", "mave_num_variants"]].rename(columns={"gene": "GeneSymbol"}),
        on="GeneSymbol", how="left",
    )
    df["has_mave_coverage"] = df["has_mave_coverage"].fillna(False)
    df["mave_num_variants"] = df["mave_num_variants"].fillna(0)

    df["gnomad_af"] = pd.to_numeric(df["gnomad_af"], errors="coerce").fillna(0.0)
    df["gnomad_af_popmax"] = pd.to_numeric(df["annovar_gnomad_af_popmax"], errors="coerce").fillna(0.0)
    df["gnomad_af_log"] = np.log1p(df["gnomad_af"])
    df["gnomad_af_popmax_log"] = np.log1p(df["gnomad_af_popmax"])
    df["low_tissue_expression_flag"] = df["low_tissue_expression_flag"].astype(bool).astype(int)
    df["n_submitters_t0"] = pd.to_numeric(df["n_submitters_t0"], errors="coerce").fillna(0).astype(int)
    df["n_submitters_2018_12"] = pd.to_numeric(df["n_submitters_2018_12"], errors="coerce").fillna(0)
    df["has_mave_coverage"] = pd.to_numeric(df["has_mave_coverage"], errors="coerce").fillna(0).astype(int)
    df["mave_num_variants_log"] = np.log1p(pd.to_numeric(df["mave_num_variants"], errors="coerce").fillna(0))
    df["submission_velocity_t0"] = pd.to_numeric(df["submission_velocity_t0"], errors="coerce").fillna(0)
    df["submitter_multiple_flag"] = pd.to_numeric(df["submitter_multiple_flag"], errors="coerce").fillna(0).astype(int)
    df["pubmed_count_t0"] = pd.to_numeric(df["pubmed_count_t0"], errors="coerce").fillna(-1)
    df["pubmed_queried"] = pd.to_numeric(df["pubmed_queried"], errors="coerce").fillna(0).astype(int)
    df["litvar2_pmids_count"] = pd.to_numeric(df["litvar2_pmids_count"], errors="coerce").fillna(-1)
    df["litvar2_queried"] = pd.to_numeric(df["litvar2_queried"], errors="coerce").fillna(0).astype(int)
    df["review_status_stars_t0"] = pd.to_numeric(df["review_status_stars_t0"], errors="coerce").fillna(0).astype(int)

    exonic_func_clean = df["annovar_exonic_func"].fillna("unknown").str.replace(" ", "_").str.lower()
    for col in EXONIC_FUNC_DUMMIES:
        df[col] = (exonic_func_clean == col[3:]).astype(int)

    chrom = df["Chromosome"].astype(str)
    for c in CHROM_LIST:
        df[f"chr_{c}"] = (chrom == c).astype(int)
    df["chr_other"] = (~chrom.isin(CHROM_LIST)).astype(int)

    df["stars_x_submitters"] = df["review_status_stars_t0"] * df["n_submitters_t0"]
    df["pubmed_x_litvar2"] = np.log1p(df["pubmed_count_t0"].clip(lower=0)) * df["litvar2_queried"]

    y = df["resolved"].astype(bool).astype(int)
    return df, y


BASE_FEATURE_COLS = (
    ["gnomad_af", "gnomad_af_log", "gnomad_af_popmax", "gnomad_af_popmax_log",
     "low_tissue_expression_flag", "n_submitters_t0", "n_submitters_2018_12",
     "has_mave_coverage", "mave_num_variants_log",
     "submission_velocity_t0", "submitter_multiple_flag", "pubmed_count_t0", "pubmed_queried",
     "litvar2_pmids_count", "litvar2_queried", "review_status_stars_t0",
     "stars_x_submitters", "pubmed_x_litvar2"]
    + EXONIC_FUNC_DUMMIES + CHROM_DUMMIES
)


def fit_gene_target_encoding(df, train_idx, target_col, global_rate, smoothing=TARGET_ENCODING_SMOOTHING):
    """Smoothed per-gene target rate, fit on the training rows only."""
    stats = df.loc[train_idx].groupby("GeneSymbol")[target_col].agg(["sum", "count"])
    stats["rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
    return stats


def oof_gene_encoding(df, train_idx, target_col, global_rate, n_splits=5, smoothing=TARGET_ENCODING_SMOOTHING):
    """Same idea as fit_gene_target_encoding, but out-of-fold so the
    training rows don't see their own label baked into the encoding."""
    train_df = df.loc[train_idx].copy()
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    encoded = pd.Series(np.nan, index=train_df.index)
    for fold_train_idx, fold_val_idx in kfold.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        stats = fold_train.groupby("GeneSymbol")[target_col].agg(["sum", "count"])
        stats["rate"] = (stats["sum"] + smoothing * global_rate) / (stats["count"] + smoothing)
        val_genes = train_df.iloc[fold_val_idx]["GeneSymbol"]
        encoded.iloc[fold_val_idx] = val_genes.map(stats["rate"]).fillna(global_rate).values
    return encoded


def fit_lgbm_xgb_rf_stack(X_train, y_train, X_test, y_test, label):
    """Trains LightGBM, XGBoost, and Random Forest, then a logistic-
    regression meta-learner on their out-of-fold predictions, and reports
    all five variants (three base models, simple average, stacked) on the
    held-out test set."""
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    lgbm = LGBMClassifier(n_estimators=800, max_depth=8, num_leaves=63, learning_rate=0.02,
                           subsample=0.8, colsample_bytree=0.8,
                           scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE, verbose=-1)
    xgb = XGBClassifier(n_estimators=1200, max_depth=6, learning_rate=0.015, subsample=0.8,
                         colsample_bytree=0.8, min_child_weight=3,
                         scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE)
    rf = RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=3,
                                 class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)

    # 5-fold out-of-fold predictions on the training set, so the stacking
    # meta-learner isn't trained on predictions the base models have
    # already seen the answer for
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros((len(X_train), 3))
    for fold_train_idx, fold_val_idx in kfold.split(X_train):
        X_fold_train, X_fold_val = X_train.iloc[fold_train_idx], X_train.iloc[fold_val_idx]
        y_fold_train = y_train.iloc[fold_train_idx]
        fold_scale_pos_weight = (y_fold_train == 0).sum() / max((y_fold_train == 1).sum(), 1)

        fold_lgbm = LGBMClassifier(n_estimators=800, max_depth=8, num_leaves=63, learning_rate=0.02,
                                    subsample=0.8, colsample_bytree=0.8, scale_pos_weight=fold_scale_pos_weight,
                                    random_state=RANDOM_STATE, verbose=-1).fit(X_fold_train, y_fold_train)
        fold_xgb = XGBClassifier(n_estimators=1200, max_depth=6, learning_rate=0.015, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=3, scale_pos_weight=fold_scale_pos_weight,
                                  eval_metric="logloss", random_state=RANDOM_STATE).fit(X_fold_train, y_fold_train)
        fold_rf = RandomForestClassifier(n_estimators=500, max_depth=14, min_samples_leaf=3,
                                          class_weight="balanced_subsample", random_state=RANDOM_STATE,
                                          n_jobs=-1).fit(X_fold_train, y_fold_train)

        oof_preds[fold_val_idx, 0] = fold_lgbm.predict_proba(X_fold_val)[:, 1]
        oof_preds[fold_val_idx, 1] = fold_xgb.predict_proba(X_fold_val)[:, 1]
        oof_preds[fold_val_idx, 2] = fold_rf.predict_proba(X_fold_val)[:, 1]

    lgbm.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta_model.fit(oof_preds, y_train)

    test_preds = np.column_stack([
        lgbm.predict_proba(X_test)[:, 1],
        xgb.predict_proba(X_test)[:, 1],
        rf.predict_proba(X_test)[:, 1],
    ])
    stacked_proba = meta_model.predict_proba(test_preds)[:, 1]
    averaged_proba = test_preds.mean(axis=1)

    rows = []
    for name, proba in [("LightGBM", test_preds[:, 0]), ("XGBoost", test_preds[:, 1]),
                         ("RandomForest", test_preds[:, 2]), ("SimpleAverage", averaged_proba),
                         ("StackedLR", stacked_proba)]:
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "model": name,
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "pr_auc": average_precision_score(y_test, proba),
            "roc_auc": roc_auc_score(y_test, proba),
        })
    comparison = pd.DataFrame(rows).set_index("model")
    print(f"\n{label}, held-out test results:")
    print(comparison.round(4))
    return {"lgbm": lgbm, "xgb": xgb, "rf": rf, "meta": meta_model}, comparison


if __name__ == "__main__":
    df, y = load_data()
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    global_rate = y.loc[train_idx].mean()

    oof_encoding = oof_gene_encoding(df, train_idx, "resolved", global_rate)
    df.loc[train_idx, "gene_resolved_rate_te"] = oof_encoding.reindex(df.loc[train_idx].index).values
    full_train_encoding = fit_gene_target_encoding(df, train_idx, "resolved", global_rate)
    df.loc[test_idx, "gene_resolved_rate_te"] = df.loc[test_idx, "GeneSymbol"].map(full_train_encoding["rate"]).fillna(global_rate)

    gene_avg_submitters = df.loc[train_idx].groupby("GeneSymbol")["n_submitters_t0"].mean()
    global_avg_submitters = df.loc[train_idx, "n_submitters_t0"].mean()
    df["gene_avg_submitters_te"] = df["GeneSymbol"].map(gene_avg_submitters).fillna(global_avg_submitters)

    reclass_feature_cols = BASE_FEATURE_COLS + ["gene_resolved_rate_te", "gene_avg_submitters_te"]

    X_train, X_test = df.loc[train_idx, reclass_feature_cols], df.loc[test_idx, reclass_feature_cols]
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]

    reclass_models, reclass_comparison = fit_lgbm_xgb_rf_stack(X_train, y_train, X_test, y_test, "reclassification-probability model")
    reclass_comparison.to_csv("data/stage2/step4_model_comparison_v11.csv")

    best_model_name = reclass_comparison["pr_auc"].idxmax()
    print(f"\nbest reclassification model: {best_model_name} "
          f"(PR-AUC={reclass_comparison.loc[best_model_name, 'pr_auc']:.4f}, "
          f"ROC-AUC={reclass_comparison.loc[best_model_name, 'roc_auc']:.4f})")

    joblib.dump(reclass_models, "data/stage2/best_model_v11.joblib")
    with open("data/stage2/best_model_features_v11.txt", "w") as f:
        f.write(",".join(reclass_feature_cols))
    with open("data/stage2/best_model_name_v11.txt", "w") as f:
        f.write(f"stack(LightGBM,XGBoost,RandomForest)+LogisticRegression meta, best_by_pr_auc={best_model_name}")
    full_train_encoding[["rate"]].rename(columns={"rate": "gene_resolved_rate_te"}).join(
        gene_avg_submitters.rename("gene_avg_submitters_te")
    ).to_csv("data/stage2/gene_target_encoding_lookup_v11.csv")
    with open("data/stage2/gene_target_encoding_global_rate_v11.txt", "w") as f:
        f.write(f"{global_rate}\n{global_avg_submitters}\n")

    # direction classifier: Pathogenic (1) vs Benign (0), resolved rows only
    resolved_mask = df["resolved_direction"].isin(["Pathogenic", "Benign"])
    direction_df = df[resolved_mask].copy()
    direction_y = (direction_df["resolved_direction"] == "Pathogenic").astype(int)
    print(f"\ndirection classifier population: {len(direction_df)} rows "
          f"({(direction_y == 1).sum()} Pathogenic / {(direction_y == 0).sum()} Benign)")

    direction_train_idx = train_idx.intersection(direction_df.index)
    direction_test_idx = test_idx.intersection(direction_df.index)
    direction_global_rate = direction_y.loc[direction_train_idx].mean()

    # separate gene-level pathogenic-rate encoding, fit only on the
    # resolved+train subset so it can't leak into the test rows
    direction_df["resolved_direction_pathogenic"] = direction_y
    direction_gene_encoding = fit_gene_target_encoding(direction_df, direction_train_idx, "resolved_direction_pathogenic", direction_global_rate)
    direction_df["gene_pathogenic_rate_te"] = direction_df["GeneSymbol"].map(direction_gene_encoding["rate"]).fillna(direction_global_rate)

    direction_feature_cols = BASE_FEATURE_COLS + ["gene_pathogenic_rate_te"]

    Xd_train, Xd_test = direction_df.loc[direction_train_idx, direction_feature_cols], direction_df.loc[direction_test_idx, direction_feature_cols]
    yd_train, yd_test = direction_y.loc[direction_train_idx], direction_y.loc[direction_test_idx]

    direction_models, direction_comparison = fit_lgbm_xgb_rf_stack(Xd_train, yd_train, Xd_test, yd_test, "direction model (Pathogenic vs Benign)")
    direction_comparison.to_csv("data/stage2/step4_direction_model_comparison_v1.csv")

    # classes are roughly balanced here, so ROC-AUC is the fairer pick metric
    best_direction_model = direction_comparison["roc_auc"].idxmax()
    print(f"\nbest direction model: {best_direction_model} "
          f"(ROC-AUC={direction_comparison.loc[best_direction_model, 'roc_auc']:.4f}, "
          f"PR-AUC={direction_comparison.loc[best_direction_model, 'pr_auc']:.4f})")

    joblib.dump(direction_models, "data/stage2/direction_model_v1.joblib")
    with open("data/stage2/direction_model_features_v1.txt", "w") as f:
        f.write(",".join(direction_feature_cols))
    direction_gene_encoding[["rate"]].rename(columns={"rate": "gene_pathogenic_rate_te"}).to_csv(
        "data/stage2/gene_pathogenic_rate_lookup_v1.csv"
    )
    with open("data/stage2/gene_pathogenic_rate_global_v1.txt", "w") as f:
        f.write(str(direction_global_rate))
    with open("data/stage2/direction_model_cindex_v1.txt", "w") as f:
        f.write(f"roc_auc={direction_comparison.loc[best_direction_model, 'roc_auc']:.4f}\n"
                f"pr_auc={direction_comparison.loc[best_direction_model, 'pr_auc']:.4f}\n")

    print("\ndone. wrote best_model_v11.joblib, direction_model_v1.joblib, plus feature/lookup/comparison files to data/stage2/.")
