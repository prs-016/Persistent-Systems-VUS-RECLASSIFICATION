"""
Step 5e — real attempt to raise Stage 1's honest (vaf-excluded) PR-AUC
above 0.5255, in response to a direct push for higher numbers.

Confirmed first (step5d) that gnomad_af's high empty-rate is real biology,
not a missed annotation step: re-running ANNOVAR locally over all 45,706
rows returned byte-identical values to what was already in the table.
gnomad_af is already fully computed and already in the model -- it was
never the bottleneck.

Two new real, leak-safe features added instead:
  1. annovar_exonic_func one-hot (same ANNOVAR run already used for
     gnomad_af, consequence-severity column simply wasn't kept before).
  2. gene-level historical germline-rate, target-encoded with a strict
     5-fold out-of-fold fit within the TRAIN split only (same leak-safe
     technique that was Stage 2's single biggest real lever) -- captures
     that real germline-pathogenic variants cluster in known
     cancer-predisposition genes (BRCA1/2, TP53, MLH1, ...) while somatic
     driver mutations cluster in a different, overlapping-but-distinct
     gene set (TTN, MUC16, PIK3CA, ...).
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, average_precision_score, roc_auc_score
from xgboost import XGBClassifier
import joblib

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import step2_annotate_vep as s2  # noqa: E402

RANDOM_STATE = 42
SMOOTHING = 20
IN_PATH = "data/tcga_training/training_table_v3.csv"

EXONIC_FUNC_DUMMIES = [
    "ef_frameshift_deletion", "ef_frameshift_substitution", "ef_nonframeshift_deletion",
    "ef_nonframeshift_substitution", "ef_nonsynonymous_snv", "ef_startloss", "ef_stopgain",
    "ef_stoploss", "ef_synonymous_snv",
]

if __name__ == "__main__":
    df = pd.read_csv(IN_PATH, low_memory=False)
    print(f"loaded {len(df)} rows")

    annovar_out = s2.annotate_with_annovar(df[["chrom", "pos", "end", "ref", "alt"]].copy())
    assert len(annovar_out) == len(df)
    clean = annovar_out["annovar_exonic_func"].fillna("unknown").astype(str).str.replace(" ", "_").str.lower().values
    for col in EXONIC_FUNC_DUMMIES:
        df[col] = (clean == col[3:]).astype(int)

    df["gnomad_af"] = pd.to_numeric(df["gnomad_af"], errors="coerce").fillna(0.0)
    df["cosmic_hotspot"] = df["cosmic_hotspot"].astype(bool).astype(int)
    df["low_tissue_expression_flag"] = df["low_tissue_expression_flag"].astype(bool).astype(int)
    y = (df["label"] == "Germline").astype(int)

    idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    global_rate = y.loc[idx_train].mean()

    train_df = df.loc[idx_train].copy()
    train_df["_y"] = y.loc[idx_train].values
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_df["gene_germline_rate_te"] = np.nan
    for fold_train_idx, fold_val_idx in kf.split(train_df):
        fold_train = train_df.iloc[fold_train_idx]
        stats = fold_train.groupby("gene")["_y"].agg(["sum", "count"])
        stats["rate"] = (stats["sum"] + SMOOTHING * global_rate) / (stats["count"] + SMOOTHING)
        train_df.iloc[fold_val_idx, train_df.columns.get_loc("gene_germline_rate_te")] = (
            train_df.iloc[fold_val_idx]["gene"].map(stats["rate"]).fillna(global_rate).values
        )
    df.loc[idx_train, "gene_germline_rate_te"] = train_df["gene_germline_rate_te"].values

    full_train_stats = df.loc[idx_train].assign(_y=y.loc[idx_train]).groupby("gene")["_y"].agg(["sum", "count"])
    full_train_stats["rate"] = (full_train_stats["sum"] + SMOOTHING * global_rate) / (full_train_stats["count"] + SMOOTHING)
    df.loc[idx_test, "gene_germline_rate_te"] = df.loc[idx_test, "gene"].map(full_train_stats["rate"]).fillna(global_rate)

    FEATURE_COLS = ["cosmic_hotspot", "low_tissue_expression_flag", "gnomad_af", "gene_germline_rate_te"] + EXONIC_FUNC_DUMMIES
    X_train, X_test = df.loc[idx_train, FEATURE_COLS], df.loc[idx_test, FEATURE_COLS]
    y_train, y_test = y.loc[idx_train], y.loc[idx_test]

    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                                  eval_metric="logloss", random_state=RANDOM_STATE),
    }
    results = []
    best_name, best_pr_auc, best_model = None, -1, None
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        row = {"model": name, "accuracy": accuracy_score(y_test, y_pred),
               "precision": precision_score(y_test, y_pred, zero_division=0),
               "recall": recall_score(y_test, y_pred, zero_division=0),
               "pr_auc": pr_auc, "roc_auc": roc_auc}
        results.append(row)
        print(row)
        if pr_auc > best_pr_auc:
            best_name, best_pr_auc, best_model = name, pr_auc, model

    comparison = pd.DataFrame(results).set_index("model")
    print("\ncomparison (vaf excluded, +exonic_func +gene_germline_rate_te):")
    print(comparison.round(4))
    comparison.to_csv("data/tcga_training/step6_model_comparison_v5_richer.csv")
    print(f"\nbest: {best_name} PR-AUC={best_pr_auc:.4f} (prior production: LogisticRegression 0.5255)")

    joblib.dump(best_model, "data/tcga_training/best_model_v5.joblib")
    with open("data/tcga_training/best_model_name_v5.txt", "w") as f:
        f.write(best_name)
    with open("data/tcga_training/best_model_features_v5.txt", "w") as f:
        f.write(",".join(FEATURE_COLS))
    full_train_stats[["rate"]].rename(columns={"rate": "gene_germline_rate_te"}).to_csv(
        "data/tcga_training/gene_germline_rate_lookup_v5.csv")
    with open("data/tcga_training/gene_germline_rate_global_v5.txt", "w") as f:
        f.write(str(global_rate))
