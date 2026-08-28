"""
Step 6b — confusion matrices + concrete misclassified examples for each of
the 3 real-data models trained in step6_train_models.py (WITH gnomad_af
feature set, the one selected for Step 7). Re-runs the exact same
GroupShuffleSplit (same random_state=42) so results are directly comparable
to step6's saved metrics — not a new split, the same held-out test set.

Written for slide-deck use: "add a confusion matrix for each model, what it
was getting wrong, example etc."
"""
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.2
FEATURE_COLS = ["vaf", "cosmic_hotspot", "low_tissue_expression_flag", "gnomad_af"]


def load_data():
    df = pd.read_csv("data/tcga_training/training_table_real.csv")
    df["cosmic_hotspot"] = df["cosmic_hotspot"].astype(int)
    df["low_tissue_expression_flag"] = df["low_tissue_expression_flag"].astype(int)
    df["gnomad_af"] = pd.to_numeric(df["gnomad_af"], errors="coerce").fillna(0.0)
    y = (df["label"] == "Germline").astype(int)
    return df, y


if __name__ == "__main__":
    df, y = load_data()
    X = df[FEATURE_COLS].copy()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=df["variant_key"]))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    meta_test = df.iloc[test_idx].reset_index(drop=True)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    models = {
        "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=RANDOM_STATE),
    }

    all_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        print(f"\n{'='*60}\n{name} confusion matrix (test n={len(y_test)})\n{'='*60}")
        print(f"                 Predicted Somatic  Predicted Germline")
        print(f"Actual Somatic   {tn:>17}  {fp:>18}")
        print(f"Actual Germline  {fn:>17}  {tp:>18}")
        print(f"\nFalse positives (Somatic predicted as Germline): {fp}")
        print(f"False negatives (real Germline predicted as Somatic): {fn}")

        wrong_mask = y_pred != y_test.values
        wrong = meta_test.loc[wrong_mask, ["chrom", "pos", "gene", "vaf", "cosmic_hotspot", "gnomad_af", "label"]].copy()
        wrong["predicted"] = ["Germline" if p == 1 else "Somatic" for p in y_pred[wrong_mask]]
        print(f"\nsample misclassified rows ({name}), up to 5:")
        print(wrong.head(5).to_string(index=False))

        all_rows.append({
            "model": name, "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "n_test": len(y_test),
        })
        wrong.to_csv(f"data/tcga_training/step6b_misclassified_{name}.csv", index=False)

    pd.DataFrame(all_rows).set_index("model").to_csv("data/tcga_training/step6b_confusion_matrices.csv")
    print("\nwrote data/tcga_training/step6b_confusion_matrices.csv and per-model misclassified CSVs")
