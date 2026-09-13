"""
Trains two candidate models (Logistic Regression as an interpretable
baseline, Random Forest as a stronger nonlinear model), evaluates both on
a held-out test set, and persists whichever has the higher ROC-AUC —
along with a metrics.json report and a feature-importance CSV — for the
inference layer to load.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
from preprocessing import build_pipeline, ALL_FEATURES, TARGET

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "loan_data.csv"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
RANDOM_STATE = 42


def evaluate(name, pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["no_default", "default"]))
    print(f"ROC-AUC: {metrics['roc_auc']}")
    return metrics


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        ),
        "random_forest": build_pipeline(
            RandomForestClassifier(
                n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )
        ),
    }

    results = {}
    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        results[name] = {
            "pipeline": pipeline,
            "metrics": evaluate(name, pipeline, X_test, y_test),
        }

    best_name = max(results, key=lambda n: results[n]["metrics"]["roc_auc"])
    best_pipeline = results[best_name]["pipeline"]
    print(f"\nBest model by ROC-AUC: {best_name}")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipeline, MODELS_DIR / "model.joblib")

    metrics_report = {name: r["metrics"] for name, r in results.items()}
    metrics_report["selected_model"] = best_name
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metrics_report, f, indent=2)

    # Feature importance (only meaningful for the tree-based model, but
    # saved whenever available so it's easy to inspect either way)
    if best_name == "random_forest":
        feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = best_pipeline.named_steps["classifier"].feature_importances_
        importance_df = pd.DataFrame({
            "feature": feature_names, "importance": importances
        }).sort_values("importance", ascending=False)
        importance_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
        print("\nTop 5 features by importance:")
        print(importance_df.head(5).to_string(index=False))

    print(f"\nSaved model -> {MODELS_DIR / 'model.joblib'}")
    print(f"Saved metrics -> {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
