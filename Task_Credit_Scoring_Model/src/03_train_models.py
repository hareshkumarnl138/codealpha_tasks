"""Train and compare baseline credit-default classifiers."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, classification_report, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay
)
from src.common import load_raw_data, feature_columns, TARGET, ID_COLUMN, MODELS, FIGURES, TABLES, setup_logging

RANDOM_STATE = 42
TEST_SIZE = 0.20
CATEGORICAL_CANDIDATES = ["sex", "education", "marriage"]

def build_pipeline(X: pd.DataFrame, estimator):
    categorical = [c for c in CATEGORICAL_CANDIDATES if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ], remainder="drop")
    return Pipeline([("preprocess", preprocess), ("model", estimator)])

def main():
    logger = setup_logging("training")
    MODELS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    X = df[feature_columns(df)].drop(columns=[ID_COLUMN], errors="ignore")
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    rows = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1_score": f1_score(y_test, pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
            "average_precision": average_precision_score(y_test, proba),
        }
        rows.append(metrics)
        fitted[name] = pipeline
        dump(pipeline, MODELS / f"{name}.joblib")
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        pd.DataFrame(report).transpose().to_csv(TABLES / f"{name}_classification_report.csv")

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(f"Confusion Matrix — {name.replace('_', ' ').title()}")
        fig.tight_layout()
        fig.savefig(FIGURES / f"{name}_confusion_matrix.png", dpi=160)
        plt.close(fig)

        logger.info("%s | Accuracy %.4f | Precision %.4f | Recall %.4f | F1 %.4f | ROC-AUC %.4f",
                    name, metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"], metrics["roc_auc"])

    results = pd.DataFrame(rows).sort_values(["roc_auc", "f1_score"], ascending=False)
    results.to_csv(TABLES / "model_comparison.csv", index=False)
    best_name = str(results.iloc[0]["model"])
    dump(fitted[best_name], MODELS / "best_model.joblib")
    with open(MODELS / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_name,
            "target": TARGET,
            "positive_class": 1,
            "negative_class": 0,
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "feature_columns": X.columns.tolist(),
            "selection_rule": "Highest ROC-AUC; F1-score breaks ties",
            "note": "Metrics are generated from the current run and held-out stratified test set."
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = results.set_index("model")[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Credit Default Model Comparison")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=160)
    plt.close(fig)

    # Curves for the selected model on the same untouched test split.
    best_pipeline = fitted[best_name]
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"ROC Curve — {best_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(FIGURES / "best_model_roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_estimator(best_pipeline, X_test, y_test, ax=ax)
    ax.set_title(f"Precision–Recall Curve — {best_name.replace('_', ' ').title()}")
    fig.tight_layout()
    fig.savefig(FIGURES / "best_model_precision_recall_curve.png", dpi=160)
    plt.close(fig)

    logger.info("Best model by ROC-AUC (F1 tie-break): %s", best_name)
    logger.info("Saved comparison table to %s", TABLES / "model_comparison.csv")
    logger.info("Saved best model to %s", MODELS / "best_model.joblib")

if __name__ == "__main__":
    main()
