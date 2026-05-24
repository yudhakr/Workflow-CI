"""
modelling.py
Training model Machine Learning dengan MLflow logging.
Digunakan sebagai entry point MLflow Project.

Usage:
    python modelling.py --n_estimators 100 --max_depth 10 --random_state 42
"""

import os
import sys
import argparse
import logging
import tempfile

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# MLflow tracking URI: fallback ke local file jika env var tidak diset
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset_preprocessing", "dataset_preprocessing.csv")
TARGET_COL = "target"
TARGET_NAMES = ["class_0", "class_1", "class_2"]
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    """Memuat dataset preprocessing."""
    if not os.path.exists(path):
        logger.error(f"Dataset tidak ditemukan: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def split_data(df: pd.DataFrame, target_col: str, random_state: int):
    """Split fitur dan target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    """Evaluasi dan return metrics."""
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro"),
        "recall_macro": recall_score(y_test, y_pred, average="macro"),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics, y_pred


def save_confusion_matrix(model, X_test, y_test, save_path: str):
    """Simpan confusion matrix dalam bentuk PNG."""
    y_pred = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TARGET_NAMES)
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    logger.info(f"Confusion matrix saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    try:
        # Load data
        df = load_data(DATA_PATH)
        X_train, X_test, y_train, y_test = split_data(df, TARGET_COL, args.random_state)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow Run ID: {run_id}")

            # Log parameters
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("test_size", TEST_SIZE)
            mlflow.log_param("model_type", "RandomForestClassifier")

            # Train model
            model = RandomForestClassifier(
                n_estimators=args.n_estimators,
                max_depth=args.max_depth,
                random_state=args.random_state
            )
            model.fit(X_train, y_train)
            logger.info("Model training completed.")

            # Evaluate
            metrics, y_pred = evaluate(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # Log classification report as text artifact
            report = classification_report(y_test, y_pred, target_names=TARGET_NAMES)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                f.write("Classification Report - Wine Dataset\n")
                f.write("=" * 50 + "\n")
                f.write(report)
                report_path = f.name
            mlflow.log_artifact(report_path, "evaluation")
            os.unlink(report_path)

            # Log confusion matrix image
            cm_path = os.path.join(tempfile.gettempdir(), "confusion_matrix.png")
            save_confusion_matrix(model, X_test, y_test, cm_path)
            mlflow.log_artifact(cm_path, "evaluation_plots")
            os.unlink(cm_path)

            # Log model
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name="Wine_RF"
            )

            # Log feature importance plot
            feature_names = [c for c in df.columns if c != TARGET_COL]
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(range(len(importances)), importances[indices], color="steelblue")
            ax.set_xticks(range(len(importances)))
            ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
            ax.set_title("Feature Importance")
            ax.set_xlabel("Features")
            ax.set_ylabel("Importance")
            plt.tight_layout()
            fi_path = os.path.join(tempfile.gettempdir(), "feature_importance.png")
            plt.savefig(fi_path, dpi=100)
            plt.close()
            mlflow.log_artifact(fi_path, "model_analysis")
            os.unlink(fi_path)

            print(f"\n{'=' * 50}")
            print(f"Run ID: {run_id}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
            print(f"{'=' * 50}")

        logger.info("Modelling completed successfully.")

    except Exception as e:
        logger.exception(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
