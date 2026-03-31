"""
Model Evaluation
- Loads saved model + scaler
- Evaluates on test set
- Prints full classification report
- Shows confusion matrix
- Finds optimal threshold
"""

import sys
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
DATA_PATH     = BASE_DIR / "data" / "creditcard.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "xgb_fraud_model.joblib"
SCALER_PATH   = ARTIFACTS_DIR / "scaler.joblib"

logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def load_artifacts():
    """Load saved model and scaler."""
    if not MODEL_PATH.exists():
        logger.error("Model not found. Run train.py first.")
        sys.exit(1)

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logger.success("Model and scaler loaded.")
    return model, scaler


def load_test_data(scaler):
    """Recreate the same test split used during training."""
    df = pd.read_csv(DATA_PATH)
    df["Amount_log"] = np.log1p(df["Amount"])
    df.drop(columns=["Time", "Amount"], inplace=True)

    X = df.drop(columns=["Class"])
    y = df["Class"].values

    X_scaled = scaler.transform(X)

    _, X_test, _, y_test = train_test_split(
        X_scaled, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )
    return X_test, y_test


def find_optimal_threshold(y_true, y_proba):
    """Find the threshold that maximizes F1 for the fraud class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx  = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def print_confusion_matrix(cm):
    """Pretty print confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    logger.info("┌─────────────────────────────────────┐")
    logger.info("│         CONFUSION MATRIX            │")
    logger.info("├──────────────┬──────────┬───────────┤")
    logger.info("│              │ Pred: 0  │  Pred: 1  │")
    logger.info("├──────────────┼──────────┼───────────┤")
    logger.info(f"│  Actual: 0   │  {tn:>6} │  {fp:>7} │")
    logger.info(f"│  Actual: 1   │  {fn:>6} │  {tp:>7} │")
    logger.info("└──────────────┴──────────┴───────────┘")
    logger.info(f"  True Negatives  (TN): {tn:,}")
    logger.info(f"  False Positives (FP): {fp:,}  ← legit flagged as fraud")
    logger.info(f"  False Negatives (FN): {fn:,}  ← fraud missed!")
    logger.info(f"  True Positives  (TP): {tp:,}  ← fraud caught ✅")


def main():
    logger.info("=" * 55)
    logger.info("   FRAUD DETECTION — MODEL EVALUATION")
    logger.info("=" * 55)

    model, scaler   = load_artifacts()
    X_test, y_test  = load_test_data(scaler)

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # ── Key Metrics ──────────────────────────────────────
    roc_auc  = roc_auc_score(y_test, y_proba)
    avg_prec = average_precision_score(y_test, y_proba)

    logger.success(f"ROC-AUC Score          : {roc_auc:.4f}")
    logger.success(f"Average Precision Score: {avg_prec:.4f}")

    # ── Optimal Threshold ────────────────────────────────
    best_thresh, best_f1 = find_optimal_threshold(y_test, y_proba)
    logger.success(f"Optimal Threshold      : {best_thresh:.4f}")
    logger.success(f"Best F1 Score          : {best_f1:.4f}")

    # Save optimal threshold to artifacts
    joblib.dump(float(best_thresh), ARTIFACTS_DIR / "optimal_threshold.joblib")
    logger.success(f"Threshold saved → {ARTIFACTS_DIR / 'optimal_threshold.joblib'}")

    # ── Classification Report ────────────────────────────
    y_pred = (y_proba >= best_thresh).astype(int)
    logger.info("\n" + classification_report(
        y_test, y_pred,
        target_names=["Legitimate", "Fraud"],
        digits=4
    ))

    # ── Confusion Matrix ─────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    print_confusion_matrix(cm)

    # ── Feature Importance ───────────────────────────────
    feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.joblib")
    importances   = model.feature_importances_
    top_features  = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1],
        reverse=True
    )[:10]

    logger.info("\n  TOP 10 IMPORTANT FEATURES:")
    for rank, (feat, score) in enumerate(top_features, 1):
        bar = "█" * int(score * 200)
        logger.info(f"  {rank:>2}. {feat:<15} {score:.4f}  {bar}")

    logger.info("=" * 55)
    logger.success("EVALUATION COMPLETE.")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()