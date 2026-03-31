"""
ML Training Pipeline
- Loads creditcard.csv
- Handles class imbalance with SMOTE
- Trains XGBoost classifier
- Saves model + scaler artifacts
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parent
DATA_PATH      = BASE_DIR / "data" / "creditcard.csv"
ARTIFACTS_DIR  = BASE_DIR / "artifacts"
MODEL_PATH     = ARTIFACTS_DIR / "xgb_fraud_model.joblib"
SCALER_PATH    = ARTIFACTS_DIR / "scaler.joblib"
FEATURES_PATH  = ARTIFACTS_DIR / "feature_names.joblib"

# ── Logger Setup ─────────────────────────────────────────────────────────────
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def load_data() -> pd.DataFrame:
    """Load and validate the creditcard dataset."""
    logger.info(f"Loading dataset from: {DATA_PATH}")

    if not DATA_PATH.exists():
        logger.error("creditcard.csv NOT FOUND. Place it in ml/data/")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    logger.success(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Validate expected columns
    assert "Class" in df.columns, "Missing 'Class' column in dataset"
    assert "Amount" in df.columns, "Missing 'Amount' column in dataset"

    fraud_count    = df["Class"].sum()
    legit_count    = len(df) - fraud_count
    fraud_pct      = (fraud_count / len(df)) * 100

    logger.info(f"Legitimate transactions : {legit_count:,}")
    logger.info(f"Fraudulent transactions : {fraud_count:,}")
    logger.info(f"Fraud percentage        : {fraud_pct:.4f}%")

    return df


def preprocess(df: pd.DataFrame):
    """
    Feature engineering + scaling.
    - Log-transform Amount (skewed)
    - Drop Time column (not useful for ML)
    - Scale all features
    """
    logger.info("Preprocessing data...")

    df = df.copy()

    # Log-transform Amount to reduce skewness
    df["Amount_log"] = np.log1p(df["Amount"])
    df.drop(columns=["Time", "Amount"], inplace=True)

    # Separate features and target
    X = df.drop(columns=["Class"])
    y = df["Class"]

    feature_names = X.columns.tolist()
    logger.info(f"Features used: {len(feature_names)}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    logger.success("Preprocessing complete.")
    return X_scaled, y.values, scaler, feature_names


def apply_smote(X_train, y_train):
    """Balance classes using SMOTE oversampling."""
    logger.info("Applying SMOTE to balance classes...")

    before_fraud = y_train.sum()
    before_total = len(y_train)

    smote = SMOTE(
        sampling_strategy=0.3,   # fraud = 30% of majority class
        random_state=42,
        k_neighbors=5
    )
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_fraud = y_resampled.sum()
    after_total = len(y_resampled)

    logger.info(f"Before SMOTE → Total: {before_total:,} | Fraud: {before_fraud:,}")
    logger.success(f"After  SMOTE → Total: {after_total:,} | Fraud: {after_fraud:,}")

    return X_resampled, y_resampled


def train_model(X_train, y_train):
    """Train XGBoost with optimized hyperparameters for fraud detection."""
    logger.info("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators      = 300,
        max_depth         = 6,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        scale_pos_weight  = 1,        # SMOTE already balanced classes
        use_label_encoder = False,
        eval_metric       = "auc",
        random_state      = 42,
        n_jobs            = -1,       # use all CPU cores
        verbosity         = 0
    )

    # Cross-validation for robust evaluation
    logger.info("Running 5-fold cross validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")

    logger.success(f"CV ROC-AUC Scores : {cv_scores.round(4)}")
    logger.success(f"Mean CV ROC-AUC   : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Final training on full training set
    model.fit(X_train, y_train)
    logger.success("Model training complete.")

    return model


def save_artifacts(model, scaler, feature_names):
    """Save model, scaler, and feature names to disk."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,         MODEL_PATH)
    joblib.dump(scaler,        SCALER_PATH)
    joblib.dump(feature_names, FEATURES_PATH)

    logger.success(f"Model   saved → {MODEL_PATH}")
    logger.success(f"Scaler  saved → {SCALER_PATH}")
    logger.success(f"Features saved → {FEATURES_PATH}")


def main():
    logger.info("=" * 55)
    logger.info("   FRAUD DETECTION — ML TRAINING PIPELINE")
    logger.info("=" * 55)

    # Step 1: Load
    df = load_data()

    # Step 2: Preprocess
    X, y, scaler, feature_names = preprocess(df)

    # Step 3: Train/Test Split (stratified to preserve class ratio)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.2,
        random_state = 42,
        stratify     = y
    )
    logger.info(f"Train size: {X_train.shape[0]:,} | Test size: {X_test.shape[0]:,}")

    # Step 4: SMOTE on training data ONLY (never on test)
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    # Step 5: Train
    model = train_model(X_train_bal, y_train_bal)

    # Step 6: Quick test set evaluation
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc      = roc_auc_score(y_test, y_pred_proba)
    logger.success(f"Test ROC-AUC: {roc_auc:.4f}")

    # Step 7: Save
    save_artifacts(model, scaler, feature_names)

    logger.info("=" * 55)
    logger.success("TRAINING COMPLETE. Run evaluate.py for full metrics.")
    logger.info("=" * 55)

    # Return test data for evaluate.py
    return X_test, y_test, model


if __name__ == "__main__":
    main()