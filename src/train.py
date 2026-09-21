"""
Model Training Module
Implements:
- Experiment A: Linguistic Baseline (Random Forest)
- Experiment B: Semantic Baseline (Logistic Regression & Random Forest)
- Experiment C: Proposed Hybrid Model (Random Forest & XGBoost)
Performs probability calibration and selects the top-performing detector.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import json
import logging
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

from src.feature_fusion import (
    ALL_FEATURE_NAMES,
    LINGUISTIC_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
TRAIN_FEAT_PATH = "data/train/train_features.csv"
VAL_FEAT_PATH = "data/processed/val_features.csv"


def load_data_and_features() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, list]]:
    """
    Loads train and validation feature CSVs along with feature names.
    """
    if not os.path.exists(TRAIN_FEAT_PATH) or not os.path.exists(VAL_FEAT_PATH):
        raise FileNotFoundError("Feature files not found. Run feature_fusion.py first.")

    train_df = pd.read_csv(TRAIN_FEAT_PATH)
    val_df = pd.read_csv(VAL_FEAT_PATH)

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_meta = json.load(f)

    return train_df, val_df, feature_meta


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Computes key classification metrics.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob))
    }


def train_and_calibrate_model(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str
) -> Tuple[Any, Dict[str, float]]:
    """
    Fits base model on training data, then calibrates on validation set.
    """
    logger.info(f"Training {model_name}...")
    base_model.fit(X_train, y_train)

    # Calibrate probabilities using sigmoid / Platt scaling on validation data
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_val, y_val)

    # Evaluate on validation set
    val_probs = calibrated_model.predict_proba(X_val)[:, 1]
    val_preds = (val_probs >= 0.50).astype(int)
    metrics = evaluate_predictions(y_val, val_preds, val_probs)

    logger.info(
        f"{model_name} - Val F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | "
        f"ROC-AUC: {metrics['roc_auc']:.4f} | Brier: {metrics['brier_score']:.4f}"
    )
    return calibrated_model, metrics


def run_training_pipeline() -> Dict[str, Any]:
    """
    Executes Experiments A, B, and C and saves model artifacts.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    train_df, val_df, feature_meta = load_data_and_features()

    y_train = train_df["label"].values
    y_val = val_df["label"].values

    results = {}
    saved_models = {}

    # -------------------------------------------------------------
    # Experiment A: Linguistic Baseline (Random Forest)
    # -------------------------------------------------------------
    logger.info("=== Experiment A: Linguistic Baseline ===")
    ling_cols = feature_meta["linguistic_features"]
    X_train_ling = train_df[ling_cols].values
    X_val_ling = val_df[ling_cols].values

    rf_ling = RandomForestClassifier(n_estimators=120, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)
    cal_rf_ling, metrics_rf_ling = train_and_calibrate_model(
        rf_ling, X_train_ling, y_train, X_val_ling, y_val, "ExpA_Linguistic_RF"
    )
    results["ExpA_Linguistic_RF"] = {
        "features": "Linguistic",
        "feature_cols": ling_cols,
        "metrics": metrics_rf_ling
    }
    saved_models["ExpA_Linguistic_RF"] = (cal_rf_ling, "baseline_linguistic_rf.joblib")

    # -------------------------------------------------------------
    # Experiment B: Semantic Baseline (Logistic Regression & RF)
    # -------------------------------------------------------------
    logger.info("=== Experiment B: Semantic Baseline ===")
    sem_cols = feature_meta["semantic_features"]
    X_train_sem = train_df[sem_cols].values
    X_val_sem = val_df[sem_cols].values

    # B1: Semantic Logistic Regression
    lr_sem = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    cal_lr_sem, metrics_lr_sem = train_and_calibrate_model(
        lr_sem, X_train_sem, y_train, X_val_sem, y_val, "ExpB_Semantic_LR"
    )
    results["ExpB_Semantic_LR"] = {
        "features": "Semantic",
        "feature_cols": sem_cols,
        "metrics": metrics_lr_sem
    }
    saved_models["ExpB_Semantic_LR"] = (cal_lr_sem, "baseline_semantic_lr.joblib")

    # B2: Semantic Random Forest
    rf_sem = RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=2, random_state=42, n_jobs=-1)
    cal_rf_sem, metrics_rf_sem = train_and_calibrate_model(
        rf_sem, X_train_sem, y_train, X_val_sem, y_val, "ExpB_Semantic_RF"
    )
    results["ExpB_Semantic_RF"] = {
        "features": "Semantic",
        "feature_cols": sem_cols,
        "metrics": metrics_rf_sem
    }
    saved_models["ExpB_Semantic_RF"] = (cal_rf_sem, "baseline_semantic_rf.joblib")

    # -------------------------------------------------------------
    # Experiment C: Proposed Hybrid Models (Random Forest & XGBoost)
    # -------------------------------------------------------------
    logger.info("=== Experiment C: Hybrid Feature Fusion ===")
    all_cols = feature_meta["all_features"]
    X_train_all = train_df[all_cols].values
    X_val_all = val_df[all_cols].values

    # C1: Hybrid Random Forest
    rf_hybrid = RandomForestClassifier(
        n_estimators=150, max_depth=14, min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
    )
    cal_rf_hybrid, metrics_rf_hybrid = train_and_calibrate_model(
        rf_hybrid, X_train_all, y_train, X_val_all, y_val, "ExpC_Hybrid_RF"
    )
    results["ExpC_Hybrid_RF"] = {
        "features": "Hybrid (Semantic + Linguistic)",
        "feature_cols": all_cols,
        "metrics": metrics_rf_hybrid
    }
    saved_models["ExpC_Hybrid_RF"] = (cal_rf_hybrid, "hybrid_random_forest.joblib")

    # C2: Hybrid XGBoost
    xgb_hybrid = XGBClassifier(
        n_estimators=160,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
    cal_xgb_hybrid, metrics_xgb_hybrid = train_and_calibrate_model(
        xgb_hybrid, X_train_all, y_train, X_val_all, y_val, "ExpC_Hybrid_XGBoost"
    )
    results["ExpC_Hybrid_XGBoost"] = {
        "features": "Hybrid (Semantic + Linguistic)",
        "feature_cols": all_cols,
        "metrics": metrics_xgb_hybrid
    }
    saved_models["ExpC_Hybrid_XGBoost"] = (cal_xgb_hybrid, "hybrid_xgboost.joblib")

    # Determine best model by Validation F1-score
    best_exp_name = max(results.keys(), key=lambda k: results[k]["metrics"]["f1"])
    logger.info(f"Top-performing model: {best_exp_name} (Val F1: {results[best_exp_name]['metrics']['f1']:.4f})")

    # Save all models
    for exp_key, (model_obj, filename) in saved_models.items():
        save_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(model_obj, save_path)
        logger.info(f"Saved {exp_key} to {save_path}")

    # Save best model reference as best_model.joblib
    best_model_obj = saved_models[best_exp_name][0]
    best_path = os.path.join(MODELS_DIR, "best_model.joblib")
    joblib.dump(best_model_obj, best_path)
    logger.info(f"Saved best model ({best_exp_name}) to {best_path}")

    # Persist training summary
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_model": best_exp_name,
            "results": results
        }, f, indent=2)
    logger.info(f"Saved training summary to {summary_path}")

    return results


if __name__ == "__main__":
    run_training_pipeline()
