"""
Comprehensive Evaluation, Benchmarking, and Ablation Study Module
Evaluates trained models on the held-out test split (2,969 claims).
Generates performance metrics, confusion matrices, ROC curves, feature importance,
efficiency benchmarks (throughput & latency), and ablation study reports.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath("."))

import time
import json
import logging
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, brier_score_loss
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.feature_fusion import (
    ALL_FEATURE_NAMES,
    SEMANTIC_FEATURE_NAMES,
    LINGUISTIC_FEATURE_NAMES,
    SURFACE_FEATURE_NAMES,
    ENTITY_FEATURE_NAMES,
    NUMERICAL_FEATURE_NAMES,
    POS_FEATURE_NAMES,
    CERTAINTY_FEATURE_NAMES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
MODELS_DIR = "models"
TEST_FEAT_PATH = "data/test/test_features.csv"
TRAIN_FEAT_PATH = "data/train/train_features.csv"
VAL_FEAT_PATH = "data/processed/val_features.csv"


def load_test_data() -> Tuple[pd.DataFrame, np.ndarray, Dict[str, list]]:
    """
    Loads test features and target labels.
    """
    if not os.path.exists(TEST_FEAT_PATH):
        raise FileNotFoundError(f"Test feature file not found at {TEST_FEAT_PATH}")

    test_df = pd.read_csv(TEST_FEAT_PATH)
    y_test = test_df["label"].values

    with open(os.path.join(MODELS_DIR, "feature_names.json"), "r", encoding="utf-8") as f:
        feature_meta = json.load(f)

    return test_df, y_test, feature_meta


def measure_throughput_and_latency(model, X_test: np.ndarray, num_runs: int = 5) -> Tuple[float, float]:
    """
    Measures inference throughput (claims/sec) and average latency per claim (ms).
    """
    # Warmup
    _ = model.predict_proba(X_test[:50])

    timings = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model.predict_proba(X_test)
        timings.append(time.perf_counter() - start)

    mean_sec = float(np.mean(timings))
    claims_per_sec = float(len(X_test) / mean_sec)
    latency_ms_per_claim = float((mean_sec / len(X_test)) * 1000.0)

    return claims_per_sec, latency_ms_per_claim


def evaluate_model_on_test(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    feature_set_name: str,
    model_path: str
) -> Dict[str, Any]:
    """
    Computes all test evaluation metrics and efficiency benchmarks for a model.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))
    cm = confusion_matrix(y_test, preds).tolist()

    claims_per_sec, latency_ms = measure_throughput_and_latency(model, X_test)
    model_size_mb = float(os.path.getsize(model_path) / (1024 * 1024)) if os.path.exists(model_path) else 0.0

    return {
        "model_name": model_name,
        "feature_set": feature_set_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "confusion_matrix": cm,
        "throughput_claims_per_sec": claims_per_sec,
        "latency_ms_per_claim": latency_ms,
        "model_size_mb": model_size_mb,
        "y_prob": probs.tolist()
    }


def plot_metrics_comparison(eval_results: List[Dict[str, Any]], output_path: str):
    """
    Plots F1, Precision, Recall, Accuracy, and ROC-AUC across models.
    """
    df = pd.DataFrame(eval_results)
    metrics = ["f1", "precision", "recall", "accuracy", "roc_auc"]
    labels = ["F1 Score", "Precision", "Recall", "Accuracy", "ROC-AUC"]

    df_plot = pd.melt(
        df,
        id_vars=["model_name"],
        value_vars=metrics,
        var_name="Metric",
        value_name="Score"
    )
    metric_map = dict(zip(metrics, labels))
    df_plot["Metric"] = df_plot["Metric"].map(metric_map)

    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("mako", n_colors=len(metrics))
    ax = sns.barplot(data=df_plot, x="model_name", y="Score", hue="Metric", palette=palette)
    plt.title("Model Performance Comparison on Held-Out Test Set (2,969 Claims)", fontsize=14, pad=15)
    plt.xlabel("Model Architecture", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.7, 1.02)
    plt.xticks(rotation=15)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved metrics comparison chart to {output_path}")


def plot_confusion_matrices(eval_results: List[Dict[str, Any]], output_path: str):
    """
    Plots confusion matrix heatmaps for all tested models.
    """
    n = len(eval_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, eval_results):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Supported (0)", "Hallucinated (1)"],
            yticklabels=["Supported (0)", "Hallucinated (1)"]
        )
        ax.set_title(f"{res['model_name']}\nAcc: {res['accuracy']:.3f} | F1: {res['f1']:.3f}", fontsize=11)
        ax.set_ylabel("True Label", fontsize=10)
        ax.set_xlabel("Predicted Label", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrices to {output_path}")


def plot_roc_curves(eval_results: List[Dict[str, Any]], y_test: np.ndarray, output_path: str):
    """
    Plots ROC curves for all models.
    """
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")

    for res in eval_results:
        probs = np.array(res["y_prob"])
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.plot(fpr, tpr, label=f"{res['model_name']} (AUC = {res['roc_auc']:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Guess (AUC = 0.50)")
    plt.title("ROC Curves - Hallucination Detection on Held-Out Test Set", fontsize=14, pad=12)
    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves to {output_path}")


def plot_feature_importance(model_path: str, feature_names: List[str], output_path: str, top_n: int = 20):
    """
    Plots top feature importances from the best tree-based detector.
    """
    calibrated_model = joblib.load(model_path)
    # Extract base estimator from CalibratedClassifierCV
    base_est = calibrated_model.estimator

    if hasattr(base_est, "feature_importances_"):
        importances = base_est.feature_importances_
    elif hasattr(base_est, "coef_"):
        importances = np.abs(base_est.coef_[0])
    else:
        logger.warning("Model does not expose feature_importances_ or coef_.")
        return

    feat_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("viridis", n_colors=top_n)
    sns.barplot(data=feat_imp, y="Feature", x="Importance", palette=palette)
    plt.title(f"Top {top_n} Most Important Model Features (Hybrid Detector)", fontsize=14, pad=12)
    plt.xlabel("Feature Importance Score", fontsize=12)
    plt.ylabel("Feature Name", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importance chart to {output_path}")


def run_ablation_study(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_chart_path: str
) -> List[Dict[str, Any]]:
    """
    Executes an ablation study by training and testing models on feature subsets:
    1. Full Hybrid (Semantic + Linguistic)
    2. Semantic Only
    3. Linguistic Only
    4. Without Entity Features
    5. Without Numerical Features
    6. Without POS Features
    7. Without Certainty Features
    """
    logger.info("=== Running Feature Ablation Study ===")
    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_test = test_df["label"].values

    ablation_configs = {
        "Full Hybrid": ALL_FEATURE_NAMES,
        "Semantic Only": SEMANTIC_FEATURE_NAMES,
        "Linguistic Only": LINGUISTIC_FEATURE_NAMES,
        "No Entities": [c for c in ALL_FEATURE_NAMES if c not in ENTITY_FEATURE_NAMES],
        "No Numerical": [c for c in ALL_FEATURE_NAMES if c not in NUMERICAL_FEATURE_NAMES],
        "No POS Ratios": [c for c in ALL_FEATURE_NAMES if c not in POS_FEATURE_NAMES],
        "No Certainty Lex": [c for c in ALL_FEATURE_NAMES if c not in CERTAINTY_FEATURE_NAMES],
    }

    ablation_results = []

    for name, cols in ablation_configs.items():
        logger.info(f"Ablation: {name} ({len(cols)} features)...")
        X_tr = train_df[cols].values
        X_te = test_df[cols].values

        clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_train)

        probs = clf.predict_proba(X_te)[:, 1]
        preds = (probs >= 0.50).astype(int)

        res = {
            "Configuration": name,
            "Num Features": len(cols),
            "F1": float(f1_score(y_test, preds, zero_division=0)),
            "Accuracy": float(accuracy_score(y_test, preds)),
            "Precision": float(precision_score(y_test, preds, zero_division=0)),
            "Recall": float(recall_score(y_test, preds, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, probs)),
        }
        ablation_results.append(res)
        logger.info(f" -> {name} | F1: {res['F1']:.4f} | Acc: {res['Accuracy']:.4f} | AUC: {res['ROC-AUC']:.4f}")

    # Plot ablation comparison
    df_abl = pd.DataFrame(ablation_results)
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df_abl, x="Configuration", y="F1", palette="crest")
    plt.title("Feature Ablation Study: Impact on Test F1 Score", fontsize=14, pad=12)
    plt.xlabel("Feature Configuration", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0.7, 1.0)
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(output_chart_path, dpi=300)
    plt.close()
    logger.info(f"Saved ablation chart to {output_chart_path}")

    return ablation_results


def run_full_evaluation():
    """
    Main evaluation pipeline coordinating test evaluation, benchmarks, ablation, and reporting.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    test_df, y_test, feature_meta = load_test_data()
    train_df = pd.read_csv(TRAIN_FEAT_PATH)
    val_df = pd.read_csv(VAL_FEAT_PATH)

    models_to_evaluate = [
        ("Random Forest (Linguistic)", "Linguistic", feature_meta["linguistic_features"], os.path.join(MODELS_DIR, "baseline_linguistic_rf.joblib")),
        ("Logistic Regression (Semantic)", "Semantic", feature_meta["semantic_features"], os.path.join(MODELS_DIR, "baseline_semantic_lr.joblib")),
        ("Random Forest (Semantic)", "Semantic", feature_meta["semantic_features"], os.path.join(MODELS_DIR, "baseline_semantic_rf.joblib")),
        ("Random Forest (Hybrid)", "Hybrid", feature_meta["all_features"], os.path.join(MODELS_DIR, "hybrid_random_forest.joblib")),
        ("XGBoost (Hybrid)", "Hybrid", feature_meta["all_features"], os.path.join(MODELS_DIR, "hybrid_xgboost.joblib")),
    ]

    eval_results = []
    for model_name, feat_type, feat_cols, model_path in models_to_evaluate:
        logger.info(f"Evaluating {model_name} on held-out test set...")
        model = joblib.load(model_path)
        X_test = test_df[feat_cols].values
        metrics = evaluate_model_on_test(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            feature_set_name=feat_type,
            model_path=model_path
        )
        eval_results.append(metrics)
        logger.info(
            f"Result: {model_name} | F1: {metrics['f1']:.4f} | Acc: {metrics['accuracy']:.4f} | "
            f"Latency: {metrics['latency_ms_per_claim']:.3f} ms/claim | Claims/s: {metrics['throughput_claims_per_sec']:.0f}"
        )

    # Visualizations
    plot_metrics_comparison(eval_results, os.path.join(RESULTS_DIR, "metrics_comparison.png"))
    plot_confusion_matrices(eval_results, os.path.join(RESULTS_DIR, "confusion_matrices.png"))
    plot_roc_curves(eval_results, y_test, os.path.join(RESULTS_DIR, "roc_curves.png"))
    plot_feature_importance(
        os.path.join(MODELS_DIR, "hybrid_xgboost.joblib"),
        feature_meta["all_features"],
        os.path.join(RESULTS_DIR, "feature_importance.png")
    )

    # Ablation Study
    ablation_results = run_ablation_study(
        train_df, val_df, test_df, os.path.join(RESULTS_DIR, "ablation_study.png")
    )

    # Clean results for serialization (remove large prob arrays)
    clean_eval_results = []
    for r in eval_results:
        entry = {k: v for k, v in r.items() if k != "y_prob"}
        clean_eval_results.append(entry)

    # Save JSON report
    report_data = {
        "test_samples": len(y_test),
        "test_distribution": pd.Series(y_test).value_counts().to_dict(),
        "model_evaluations": clean_eval_results,
        "ablation_study": ablation_results
    }
    with open(os.path.join(RESULTS_DIR, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    # Generate Markdown Table Report
    md_table = "# Empirical Evaluation Results on Held-Out Test Set (2,969 Claims)\n\n"
    md_table += "| Model | Feature Set | F1-Score | Precision | Recall | Accuracy | ROC-AUC | Latency (ms/claim) | Throughput (claims/s) | Model Size (MB) |\n"
    md_table += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for r in clean_eval_results:
        md_table += (
            f"| **{r['model_name']}** | {r['feature_set']} | {r['f1']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['accuracy']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['latency_ms_per_claim']:.3f} | {r['throughput_claims_per_sec']:,.0f} | {r['model_size_mb']:.2f} |\n"
        )

    md_table += "\n## Feature Ablation Study\n\n"
    md_table += "| Configuration | Features | F1-Score | Accuracy | Precision | Recall | ROC-AUC |\n"
    md_table += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for a in ablation_results:
        md_table += (
            f"| **{a['Configuration']}** | {a['Num Features']} | {a['F1']:.4f} | {a['Accuracy']:.4f} | "
            f"{a['Precision']:.4f} | {a['Recall']:.4f} | {a['ROC-AUC']:.4f} |\n"
        )

    with open(os.path.join(RESULTS_DIR, "experiment_results.md"), "w", encoding="utf-8") as f:
        f.write(md_table)

    logger.info(f"Saved evaluation markdown report to {os.path.join(RESULTS_DIR, 'experiment_results.md')}")
    print("\n" + md_table)


if __name__ == "__main__":
    run_full_evaluation()
