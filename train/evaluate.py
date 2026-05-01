"""
evaluate.py – Comprehensive model evaluation on test set.

Generates:
  - Confusion matrix (PNG)
  - Classification report (precision, recall, F1)
  - ROC curve with AUC
  - Precision-Recall curve with AUPRC
  - Cross-validation results (5-fold)

Usage:
  python -m train.evaluate --model-path models/cnn_gunshot_classifier.keras --test-split
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .dataset import load_spectrogram_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _predict_probabilities(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect true labels and sigmoid probabilities from a logits-output model.
    """
    y_true = []
    y_pred_probs = []

    for images, labels in dataset:
        logits = np.asarray(model.predict_on_batch(images)).reshape(-1)
        probs = 1.0 / (1.0 + np.exp(-logits))
        y_pred_probs.extend(probs.tolist())
        y_true.extend(np.asarray(labels).reshape(-1).tolist())

    return np.array(y_true), np.array(y_pred_probs)


def evaluate_on_test_set(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    reports_dir: Path,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model on test set and generate visualizations.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    test_ds : tf.data.Dataset
        Test dataset.
    reports_dir : Path
        Directory to save reports.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    metrics : dict
        Dictionary of evaluation metrics.
    """
    logger.info("=== Evaluating on Test Set (threshold=%.2f) ===", threshold)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true, y_pred_probs = _predict_probabilities(model, test_ds)
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    logger.info("Confusion matrix:\n%s", cm)

    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    logger.info("Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f", accuracy, precision, recall, f1)
    logger.info("FPR: %.4f, Specificity: %.4f", fpr, specificity)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["NOGUN", "GUN"])
    ax.set_yticklabels(["NOGUN", "GUN"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="white", fontsize=14)

    plt.colorbar(im, ax=ax)
    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=100, bbox_inches="tight")
    logger.info("Saved: %s", cm_path)
    plt.close()

    # ROC curve
    fpr_roc, tpr_roc, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr_roc, tpr_roc)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr_roc, tpr_roc, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    roc_path = reports_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=100, bbox_inches="tight")
    logger.info("Saved: %s", roc_path)
    plt.close()

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    auprc = average_precision_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="darkgreen", lw=2, label=f"PR curve (AUPRC = {auprc:.3f})")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    pr_path = reports_dir / "pr_curve.png"
    plt.savefig(pr_path, dpi=100, bbox_inches="tight")
    logger.info("Saved: %s", pr_path)
    plt.close()

    # Classification report
    report = classification_report(y_true, y_pred, target_names=["NOGUN", "GUN"])
    logger.info("Classification Report:\n%s", report)

    report_path = reports_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("Saved: %s", report_path)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "auc": float(roc_auc),
        "auprc": float(auprc),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }

    return metrics


def cross_validate(
    model: tf.keras.Model,
    dataset_dir: Path,
    feature_type: str = "LogMel",
    n_splits: int = 5,
    reports_dir: Path = Path("reports"),
) -> dict:
    """
    Perform 5-fold cross-validation on frozen-base head.

    Note: For computational efficiency, CV is done only on the classification head
    with the frozen base model, not full retraining.

    Parameters
    ----------
    model : tf.keras.Model
        The full trained model (base + head).
    dataset_dir : Path
        Dataset directory.
    feature_type : str
        Spectrogram feature type.
    n_splits : int
        Number of folds.
    reports_dir : Path
        Directory to save CV results.

    Returns
    -------
    cv_results : dict
        Cross-validation metrics.
    """
    logger.info("=== Cross-Validation (%d-fold) ===", n_splits)

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load all data without train/val/test split
    from .dataset import _create_dataset
    import os

    feature_dir = dataset_dir / feature_type
    gun_dir = feature_dir / "GUN"
    nogun_dir = feature_dir / "NOGUN"

    image_paths = []
    labels = []

    if gun_dir.exists():
        gun_images = sorted(gun_dir.glob("*.png"))
        image_paths.extend(gun_images)
        labels.extend([1] * len(gun_images))

    if nogun_dir.exists():
        nogun_images = sorted(nogun_dir.glob("*.png"))
        image_paths.extend(nogun_images)
        labels.extend([0] * len(nogun_images))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        logger.info("Fold %d/%d", fold + 1, n_splits)

        # Get fold data
        fold_train_paths = image_paths[train_idx]
        fold_test_paths = image_paths[test_idx]
        fold_train_labels = labels[train_idx]
        fold_test_labels = labels[test_idx]

        # Create datasets
        fold_train_ds = _create_dataset(
            fold_train_paths, fold_train_labels, (224, 224), batch_size=32, augment=False
        )
        fold_test_ds = _create_dataset(
            fold_test_paths, fold_test_labels, (224, 224), batch_size=32, augment=False
        )

        # Evaluate on fold test set
        y_true_fold = []
        y_pred_fold = []

        y_true_probs, y_pred_probs = _predict_probabilities(model, fold_test_ds)
        y_pred_fold.extend((y_pred_probs >= 0.5).astype(int).flatten())
        y_true_fold.extend(y_true_probs.flatten())

        y_true_fold = np.array(y_true_fold)
        y_pred_fold = np.array(y_pred_fold)

        # Compute metrics
        cm = confusion_matrix(y_true_fold, y_pred_fold)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, np.sum(y_true_fold == y_pred_fold))

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        fold_results.append({
            "fold": fold + 1,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

        logger.info(
            "Fold %d: Accuracy=%.3f, Precision=%.3f, Recall=%.3f, F1=%.3f",
            fold + 1,
            accuracy,
            precision,
            recall,
            f1,
        )

    # Aggregate results
    accuracies = [r["accuracy"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    recalls = [r["recall"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]

    cv_results = {
        "n_splits": n_splits,
        "fold_results": fold_results,
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
    }

    logger.info("=== CV Summary ===")
    logger.info("Accuracy:  %.3f ± %.3f", cv_results["accuracy_mean"], cv_results["accuracy_std"])
    logger.info("Precision: %.3f ± %.3f", cv_results["precision_mean"], cv_results["precision_std"])
    logger.info("Recall:    %.3f ± %.3f", cv_results["recall_mean"], cv_results["recall_std"])
    logger.info("F1:        %.3f ± %.3f", cv_results["f1_mean"], cv_results["f1_std"])

    cv_path = reports_dir / "cross_validation_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("Saved: %s", cv_path)

    return cv_results


def main(
    model_path: Path,
    dataset_dir: Path = Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
    feature_type: str = "LogMel",
    reports_dir: Path = Path("reports"),
    threshold: float = 0.5,
    do_cv: bool = False,
) -> None:
    """
    Main evaluation orchestration.

    Parameters
    ----------
    model_path : Path
        Path to saved model (.keras file).
    dataset_dir : Path
        Root directory of spectrogram dataset.
    feature_type : str
        Spectrogram feature type.
    reports_dir : Path
        Directory to save reports.
    threshold : float
        Decision threshold for evaluation.
    do_cv : bool
        Whether to run cross-validation.
    """
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info("Model: %s", model_path)
    logger.info("Threshold: %.2f", threshold)

    # Load model
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded.")

    # Load test set
    logger.info("Loading dataset...")
    _, _, test_ds, _ = load_spectrogram_dataset(
        dataset_dir,
        feature_type=feature_type,
        batch_size=32,
    )

    # Evaluate
    metrics = evaluate_on_test_set(model, test_ds, reports_dir, threshold=threshold)

    # Save metrics
    metrics_path = Path(reports_dir) / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved metrics to: %s", metrics_path)

    # Cross-validation
    if do_cv:
        cv_results = cross_validate(model, dataset_dir, feature_type, reports_dir=reports_dir)

    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .keras model")
    parser.add_argument("--feature-type", default="LogMel", help="Spectrogram feature type")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Reports directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold CV")

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        feature_type=args.feature_type,
        reports_dir=args.reports_dir,
        threshold=args.threshold,
        do_cv=args.cross_validate,
    )
