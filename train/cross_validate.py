"""
cross_validate.py – Standalone 5-fold cross-validation script.

Performs cross-validation on the classification head with frozen base,
and reports mean ± std metrics across folds.

Usage:
  python -m train.cross_validate --model-path models/cnn_gunshot_classifier.keras
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from .dataset import _create_dataset
from .dataset import load_spectrogram_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def cross_validate_model(
    model: tf.keras.Model,
    dataset_dir: Path,
    feature_type: str = "LogMel",
    n_splits: int = 5,
    batch_size: int = 32,
    threshold: float = 0.5,
) -> dict:
    """
    Perform n-fold cross-validation.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    dataset_dir : Path
        Root directory of spectrogram dataset.
    feature_type : str
        Spectrogram feature type.
    n_splits : int
        Number of folds.
    batch_size : int
        Batch size for datasets.
    threshold : float
        Decision threshold for binary classification.

    Returns
    -------
    cv_results : dict
        Cross-validation metrics and fold results.
    """
    logger.info("=== %d-Fold Cross-Validation ===", n_splits)
    logger.info("Threshold: %.2f", threshold)

    # Load all data
    feature_dir = dataset_dir / feature_type
    gun_dir = feature_dir / "GUN"
    nogun_dir = feature_dir / "NOGUN"

    image_paths = []
    labels = []

    if gun_dir.exists():
        gun_images = sorted(gun_dir.glob("*.png"))
        image_paths.extend(gun_images)
        labels.extend([1] * len(gun_images))
        logger.info("Loaded %d GUN images", len(gun_images))

    if nogun_dir.exists():
        nogun_images = sorted(nogun_dir.glob("*.png"))
        image_paths.extend(nogun_images)
        labels.extend([0] * len(nogun_images))
        logger.info("Loaded %d NOGUN images", len(nogun_images))

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    logger.info("Total images: %d", len(image_paths))

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(image_paths, labels)):
        logger.info("\n--- Fold %d/%d ---", fold + 1, n_splits)

        # Get fold data
        fold_test_paths = image_paths[test_idx]
        fold_test_labels = labels[test_idx]

        logger.info("Test set size: %d", len(fold_test_paths))

        # Create test dataset
        fold_test_ds = _create_dataset(
            fold_test_paths, fold_test_labels, (224, 224), batch_size=batch_size, augment=False
        )

        # Predict
        y_true_fold = []
        y_pred_probs_fold = []

        for images, labels_batch in fold_test_ds:
            probs = model.predict(images, verbose=0)
            y_pred_probs_fold.extend(probs.flatten())
            y_true_fold.extend(labels_batch.numpy().flatten())

        y_true_fold = np.array(y_true_fold)
        y_pred_probs_fold = np.array(y_pred_probs_fold)
        y_pred_fold = (y_pred_probs_fold >= threshold).astype(int)

        # Compute metrics
        accuracy = accuracy_score(y_true_fold, y_pred_fold)
        precision = precision_score(y_true_fold, y_pred_fold, zero_division=0)
        recall = recall_score(y_true_fold, y_pred_fold, zero_division=0)
        f1 = f1_score(y_true_fold, y_pred_fold, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true_fold, y_pred_fold)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        fold_result = {
            "fold": fold + 1,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fpr),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        fold_results.append(fold_result)

        logger.info("Fold %d: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, FPR=%.4f",
                   fold + 1, accuracy, precision, recall, f1, fpr)

    # Aggregate results
    accuracies = [r["accuracy"] for r in fold_results]
    precisions = [r["precision"] for r in fold_results]
    recalls = [r["recall"] for r in fold_results]
    f1s = [r["f1"] for r in fold_results]
    fprs = [r["fpr"] for r in fold_results]

    cv_results = {
        "n_splits": n_splits,
        "threshold": float(threshold),
        "fold_results": fold_results,
        "accuracy": {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
        },
        "precision": {
            "mean": float(np.mean(precisions)),
            "std": float(np.std(precisions)),
        },
        "recall": {
            "mean": float(np.mean(recalls)),
            "std": float(np.std(recalls)),
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s)),
        },
        "fpr": {
            "mean": float(np.mean(fprs)),
            "std": float(np.std(fprs)),
        },
    }

    logger.info("\n=== Cross-Validation Summary ===")
    logger.info("Accuracy:  %.4f ± %.4f", cv_results["accuracy"]["mean"], cv_results["accuracy"]["std"])
    logger.info("Precision: %.4f ± %.4f", cv_results["precision"]["mean"], cv_results["precision"]["std"])
    logger.info("Recall:    %.4f ± %.4f", cv_results["recall"]["mean"], cv_results["recall"]["std"])
    logger.info("F1:        %.4f ± %.4f", cv_results["f1"]["mean"], cv_results["f1"]["std"])
    logger.info("FPR:       %.4f ± %.4f", cv_results["fpr"]["mean"], cv_results["fpr"]["std"])

    return cv_results


def main(
    model_path: Path,
    dataset_dir: Path = Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
    feature_type: str = "LogMel",
    n_splits: int = 5,
    batch_size: int = 32,
    threshold: float = 0.5,
    output_dir: Path = Path("reports"),
) -> None:
    """
    Main cross-validation orchestration.

    Parameters
    ----------
    model_path : Path
        Path to saved model.
    dataset_dir : Path
        Dataset directory.
    feature_type : str
        Spectrogram feature type.
    n_splits : int
        Number of folds.
    batch_size : int
        Batch size.
    threshold : float
        Decision threshold.
    output_dir : Path
        Output directory for results.
    """
    logger.info("=" * 70)
    logger.info("CROSS-VALIDATION")
    logger.info("=" * 70)
    logger.info("Model: %s", model_path)
    logger.info("Dataset: %s", dataset_dir)
    logger.info("Feature type: %s", feature_type)

    # Load model
    model = tf.keras.models.load_model(str(model_path))
    logger.info("Model loaded.")

    # Cross-validate
    cv_results = cross_validate_model(
        model,
        dataset_dir,
        feature_type=feature_type,
        n_splits=n_splits,
        batch_size=batch_size,
        threshold=threshold,
    )

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_path = output_dir / "cross_validation_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_results, f, indent=2)
    logger.info("Results saved to: %s", cv_path)

    logger.info("\n" + "=" * 70)
    logger.info("CROSS-VALIDATION COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-validate trained model")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to .keras model")
    parser.add_argument("--feature-type", default="LogMel", help="Spectrogram feature type")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Output directory")

    args = parser.parse_args()

    main(
        model_path=args.model_path,
        feature_type=args.feature_type,
        n_splits=args.n_splits,
        batch_size=args.batch_size,
        threshold=args.threshold,
        output_dir=args.output_dir,
    )
