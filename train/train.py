"""
train.py – Top-level training orchestration.

Two-stage training:
  Stage A: Train head only (15 epochs)
  Stage B: Fine-tune a small set of top layers (20 epochs)
  Then: Threshold sweep to find optimal decision boundary

Usage:
  python -m train.train [--feature-type LogMel] [--batch-size 32] [--threshold-sweep]
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import tensorflow as tf

from .dataset import load_spectrogram_dataset
from .model import build_cnn_model, build_metrics, compile_model, unfreeze_top_layers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def train_stage_a(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: dict,
    epochs: int = 15,
    checkpoint_path: Path = Path("models/checkpoint_stage_a.keras"),
) -> tf.keras.callbacks.History:
    """
    Stage A: Train classification head only (frozen base).

    Parameters
    ----------
    model : tf.keras.Model
        The model with frozen base.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    class_weights : dict
        Class weight dictionary for imbalanced data.
    epochs : int
        Number of epochs to train.

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history.
    """
    logger.info("=== STAGE A: Train Head (Base Frozen) ===")
    logger.info("Epochs: %d, Learning rate: 1e-3", epochs)

    checkpoint_path = Path(checkpoint_path)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_auprc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auprc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Stage A complete.")
    return history


def train_stage_b(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    class_weights: dict,
    epochs: int = 20,
    checkpoint_path: Path = Path("models/checkpoint_stage_b.keras"),
) -> tf.keras.callbacks.History:
    """
    Stage B: Fine-tune a small set of top layers with a conservative LR.

    Parameters
    ----------
    model : tf.keras.Model
        The model from Stage A.
    base_model : tf.keras.Model
        The EfficientNetB0 base model.
    train_ds : tf.data.Dataset
        Training dataset.
    val_ds : tf.data.Dataset
        Validation dataset.
    class_weights : dict
        Class weight dictionary.
    epochs : int
        Number of epochs to train.

    Returns
    -------
    history : tf.keras.callbacks.History
        Training history.
    """
    logger.info("=== STAGE B: Fine-Tune (Conservative Unfreezing) ===")

    checkpoint_path = Path(checkpoint_path)

    # Stage B was degrading validation quality with an aggressive setup.
    # Fine-tune fewer layers and use a much smaller LR to preserve pretrained features.
    unfreeze_top_layers(base_model, num_layers=20)

    initial_lr = 1e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    metrics = build_metrics()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    logger.info("Epochs: %d, Initial LR: %.0e", epochs, initial_lr)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_auprc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auprc",
            mode="max",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auprc",
            mode="max",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    logger.info("Stage B complete.")
    return history


def _collect_logits_and_labels(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect raw logits and labels for a dataset.
    """
    y_true = []
    logits = []

    for images, labels in dataset:
        batch_logits = np.asarray(model.predict_on_batch(images)).reshape(-1)
        logits.extend(batch_logits.tolist())
        y_true.extend(np.asarray(labels).reshape(-1).tolist())

    return np.array(y_true), np.array(logits)


def _compute_threshold_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute confusion-matrix-derived metrics at a single decision threshold.
    """
    y_pred = (y_pred_probs >= threshold).astype(int)

    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    balanced_accuracy = (recall + specificity) / 2.0

    return {
        "recall": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_model_on_dataset(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate a logits-output model on a dataset using sigmoid probabilities.
    """
    y_true, logits = _collect_logits_and_labels(model, dataset)
    y_pred_probs = 1.0 / (1.0 + np.exp(-logits))
    metrics = _compute_threshold_metrics(y_true, y_pred_probs, threshold)
    metrics["auc"] = float(tf.keras.metrics.AUC(from_logits=False)(y_true, y_pred_probs).numpy())
    metrics["auprc"] = float(
        tf.keras.metrics.AUC(curve="PR", from_logits=False)(y_true, y_pred_probs).numpy()
    )
    metrics["threshold"] = float(threshold)
    metrics["logit_mean"] = float(np.mean(logits))
    metrics["logit_std"] = float(np.std(logits))
    return metrics


def threshold_sweep(
    model: tf.keras.Model,
    val_ds: tf.data.Dataset,
    thresholds: np.ndarray = np.arange(0.30, 0.71, 0.05),
    target_recall: float = 0.90,
    max_fpr: float = 0.25,
) -> Tuple[float, dict]:
    """
    Sweep decision thresholds to find the one maximizing recall >= 0.90
    with minimum false positive rate.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    val_ds : tf.data.Dataset
        Validation dataset.
    thresholds : np.ndarray
        Array of thresholds to test.

    Returns
    -------
    best_threshold : float
        Optimal decision threshold.
    results : dict
        Metrics at each threshold.
    """
    logger.info("=== Threshold Sweep ===")

    # Predict on validation set
    y_true, logits = _collect_logits_and_labels(model, val_ds)
    y_pred_probs = 1.0 / (1.0 + np.exp(-logits))

    results = {}
    preferred_candidate = None
    fallback_candidate = None

    for threshold in thresholds:
        metrics = _compute_threshold_metrics(y_true, y_pred_probs, threshold)
        results[float(threshold)] = metrics

        logger.info(
            "Threshold=%.2f: Recall=%.3f, FPR=%.3f, Precision=%.3f, F1=%.3f, BalAcc=%.3f",
            threshold,
            metrics["recall"],
            metrics["fpr"],
            metrics["precision"],
            metrics["f1"],
            metrics["balanced_accuracy"],
        )

        meets_target = (
            metrics["recall"] >= target_recall and metrics["fpr"] <= max_fpr
        )
        preferred_score = (metrics["f1"], metrics["balanced_accuracy"], -metrics["fpr"])
        fallback_score = (metrics["balanced_accuracy"], metrics["f1"], -metrics["fpr"])

        if meets_target:
            if preferred_candidate is None or preferred_score > preferred_candidate[0]:
                preferred_candidate = (preferred_score, threshold)
        elif fallback_candidate is None or fallback_score > fallback_candidate[0]:
            fallback_candidate = (fallback_score, threshold)

    if preferred_candidate is not None:
        best_threshold = preferred_candidate[1]
    else:
        best_threshold = fallback_candidate[1] if fallback_candidate is not None else 0.5
        logger.warning(
            "No threshold met recall >= %.2f with FPR <= %.2f; falling back to best balanced-accuracy threshold.",
            target_recall,
            max_fpr,
        )

    logger.info("Best threshold: %.2f (recall=%.3f, fpr=%.3f, f1=%.3f)",
               best_threshold,
               results[best_threshold]["recall"],
               results[best_threshold]["fpr"],
               results[best_threshold]["f1"])

    return best_threshold, results


def main(
    dataset_dir: Path = Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
    feature_type: str = "LogMel",
    batch_size: int = 32,
    output_dir: Path = Path("models"),
    epochs_stage_a: int = 15,
    epochs_stage_b: int = 20,
    do_threshold_sweep: bool = True,
    skip_stage_b: bool = False,
) -> None:
    """
    Main training orchestration.

    Parameters
    ----------
    dataset_dir : Path
        Root directory of spectrogram dataset.
    feature_type : str
        Spectrogram type: "LogMel", "FFT", or "MFCC".
    batch_size : int
        Batch size for training.
    output_dir : Path
        Directory to save models and results.
    epochs_stage_a : int
        Epochs for Stage A.
    epochs_stage_b : int
        Epochs for Stage B.
    do_threshold_sweep : bool
        Whether to run threshold sweep after training.
    skip_stage_b : bool
        If True, keep the best Stage A model and skip fine-tuning.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("GUNSHOT DETECTION CNN TRAINING")
    logger.info("=" * 70)
    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Feature type: %s", feature_type)
    logger.info("Batch size: %d", batch_size)
    logger.info("Output dir: %s", output_dir)

    # Load dataset
    logger.info("\n--- Loading Dataset ---")
    train_ds, val_ds, test_ds, class_weights = load_spectrogram_dataset(
        dataset_dir,
        feature_type=feature_type,
        batch_size=batch_size,
    )

    # Build model
    logger.info("\n--- Building Model ---")
    model, base_model = build_cnn_model()
    compile_model(model, learning_rate=1e-3)

    # Stage A: Train head
    logger.info("\n--- Stage A Training ---")
    stage_a_checkpoint = output_dir / "checkpoint_stage_a.keras"
    stage_b_checkpoint = output_dir / "checkpoint_stage_b.keras"
    history_a = train_stage_a(
        model,
        train_ds,
        val_ds,
        class_weights,
        epochs=epochs_stage_a,
        checkpoint_path=stage_a_checkpoint,
    )
    stage_a_metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.5)
    logger.info(
        "Stage A validation: AUROC=%.3f, AUPRC=%.3f, F1@0.50=%.3f",
        stage_a_metrics["auc"],
        stage_a_metrics["auprc"],
        stage_a_metrics["f1"],
    )

    stage_b_metrics = None
    if skip_stage_b or epochs_stage_b <= 0:
        selected_stage = "stage_a"
        selected_stage_metrics = stage_a_metrics
        logger.info("Skipping Stage B fine-tuning; keeping Stage A checkpoint.")
    else:
        logger.info("\n--- Stage B Training ---")
        history_b = train_stage_b(
            model,
            base_model,
            train_ds,
            val_ds,
            class_weights,
            epochs=epochs_stage_b,
            checkpoint_path=stage_b_checkpoint,
        )
        stage_b_metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.5)
        logger.info(
            "Stage B validation: AUROC=%.3f, AUPRC=%.3f, F1@0.50=%.3f",
            stage_b_metrics["auc"],
            stage_b_metrics["auprc"],
            stage_b_metrics["f1"],
        )

        selected_stage = "stage_b"
        selected_stage_metrics = stage_b_metrics
        if stage_b_metrics["auprc"] + 1e-6 < stage_a_metrics["auprc"]:
            logger.warning(
                "Stage B degraded validation AUPRC from %.3f to %.3f. Reverting to Stage A checkpoint.",
                stage_a_metrics["auprc"],
                stage_b_metrics["auprc"],
            )
            model = tf.keras.models.load_model(str(stage_a_checkpoint))
            selected_stage = "stage_a"
            selected_stage_metrics = stage_a_metrics

    # Save model
    model_path = output_dir / "cnn_gunshot_classifier.keras"
    model.save(str(model_path))
    logger.info("Model saved to: %s", model_path)

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = output_dir / "cnn_gunshot_classifier.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("TFLite model saved to: %s", tflite_path)

    # Threshold sweep
    best_threshold = 0.5
    threshold_results = {}
    if do_threshold_sweep:
        logger.info("\n--- Threshold Sweep ---")
        best_threshold, threshold_results = threshold_sweep(model, val_ds)

    # Save summary
    summary = {
        "model_path": str(model_path),
        "feature_type": feature_type,
        "best_threshold": float(best_threshold),
        "threshold_sweep_results": threshold_results,
        "class_weights": class_weights,
        "epochs_stage_a": epochs_stage_a,
        "epochs_stage_b": epochs_stage_b,
        "selected_stage": selected_stage,
        "stage_a_validation_metrics": stage_a_metrics,
        "stage_b_validation_metrics": stage_b_metrics,
        "selected_stage_validation_metrics": selected_stage_metrics,
    }

    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to: %s", summary_path)

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Model: %s", model_path)
    logger.info("Best threshold: %.2f", best_threshold)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train gunshot detection CNN")
    parser.add_argument(
        "--feature-type",
        default="LogMel",
        choices=["LogMel", "FFT", "MFCC"],
        help="Spectrogram feature type",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
        help="Root directory of the spectrogram dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for models",
    )
    parser.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Run threshold sweep after training",
    )
    parser.add_argument(
        "--epochs-a",
        type=int,
        default=15,
        help="Epochs for Stage A (head training)",
    )
    parser.add_argument(
        "--epochs-b",
        type=int,
        default=20,
        help="Epochs for Stage B (fine-tuning)",
    )
    parser.add_argument(
        "--skip-stage-b",
        action="store_true",
        help="Skip fine-tuning and keep the best Stage A checkpoint.",
    )

    args = parser.parse_args()

    main(
        dataset_dir=args.dataset_dir,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        epochs_stage_a=args.epochs_a,
        epochs_stage_b=args.epochs_b,
        do_threshold_sweep=args.threshold_sweep,
        skip_stage_b=args.skip_stage_b,
    )
