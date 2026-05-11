"""
finetune_realworld.py — Stage C: fine-tune the CNN on real audio to close the domain gap.

Root cause of the domain gap: the CNN was trained on pre-rendered PNG spectrograms
from Kaggle, which used different rendering parameters than the inference pipeline's
waveform_to_rgb_image().  The model learned spectrogram rendering artefacts rather
than gunshot acoustics, causing recall to collapse to 11-15% on real VOICe audio.

Fix: render all real training audio through the exact same waveform_to_rgb_image()
pipeline used at inference, so training and inference see identical representations.

Mixed dataset composition:
  - Real positives:  VOICe gunshot segments (rendered on-the-fly)
  - Real negatives:  VOICe babycry segments + ReaLISED audio (hard negatives)
  - Synthetic anchor: 30% random sample of Kaggle FFT PNGs (prevents catastrophic
    forgetting of the 98.8% synthetic-domain performance)

Metric gate: recall on voice_validation should rise from 11% to at least 45%.

Usage:
    python -m train.finetune_realworld [--epochs 10] [--batch-size 16]
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Iterator, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from impulsive_sound_detection import config
from impulsive_sound_detection.augmentor import RobustAugmentor
from impulsive_sound_detection.data_loader import (
    discover_voice_dataset,
    load_wav,
)
from impulsive_sound_detection.spectrogram_utils import waveform_to_rgb_image

from .external_data import collect_audio_files, ensure_realised_dataset
from .model import build_metrics, unfreeze_top_layers
from .train import train_stage_c, threshold_sweep, evaluate_model_on_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

IMAGE_SIZE = (224, 224)
FEATURE_TYPE = "FFT"


# ─────────────────────────────────────────────────────────────────────────────
# Waveform → image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _waveform_to_uint8_image(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Render one waveform to a uint8 RGB spectrogram using the inference pipeline."""
    img = waveform_to_rgb_image(
        waveform,
        sample_rate,
        feature_type=FEATURE_TYPE,
        image_size=IMAGE_SIZE,
        target_duration_sec=5.0,
    )
    return img.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Example generators
# ─────────────────────────────────────────────────────────────────────────────

def _voice_generator(
    file_list_name: str | None = None,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (image_uint8, label) from VOICe annotated gunshot/babycry segments."""
    file_list = None
    if file_list_name:
        candidate = config.VOICE_SOURCE_DIR / file_list_name
        if candidate.exists():
            file_list = candidate

    bundle = discover_voice_dataset(
        file_list=file_list,
        positive_labels=frozenset({"gunshot"}),
        negative_labels=config.NEGATIVE_LABELS,
    )

    for seg in bundle.positive:
        try:
            img = _waveform_to_uint8_image(seg.waveform, seg.sample_rate)
            yield img, 1
        except Exception as exc:
            logger.warning("Skipping VOICe positive segment: %s", exc)

    for seg in bundle.negative:
        try:
            img = _waveform_to_uint8_image(seg.waveform, seg.sample_rate)
            yield img, 0
        except Exception as exc:
            logger.warning("Skipping VOICe negative segment: %s", exc)


def _realised_generator(dataset_root: Path) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield (image_uint8, label=0) for all ReaLISED audio files (hard negatives)."""
    realised_dir = ensure_realised_dataset(dataset_root)
    audio_files = collect_audio_files(realised_dir)
    logger.info("ReaLISED: %d audio files found", len(audio_files))
    for audio_path in audio_files:
        try:
            waveform, sr = load_wav(audio_path)
            img = _waveform_to_uint8_image(waveform, sr)
            yield img, 0
        except Exception as exc:
            logger.warning("Skipping ReaLISED file %s: %s", audio_path.name, exc)


def _synthetic_png_generator(
    kaggle_dir: Path,
    sample_fraction: float = config.KAGGLE_SAMPLE_FRACTION,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, int]]:
    """Yield a random subset of Kaggle FFT PNGs as (image_uint8, label) pairs."""
    gun_dir = kaggle_dir / FEATURE_TYPE / "GUN"
    nogun_dir = kaggle_dir / FEATURE_TYPE / "NOGUN"

    gun_paths = sorted(gun_dir.glob("*.png")) if gun_dir.exists() else []
    nogun_paths = sorted(nogun_dir.glob("*.png")) if nogun_dir.exists() else []

    rng = random.Random(seed)
    gun_sample = rng.sample(gun_paths, max(1, int(len(gun_paths) * sample_fraction)))
    nogun_sample = rng.sample(nogun_paths, max(1, int(len(nogun_paths) * sample_fraction)))

    logger.info(
        "Synthetic anchor: %d GUN + %d NOGUN PNGs (%.0f%% of each class)",
        len(gun_sample), len(nogun_sample), sample_fraction * 100,
    )

    all_examples: List[Tuple[Path, int]] = (
        [(p, 1) for p in gun_sample] + [(p, 0) for p in nogun_sample]
    )
    rng.shuffle(all_examples)

    for png_path, label in all_examples:
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(png_path).convert("RGB").resize(IMAGE_SIZE), dtype=np.uint8)
            yield img, label
        except Exception as exc:
            logger.warning("Skipping PNG %s: %s", png_path.name, exc)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def _generator_to_tf_dataset(
    gen_fn,
    batch_size: int,
    shuffle: bool = True,
    buffer_size: int = 500,
) -> tf.data.Dataset:
    """Wrap a Python generator in a batched, prefetched tf.data.Dataset."""
    ds = tf.data.Dataset.from_generator(
        gen_fn,
        output_signature=(
            tf.TensorSpec(shape=(*IMAGE_SIZE, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_training_dataset(
    dataset_root: Path,
    batch_size: int = 16,
    augment: bool = True,
) -> Tuple[tf.data.Dataset, dict]:
    """
    Build the mixed real+synthetic training dataset.

    When augment=True, each real-audio waveform is passed through
    RobustAugmentor before rendering, simulating varied acoustic conditions.
    Gunshot positives get 3 augmented copies (to compensate for class imbalance
    since real gunshot recordings are scarce); negatives get 1 copy.

    Returns train_ds and class_weights computed from collected labels.
    """
    logger.info("Building mixed real+synthetic training dataset (augment=%s) …", augment)

    augmentor = RobustAugmentor(sample_rate=config.SAMPLE_RATE) if augment else None

    def _render_with_augmentation(waveform: np.ndarray, sample_rate: int, label: int, n_copies: int = 1):
        """Yield n_copies of a waveform, each optionally augmented before rendering."""
        for _ in range(n_copies):
            wav = augmentor(waveform) if augmentor is not None else waveform
            try:
                img = _waveform_to_uint8_image(wav, sample_rate)
                yield img, label
            except Exception as exc:
                logger.warning("Skipping augmented example (label=%d): %s", label, exc)

    # Collect all examples eagerly to compute class weights before creating the dataset
    all_images: List[np.ndarray] = []
    all_labels: List[int] = []

    # VOICe: 3 augmented copies for positives (scarce gunshot data), 1 for negatives
    bundle_voice = discover_voice_dataset(
        file_list=None,
        positive_labels=frozenset({"gunshot"}),
        negative_labels=config.NEGATIVE_LABELS,
    )
    for seg in bundle_voice.positive:
        for img, lbl in _render_with_augmentation(seg.waveform, seg.sample_rate, label=1, n_copies=3):
            all_images.append(img)
            all_labels.append(lbl)
    for seg in bundle_voice.negative:
        for img, lbl in _render_with_augmentation(seg.waveform, seg.sample_rate, label=0, n_copies=1):
            all_images.append(img)
            all_labels.append(lbl)

    # ReaLISED hard negatives (1 copy each; augmentation adds diversity)
    realised_dir = ensure_realised_dataset(dataset_root)
    for audio_path in collect_audio_files(realised_dir):
        try:
            waveform, sr = load_wav(audio_path)
            for img, lbl in _render_with_augmentation(waveform, sr, label=0, n_copies=1):
                all_images.append(img)
                all_labels.append(lbl)
        except Exception as exc:
            logger.warning("Skipping ReaLISED file: %s", exc)

    for img, label in _synthetic_png_generator(config.GUNSHOT_SPECTROGRAM_DIR):
        all_images.append(img)
        all_labels.append(label)

    labels_arr = np.array(all_labels)
    images_arr = np.stack(all_images, axis=0)

    n_pos = int(np.sum(labels_arr))
    n_neg = int(np.sum(labels_arr == 0))
    logger.info("Training set: %d positive, %d negative (%d total)", n_pos, n_neg, len(labels_arr))

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels_arr)
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    logger.info("Class weights: %s", class_weights)

    # Shuffle once before creating the dataset
    indices = np.random.default_rng(seed=42).permutation(len(labels_arr))
    images_arr = images_arr[indices]
    labels_arr = labels_arr[indices]

    ds = tf.data.Dataset.from_tensor_slices((images_arr, labels_arr.astype(np.int32)))
    ds = ds.shuffle(buffer_size=min(1000, len(labels_arr)), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return ds, class_weights


def build_val_dataset(batch_size: int = 16) -> tf.data.Dataset:
    """
    Build a validation dataset from VOICe source-domain validation file list.

    Uses the same waveform_to_rgb_image() pipeline as training to ensure
    the validation distribution matches the training distribution.
    """
    logger.info("Building real-audio validation dataset (VOICe source validation) …")

    def gen():
        yield from _voice_generator(file_list_name="synthetic_source_validation.txt")

    return _generator_to_tf_dataset(gen, batch_size=batch_size, shuffle=False)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    base_model_path: Path = Path(
        r"C:\Users\holde\Documents\MLProject\models\feature_sweep_stagea6\fft\cnn_gunshot_classifier.keras"
    ),
    dataset_root: Path = Path(r"C:\Users\holde\Documents\MLProject\external_data"),
    output_dir: Path = config.FINETUNED_MODEL_PATH.parent,
    epochs: int = config.REALWORLD_FINETUNE_EPOCHS,
    learning_rate: float = config.REALWORLD_FINETUNE_LR,
    batch_size: int = 16,
    do_threshold_sweep: bool = True,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("STAGE C: REAL-WORLD DOMAIN ADAPTATION FINE-TUNING")
    logger.info("=" * 70)
    logger.info("Base model: %s", base_model_path)
    logger.info("Output dir: %s", output_dir)

    # Load the best synthetic-domain checkpoint
    logger.info("Loading base model …")
    model = tf.keras.models.load_model(str(base_model_path))

    # Retrieve the EfficientNetB0 sub-model by name for selective unfreezing
    base_model = next(
        (layer for layer in model.layers if "efficientnetb0" in layer.name.lower()),
        None,
    )
    if base_model is None:
        raise RuntimeError(
            "Could not locate EfficientNetB0 sub-model in the loaded model. "
            "Check that the model was built with build_cnn_model()."
        )
    logger.info("Found EfficientNetB0 sub-model: %s", base_model.name)

    # Compile first with all layers frozen to verify forward pass
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=build_metrics(),
    )

    # Build datasets
    train_ds, class_weights = build_training_dataset(dataset_root, batch_size=batch_size)
    val_ds = build_val_dataset(batch_size=batch_size)

    # Evaluate baseline before fine-tuning
    logger.info("Evaluating base model on real-audio val set (before Stage C) …")
    baseline_metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.65)
    logger.info(
        "Baseline — Recall=%.3f, Precision=%.3f, F1=%.3f, AUC=%.3f",
        baseline_metrics["recall"], baseline_metrics["precision"],
        baseline_metrics["f1"], baseline_metrics["auc"],
    )

    # Stage C fine-tuning
    checkpoint_path = output_dir / "checkpoint_stage_c.keras"
    train_stage_c(
        model=model,
        base_model=base_model,
        train_ds=train_ds,
        val_ds=val_ds,
        class_weights=class_weights,
        epochs=epochs,
        learning_rate=learning_rate,
        checkpoint_path=checkpoint_path,
    )

    # Post-fine-tune evaluation
    logger.info("Evaluating fine-tuned model on real-audio val set (after Stage C) …")
    post_metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.5)
    logger.info(
        "After Stage C — Recall=%.3f, Precision=%.3f, F1=%.3f, AUC=%.3f",
        post_metrics["recall"], post_metrics["precision"],
        post_metrics["f1"], post_metrics["auc"],
    )

    # Threshold sweep on real-audio val set
    best_threshold = 0.5
    threshold_results = {}
    if do_threshold_sweep:
        logger.info("Running threshold sweep on real-audio val set …")
        best_threshold, threshold_results = threshold_sweep(
            model,
            val_ds,
            target_recall=0.80,  # lower target than synthetic — real audio is harder
            max_fpr=0.30,
        )

    # Save fine-tuned model
    model_path = output_dir / "cnn_gunshot_classifier.keras"
    model.save(str(model_path))
    logger.info("Fine-tuned model saved to: %s", model_path)

    # Save TFLite export
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path = output_dir / "cnn_gunshot_classifier.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    logger.info("TFLite model saved to: %s", tflite_path)

    # Save summary
    summary = {
        "base_model_path": str(base_model_path),
        "fine_tuned_model_path": str(model_path),
        "feature_type": FEATURE_TYPE,
        "best_threshold": float(best_threshold),
        "threshold_sweep_results": threshold_results,
        "class_weights": class_weights,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "baseline_real_audio_metrics": baseline_metrics,
        "post_finetune_real_audio_metrics": post_metrics,
    }
    summary_path = output_dir / "finetune_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to: %s", summary_path)

    recall_delta = post_metrics["recall"] - baseline_metrics["recall"]
    logger.info("\n" + "=" * 70)
    logger.info("STAGE C COMPLETE")
    logger.info("=" * 70)
    logger.info("Real-audio recall: %.1f%% → %.1f%%  (Δ%.1f%%)",
                baseline_metrics["recall"] * 100,
                post_metrics["recall"] * 100,
                recall_delta * 100)
    logger.info("Best threshold: %.2f", best_threshold)
    logger.info("=" * 70)

    if post_metrics["recall"] < 0.45:
        logger.warning(
            "Recall (%.1f%%) did not reach the minimum gate of 45%%. "
            "Consider: (1) running more epochs, (2) adding more VOICe data, "
            "or (3) checking that the base model path points to the FFT checkpoint.",
            post_metrics["recall"] * 100,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage C: fine-tune CNN on real audio to close the domain gap"
    )
    parser.add_argument("--base-model-path", type=Path,
                        default=Path(r"C:\Users\holde\Documents\MLProject\models\feature_sweep_stagea6\fft\cnn_gunshot_classifier.keras"),
                        help="Path to the Stage A/B .keras checkpoint to start from")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(r"C:\Users\holde\Documents\MLProject\external_data"),
                        help="Directory for external datasets (ReaLISED auto-downloaded here)")
    parser.add_argument("--output-dir", type=Path,
                        default=config.FINETUNED_MODEL_PATH.parent,
                        help="Where to save the fine-tuned model and summary")
    parser.add_argument("--epochs", type=int, default=config.REALWORLD_FINETUNE_EPOCHS,
                        help="Maximum Stage C epochs")
    parser.add_argument("--learning-rate", type=float, default=config.REALWORLD_FINETUNE_LR,
                        help="Starting learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--no-threshold-sweep", action="store_true",
                        help="Skip the threshold sweep after training")

    args = parser.parse_args()
    main(
        base_model_path=args.base_model_path,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        do_threshold_sweep=not args.no_threshold_sweep,
    )
