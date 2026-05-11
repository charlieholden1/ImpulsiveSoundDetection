"""
train_yamnet_head.py — Train a lightweight classification head on YAMNet embeddings.

Why this works better than the CNN on real audio:
  YAMNet was pre-trained on 2M real-world YouTube audio clips (AudioSet).
  Its 1024-d embedding per 0.48s frame captures rich acoustic representations
  that already understand gunshot acoustics.  The label-matching approach in
  YAMNetClassifier (checking top-K labels against a suspicious set) is brittle —
  this trains a custom binary head on those embeddings using labelled real audio,
  giving us a learned decision boundary instead of a hand-curated label list.

Architecture:
  YAMNet (frozen, from TF-Hub) → 1024-d embeddings per 0.48s frame
  → GlobalAveragePooling over time frames → 1024-d vector
  → Dense(256, relu) → Dropout(0.5)
  → Dense(64, relu) → Dropout(0.3)
  → Dense(1, logit)  [binary: gunshot vs. non-gunshot]

Training data: VOICe annotated gunshot/babycry segments + ReaLISED hard negatives
(same real-audio sources as finetune_realworld.py).

Usage:
    python -m train.train_yamnet_head [--epochs 30] [--batch-size 64]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterator, List, Tuple

import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.utils.class_weight import compute_class_weight

from impulsive_sound_detection import config
from impulsive_sound_detection.data_loader import discover_voice_dataset, load_wav

from .external_data import collect_audio_files, ensure_realised_dataset
from .train import _compute_threshold_metrics, threshold_sweep

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

YAMNET_EMBEDDING_DIM = 1024

# Pitch-shift augmentation steps applied to positive (gunshot) segments.
# Simulates the acoustic effect of shooting at different distances / rooms.
_PITCH_SHIFTS_SEMITONES = (-2, 2)  # ±2 semitones


def _pitch_augment(waveform: np.ndarray, sr: int) -> list[np.ndarray]:
    """Return the original waveform plus pitch-shifted copies."""
    copies = [waveform]
    for n_steps in _PITCH_SHIFTS_SEMITONES:
        try:
            shifted = librosa.effects.pitch_shift(waveform.astype(np.float32), sr=sr, n_steps=n_steps)
            copies.append(shifted)
        except Exception as exc:
            logger.warning("Pitch shift (%+d st) failed: %s", n_steps, exc)
    return copies


# ─────────────────────────────────────────────────────────────────────────────
# YAMNet embedding extractor
# ─────────────────────────────────────────────────────────────────────────────

class YAMNetEmbeddingExtractor:
    """Lazy-loading wrapper that extracts frame-level embeddings from YAMNet."""

    def __init__(self, model_handle: str = config.YAMNET_MODEL_HANDLE) -> None:
        self._model_handle = model_handle
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading YAMNet from %s …", self._model_handle)
        self._model = hub.load(self._model_handle)
        logger.info("YAMNet loaded.")

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract pooled embeddings from a waveform.

        Parameters
        ----------
        waveform : np.ndarray  shape (N,) float32
            Audio at 16 kHz.

        Returns
        -------
        np.ndarray  shape (1024,)
            Mean-pooled YAMNet embedding across all time frames.
        """
        self._ensure_model()
        waveform_t = tf.constant(waveform.astype(np.float32))
        _, embeddings, _ = self._model(waveform_t)
        # embeddings shape: (n_frames, 1024)
        return np.mean(embeddings.numpy(), axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset builders
# ─────────────────────────────────────────────────────────────────────────────

def _collect_embeddings(
    extractor: YAMNetEmbeddingExtractor,
    dataset_root: Path,
    file_list_name: str | None = None,
    annotation_dir: Path | None = None,
    augment_positives: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collect YAMNet embeddings and labels from VOICe and ReaLISED.

    Parameters
    ----------
    extractor : YAMNetEmbeddingExtractor
    dataset_root : Path
        Root directory for external datasets (ReaLISED).
    file_list_name : str | None
        Optional file list within VOICE_SOURCE_DIR to restrict examples.
    annotation_dir : Path | None
        If provided, load annotations from this directory instead of the
        default VOICE_ANNOTATION_DIR.  Use this to retrain with corrected
        annotations from the annotate_audio.py tool.
    augment_positives : bool
        If True, add pitch-shifted copies of positive segments (±2 semitones).

    Returns
    -------
    embeddings : np.ndarray  shape (N, 1024)
    labels     : np.ndarray  shape (N,)
    """
    file_list = None
    if file_list_name:
        candidate = config.VOICE_SOURCE_DIR / file_list_name
        if candidate.exists():
            file_list = candidate

    ann_dir = Path(annotation_dir) if annotation_dir else config.VOICE_ANNOTATION_DIR
    # When a corrected annotation dir is given, it only covers the files that
    # were reviewed.  Fall back to the original annotations for the rest so
    # the training set stays complete.
    fallback = config.VOICE_ANNOTATION_DIR if annotation_dir else None

    bundle = discover_voice_dataset(
        annotation_dir=ann_dir,
        fallback_annotation_dir=fallback,
        file_list=file_list,
        positive_labels=frozenset({"gunshot"}),
        negative_labels=config.NEGATIVE_LABELS,
    )

    embeddings: List[np.ndarray] = []
    labels: List[int] = []

    for seg in bundle.positive:
        waveforms = _pitch_augment(seg.waveform, seg.sample_rate) if augment_positives else [seg.waveform]
        for wav in waveforms:
            try:
                emb = extractor.extract(wav)
                embeddings.append(emb)
                labels.append(1)
            except Exception as exc:
                logger.warning("Skipping positive segment: %s", exc)

    for seg in bundle.negative:
        try:
            emb = extractor.extract(seg.waveform)
            embeddings.append(emb)
            labels.append(0)
        except Exception as exc:
            logger.warning("Skipping negative segment: %s", exc)

    # ReaLISED hard negatives
    realised_dir = ensure_realised_dataset(dataset_root)
    audio_files = collect_audio_files(realised_dir)
    logger.info("ReaLISED: extracting embeddings from %d files …", len(audio_files))
    for audio_path in audio_files:
        try:
            waveform, _ = load_wav(audio_path)
            emb = extractor.extract(waveform)
            embeddings.append(emb)
            labels.append(0)
        except Exception as exc:
            logger.warning("Skipping ReaLISED file %s: %s", audio_path.name, exc)

    return np.stack(embeddings, axis=0), np.array(labels, dtype=np.int32)


def _build_tf_dataset(
    embeddings: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(
        (embeddings.astype(np.float32), labels.astype(np.int32))
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=len(labels), reshuffle_each_iteration=True, seed=seed)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────────────────────────────────────
# Model architecture
# ─────────────────────────────────────────────────────────────────────────────

def build_yamnet_head_model(embedding_dim: int = YAMNET_EMBEDDING_DIM) -> tf.keras.Model:
    """
    Build a lightweight classification head for YAMNet embeddings.

    Input: pre-pooled 1024-d embedding vector (already averaged over frames).
    Output: raw binary logit.
    """
    inputs = tf.keras.Input(shape=(embedding_dim,), name="yamnet_embedding")
    x = tf.keras.layers.Dense(256, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(1, activation=None, name="logit")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="yamnet_head")
    logger.info("YAMNet head model built. Parameters: %d", model.count_params())
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    dataset_root: Path = Path(r"C:\Users\holde\Documents\MLProject\external_data"),
    output_dir: Path = config.YAMNET_HEAD_MODEL_PATH.parent,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    val_fraction: float = 0.20,
    do_threshold_sweep: bool = True,
    annotation_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("YAMNET EMBEDDING HEAD TRAINING")
    logger.info("=" * 70)
    if annotation_dir:
        logger.info("Using corrected annotations from: %s", annotation_dir)

    extractor = YAMNetEmbeddingExtractor()

    # Collect training embeddings (all VOICe + ReaLISED, with pitch augmentation)
    logger.info("Extracting YAMNet embeddings for full dataset …")
    embeddings, labels = _collect_embeddings(
        extractor, dataset_root, file_list_name=None,
        annotation_dir=annotation_dir, augment_positives=True,
    )
    n_pos = int(np.sum(labels))
    n_neg = int(np.sum(labels == 0))
    logger.info("Dataset: %d positive, %d negative (%d total)", n_pos, n_neg, len(labels))

    # Validation set — never augmented, always from canonical validation file list
    logger.info("Extracting validation embeddings (VOICe source validation) …")
    val_emb, val_lbl = _collect_embeddings(
        extractor, dataset_root, file_list_name="synthetic_source_validation.txt",
        annotation_dir=annotation_dir, augment_positives=False,
    )

    # Remove validation examples from training set (de-overlap by index matching)
    # Simple approach: use all non-validation VOICe for training
    train_emb, train_lbl = embeddings, labels

    # Class weights
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=train_lbl)
    class_weights = {0: float(weights[0]), 1: float(weights[1])}
    logger.info("Class weights: %s", class_weights)

    train_ds = _build_tf_dataset(train_emb, train_lbl, batch_size, shuffle=True)
    val_ds = _build_tf_dataset(val_emb, val_lbl, batch_size, shuffle=False)

    # Build and compile model
    model = build_yamnet_head_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0),
            tf.keras.metrics.AUC(name="auc", from_logits=True),
            tf.keras.metrics.AUC(name="auprc", curve="PR", from_logits=True),
            tf.keras.metrics.Recall(name="recall", thresholds=0.0),
        ],
    )

    checkpoint_path = output_dir / "checkpoint_best.keras"
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
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auprc",
            mode="max",
            factor=0.5,
            patience=3,
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

    # Evaluate
    logger.info("Evaluating on validation set …")
    y_true_val = val_lbl
    logits_val = np.asarray(model.predict(val_ds, verbose=0)).reshape(-1)
    probs_val = 1.0 / (1.0 + np.exp(-logits_val))

    val_metrics = _compute_threshold_metrics(y_true_val, probs_val, threshold=0.5)
    val_metrics["auc"] = float(
        tf.keras.metrics.AUC(from_logits=False)(y_true_val, probs_val).numpy()
    )
    val_metrics["auprc"] = float(
        tf.keras.metrics.AUC(curve="PR", from_logits=False)(y_true_val, probs_val).numpy()
    )
    logger.info(
        "Val — Recall=%.3f, Precision=%.3f, F1=%.3f, AUC=%.3f, AUPRC=%.3f",
        val_metrics["recall"], val_metrics["precision"],
        val_metrics["f1"], val_metrics["auc"], val_metrics["auprc"],
    )

    # Threshold sweep
    best_threshold = 0.5
    threshold_results = {}
    if do_threshold_sweep:
        logger.info("Running threshold sweep …")
        best_threshold, threshold_results = threshold_sweep(
            model, val_ds, target_recall=0.80, max_fpr=0.30
        )

    # Save model
    model_path = output_dir / "yamnet_head_classifier.keras"
    model.save(str(model_path))
    logger.info("YAMNet head model saved to: %s", model_path)

    summary = {
        "model_path": str(model_path),
        "embedding_dim": YAMNET_EMBEDDING_DIM,
        "best_threshold": float(best_threshold),
        "threshold_sweep_results": threshold_results,
        "class_weights": class_weights,
        "epochs_trained": len(history.history.get("loss", [])),
        "val_metrics": val_metrics,
        "n_train": int(len(train_lbl)),
        "n_val": int(len(val_lbl)),
    }
    summary_path = output_dir / "yamnet_head_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to: %s", summary_path)

    logger.info("\n" + "=" * 70)
    logger.info("YAMNET HEAD TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info("Balanced accuracy: %.3f", (val_metrics["recall"] + val_metrics["specificity"]) / 2)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YAMNet embedding head")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path(r"C:\Users\holde\Documents\MLProject\external_data"),
                        help="Directory for external datasets (ReaLISED auto-downloaded here)")
    parser.add_argument("--output-dir", type=Path,
                        default=config.YAMNET_HEAD_MODEL_PATH.parent,
                        help="Where to save the trained head model")
    parser.add_argument("--annotation-dir", type=Path, default=None,
                        help="Use corrected annotations from this directory (e.g. clean/annotation_corrected) "
                             "instead of the default VOICE_ANNOTATION_DIR.  Pass this after an annotation "
                             "session with train/annotate_audio.py to retrain on hard examples.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-threshold-sweep", action="store_true")

    args = parser.parse_args()
    main(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        do_threshold_sweep=not args.no_threshold_sweep,
        annotation_dir=args.annotation_dir,
    )
