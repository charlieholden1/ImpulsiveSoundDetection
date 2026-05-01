"""
Benchmark CNN checkpoints on raw-audio datasets rendered through the shared
spectrogram pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf

from impulsive_sound_detection import config
from impulsive_sound_detection.data_loader import discover_voice_dataset, load_wav
from impulsive_sound_detection.spectrogram_utils import waveform_to_rgb_image

from .external_data import collect_audio_files, ensure_realised_dataset
from .train import _compute_threshold_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _predict_probability(
    model: tf.keras.Model,
    waveform: np.ndarray,
    sample_rate: int,
    feature_type: str,
) -> float:
    image = waveform_to_rgb_image(
        waveform,
        sample_rate,
        feature_type=feature_type,
        image_size=(224, 224),
        target_duration_sec=5.0,
    )
    logits = np.asarray(model.predict_on_batch(np.expand_dims(image, axis=0))).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    return float(probs[0])


def _voice_examples(file_list_name: str) -> list[tuple[np.ndarray, int, int, str]]:
    file_list = config.VOICE_SOURCE_DIR / file_list_name
    bundle = discover_voice_dataset(
        file_list=file_list,
        positive_labels=frozenset({"gunshot"}),
        negative_labels=config.NEGATIVE_LABELS,
    )
    examples = []
    for seg in bundle.positive:
        examples.append((seg.waveform, seg.sample_rate, 1, "voice_gunshot"))
    for seg in bundle.negative:
        examples.append((seg.waveform, seg.sample_rate, 0, "voice_babycry"))
    return examples


def _realised_examples(dataset_root: Path, limit: int | None = None) -> list[tuple[np.ndarray, int, int, str]]:
    realised_dir = ensure_realised_dataset(dataset_root)
    examples = []
    for audio_path in collect_audio_files(realised_dir, limit=limit):
        waveform, sample_rate = load_wav(audio_path)
        examples.append((waveform, sample_rate, 0, "realised_negative"))
    return examples


def benchmark_model(
    model_path: Path,
    threshold: float,
    dataset_root: Path,
    feature_type: str = "LogMel",
    realised_limit: int | None = None,
) -> dict:
    """
    Evaluate a model on VOICe validation/test segments and ReaLISED negatives.
    """
    model = tf.keras.models.load_model(str(model_path))
    datasets = {
        "voice_validation": _voice_examples("synthetic_source_validation.txt"),
        "voice_test": _voice_examples("synthetic_source_test.txt"),
        "realised_negative": _realised_examples(dataset_root, limit=realised_limit),
    }

    results = {
        "model_path": str(model_path),
        "feature_type": feature_type,
        "threshold": float(threshold),
        "datasets": {},
    }

    for name, examples in datasets.items():
        logger.info("Benchmarking %s on %d examples", name, len(examples))
        y_true = []
        y_pred_probs = []
        for waveform, sample_rate, label, _source in examples:
            prob = _predict_probability(model, waveform, sample_rate, feature_type)
            y_true.append(label)
            y_pred_probs.append(prob)

        y_true = np.array(y_true)
        y_pred_probs = np.array(y_pred_probs)
        metrics = _compute_threshold_metrics(y_true, y_pred_probs, threshold)
        metrics["auc"] = float(tf.keras.metrics.AUC(from_logits=False)(y_true, y_pred_probs).numpy())
        metrics["auprc"] = float(
            tf.keras.metrics.AUC(curve="PR", from_logits=False)(y_true, y_pred_probs).numpy()
        )
        metrics["count"] = int(len(y_true))
        metrics["positive_rate"] = float(np.mean(y_true))
        metrics["mean_probability"] = float(np.mean(y_pred_probs))
        results["datasets"][name] = metrics

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a CNN checkpoint on raw audio")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the .keras model")
    parser.add_argument("--threshold", type=float, required=True, help="Decision threshold")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("external_data"),
        help="Directory that stores downloaded external datasets",
    )
    parser.add_argument("--feature-type", default="LogMel", help="Feature type to render")
    parser.add_argument(
        "--realised-limit",
        type=int,
        default=None,
        help="Optional cap on the number of ReaLISED negatives",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/raw_benchmark.json"),
        help="Where to save benchmark results",
    )

    args = parser.parse_args()

    results = benchmark_model(
        model_path=args.model_path,
        threshold=args.threshold,
        dataset_root=args.dataset_root,
        feature_type=args.feature_type,
        realised_limit=args.realised_limit,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Saved benchmark results to %s", args.output_path)
