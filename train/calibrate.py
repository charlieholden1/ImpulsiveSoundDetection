"""
calibrate.py — Platt scaling calibration for the CNN classifier.

The FFT model has logit std ≈ 7.95 (very high), pushing sigmoid outputs to
the extremes (near 0 or 1).  This makes the raw confidence scores unreliable
for ensemble weighting: a CNN score of 0.85 and a YAMNet score of 0.85 don't
represent the same actual gunshot probability.

Platt scaling fits a logistic regression (a + b*logit → calibrated_probability)
on a held-out calibration set, producing confidence scores that are meaningful
probabilities rather than clipped outputs.

Usage:
    python -m train.calibrate \
        --model-path models/feature_sweep_stagea6/fft/cnn_gunshot_classifier.keras \
        --output-path models/calibration_params.json

After running, CNNClassifier in classifier.py will automatically load and apply
these parameters if CNN_CALIBRATION_PATH points to the output file.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from impulsive_sound_detection import config
from impulsive_sound_detection.data_loader import discover_voice_dataset
from impulsive_sound_detection.spectrogram_utils import waveform_to_rgb_image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def collect_logits_on_real_audio(
    model: tf.keras.Model,
    feature_type: str = "FFT",
    file_list_name: str | None = "synthetic_source_validation.txt",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the model on VOICe validation segments and return (logits, true_labels).

    Uses the real-audio rendering pipeline (waveform_to_rgb_image) rather than
    pre-computed PNGs, so calibration is performed in the same domain as inference.

    Parameters
    ----------
    model : tf.keras.Model
        CNN model with raw logit output.
    feature_type : str
        Spectrogram type (must match what the model was trained on).
    file_list_name : str | None
        File list within VOICE_SOURCE_DIR to restrict the calibration set.
        If None, all VOICe files are used.

    Returns
    -------
    logits : np.ndarray  shape (N,)
    labels : np.ndarray  shape (N,)
    """
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

    all_logits = []
    all_labels = []

    for seg, label in [(s, 1) for s in bundle.positive] + [(s, 0) for s in bundle.negative]:
        try:
            img = waveform_to_rgb_image(
                seg.waveform,
                seg.sample_rate,
                feature_type=feature_type,
                image_size=(224, 224),
                target_duration_sec=5.0,
            )
            batch = np.expand_dims(img.astype(np.uint8), axis=0)
            logit = float(np.asarray(model.predict_on_batch(batch)).reshape(-1)[0])
            all_logits.append(logit)
            all_labels.append(label)
        except Exception as exc:
            logger.warning("Skipping segment: %s", exc)

    if len(all_logits) < 10:
        raise RuntimeError(
            f"Too few calibration examples ({len(all_logits)}). "
            "Check that the VOICe dataset is present and the file list is correct."
        )

    logger.info(
        "Collected %d calibration examples (%d positive, %d negative)",
        len(all_logits),
        int(np.sum(all_labels)),
        int(np.sum(np.array(all_labels) == 0)),
    )
    return np.array(all_logits, dtype=np.float32), np.array(all_labels, dtype=np.int32)


def fit_platt_scaling(
    logits: np.ndarray,
    labels: np.ndarray,
) -> dict:
    """
    Fit Platt scaling (logistic regression on logits) to calibrate probabilities.

    Parameters
    ----------
    logits : np.ndarray  shape (N,)
        Raw model logits.
    labels : np.ndarray  shape (N,)
        True binary labels.

    Returns
    -------
    dict with keys "coef" and "intercept" (both floats).
    """
    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    lr.fit(logits.reshape(-1, 1), labels)

    coef = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])

    logger.info("Platt scaling fitted: coef=%.4f, intercept=%.4f", coef, intercept)
    return {"coef": coef, "intercept": intercept}


def apply_calibration(logit: float, coef: float, intercept: float) -> float:
    """Map a raw logit to a calibrated probability using Platt scaling."""
    return float(1.0 / (1.0 + np.exp(-(coef * logit + intercept))))


def _compute_ece(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (lower is better)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = float(probs[mask].mean())
        bin_acc = float(labels[mask].mean())
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def main(
    model_path: Path,
    output_path: Path = config.CNN_CALIBRATION_PATH,
    feature_type: str = "FFT",
    file_list_name: str | None = "synthetic_source_validation.txt",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading model from %s …", model_path)
    model = tf.keras.models.load_model(str(model_path))

    logger.info("Collecting logits on real-audio calibration set …")
    logits, labels = collect_logits_on_real_audio(model, feature_type, file_list_name)

    # Before calibration
    raw_probs = 1.0 / (1.0 + np.exp(-logits))
    ece_before = _compute_ece(labels, raw_probs)
    logger.info("ECE before calibration: %.4f", ece_before)

    # Fit Platt scaling
    params = fit_platt_scaling(logits, labels)

    # After calibration
    cal_probs = np.array([apply_calibration(l, params["coef"], params["intercept"]) for l in logits])
    ece_after = _compute_ece(labels, cal_probs)
    logger.info("ECE after  calibration: %.4f", ece_after)

    # Reliability diagram data (fraction of positives per confidence bin)
    try:
        fraction_of_positives, mean_predicted = calibration_curve(
            labels, cal_probs, n_bins=10, strategy="uniform"
        )
        logger.info("Calibration curve (mean_predicted → fraction_positives):")
        for mp, fp in zip(mean_predicted, fraction_of_positives):
            logger.info("  %.2f → %.2f", mp, fp)
    except Exception:
        pass

    params["ece_before"] = ece_before
    params["ece_after"] = ece_after
    params["feature_type"] = feature_type
    params["n_calibration_examples"] = int(len(labels))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    logger.info("Calibration parameters saved to: %s", output_path)

    if ece_after >= 0.15:
        logger.warning(
            "ECE after calibration (%.4f) did not reach the target of <0.08. "
            "Consider using a larger or more balanced calibration set.",
            ece_after,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit Platt scaling for CNN confidence calibration")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to the .keras model file")
    parser.add_argument("--output-path", type=Path, default=config.CNN_CALIBRATION_PATH,
                        help="Where to save the calibration parameters JSON")
    parser.add_argument("--feature-type", default="FFT",
                        choices=["FFT", "LogMel", "MFCC"],
                        help="Spectrogram feature type used by the model")
    parser.add_argument("--file-list", default="synthetic_source_validation.txt",
                        help="VOICe file list to use for calibration (from VOICE_SOURCE_DIR)")

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        output_path=args.output_path,
        feature_type=args.feature_type,
        file_list_name=args.file_list,
    )
