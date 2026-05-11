"""
classifier.py – YAMNet-based classification and suspicious-label filtering
(Stage 2).

Loads the YAMNet model from TensorFlow Hub, runs inference on a 0.975 s
audio window, extracts the top-K predictions, and maps them to a binary
"suspicious" / "non-suspicious" label.

Public API
----------
YAMNetClassifier
    Stateful wrapper that lazily loads the model on first use.
ClassificationResult
    Structured output of a single inference call.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import time

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from . import config
from .spectrogram_utils import waveform_to_rgb_image

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    """Output of a classification inference."""

    timestamp: float
    onset_index: int
    label: str
    confidence: float
    is_suspicious: bool
    top_k: List[dict] = field(default_factory=list)
    event_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    severity: str = "LOW"
    session_id: Optional[str] = None
    # ── MQTT / multi-node fields ──────────────────────────────────────
    node_id: str = field(default_factory=lambda: config.NODE_ID)
    wall_clock_time: float = field(default_factory=time.time)
    received_at_host: float = 0.0   # set by host_subscriber on arrival

    def __post_init__(self):
        """Compute severity from confidence."""
        if self.confidence < 0.6:
            self.severity = "LOW"
        elif self.confidence < 0.85:
            self.severity = "MEDIUM"
        else:
            self.severity = "HIGH"

    def to_json(self) -> str:
        """Serialise to a compact wire-format JSON string for MQTT.

        Omits top_k to keep the payload small.  Full payload (incl.
        top_k) is written to the local JSONL log via to_json_full().

        Returns
        -------
        str
            JSON representation of the detection event.
        """
        payload = {
            "event_uuid":       self.event_uuid,
            "node_id":          self.node_id,
            "timestamp_unix":   self.timestamp,
            "wall_clock_time":  round(self.wall_clock_time, 6),
            "timestamp_iso":    datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "onset_index":      self.onset_index,
            "label":            self.label,
            "confidence":       round(self.confidence, 4),
            "is_suspicious":    self.is_suspicious,
            "severity":         self.severity,
            "session_id":       self.session_id,
        }
        return json.dumps(payload)

    def to_json_full(self) -> str:
        """Serialise including top_k – used for local JSONL logs.

        Returns
        -------
        str
        """
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        """Return a plain dictionary for logging or aggregation.

        Returns
        -------
        dict
        """
        return asdict(self)

    @classmethod
    def from_mqtt_payload(cls, payload: str, received_at: float) -> "ClassificationResult":
        """Deserialise a JSON string received from MQTT on the host.

        Parameters
        ----------
        payload : str
            JSON string as published by the node via to_json().
        received_at : float
            time.time() captured by the host subscriber on arrival.
            Used by the Sound Localization team for TDOA calculations.

        Returns
        -------
        ClassificationResult
        """
        data = json.loads(payload)
        obj = cls(
            timestamp=data["timestamp_unix"],
            onset_index=data.get("onset_index", 0),
            label=data["label"],
            confidence=data["confidence"],
            is_suspicious=data["is_suspicious"],
            top_k=[],
            event_uuid=data.get("event_uuid", str(uuid.uuid4())),
            severity=data.get("severity", "LOW"),
            session_id=data.get("session_id"),
            node_id=data.get("node_id", "unknown"),
            wall_clock_time=data.get("wall_clock_time", 0.0),
            received_at_host=received_at,
        )
        return obj


# ──────────────────────────────────────────────────────────────────────
# YAMNet classifier wrapper
# ──────────────────────────────────────────────────────────────────────
class YAMNetClassifier:
    """Lazy-loading YAMNet wrapper with suspicious-label filtering.

    Parameters
    ----------
    model_handle : str
        TensorFlow Hub URL for the YAMNet model.
    top_k : int
        Number of top predictions to retain.
    suspicious_labels : frozenset
        Set of YAMNet class names that should be flagged suspicious.
    """

    def __init__(
        self,
        model_handle: str = config.YAMNET_MODEL_HANDLE,
        top_k: int = config.TOP_K,
        suspicious_labels: frozenset = config.SUSPICIOUS_LABELS,
        node_id: str = config.NODE_ID,
    ) -> None:
        self._model_handle = model_handle
        self._top_k = top_k
        self._suspicious_labels = suspicious_labels
        self._node_id = node_id
        self._model = None
        self._class_names: List[str] = []

    # ── lazy model loading ────────────────────────────────────────────
    def _ensure_model(self) -> None:
        """Load YAMNet from TF-Hub if not already cached.

        Raises
        ------
        RuntimeError
            If the model cannot be loaded.
        """
        if self._model is not None:
            return
        logger.info("Loading YAMNet from %s …", self._model_handle)
        try:
            self._model = hub.load(self._model_handle)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YAMNet: {exc}"
            ) from exc

        # Retrieve the class-name map shipped inside the SavedModel
        class_map_path = self._model.class_map_path().numpy().decode("utf-8")
        with open(class_map_path, "r", encoding="utf-8") as fh:
            # CSV: index, mid, display_name
            lines = fh.read().strip().splitlines()
        self._class_names = []
        for line in lines[1:]:                       # skip header
            parts = line.split(",")
            if len(parts) >= 3:
                self._class_names.append(parts[2].strip('" '))
            else:
                self._class_names.append(parts[-1].strip('" '))
        logger.info(
            "YAMNet loaded – %d class labels available",
            len(self._class_names),
        )

    # ── inference ─────────────────────────────────────────────────────
    def classify(
        self,
        waveform: np.ndarray,
        timestamp: float = 0.0,
        onset_index: int = 0,
    ) -> ClassificationResult:
        """Run YAMNet inference on a single audio window.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 array at 16 kHz.  Ideally 0.975 s
            (15 600 samples) but YAMNet tolerates slightly different
            lengths.
        timestamp : float
            Playback timestamp (seconds) for logging / JSON output.
        onset_index : int
            Absolute sample index of the trigger onset.

        Returns
        -------
        ClassificationResult
        """
        self._ensure_model()

        # YAMNet expects a 1-D float32 tensor in [-1, 1]
        waveform = waveform.astype(np.float32)
        scores, embeddings, spectrogram = self._model(waveform)
        scores_np: np.ndarray = scores.numpy()

        # Average over time frames → single score vector
        mean_scores = scores_np.mean(axis=0)
        top_indices = mean_scores.argsort()[::-1][: self._top_k]

        top_k_list = []
        for idx in top_indices:
            name = (
                self._class_names[idx]
                if idx < len(self._class_names)
                else f"class_{idx}"
            )
            top_k_list.append(
                {"class": name, "score": float(round(mean_scores[idx], 4))}
            )

        best = top_k_list[0]
        is_suspicious = self._is_suspicious(top_k_list)

        if is_suspicious:
            matching = next(
                entry for entry in top_k_list
                if any(s.lower() in entry["class"].lower() for s in self._suspicious_labels)
            )
            label = matching["class"]
            confidence = matching["score"]
        else:
            label = best["class"]
            confidence = best["score"]

        result = ClassificationResult(
            timestamp=round(timestamp, 4),
            onset_index=onset_index,
            label=label,
            confidence=confidence,
            is_suspicious=is_suspicious,
            top_k=top_k_list,
            node_id=self._node_id,
            wall_clock_time=time.time(),
        )
        logger.info(
            "t=%.3f s  →  %s (%.2f)  suspicious=%s",
            timestamp,
            label,
            confidence,
            is_suspicious,
        )
        return result

    def _is_suspicious(self, top_k: List[dict]) -> bool:
        """Check whether any top-K label matches the suspicious set.

        Parameters
        ----------
        top_k : list[dict]
            List of ``{"class": str, "score": float}`` dicts.

        Returns
        -------
        bool
        """
        for entry in top_k:
            for susp in self._suspicious_labels:
                if susp.lower() in entry["class"].lower():
                    return True
        return False


# ──────────────────────────────────────────────────────────────────────
# CNN-based classifier
# ──────────────────────────────────────────────────────────────────────
class CNNClassifier:
    """Trained EfficientNetB0 CNN for binary gunshot classification.

    Takes a raw waveform, computes log-mel spectrogram, and runs inference
    on the trained model.

    Parameters
    ----------
    model_path : str
        Path to the .keras model file.
    decision_threshold : float
        Sigmoid threshold for binary decision (default 0.5).
    calibration_path : str | None
        Optional path to a JSON file with Platt scaling parameters
        ({"coef": float, "intercept": float}).  When provided, raw logits
        are mapped through the calibration transform before the threshold
        is applied, producing calibrated probabilities.
    """

    def __init__(
        self,
        model_path: str = None,
        decision_threshold: float = 0.5,
        feature_type: str = None,
        node_id: str = config.NODE_ID,
        calibration_path: str = None,
    ) -> None:
        self._model_path = model_path or str(config.CNN_MODEL_PATH)
        self._decision_threshold = decision_threshold
        self._feature_type = feature_type or config.CNN_FEATURE_TYPE
        self._node_id = node_id
        self._model = None

        # Load Platt scaling parameters if a calibration file is provided or configured
        cal_path = Path(calibration_path or str(config.CNN_CALIBRATION_PATH))
        if cal_path.exists():
            with open(cal_path, "r", encoding="utf-8") as f:
                self._calibration = json.load(f)
            logger.info(
                "CNN Classifier: loaded calibration params from %s (coef=%.4f, intercept=%.4f)",
                cal_path,
                self._calibration.get("coef", 1.0),
                self._calibration.get("intercept", 0.0),
            )
        else:
            self._calibration = None
            if calibration_path:
                logger.warning("Calibration file not found: %s — using raw sigmoid", cal_path)

        logger.info(
            "CNN Classifier initialized. Model path: %s, Feature: %s, Threshold: %.2f, Calibrated: %s",
            self._model_path,
            self._feature_type,
            decision_threshold,
            self._calibration is not None,
        )

    def _ensure_model(self) -> None:
        """Load model if not already cached."""
        if self._model is not None:
            return
        logger.info("Loading CNN model from %s", self._model_path)
        try:
            self._model = tf.keras.models.load_model(self._model_path)
            logger.info("CNN model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load CNN model: {exc}") from exc

    def classify(
        self,
        waveform: np.ndarray,
        timestamp: float = 0.0,
        onset_index: int = 0,
    ) -> ClassificationResult:
        """Run CNN inference on a single audio window.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 array at 16 kHz. Ideally 0.975 s (15,600 samples).
        timestamp : float
            Playback timestamp (seconds) for logging.
        onset_index : int
            Absolute sample index of the trigger onset.

        Returns
        -------
        ClassificationResult
        """
        self._ensure_model()

        # Compute log-mel spectrogram
        try:
            img_rgb = waveform_to_rgb_image(
                waveform,
                config.SAMPLE_RATE,
                feature_type=self._feature_type,
                image_size=(224, 224),
                target_duration_sec=5.0,
            )
            img_rgb = np.expand_dims(img_rgb, axis=0)

        except Exception as exc:
            logger.error("Failed to compute spectrogram: %s", exc)
            raise RuntimeError(f"Spectrogram computation failed: {exc}") from exc

        # Run inference
        try:
            logit = self._model.predict(img_rgb, verbose=0)[0, 0]
        except Exception as exc:
            logger.error("Model inference failed: %s", exc)
            raise RuntimeError(f"Model inference failed: {exc}") from exc

        # Apply Platt scaling calibration if available, otherwise use raw sigmoid
        if self._calibration is not None:
            coef = self._calibration["coef"]
            intercept = self._calibration["intercept"]
            sigmoid_output = 1.0 / (1.0 + np.exp(-(coef * logit + intercept)))
        else:
            sigmoid_output = 1.0 / (1.0 + np.exp(-logit))

        # Apply threshold
        is_suspicious = sigmoid_output >= self._decision_threshold
        label = "GUNSHOT" if is_suspicious else "NOGUN"

        result = ClassificationResult(
            timestamp=round(timestamp, 4),
            onset_index=onset_index,
            label=label,
            confidence=float(sigmoid_output),
            is_suspicious=bool(is_suspicious),
            top_k=[
                {"class": label, "score": float(sigmoid_output)},
            ],
            node_id=self._node_id,
            wall_clock_time=time.time(),
        )

        logger.info(
            "t=%.3f s  →  %s (%.4f)  suspicious=%s",
            timestamp,
            label,
            sigmoid_output,
            is_suspicious,
        )

        return result


# ──────────────────────────────────────────────────────────────────────
# YAMNet embedding head classifier
# ──────────────────────────────────────────────────────────────────────
class YAMNetHeadClassifier:
    """Classifier that runs YAMNet embeddings through a trained binary head.

    Unlike YAMNetClassifier (which uses a hand-curated suspicious-label list)
    this class uses a custom head trained on real gunshot audio, giving a
    learned decision boundary on YAMNet's 1024-d embedding space.

    Parameters
    ----------
    head_model_path : str
        Path to the trained head .keras file
        (output of train/train_yamnet_head.py).
    yamnet_handle : str
        TF-Hub URL for the YAMNet model.
    decision_threshold : float
        Sigmoid threshold for the binary decision.
    """

    def __init__(
        self,
        head_model_path: str = None,
        yamnet_handle: str = config.YAMNET_MODEL_HANDLE,
        decision_threshold: float = 0.50,
    ) -> None:
        self._head_path = head_model_path or str(config.YAMNET_HEAD_MODEL_PATH)
        self._yamnet_handle = yamnet_handle
        self._decision_threshold = decision_threshold
        self._yamnet = None
        self._head = None

    def _ensure_models(self) -> None:
        if self._yamnet is not None:
            return
        logger.info("Loading YAMNet from %s …", self._yamnet_handle)
        self._yamnet = hub.load(self._yamnet_handle)
        logger.info("Loading YAMNet head from %s …", self._head_path)
        self._head = tf.keras.models.load_model(self._head_path)
        logger.info("YAMNetHeadClassifier ready.")

    def classify(
        self,
        waveform: np.ndarray,
        timestamp: float = 0.0,
        onset_index: int = 0,
    ) -> ClassificationResult:
        """Run inference using YAMNet embeddings + trained head.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 array at 16 kHz.
        timestamp : float
            Playback timestamp (seconds).
        onset_index : int
            Absolute sample index of the trigger.

        Returns
        -------
        ClassificationResult
        """
        self._ensure_models()

        waveform = waveform.astype(np.float32)
        _, embeddings, _ = self._yamnet(waveform)
        # Average over time frames → fixed 1024-d vector
        pooled = np.mean(embeddings.numpy(), axis=0, keepdims=True).astype(np.float32)

        logit = float(np.asarray(self._head.predict_on_batch(pooled)).reshape(-1)[0])
        sigmoid_output = float(1.0 / (1.0 + np.exp(-logit)))

        is_suspicious = sigmoid_output >= self._decision_threshold
        label = "GUNSHOT" if is_suspicious else "NOGUN"

        result = ClassificationResult(
            timestamp=round(timestamp, 4),
            onset_index=onset_index,
            label=label,
            confidence=sigmoid_output,
            is_suspicious=bool(is_suspicious),
            top_k=[{"class": label, "score": sigmoid_output}],
        )
        logger.info(
            "t=%.3f s  →  %s (%.4f)  suspicious=%s  [YAMNetHead]",
            timestamp, label, sigmoid_output, is_suspicious,
        )
        return result
