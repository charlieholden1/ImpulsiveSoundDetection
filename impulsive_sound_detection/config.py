"""
config.py – Global constants, paths, and tunable parameters.

All Windows file-system paths use raw strings (r"") to avoid
back-slash escape issues.  Every numeric knob used by the detection
pipeline is collected here so experiments can be re-run by editing a
single file.
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# 1. DATA PATHS
# ──────────────────────────────────────────────────────────────────────
GUNSHOT_SPECTROGRAM_DIR: Path = Path(
    r"C:\Users\holde\Documents\MLProject"
    r"\Gunshot Audio Spectrogram Dataset for Binary Class"
)

VOICE_DATASET_DIR: Path = Path(
    r"C:\Users\holde\Documents\MLProject\clean"
)

VOICE_AUDIO_DIR: Path = VOICE_DATASET_DIR / "audio"
VOICE_ANNOTATION_DIR: Path = VOICE_DATASET_DIR / "annotation"
VOICE_SOURCE_DIR: Path = VOICE_DATASET_DIR / "source"
VOICE_TARGET_DIR: Path = VOICE_DATASET_DIR / "target"

# ──────────────────────────────────────────────────────────────────────
# 2. AUDIO PARAMETERS
# ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000          # Hz – YAMNet requirement
MONO: bool = True
YAMNET_WINDOW_SEC: float = 0.975   # seconds per YAMNet frame
YAMNET_WINDOW_SAMPLES: int = int(SAMPLE_RATE * YAMNET_WINDOW_SEC)  # 15600

# ──────────────────────────────────────────────────────────────────────
# 3. STREAM MONITOR (Stage 1) PARAMETERS
# ──────────────────────────────────────────────────────────────────────
RMS_FRAME_SIZE: int = 512          # sliding-window hop for RMS
ROLLING_WINDOW_SEC: float = 10.0   # seconds of history for baseline
ENERGY_MULTIPLIER: float = 2.0     # trigger when energy > N × baseline
MIN_RETRIGGER_SEC: float = 0.5     # dead-time after a trigger

# ──────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION (Stage 2) PARAMETERS
# ──────────────────────────────────────────────────────────────────────
YAMNET_MODEL_HANDLE: str = (
    "https://tfhub.dev/google/yamnet/1"
)
TOP_K: int = 5                     # number of top predictions to keep

# Labels that YAMNet may output which we consider "suspicious"
SUSPICIOUS_LABELS: frozenset = frozenset({
    "Explosion",
    "Gunshot, gunfire",
    "Machine gun",
    "Fusillade",
    "Cap gun",
    "Gunshot",
    "Gunfire",
    "Boom",
    "Glass",
    "Shatter",
    "Breaking",
    "Bang",
    "Burst, pop",
    "Slam",
})

# ──────────────────────────────────────────────────────────────────────
# 5. VOICE DATASET LABEL MAPPING
# ──────────────────────────────────────────────────────────────────────
# Labels present in the VOICe annotation files
POSITIVE_LABELS: frozenset = frozenset({"gunshot", "glassbreak"})
NEGATIVE_LABELS: frozenset = frozenset({"babycry"})

# ──────────────────────────────────────────────────────────────────────
# 6. AUGMENTATION DEFAULTS
# ──────────────────────────────────────────────────────────────────────
GAUSSIAN_NOISE_MIN_AMP: float = 0.001
GAUSSIAN_NOISE_MAX_AMP: float = 0.015
TIME_SHIFT_MIN_MS: int = -200
TIME_SHIFT_MAX_MS: int = 200
BACKGROUND_NOISE_SNR_DB_MIN: float = 3.0
BACKGROUND_NOISE_SNR_DB_MAX: float = 12.0

# ──────────────────────────────────────────────────────────────────────
# 7. BUFFER-OVERFLOW GUARD
# ──────────────────────────────────────────────────────────────────────
MAX_QUEUE_SIZE: int = 64           # max pending trigger windows
INFERENCE_TIMEOUT_SEC: float = 5.0 # per-window inference budget

# ──────────────────────────────────────────────────────────────────────
# 8. CNN CLASSIFIER (STAGE 2 ALTERNATIVE)
# ──────────────────────────────────────────────────────────────────────
CNN_MODEL_PATH: Path = Path(
    r"C:\Users\holde\Documents\MLProject\models\feature_sweep_stagea6\logmel\cnn_gunshot_classifier.keras"
)
CNN_FEATURE_TYPE: str = "LogMel"
CNN_DECISION_THRESHOLD: float = 0.15
CLASSIFIER_MODE: str = "yamnet"  # Options: "cnn", "yamnet", "ensemble"

# ──────────────────────────────────────────────────────────────────────
# 9. DATABASE LOGGING
# ──────────────────────────────────────────────────────────────────────
SQLITE_LOG_PATH: Path = Path(r"C:\Users\holde\Documents\MLProject\logs\detections.db")

# ──────────────────────────────────────────────────────────────────────
# 10. LOGGING / OUTPUT
# ──────────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
