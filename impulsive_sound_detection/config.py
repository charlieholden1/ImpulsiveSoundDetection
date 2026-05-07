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
#    Training / dataset paths (development machine).
#    On deployed RPi nodes these are not used at runtime.
# ──────────────────────────────────────────────────────────────────────
ISD_ROOT: Path = Path(r"C:\ImpulsiveSoundDetection")

GUNSHOT_SPECTROGRAM_DIR: Path = (
    ISD_ROOT / "Gunshot Audio Spectrogram Dataset for Binary Class"
)

VOICE_DATASET_DIR: Path = ISD_ROOT / "clean"

VOICE_AUDIO_DIR: Path      = VOICE_DATASET_DIR / "audio"
VOICE_ANNOTATION_DIR: Path = VOICE_DATASET_DIR / "annotation"
VOICE_SOURCE_DIR: Path     = VOICE_DATASET_DIR / "source"
VOICE_TARGET_DIR: Path     = VOICE_DATASET_DIR / "target"

# ──────────────────────────────────────────────────────────────────────
# 2. AUDIO PARAMETERS
# ──────────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 16_000           # Hz – YAMNet requirement
MONO: bool = True
YAMNET_WINDOW_SEC: float = 0.975    # seconds per YAMNet frame
YAMNET_WINDOW_SAMPLES: int = int(SAMPLE_RATE * YAMNET_WINDOW_SEC)  # 15600

# ──────────────────────────────────────────────────────────────────────
# 3. STREAM MONITOR (Stage 1) PARAMETERS
# ──────────────────────────────────────────────────────────────────────
RMS_FRAME_SIZE: int = 512           # sliding-window hop for RMS
ROLLING_WINDOW_SEC: float = 10.0    # seconds of history for baseline
ENERGY_MULTIPLIER: float = 2.0      # trigger when energy > N × baseline
MIN_RETRIGGER_SEC: float = 0.5      # dead-time after a trigger

# ──────────────────────────────────────────────────────────────────────
# 4. CLASSIFICATION (Stage 2) PARAMETERS
# ──────────────────────────────────────────────────────────────────────
YAMNET_MODEL_HANDLE: str = "https://tfhub.dev/google/yamnet/1"
TOP_K: int = 5                      # number of top predictions to keep

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
MAX_QUEUE_SIZE: int = 64            # max pending trigger windows
INFERENCE_TIMEOUT_SEC: float = 5.0  # per-window inference budget

# ──────────────────────────────────────────────────────────────────────
# 8. CNN CLASSIFIER (STAGE 2 ALTERNATIVE)
#    Model path is relative to ISD_ROOT so it works on any machine.
# ──────────────────────────────────────────────────────────────────────
CNN_MODEL_PATH: Path = (
    ISD_ROOT / "models" / "feature_sweep_stagea6" / "logmel"
    / "cnn_gunshot_classifier.keras"
)
CNN_FEATURE_TYPE: str = "LogMel"
CNN_DECISION_THRESHOLD: float = 0.15
CLASSIFIER_MODE: str = "yamnet"     # Options: "cnn", "yamnet", "ensemble"

# ──────────────────────────────────────────────────────────────────────
# 9. DATABASE / LOG PATHS
# ──────────────────────────────────────────────────────────────────────
# SQLite detection log (used by EventLogger)
SQLITE_LOG_PATH: Path = ISD_ROOT / "logs" / "detections.db"

# Per-node JSONL log directory (one file per node_id on RPi)
LOG_DIR: Path = ISD_ROOT / "logs"

# ──────────────────────────────────────────────────────────────────────
# 10. NODE IDENTITY  ← edit per-device on each RPi before deployment
# ──────────────────────────────────────────────────────────────────────
NODE_ID: str = "node_test"
NODE_LOCATION: str = "Local Dev Machine"

# ──────────────────────────────────────────────────────────────────────
# 11. MQTT  ← shared broker settings
# ──────────────────────────────────────────────────────────────────────
# Set MQTT_BROKER_HOST to the LAN IP of the host machine running
# the dashboard.  All RPi nodes publish to this address.
MQTT_BROKER_HOST: str = "127.0.0.1"   # localhost instead of the real host IP
MQTT_BROKER_PORT: int = 1883
MQTT_KEEPALIVE_SEC: int = 60

# Topic templates (formatted with NODE_ID at runtime)
MQTT_TOPIC_DETECTION:    str = f"isd/node/{NODE_ID}/detection"
MQTT_TOPIC_RMS:          str = f"isd/node/{NODE_ID}/rms"
MQTT_TOPIC_HEARTBEAT:    str = f"isd/node/{NODE_ID}/heartbeat"
MQTT_TOPIC_LOCALIZATION: str = "isd/localization/result"  # host subscribes

# Publish one RMS frame to MQTT every N frames (~5 publishes/sec at 512 frame)
MQTT_RMS_PUBLISH_EVERY_N_FRAMES: int = 6

# ──────────────────────────────────────────────────────────────────────
# 12. LOGGING / OUTPUT
# ──────────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"
