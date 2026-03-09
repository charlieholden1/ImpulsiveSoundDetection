# Robust Impulsive Sound Detection System

A two-stage acoustic event detection pipeline that identifies impulsive sounds (gunshots, glass breaks) in real time and from pre-recorded audio, using energy-based triggering (Stage 1) and YAMNet-based classification (Stage 2).

---

## Requirements

- Python 3.10 or 3.11
- Windows, macOS, or Linux
- A working microphone (for `live` and `gui` modes)

---

## Setup

### 1. Clone the repository

```bash
git clone <repo-url>
cd MLProject
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install tensorflow tensorflow-hub librosa audiomentations \
            sounddevice soundfile customtkinter matplotlib \
            numpy scipy scikit-learn rich
```

> **Note:** TensorFlow on Windows requires a 64-bit Python build. TensorFlow Hub will download the YAMNet model (~25 MB) on first run and cache it locally.

### 4. Obtain the datasets

The following data directories are excluded from version control due to their size. Place them in the project root before using `prepare` or `demo` mode:

| Directory | Contents |
|-----------|----------|
| `clean/audio/` | VOICe dataset WAV files |
| `clean/source/` | Source file lists (`synthetic_source_training.txt`, etc.) |
| `clean/target/` | Target/reference files |
| `Gunshot Audio Spectrogram Dataset for Binary Class/FFT/` | FFT spectrogram PNGs |
| `Gunshot Audio Spectrogram Dataset for Binary Class/LogMel/` | Log-Mel spectrogram PNGs |
| `Gunshot Audio Spectrogram Dataset for Binary Class/MFCC/` | MFCC spectrogram PNGs |

### 5. (Optional) Configure paths and parameters

All tunable constants live in [`impulsive_sound_detection/config.py`](impulsive_sound_detection/config.py). Update `GUNSHOT_SPECTROGRAM_DIR` and `VOICE_DATASET_DIR` if your data lives outside the default project root.

---

## Usage

All modes are accessed through the package entry point:

```bash
python -m impulsive_sound_detection.main <mode> [options]
```

---

### `detect` — Run the pipeline on WAV file(s)

```bash
python -m impulsive_sound_detection.main detect --wav path/to/file.wav
```

| Flag | Description |
|------|-------------|
| `--wav` | One or more `.wav` file paths (required) |
| `--log <path>` | Write JSON detections to a file |
| `--no-viz` | Skip the spectrogram visualisation plot |

**Example — multiple files, save detections:**
```bash
python -m impulsive_sound_detection.main detect \
    --wav clip1.wav clip2.wav \
    --log detections.json
```

---

### `prepare` — Prepare the VOICe dataset

Discovers and segments the VOICe dataset, with optional on-the-fly augmentation of the positive (gunshot / glass-break) class.

```bash
python -m impulsive_sound_detection.main prepare
python -m impulsive_sound_detection.main prepare --augment --n-aug 5
```

| Flag | Description |
|------|-------------|
| `--augment` | Augment positive-class waveforms |
| `--n-aug <int>` | Number of augmented copies per sample (default: 3) |

---

### `demo` — Quick visualisation demo

Runs detection on a single VOICe training file and displays the result plot. Useful for a quick sanity-check after setup.

```bash
python -m impulsive_sound_detection.main demo
python -m impulsive_sound_detection.main demo --file-index 5
```

| Flag | Description |
|------|-------------|
| `--file-index <int>` | Index of the file from the training list (default: 0) |

---

### `live` — Real-time microphone detection

Opens the default system microphone and runs the two-stage pipeline continuously, printing a colour-coded terminal dashboard.

```bash
python -m impulsive_sound_detection.main live
python -m impulsive_sound_detection.main live --threshold-multiplier 4.0
```

| Flag | Description |
|------|-------------|
| `--threshold-multiplier <float>` | Energy trigger sensitivity (default: 3.0 — lower = more sensitive) |

Press **Ctrl+C** to stop.

---

### `gui` — Graphical dashboard

Launches a dark-mode GUI (customtkinter) with a live scrolling RMS energy plot, colour-coded event log, and real-time parameter sliders.

```bash
python -m impulsive_sound_detection.main gui
```

No additional flags. Microphone device and threshold can be adjusted from within the GUI.

---

## Project Structure

```
MLProject/
├── impulsive_sound_detection/   # Main package
│   ├── main.py                  # CLI entry point
│   ├── config.py                # All constants and paths
│   ├── pipeline.py              # Stage 1 + 2 orchestration
│   ├── stream_monitor.py        # Stage 1: energy-based trigger
│   ├── classifier.py            # Stage 2: YAMNet classification
│   ├── data_loader.py           # VOICe dataset loader
│   ├── augmentor.py             # Audio augmentation (audiomentations)
│   ├── visualizer.py            # Detection plots (matplotlib)
│   ├── live_stream.py           # Real-time mic stream
│   ├── dashboard.py             # Terminal dashboard (rich)
│   └── gui.py                   # GUI dashboard (customtkinter)
├── clean/                       # VOICe dataset (not tracked by git)
│   ├── annotation/              # Label text files
│   ├── audio/                   # WAV files
│   ├── source/                  # File lists
│   └── target/
└── Gunshot Audio Spectrogram Dataset for Binary Class/   # (not tracked)
    ├── FFT/
    ├── LogMel/
    └── MFCC/
```

---

## How It Works

1. **Stage 1 — Stream Monitor:** A sliding-window RMS energy tracker maintains a rolling baseline. When the instantaneous energy exceeds `ENERGY_MULTIPLIER × baseline`, a 0.975 s audio window is queued for classification.

2. **Stage 2 — YAMNet Classifier:** The queued window is fed to Google's YAMNet model (loaded from TensorFlow Hub). The top-K predicted labels are checked against a configurable set of `SUSPICIOUS_LABELS`; a match produces a `ClassificationResult` with `is_suspicious = True`.

---

## License

Code in `impulsive_sound_detection/` is released under MIT. The bundled dataset folders carry their own licences — see [`Gunshot Audio Spectrogram Dataset for Binary Class/README.md`](Gunshot%20Audio%20Spectrogram%20Dataset%20for%20Binary%20Class/README.md) for details.
