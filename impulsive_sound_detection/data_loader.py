"""
data_loader.py – Data discovery, loading, and segment extraction.

Uses *pathlib* exclusively for filesystem traversal so that every path
operation is OS-agnostic (and the raw-string Windows literals in
``config.py`` work seamlessly).

Public API
----------
load_wav(path, sr, mono)
    Load a single WAV file via librosa.

parse_annotation(annotation_path)
    Parse a VOICe-style tab-separated annotation file.

extract_segments(wav, sr, annotations, labels)
    Slice labelled segments out of a full-length waveform.

discover_voice_dataset(audio_dir, annotation_dir, positive, negative)
    Walk the VOICe dataset and return (positive_segments, negative_segments).

discover_gunshot_spectrograms(root_dir)
    Walk the Gunshot Spectrogram Dataset and return paths grouped by class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────
@dataclass
class AnnotationEntry:
    """One labelled time-span inside an audio file."""

    start_sec: float
    end_sec: float
    label: str


@dataclass
class AudioSegment:
    """A labelled chunk of audio data ready for training / inference."""

    source_file: Path
    label: str
    waveform: np.ndarray
    sample_rate: int
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class DatasetBundle:
    """Aggregated positive and negative audio segments."""

    positive: List[AudioSegment] = field(default_factory=list)
    negative: List[AudioSegment] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────
# Core I/O helpers
# ──────────────────────────────────────────────────────────────────────
def load_wav(
    path: Path,
    sr: int = config.SAMPLE_RATE,
    mono: bool = config.MONO,
) -> Tuple[np.ndarray, int]:
    """Load an audio file and resample to *sr* Hz mono.

    Parameters
    ----------
    path : Path
        Absolute or relative path to a ``.wav`` file.
    sr : int
        Target sample rate (default ``16 000``).
    mono : bool
        If ``True`` (default), mix to single channel.

    Returns
    -------
    tuple[np.ndarray, int]
        ``(waveform, sample_rate)`` where *waveform* is a 1-D float32
        array normalised to [-1, 1].

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    waveform, out_sr = librosa.load(str(path), sr=sr, mono=mono)
    logger.debug("Loaded %s  (%d samples @ %d Hz)", path.name, len(waveform), out_sr)
    return waveform.astype(np.float32), out_sr


def parse_annotation(annotation_path: Path) -> List[AnnotationEntry]:
    """Parse a VOICe annotation file (TSV: start  end  label).

    Parameters
    ----------
    annotation_path : Path
        Path to a ``.txt`` annotation file.

    Returns
    -------
    list[AnnotationEntry]
        Chronologically ordered entries.
    """
    annotation_path = Path(annotation_path)
    entries: List[AnnotationEntry] = []
    with annotation_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                logger.warning(
                    "%s:%d – skipping malformed line: %r",
                    annotation_path.name,
                    lineno,
                    line,
                )
                continue
            try:
                start = float(parts[0])
                end = float(parts[1])
                label = parts[2].strip().lower()
                entries.append(AnnotationEntry(start, end, label))
            except ValueError:
                logger.warning(
                    "%s:%d – cannot parse floats: %r",
                    annotation_path.name,
                    lineno,
                    line,
                )
    return entries


# ──────────────────────────────────────────────────────────────────────
# Segment extraction
# ──────────────────────────────────────────────────────────────────────
def extract_segments(
    waveform: np.ndarray,
    sr: int,
    annotations: Sequence[AnnotationEntry],
    target_labels: Optional[frozenset] = None,
    source_file: Optional[Path] = None,
) -> List[AudioSegment]:
    """Slice labelled segments out of a waveform.

    Parameters
    ----------
    waveform : np.ndarray
        Full waveform (1-D float32).
    sr : int
        Sample rate of *waveform*.
    annotations : Sequence[AnnotationEntry]
        Parsed annotation entries.
    target_labels : frozenset | None
        If given, only keep segments whose label is in this set.
    source_file : Path | None
        Original file path (stored for provenance).

    Returns
    -------
    list[AudioSegment]
    """
    segments: List[AudioSegment] = []
    total_samples = len(waveform)
    for ann in annotations:
        if target_labels and ann.label not in target_labels:
            continue
        start_idx = int(ann.start_sec * sr)
        end_idx = int(ann.end_sec * sr)
        # Clamp to waveform bounds
        start_idx = max(0, start_idx)
        end_idx = min(total_samples, end_idx)
        if end_idx <= start_idx:
            continue
        chunk = waveform[start_idx:end_idx].copy()
        segments.append(
            AudioSegment(
                source_file=source_file or Path("unknown"),
                label=ann.label,
                waveform=chunk,
                sample_rate=sr,
                start_sec=ann.start_sec,
                end_sec=ann.end_sec,
            )
        )
    return segments


# ──────────────────────────────────────────────────────────────────────
# Dataset-level discovery
# ──────────────────────────────────────────────────────────────────────
def discover_voice_dataset(
    audio_dir: Path = config.VOICE_AUDIO_DIR,
    annotation_dir: Path = config.VOICE_ANNOTATION_DIR,
    positive_labels: frozenset = config.POSITIVE_LABELS,
    negative_labels: frozenset = config.NEGATIVE_LABELS,
    file_list: Optional[Path] = None,
) -> DatasetBundle:
    """Walk the VOICe dataset and return positive / negative segments.

    Parameters
    ----------
    audio_dir : Path
        Directory containing ``.wav`` files.
    annotation_dir : Path
        Directory containing matching ``.txt`` annotation files.
    positive_labels : frozenset
        Labels that count as *suspicious* (e.g. ``gunshot``, ``glassbreak``).
    negative_labels : frozenset
        Labels that count as *non-suspicious* (e.g. ``babycry``).
    file_list : Path | None
        If provided, only process filenames listed in this text file.

    Returns
    -------
    DatasetBundle
        Aggregated positive and negative ``AudioSegment`` lists.
    """
    audio_dir = Path(audio_dir)
    annotation_dir = Path(annotation_dir)

    # Determine which files to process
    if file_list and file_list.exists():
        wav_names = [
            line.strip()
            for line in file_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        wav_names = sorted(p.name for p in audio_dir.glob("*.wav"))

    bundle = DatasetBundle()
    for wav_name in wav_names:
        wav_path = audio_dir / wav_name
        ann_path = annotation_dir / wav_name.replace(".wav", ".txt")

        if not wav_path.exists():
            logger.warning("Missing audio file: %s", wav_path)
            continue
        if not ann_path.exists():
            logger.warning("Missing annotation file: %s", ann_path)
            continue

        waveform, sr = load_wav(wav_path)
        annotations = parse_annotation(ann_path)

        pos_segments = extract_segments(
            waveform, sr, annotations, positive_labels, wav_path
        )
        neg_segments = extract_segments(
            waveform, sr, annotations, negative_labels, wav_path
        )
        bundle.positive.extend(pos_segments)
        bundle.negative.extend(neg_segments)
        logger.info(
            "%s → %d positive, %d negative segments",
            wav_name,
            len(pos_segments),
            len(neg_segments),
        )

    logger.info(
        "VOICe dataset: %d positive, %d negative segments total",
        len(bundle.positive),
        len(bundle.negative),
    )
    return bundle


def discover_gunshot_spectrograms(
    root_dir: Path = config.GUNSHOT_SPECTROGRAM_DIR,
) -> Dict[str, List[Path]]:
    """Walk the Gunshot Spectrogram Dataset and return paths by class.

    Parameters
    ----------
    root_dir : Path
        Root directory of the spectrogram dataset.

    Returns
    -------
    dict[str, list[Path]]
        Keys are ``"GUN"`` and ``"NOGUN"``; values are lists of
        ``.png`` file paths.
    """
    root_dir = Path(root_dir)
    result: Dict[str, List[Path]] = {"GUN": [], "NOGUN": []}
    for feature_dir in ("FFT", "LogMel", "MFCC"):
        for class_label in ("GUN", "NOGUN"):
            class_dir = root_dir / feature_dir / class_label
            if not class_dir.exists():
                logger.warning("Directory not found: %s", class_dir)
                continue
            pngs = sorted(class_dir.glob("*.png"))
            result[class_label].extend(pngs)
            logger.info(
                "%s/%s → %d images", feature_dir, class_label, len(pngs)
            )
    return result
