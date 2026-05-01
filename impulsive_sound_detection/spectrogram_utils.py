"""
Utilities for rendering audio waveforms into CNN-ready spectrogram images.
"""

from __future__ import annotations

from typing import Tuple

import librosa
import numpy as np
from matplotlib import colormaps
from PIL import Image


def standardize_waveform_duration(
    waveform: np.ndarray,
    sample_rate: int,
    target_duration_sec: float = 5.0,
) -> np.ndarray:
    """
    Trim or zero-pad a waveform to a fixed target duration.
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target_samples = int(round(sample_rate * target_duration_sec))
    if len(waveform) == target_samples:
        return waveform
    if len(waveform) > target_samples:
        return waveform[:target_samples]

    padded = np.zeros(target_samples, dtype=np.float32)
    padded[: len(waveform)] = waveform
    return padded


def compute_feature_image(
    waveform: np.ndarray,
    sample_rate: int,
    feature_type: str = "LogMel",
) -> np.ndarray:
    """
    Compute a 2-D audio feature image.
    """
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    feature_key = feature_type.lower()

    if feature_key == "logmel":
        feature = librosa.feature.melspectrogram(
            y=waveform,
            sr=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
        )
        feature = librosa.power_to_db(feature, ref=np.max)
    elif feature_key == "mfcc":
        feature = librosa.feature.mfcc(
            y=waveform,
            sr=sample_rate,
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
        )
    elif feature_key == "fft":
        stft = np.abs(librosa.stft(waveform, n_fft=2048, hop_length=512))
        feature = librosa.amplitude_to_db(stft, ref=np.max)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return np.asarray(feature, dtype=np.float32)


def feature_to_rgb_image(
    feature: np.ndarray,
    image_size: Tuple[int, int] = (224, 224),
    cmap_name: str = "viridis",
) -> np.ndarray:
    """
    Render a feature matrix into an RGB image matching CNN training inputs.
    """
    feature = np.asarray(feature, dtype=np.float32)
    feature = np.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

    min_val = float(feature.min())
    max_val = float(feature.max())
    if max_val > min_val:
        normalized = (feature - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(feature, dtype=np.float32)

    colored = colormaps[cmap_name](normalized)[..., :3]
    rgb = (colored * 255.0).clip(0, 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize(image_size, Image.LANCZOS)
    return np.asarray(image, dtype=np.uint8)


def waveform_to_rgb_image(
    waveform: np.ndarray,
    sample_rate: int,
    feature_type: str = "LogMel",
    image_size: Tuple[int, int] = (224, 224),
    target_duration_sec: float = 5.0,
    cmap_name: str = "viridis",
) -> np.ndarray:
    """
    Convert a waveform directly into an RGB spectrogram image.
    """
    standardized = standardize_waveform_duration(
        waveform,
        sample_rate,
        target_duration_sec=target_duration_sec,
    )
    feature = compute_feature_image(
        standardized,
        sample_rate,
        feature_type=feature_type,
    )
    return feature_to_rgb_image(feature, image_size=image_size, cmap_name=cmap_name)
