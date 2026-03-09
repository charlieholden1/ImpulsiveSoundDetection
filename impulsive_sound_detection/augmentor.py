"""
augmentor.py – Audio augmentation pipeline using *audiomentations*.

Provides the ``RobustAugmentor`` class which chains together transforms
that simulate realistic degradation found in a school environment:

* Additive Gaussian noise  (HVAC hum, electrical noise)
* Random time shifts        (microphone placement variation)
* Background-noise mixing   (hallway chatter, bells, etc.)

Usage
-----
>>> from impulsive_sound_detection.augmentor import RobustAugmentor
>>> aug = RobustAugmentor(sample_rate=16000)
>>> augmented = aug(waveform)           # numpy float32 array
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    Compose,
    Shift,
)

from . import config

logger = logging.getLogger(__name__)


class RobustAugmentor:
    """Configurable audio augmentation pipeline for school-environment
    robustness training.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of incoming waveforms.
    background_noise_dir : Path | None
        Directory of ``.wav`` files used as background noise.  If
        ``None`` the ``AddBackgroundNoise`` transform is skipped.
    gaussian_min_amplitude : float
        Lower bound of Gaussian noise amplitude.
    gaussian_max_amplitude : float
        Upper bound of Gaussian noise amplitude.
    time_shift_min_ms : int
        Minimum shift in milliseconds (negative = shift left).
    time_shift_max_ms : int
        Maximum shift in milliseconds.
    bg_snr_min_db : float
        Minimum signal-to-noise ratio (dB) for background mixing.
    bg_snr_max_db : float
        Maximum signal-to-noise ratio (dB) for background mixing.
    p : float
        Per-transform probability of application.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        background_noise_dir: Optional[Path] = None,
        gaussian_min_amplitude: float = config.GAUSSIAN_NOISE_MIN_AMP,
        gaussian_max_amplitude: float = config.GAUSSIAN_NOISE_MAX_AMP,
        time_shift_min_ms: int = config.TIME_SHIFT_MIN_MS,
        time_shift_max_ms: int = config.TIME_SHIFT_MAX_MS,
        bg_snr_min_db: float = config.BACKGROUND_NOISE_SNR_DB_MIN,
        bg_snr_max_db: float = config.BACKGROUND_NOISE_SNR_DB_MAX,
        p: float = 0.5,
    ) -> None:
        self.sample_rate = sample_rate

        transforms = [
            AddGaussianNoise(
                min_amplitude=gaussian_min_amplitude,
                max_amplitude=gaussian_max_amplitude,
                p=p,
            ),
            Shift(
                min_shift=time_shift_min_ms / 1000.0,
                max_shift=time_shift_max_ms / 1000.0,
                p=p,
            ),
        ]

        if background_noise_dir is not None:
            bg_dir = Path(background_noise_dir)
            if bg_dir.exists() and any(bg_dir.glob("*.wav")):
                transforms.append(
                    AddBackgroundNoise(
                        sounds_path=str(bg_dir),
                        min_snr_in_db=bg_snr_min_db,
                        max_snr_in_db=bg_snr_max_db,
                        p=p,
                    )
                )
                logger.info(
                    "Background noise enabled from %s", bg_dir
                )
            else:
                logger.warning(
                    "Background noise dir missing / empty: %s – "
                    "skipping AddBackgroundNoise",
                    bg_dir,
                )

        self._pipeline = Compose(transforms)
        logger.info(
            "RobustAugmentor initialised with %d transforms",
            len(transforms),
        )

    # ── public interface ──────────────────────────────────────────────
    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """Apply the full augmentation pipeline to *waveform*.

        Parameters
        ----------
        waveform : np.ndarray
            1-D float32 array in [-1, 1].

        Returns
        -------
        np.ndarray
            Augmented waveform (same shape, dtype, and range).
        """
        return self._pipeline(
            samples=waveform, sample_rate=self.sample_rate
        )

    def augment_batch(
        self,
        waveforms: list[np.ndarray],
        n_augmentations: int = 1,
    ) -> list[np.ndarray]:
        """Generate *n_augmentations* variants per input waveform.

        Parameters
        ----------
        waveforms : list[np.ndarray]
            List of 1-D float32 waveforms.
        n_augmentations : int
            How many augmented copies to produce for each original.

        Returns
        -------
        list[np.ndarray]
            Flat list of augmented waveforms (length =
            ``len(waveforms) * n_augmentations``).
        """
        augmented: list[np.ndarray] = []
        for wav in waveforms:
            for _ in range(n_augmentations):
                augmented.append(self(wav))
        logger.info(
            "Generated %d augmented waveforms from %d originals",
            len(augmented),
            len(waveforms),
        )
        return augmented
