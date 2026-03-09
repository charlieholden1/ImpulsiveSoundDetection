"""
visualizer.py – Matplotlib waveform + onset visualisation.

Provides a single entry-point function ``plot_detections`` that draws
the full waveform, overlays vertical markers at each detected onset,
and colour-codes them by suspicious / non-suspicious.

Usage
-----
>>> from impulsive_sound_detection.visualizer import plot_detections
>>> plot_detections(waveform, sr=16000, results=results)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .classifier import ClassificationResult

logger = logging.getLogger(__name__)


def plot_detections(
    waveform: np.ndarray,
    sr: int,
    results: List[ClassificationResult],
    title: str = "Impulsive Sound Detections",
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 5),
    show: bool = True,
) -> plt.Figure:
    """Plot the waveform and highlight detected onset points.

    Parameters
    ----------
    waveform : np.ndarray
        Full audio waveform (1-D float32).
    sr : int
        Sample rate of *waveform*.
    results : list[ClassificationResult]
        Classification outputs from the pipeline.
    title : str
        Plot title.
    save_path : Path | None
        If given, save the figure to this path (PNG / PDF / SVG).
    figsize : tuple
        Matplotlib figure size ``(width, height)`` in inches.
    show : bool
        Whether to call ``plt.show()`` (set ``False`` in scripts).

    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    """
    duration_sec = len(waveform) / sr
    time_axis = np.linspace(0.0, duration_sec, num=len(waveform))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, waveform, color="#4a90d9", linewidth=0.35, alpha=0.8)
    ax.set_xlim(0, duration_sec)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # Overlay onset markers
    suspicious_plotted = False
    safe_plotted = False
    for res in results:
        if res.is_suspicious:
            colour = "#e74c3c"  # red
            label = "Suspicious" if not suspicious_plotted else None
            suspicious_plotted = True
        else:
            colour = "#27ae60"  # green
            label = "Non-suspicious" if not safe_plotted else None
            safe_plotted = True

        ax.axvline(
            x=res.timestamp,
            color=colour,
            linestyle="--",
            linewidth=1.2,
            alpha=0.85,
            label=label,
        )
        # Annotate with label text
        ax.annotate(
            f"{res.label}\n{res.confidence:.2f}",
            xy=(res.timestamp, 0),
            xytext=(res.timestamp + duration_sec * 0.005, 0.6),
            fontsize=6,
            color=colour,
            rotation=45,
            ha="left",
            va="bottom",
        )

    if results:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.8)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        logger.info("Figure saved to %s", save_path)

    if show:
        plt.show()

    return fig


def plot_rms_energy(
    rms_values: List[float],
    frame_size: int,
    sr: int,
    threshold_values: Optional[List[float]] = None,
    title: str = "RMS Energy over Time",
    save_path: Optional[Path] = None,
    figsize: tuple = (16, 4),
    show: bool = True,
) -> plt.Figure:
    """Plot the RMS energy curve alongside the dynamic threshold.

    Parameters
    ----------
    rms_values : list[float]
        Per-frame RMS values from ``StreamMonitor``.
    frame_size : int
        Samples per frame (to derive the time axis).
    sr : int
        Sample rate.
    threshold_values : list[float] | None
        Per-frame dynamic threshold values.
    title : str
        Plot title.
    save_path : Path | None
        Optional save path.
    figsize : tuple
        Figure size.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    times = np.arange(len(rms_values)) * frame_size / sr

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(times, rms_values, color="#2980b9", linewidth=0.6, label="RMS Energy")

    if threshold_values is not None and len(threshold_values) == len(rms_values):
        ax.plot(
            times,
            threshold_values,
            color="#e74c3c",
            linewidth=0.8,
            linestyle="--",
            label="Dynamic Threshold",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS Energy")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150)
        logger.info("Figure saved to %s", save_path)

    if show:
        plt.show()

    return fig
