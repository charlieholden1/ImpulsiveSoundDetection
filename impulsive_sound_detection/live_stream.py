"""
live_stream.py – Live microphone capture via *sounddevice*.

Provides the ``LiveMicStreamer`` class which opens a
``sounddevice.InputStream``, feeds each audio callback directly into
``StreamMonitor.feed()``, and renders a real-time terminal dashboard.

The YAMNet classification runs on a **separate thread** (managed by
``DetectionPipeline.start_inference_worker`` in ``pipeline.py``) so
that the audio callback is never blocked by model inference.

Usage
-----
>>> from impulsive_sound_detection.live_stream import LiveMicStreamer
>>> streamer = LiveMicStreamer()
>>> streamer.start()    # blocks until Ctrl+C
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]

from . import config
from .classifier import YAMNetClassifier
from .dashboard import LiveDashboard
from .pipeline import DetectionPipeline
from .stream_monitor import StreamMonitor

logger = logging.getLogger(__name__)


class LiveMicStreamer:
    """Capture live microphone audio and run two-stage detection.

    Parameters
    ----------
    sample_rate : int
        Capture sample rate (must match YAMNet = 16 kHz).
    block_size : int
        Samples per ``sounddevice`` callback (default ``1024``).
    device : int | str | None
        Microphone device index / name.  ``None`` = system default.
    energy_multiplier : float
        Override for the dynamic-threshold multiplier.
    log_path : str | None
        Optional JSONL file path for persisting detections.
    enable_colour : bool
        Enable ANSI colour codes in the dashboard.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        block_size: int = 1024,
        device: Optional[int | str] = None,
        energy_multiplier: float = config.ENERGY_MULTIPLIER,
        log_path: Optional[str] = None,
        enable_colour: bool = True,
    ) -> None:
        if sd is None:
            raise ImportError(
                "sounddevice is required for live mode.  "
                "Install it with:  pip install sounddevice"
            )

        self._sample_rate = sample_rate
        self._block_size = block_size
        self._device = device

        # Build the detection stack
        self._monitor = StreamMonitor(
            sample_rate=sample_rate,
            energy_multiplier=energy_multiplier,
        )
        self._classifier = YAMNetClassifier()
        self._pipeline = DetectionPipeline(
            monitor=self._monitor,
            classifier=self._classifier,
            log_path=log_path,
        )

        # Dashboard
        self._dashboard = LiveDashboard(enable_colour=enable_colour)

        # Bookkeeping
        self._running = threading.Event()
        self._start_time: float = 0.0
        self._last_rms: float = 0.0
        self._last_baseline: float = 0.0

    # ── sounddevice callback ──────────────────────────────────────────
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """Called by sounddevice for every audio block.

        This runs on a **high-priority audio thread** and must be
        fast.  Heavy work (YAMNet inference) happens on the pipeline's
        inference worker thread instead.

        Parameters
        ----------
        indata : np.ndarray
            Incoming audio block (shape ``(frames, channels)``).
        frames : int
            Number of frames in this block.
        time_info
            Timing information (unused).
        status
            ``sounddevice`` status flags.
        """
        if status:
            logger.warning("sounddevice status: %s", status)

        # Convert to 1-D float32 mono
        chunk = indata[:, 0].astype(np.float32)

        # Feed Stage 1 (fast – just RMS + ring-buffer write)
        self._monitor.feed(chunk)

        # Grab the latest RMS / baseline for the dashboard
        if self._monitor._rms_history:
            self._last_rms = self._monitor._rms_history[-1]
        self._last_baseline = self._monitor._rolling_mean()

    # ── public API ────────────────────────────────────────────────────
    def start(self) -> None:
        """Open the microphone stream and block until Ctrl+C.

        The method:
        1. Prints the dashboard banner.
        2. Pre-loads YAMNet (so the first detection isn't delayed).
        3. Starts the pipeline inference worker thread.
        4. Opens the sounddevice InputStream.
        5. Runs a main-thread loop that updates the dashboard meter.
        """
        self._dashboard.print_banner()

        # Pre-load YAMNet so the first trigger isn't delayed
        sys.stdout.write("  Loading YAMNet model … ")
        sys.stdout.flush()
        self._classifier._ensure_model()
        sys.stdout.write("done.\n\n")
        sys.stdout.flush()

        # Start the background inference worker
        self._pipeline.start_inference_worker(
            dashboard=self._dashboard,
        )

        self._running.set()
        self._start_time = time.monotonic()

        stream = sd.InputStream(
            samplerate=self._sample_rate,
            blocksize=self._block_size,
            channels=1,
            dtype="float32",
            device=self._device,
            callback=self._audio_callback,
        )

        try:
            with stream:
                logger.info(
                    "Microphone open – device=%s, sr=%d, block=%d",
                    self._device or "default",
                    self._sample_rate,
                    self._block_size,
                )
                self._main_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _main_loop(self) -> None:
        """Dashboard refresh loop running on the main thread.

        Updates the RMS meter ~20 times/sec and prints a periodic
        status line every 30 seconds.
        """
        status_interval = 30.0
        last_status = time.monotonic()

        while self._running.is_set():
            elapsed = time.monotonic() - self._start_time
            threshold = (
                self._last_baseline * self._monitor.energy_multiplier
            )
            self._dashboard.update_meter(
                rms=self._last_rms,
                baseline=self._last_baseline,
                threshold=threshold,
            )

            if time.monotonic() - last_status >= status_interval:
                self._dashboard.show_status(elapsed)
                last_status = time.monotonic()

            time.sleep(0.05)

    def _shutdown(self) -> None:
        """Clean up: stop the inference worker and print the summary."""
        self._running.clear()
        elapsed = time.monotonic() - self._start_time
        self._pipeline.stop_inference_worker()
        self._dashboard.print_shutdown(elapsed)

    @staticmethod
    def list_devices() -> None:
        """Print available audio input devices.

        Useful for choosing the ``--device`` argument.
        """
        if sd is None:
            print("sounddevice is not installed.")
            return
        print(sd.query_devices())
