"""
stream_monitor.py – Real-time energy-based trigger (Stage 1).

Implements a ``StreamMonitor`` that processes a continuous audio stream
in small frames, computes RMS energy, maintains a rolling baseline, and
fires a trigger when a transient energy spike exceeds a dynamic
threshold.

Trigger rule
~~~~~~~~~~~~
    ``current_rms  >  ENERGY_MULTIPLIER  ×  rolling_mean(last N seconds)``

A *dead-time* window prevents re-triggering within
``MIN_RETRIGGER_SEC`` of the previous trigger.

Buffer-overflow guard
~~~~~~~~~~~~~~~~~~~~~
Detected trigger windows are placed into a bounded ``queue.Queue``.  If
the queue is full (consumer – i.e. YAMNet inference – cannot keep up),
the oldest pending window is *dropped* and a warning is logged.  This
prevents unbounded memory growth when inference is slower than
real-time.
"""

from __future__ import annotations

import logging
import queue
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Data container for a detected trigger event
# ──────────────────────────────────────────────────────────────────────
@dataclass
class TriggerEvent:
    """Metadata + audio window associated with a single trigger."""

    onset_index: int
    timestamp_sec: float
    window: np.ndarray          # 0.975 s clip centred on the onset
    rms_energy: float
    baseline_energy: float


# ──────────────────────────────────────────────────────────────────────
# Stream monitor
# ──────────────────────────────────────────────────────────────────────
class StreamMonitor:
    """Sliding-window energy detector for impulsive sound onsets.

    Parameters
    ----------
    sample_rate : int
        Expected sample rate of the incoming stream.
    frame_size : int
        Number of samples per RMS frame (default ``512``).
    rolling_window_sec : float
        How many seconds of RMS history to keep for the baseline.
    energy_multiplier : float
        Trigger when current RMS > multiplier × baseline.
    min_retrigger_sec : float
        Minimum gap (seconds) between consecutive triggers.
    max_queue_size : int
        Maximum number of pending trigger windows before dropping.
    """

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        frame_size: int = config.RMS_FRAME_SIZE,
        rolling_window_sec: float = config.ROLLING_WINDOW_SEC,
        energy_multiplier: float = config.ENERGY_MULTIPLIER,
        min_retrigger_sec: float = config.MIN_RETRIGGER_SEC,
        max_queue_size: int = config.MAX_QUEUE_SIZE,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.energy_multiplier = energy_multiplier
        self.min_retrigger_sec = min_retrigger_sec

        # Number of RMS values to keep in the rolling window
        frames_per_sec = sample_rate / frame_size
        self._rolling_len = int(rolling_window_sec * frames_per_sec)
        self._rms_history: List[float] = []

        # Ring buffer for raw audio (to extract the ±0.975 s window)
        self._yamnet_half = config.YAMNET_WINDOW_SAMPLES // 2
        ring_len = config.YAMNET_WINDOW_SAMPLES + sample_rate  # ~2 s
        self._ring = np.zeros(ring_len, dtype=np.float32)
        self._ring_write: int = 0          # next write position in ring

        # Bookkeeping
        self._global_sample_idx: int = 0   # absolute sample count
        self._last_trigger_time: float = -999.0

        # Bounded output queue (buffer-overflow guard)
        self.trigger_queue: queue.Queue[TriggerEvent] = queue.Queue(
            maxsize=max_queue_size
        )

        logger.info(
            "StreamMonitor ready  (frame=%d, rolling=%d frames, "
            "multiplier=%.1f×, queue_max=%d)",
            frame_size,
            self._rolling_len,
            energy_multiplier,
            max_queue_size,
        )

    # ── internal helpers ──────────────────────────────────────────────
    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        """Compute root-mean-square energy of a frame.

        Parameters
        ----------
        frame : np.ndarray
            Audio samples.

        Returns
        -------
        float
            RMS energy value.
        """
        return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))

    def _rolling_mean(self) -> float:
        """Return the rolling mean of RMS history.

        Returns
        -------
        float
            Mean RMS over the history window, or ``1e-8`` if no
            history yet (to avoid division by zero).
        """
        if not self._rms_history:
            return 1e-8
        return float(np.mean(self._rms_history[-self._rolling_len:]))

    def _write_ring(self, samples: np.ndarray) -> None:
        """Append *samples* to the internal ring buffer.

        Parameters
        ----------
        samples : np.ndarray
            Chunk of incoming audio (float32).
        """
        n = len(samples)
        ring_len = len(self._ring)
        if n >= ring_len:
            # If chunk is larger than ring, keep only the tail
            self._ring[:] = samples[-ring_len:]
            self._ring_write = 0
        else:
            space = ring_len - self._ring_write
            if n <= space:
                self._ring[self._ring_write:self._ring_write + n] = samples
            else:
                self._ring[self._ring_write:] = samples[:space]
                self._ring[:n - space] = samples[space:]
            self._ring_write = (self._ring_write + n) % ring_len

    def _extract_window(self, centre_sample: int) -> np.ndarray:
        """Extract a 0.975 s window centred on *centre_sample* from
        the ring buffer.

        Parameters
        ----------
        centre_sample : int
            Absolute sample index of the trigger point.

        Returns
        -------
        np.ndarray
            Audio window of length ``YAMNET_WINDOW_SAMPLES``.
        """
        ring_len = len(self._ring)
        half = self._yamnet_half
        win_len = config.YAMNET_WINDOW_SAMPLES

        # Map absolute index → ring position
        ring_pos = centre_sample % ring_len
        start = (ring_pos - half) % ring_len

        window = np.empty(win_len, dtype=np.float32)
        if start + win_len <= ring_len:
            window[:] = self._ring[start:start + win_len]
        else:
            first = ring_len - start
            window[:first] = self._ring[start:]
            window[first:] = self._ring[:win_len - first]
        return window

    # ── public interface ──────────────────────────────────────────────
    def feed(self, chunk: np.ndarray) -> List[TriggerEvent]:
        """Ingest a chunk of audio and return any new triggers.

        This is the main entry point.  Call it repeatedly with
        successive chunks from a microphone or file reader.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 audio samples at ``self.sample_rate``.

        Returns
        -------
        list[TriggerEvent]
            Zero or more trigger events detected in this chunk.
        """
        self._write_ring(chunk)
        triggers: List[TriggerEvent] = []

        offset = 0
        while offset + self.frame_size <= len(chunk):
            frame = chunk[offset:offset + self.frame_size]
            rms = self._rms(frame)
            baseline = self._rolling_mean()
            self._rms_history.append(rms)

            # Keep history bounded
            if len(self._rms_history) > self._rolling_len * 2:
                self._rms_history = self._rms_history[-self._rolling_len:]

            current_time = self._global_sample_idx / self.sample_rate

            if rms > self.energy_multiplier * baseline:
                gap = current_time - self._last_trigger_time
                if gap >= self.min_retrigger_sec:
                    window = self._extract_window(self._global_sample_idx)
                    event = TriggerEvent(
                        onset_index=self._global_sample_idx,
                        timestamp_sec=round(current_time, 4),
                        window=window,
                        rms_energy=round(rms, 6),
                        baseline_energy=round(baseline, 6),
                    )
                    triggers.append(event)
                    self._last_trigger_time = current_time

                    # Enqueue with overflow protection
                    try:
                        self.trigger_queue.put_nowait(event)
                    except queue.Full:
                        # Drop the OLDEST pending event
                        try:
                            dropped = self.trigger_queue.get_nowait()
                            logger.warning(
                                "Buffer overflow – dropped trigger at "
                                "t=%.3f s to make room",
                                dropped.timestamp_sec,
                            )
                        except queue.Empty:
                            pass
                        self.trigger_queue.put_nowait(event)

                    logger.debug(
                        "TRIGGER at t=%.3f s  (RMS=%.4f, baseline=%.4f)",
                        current_time,
                        rms,
                        baseline,
                    )

            self._global_sample_idx += self.frame_size
            offset += self.frame_size

        return triggers

    def reset(self) -> None:
        """Clear all internal state and the trigger queue.

        Call this between files or when restarting the stream.
        """
        self._rms_history.clear()
        self._ring[:] = 0.0
        self._ring_write = 0
        self._global_sample_idx = 0
        self._last_trigger_time = -999.0
        while not self.trigger_queue.empty():
            try:
                self.trigger_queue.get_nowait()
            except queue.Empty:
                break
        logger.info("StreamMonitor reset")
