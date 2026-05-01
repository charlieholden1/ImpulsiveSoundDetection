"""
pipeline.py – End-to-end orchestration of Stages 1 & 2.

Ties together the ``StreamMonitor`` (Stage 1 – energy trigger) and
``YAMNetClassifier`` (Stage 2 – classification) into a single
``DetectionPipeline`` that can process:

* a pre-recorded ``.wav`` file       (``run_on_file``)
* a simulated live stream             (``run_on_stream``)
* a real-time microphone stream       (via ``start_inference_worker``)

Every detection is emitted as a JSON line and optionally logged to a
file.

Threaded inference (live mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For live microphone capture the audio callback cannot block on YAMNet.
``start_inference_worker()`` launches a daemon ``threading.Thread``
that continuously drains ``StreamMonitor.trigger_queue`` and runs
Stage 2 classification independently of the audio thread.

Buffer-overflow guard
~~~~~~~~~~~~~~~~~~~~~
If the YAMNet inference cannot keep up with the trigger rate, the
oldest unprocessed triggers are silently dropped (with a warning) so
that memory consumption stays bounded.  See
``StreamMonitor.trigger_queue`` in ``stream_monitor.py``.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from . import config
from .classifier import ClassificationResult, YAMNetClassifier, CNNClassifier
from .data_loader import load_wav
from .event_logger import EventLogger
from .stream_monitor import StreamMonitor, TriggerEvent
from .visualizer import plot_detections

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """Top-level controller that feeds audio through Stage 1 → Stage 2.

    Parameters
    ----------
    monitor : StreamMonitor | None
        Custom monitor instance.  If ``None`` a default one is created.
    classifier : YAMNetClassifier | CNNClassifier | None
        Custom classifier instance.  If ``None`` a default one is
        created based on config.CLASSIFIER_MODE.
    log_path : Path | None
        If given, each JSON detection line is appended to this file.
    classifier_mode : str
        One of: "cnn", "yamnet", "ensemble". Ignored if classifier is provided.
    """

    def __init__(
        self,
        monitor: Optional[StreamMonitor] = None,
        classifier: Optional[Union[YAMNetClassifier, CNNClassifier]] = None,
        log_path: Optional[Path] = None,
        classifier_mode: Optional[str] = None,
    ) -> None:
        self.monitor = monitor or StreamMonitor()

        # Initialize classifier based on mode if not provided
        if classifier is not None:
            self.classifier = classifier
        else:
            mode = classifier_mode or config.CLASSIFIER_MODE
            if mode == "cnn":
                logger.info("Using CNN classifier")
                self.classifier = CNNClassifier(
                    model_path=str(config.CNN_MODEL_PATH),
                    decision_threshold=config.CNN_DECISION_THRESHOLD,
                )
            elif mode == "ensemble":
                logger.info("Using ensemble classifier (CNN + YAMNet)")
                self.classifier = [
                    CNNClassifier(
                        model_path=str(config.CNN_MODEL_PATH),
                        decision_threshold=config.CNN_DECISION_THRESHOLD,
                    ),
                    YAMNetClassifier(),
                ]
            else:  # yamnet or default
                logger.info("Using YAMNet classifier")
                self.classifier = YAMNetClassifier()

        self._log_path = Path(log_path) if log_path else None
        self._results: List[ClassificationResult] = []
        self._results_lock = threading.Lock()

        # Event logger for SQLite + JSONL
        self._event_logger = EventLogger(
            sqlite_path=config.SQLITE_LOG_PATH,
            jsonl_path=self._log_path,
        )

        # Inference worker state (used in live mode)
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._on_result_callback: Optional[
            Callable[[ClassificationResult], None]
        ] = None

        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Detection log → %s", self._log_path)

    # ── file-level processing ─────────────────────────────────────────
    def run_on_file(
        self,
        wav_path: Path,
        chunk_size: int = 4096,
        visualize: bool = True,
        fig_save_path: Optional[Path] = None,
    ) -> List[ClassificationResult]:
        """Process a pre-recorded WAV file end-to-end.

        Parameters
        ----------
        wav_path : Path
            Path to a ``.wav`` file (any sample rate – will be
            resampled to 16 kHz).
        chunk_size : int
            Number of samples to feed per call (simulates streaming).
        visualize : bool
            If ``True``, show the waveform plot with onsets at the end.
        fig_save_path : Path | None
            Optional path to save the visualisation figure.

        Returns
        -------
        list[ClassificationResult]
            All detections found in the file.
        """
        wav_path = Path(wav_path)
        waveform, sr = load_wav(wav_path)
        logger.info(
            "Processing %s  (%d samples, %.2f s)",
            wav_path.name,
            len(waveform),
            len(waveform) / sr,
        )

        self.monitor.reset()
        self._results.clear()

        # Stream simulation: feed in chunks
        offset = 0
        total = len(waveform)
        while offset < total:
            end = min(offset + chunk_size, total)
            chunk = waveform[offset:end]
            triggers = self.monitor.feed(chunk)
            for trigger in triggers:
                result = self._classify_trigger(trigger)
                self._results.append(result)
                self._emit(result)
            offset = end

        # Drain anything left in the queue (in case feed missed it)
        self._drain_queue()

        logger.info(
            "Finished %s – %d detections", wav_path.name, len(self._results)
        )

        if visualize or fig_save_path:
            plot_detections(
                waveform,
                sr,
                self._results,
                title=f"Detections – {wav_path.name}",
                save_path=fig_save_path,
                show=visualize,
            )

        return list(self._results)

    # ── stream-level processing ───────────────────────────────────────
    def run_on_stream(
        self,
        stream_iterator,
        max_duration_sec: Optional[float] = None,
    ) -> List[ClassificationResult]:
        """Process an iterable of audio chunks (simulated live stream).

        Parameters
        ----------
        stream_iterator
            Iterable yielding 1-D float32 numpy arrays.
        max_duration_sec : float | None
            Stop after this many seconds (safety net).

        Returns
        -------
        list[ClassificationResult]
        """
        self.monitor.reset()
        self._results.clear()
        start = time.monotonic()

        for chunk in stream_iterator:
            triggers = self.monitor.feed(chunk)
            for trigger in triggers:
                result = self._classify_trigger(trigger)
                self._results.append(result)
                self._emit(result)

            if max_duration_sec is not None:
                elapsed = time.monotonic() - start
                if elapsed >= max_duration_sec:
                    logger.info("Stream cap reached (%.1f s)", elapsed)
                    break

        self._drain_queue()
        return list(self._results)

    # ── internal helpers ──────────────────────────────────────────────
    def _classify_trigger(self, trigger: TriggerEvent) -> ClassificationResult:
        """Run Stage 2 classification on a trigger event.

        Parameters
        ----------
        trigger : TriggerEvent
            Output of Stage 1.

        Returns
        -------
        ClassificationResult
        """
        t0 = time.monotonic()

        # Handle ensemble mode (list of classifiers)
        if isinstance(self.classifier, list):
            # Ensemble: CNN AND YAMNet (both must agree)
            results = [
                c.classify(
                    waveform=trigger.window,
                    timestamp=trigger.timestamp_sec,
                    onset_index=trigger.onset_index,
                )
                for c in self.classifier
            ]
            # Combine: is_suspicious only if BOTH agree
            is_suspicious = all(r.is_suspicious for r in results)
            # Use CNN result as primary, note ensemble decision
            result = results[0]
            result.is_suspicious = is_suspicious
            result.label = "GUNSHOT" if is_suspicious else "NOGUN"
        else:
            # Single classifier (CNN or YAMNet)
            result = self.classifier.classify(
                waveform=trigger.window,
                timestamp=trigger.timestamp_sec,
                onset_index=trigger.onset_index,
            )

        elapsed = time.monotonic() - t0
        if elapsed > config.INFERENCE_TIMEOUT_SEC:
            logger.warning(
                "Inference took %.2f s (budget %.1f s) – risk of "
                "buffer overflow",
                elapsed,
                config.INFERENCE_TIMEOUT_SEC,
            )
        return result

    def _drain_queue(self) -> None:
        """Process any remaining triggers still sitting in the queue.

        This handles the edge case where ``feed`` enqueued events but
        the main loop exited before they were consumed.
        """
        while not self.monitor.trigger_queue.empty():
            try:
                trigger = self.monitor.trigger_queue.get_nowait()
            except Exception:
                break
            result = self._classify_trigger(trigger)
            self._results.append(result)
            self._emit(result)

    def _emit(self, result: ClassificationResult) -> None:
        """Emit a detection result via callback, print, and log.

        If a dashboard callback is registered (live mode) it is
        called *instead* of the plain ``print()``, keeping the
        terminal output clean. Event is logged to both SQLite and JSONL.

        Parameters
        ----------
        result : ClassificationResult
            Detection to emit.
        """
        # Live-mode dashboard callback
        if self._on_result_callback is not None:
            try:
                self._on_result_callback(result)
            except Exception:
                logger.exception("Dashboard callback failed")
        else:
            print(result.to_json())

        # Log to SQLite + JSONL via EventLogger
        try:
            self._event_logger.log(result)
        except Exception:
            logger.exception("Failed to log event")

    # ── threaded inference worker (live mode) ─────────────────────────
    def start_inference_worker(
        self,
        dashboard=None,
    ) -> None:
        """Launch a daemon thread that drains the trigger queue.

        The worker continuously polls ``self.monitor.trigger_queue``,
        runs YAMNet classification on each trigger, and emits the
        result – all without blocking the audio callback thread.

        Parameters
        ----------
        dashboard : LiveDashboard | None
            If provided, ``dashboard.show_alert()`` is called for
            every classification result (instead of plain JSON print).
        """
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.warning("Inference worker already running")
            return

        if dashboard is not None:
            self._on_result_callback = dashboard.show_alert

        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._inference_loop,
            name="yamnet-inference-worker",
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("Inference worker thread started")

    def stop_inference_worker(self, timeout: float = 5.0) -> None:
        """Signal the inference worker to stop and wait for it.

        Parameters
        ----------
        timeout : float
            Maximum seconds to wait for the thread to join.
        """
        if self._worker_thread is None:
            return
        self._worker_stop.set()
        self._worker_thread.join(timeout=timeout)
        if self._worker_thread.is_alive():
            logger.warning(
                "Inference worker did not stop within %.1f s", timeout
            )
        else:
            logger.info("Inference worker stopped")
        self._worker_thread = None
        self._on_result_callback = None

    def _inference_loop(self) -> None:
        """Background loop that processes queued triggers.

        Runs on the inference worker thread.  Blocks on the trigger
        queue with a short timeout so that the stop event can be
        checked periodically.
        """
        logger.info("Inference loop running")
        while not self._worker_stop.is_set():
            try:
                trigger: TriggerEvent = (
                    self.monitor.trigger_queue.get(timeout=0.1)
                )
            except queue.Empty:
                continue

            result = self._classify_trigger(trigger)
            with self._results_lock:
                self._results.append(result)
            self._emit(result)

        # Drain remaining items before exiting
        while not self.monitor.trigger_queue.empty():
            try:
                trigger = self.monitor.trigger_queue.get_nowait()
            except queue.Empty:
                break
            result = self._classify_trigger(trigger)
            with self._results_lock:
                self._results.append(result)
            self._emit(result)

        logger.info("Inference loop exiting")

    @property
    def results(self) -> List[ClassificationResult]:
        """Return a copy of all results collected so far.

        Returns
        -------
        list[ClassificationResult]
        """
        with self._results_lock:
            return list(self._results)
