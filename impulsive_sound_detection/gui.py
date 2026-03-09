"""
gui.py – Professional dark-mode GUI dashboard using *customtkinter*.

Embeds a live scrolling Matplotlib plot (RMS energy vs. dynamic
threshold), a colour-coded event log, and a control sidebar with
real-time parameter tuning – all without blocking the audio or
inference threads.

Threading architecture
~~~~~~~~~~~~~~~~~~~~~~
::

    Main Thread (Tk)          Audio Thread (sounddevice)   Inference Thread
    ────────────────          ───────────────────────────  ────────────────
    CTk mainloop              InputStream callback         _inference_loop()
      ├ _poll_gui_queue()       └ StreamMonitor.feed()       └ YAMNet classify
      │   (root.after 70ms)                                     └ gui_queue.put()
      └ _animate_plot()
          (root.after 80ms)

* ``sounddevice`` callback → feeds audio into ``StreamMonitor``
  (Stage 1, fast – runs on its own thread managed by PortAudio).
* ``_inference_loop`` → pulls from ``trigger_queue``, runs YAMNet,
  pushes ``ClassificationResult`` into ``gui_queue``.
* ``_poll_gui_queue`` / ``_animate_plot`` → scheduled on the **main
  thread** via ``root.after()``; safe to touch Tk widgets.

Usage
-----
    python -m impulsive_sound_detection.main gui
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import config
from .classifier import ClassificationResult, YAMNetClassifier
from .data_loader import load_wav
from .stream_monitor import StreamMonitor

logger = logging.getLogger(__name__)

# ── Try importing sounddevice at module level ─────────────────────────
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]


# ======================================================================
#  Constants for the GUI layout and animation
# ======================================================================
_PLOT_HISTORY_SEC: float = 15.0      # Seconds of RMS shown on x-axis
_PLOT_FPS: int = 12                  # Target frames per second for plot
_PLOT_INTERVAL_MS: int = int(1000 / _PLOT_FPS)
_QUEUE_POLL_MS: int = 70             # How often to check gui_queue
_SIDEBAR_WIDTH: int = 260            # Pixels
_WINDOW_MIN_W: int = 1100
_WINDOW_MIN_H: int = 700

# Colour palette (hex) – consistent dark theme
_CLR_BG = "#1a1a2e"
_CLR_SIDEBAR = "#16213e"
_CLR_ACCENT = "#0f3460"
_CLR_ALERT_RED = "#e74c3c"
_CLR_ALERT_GREEN = "#27ae60"
_CLR_TEXT = "#e0e0e0"
_CLR_TEXT_DIM = "#8899aa"
_CLR_PLOT_RMS = "#00b4d8"
_CLR_PLOT_THR = "#e74c3c"


# ======================================================================
#  Main application class
# ======================================================================
class DetectionGUI(ctk.CTk):
    """Top-level customtkinter window for the detection dashboard.

    Parameters
    ----------
    log_path : str | None
        Optional JSONL log path for persisting results.
    """

    # ------------------------------------------------------------------
    #  Initialisation
    # ------------------------------------------------------------------
    def __init__(self, log_path: Optional[str] = None) -> None:
        super().__init__()

        # ── Window basics ─────────────────────────────────────────────
        self.title("Impulsive Sound Detection – Dashboard")
        self.minsize(_WINDOW_MIN_W, _WINDOW_MIN_H)
        self.geometry("1200x750")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # ── Shared state ──────────────────────────────────────────────
        self._log_path = Path(log_path) if log_path else None
        self._monitor: Optional[StreamMonitor] = None
        self._classifier: Optional[YAMNetClassifier] = None
        self._sd_stream: object = None  # sounddevice.InputStream or None
        self._sd_output: object = None  # sounddevice.OutputStream or None
        self._wav_thread: Optional[threading.Thread] = None

        # Inference worker
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()

        # Thread-safe queue: inference thread → GUI thread
        self._gui_queue: queue.Queue[ClassificationResult] = queue.Queue()

        # Plot data buffers (ring arrays)
        self._plot_maxlen = int(
            _PLOT_HISTORY_SEC * config.SAMPLE_RATE / config.RMS_FRAME_SIZE
        )
        self._rms_buf: List[float] = []
        self._thr_buf: List[float] = []
        self._time_buf: List[float] = []
        self._stream_start: float = 0.0

        # Counters
        self._total_alerts: int = 0
        self._suspicious_alerts: int = 0
        self._is_streaming: bool = False

        # ── Build UI ──────────────────────────────────────────────────
        self._build_sidebar()
        self._build_main_area()

        # ── Protocol: clean shutdown on window close ──────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ==================================================================
    #  UI Construction
    # ==================================================================
    def _build_sidebar(self) -> None:
        """Construct the left sidebar with controls and sliders."""
        sidebar = ctk.CTkFrame(self, width=_SIDEBAR_WIDTH, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        # ── Title ─────────────────────────────────────────────────────
        ctk.CTkLabel(
            sidebar,
            text="🎛  Controls",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(18, 10), padx=14, anchor="w")

        # ── Input mode dropdown ───────────────────────────────────────
        ctk.CTkLabel(
            sidebar, text="Input Source", font=ctk.CTkFont(size=12)
        ).pack(padx=14, anchor="w")
        self._input_var = ctk.StringVar(value="Live Microphone")
        self._input_dropdown = ctk.CTkOptionMenu(
            sidebar,
            variable=self._input_var,
            values=["Live Microphone", "WAV File Playback"],
            width=220,
            command=self._on_input_mode_changed,
        )
        self._input_dropdown.pack(padx=14, pady=(2, 8))

        # ── WAV file chooser (hidden by default) ──────────────────────
        self._wav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        self._wav_path_var = ctk.StringVar(value="")
        self._wav_entry = ctk.CTkEntry(
            self._wav_frame,
            textvariable=self._wav_path_var,
            placeholder_text="Path to .wav file",
            width=150,
        )
        self._wav_entry.pack(side="left", padx=(0, 4))
        self._wav_browse_btn = ctk.CTkButton(
            self._wav_frame,
            text="…",
            width=36,
            command=self._browse_wav,
        )
        self._wav_browse_btn.pack(side="left")
        # (not packed yet – shown only when mode == WAV)

        # ── Start / Stop ──────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(padx=14, pady=10, fill="x")
        self._start_btn = ctk.CTkButton(
            btn_frame,
            text="▶  Start Stream",
            fg_color="#27ae60",
            hover_color="#219150",
            command=self._on_start,
        )
        self._start_btn.pack(fill="x", pady=(0, 6))
        self._stop_btn = ctk.CTkButton(
            btn_frame,
            text="■  Stop Stream",
            fg_color="#c0392b",
            hover_color="#96281b",
            state="disabled",
            command=self._on_stop,
        )
        self._stop_btn.pack(fill="x")

        # ── Separator ─────────────────────────────────────────────────
        ctk.CTkLabel(
            sidebar,
            text="─── Live Tuning ───",
            font=ctk.CTkFont(size=11),
            text_color=_CLR_TEXT_DIM,
        ).pack(pady=(16, 4))

        # ── Energy multiplier slider ──────────────────────────────────
        ctk.CTkLabel(
            sidebar, text="Threshold Multiplier", font=ctk.CTkFont(size=12)
        ).pack(padx=14, anchor="w")
        self._mult_var = tk.DoubleVar(value=config.ENERGY_MULTIPLIER)
        self._mult_label = ctk.CTkLabel(
            sidebar,
            text=f"{config.ENERGY_MULTIPLIER:.1f}×",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._mult_label.pack(padx=14, anchor="e")
        self._mult_slider = ctk.CTkSlider(
            sidebar,
            from_=1.5,
            to=10.0,
            number_of_steps=85,
            variable=self._mult_var,
            width=220,
            command=self._on_mult_changed,
        )
        self._mult_slider.pack(padx=14, pady=(0, 10))

        # ── Dead-time slider ──────────────────────────────────────────
        ctk.CTkLabel(
            sidebar, text="Min Re-trigger (s)", font=ctk.CTkFont(size=12)
        ).pack(padx=14, anchor="w")
        self._dead_var = tk.DoubleVar(value=config.MIN_RETRIGGER_SEC)
        self._dead_label = ctk.CTkLabel(
            sidebar,
            text=f"{config.MIN_RETRIGGER_SEC:.2f} s",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self._dead_label.pack(padx=14, anchor="e")
        self._dead_slider = ctk.CTkSlider(
            sidebar,
            from_=0.1,
            to=3.0,
            number_of_steps=58,
            variable=self._dead_var,
            width=220,
            command=self._on_dead_changed,
        )
        self._dead_slider.pack(padx=14, pady=(0, 16))

        # ── Status counters ───────────────────────────────────────────
        ctk.CTkLabel(
            sidebar,
            text="─── Session Stats ───",
            font=ctk.CTkFont(size=11),
            text_color=_CLR_TEXT_DIM,
        ).pack(pady=(8, 4))
        self._status_label = ctk.CTkLabel(
            sidebar,
            text="Alerts: 0  |  Suspicious: 0",
            font=ctk.CTkFont(size=12),
        )
        self._status_label.pack(padx=14, anchor="w", pady=4)
        self._elapsed_label = ctk.CTkLabel(
            sidebar,
            text="Elapsed: 0.0 s",
            font=ctk.CTkFont(size=12),
            text_color=_CLR_TEXT_DIM,
        )
        self._elapsed_label.pack(padx=14, anchor="w")

    def _build_main_area(self) -> None:
        """Construct the right-hand main area: plot canvas + event log."""
        main = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        main.pack(side="right", fill="both", expand=True)

        # ── Top: Matplotlib canvas ────────────────────────────────────
        plot_frame = ctk.CTkFrame(main, corner_radius=8)
        plot_frame.pack(fill="both", expand=True, padx=10, pady=(10, 5))

        self._fig = Figure(figsize=(9, 3), dpi=100, facecolor="#0e1117")
        self._ax = self._fig.add_subplot(111)
        self._setup_plot_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        # ── Bottom: Event log ─────────────────────────────────────────
        log_header = ctk.CTkFrame(main, fg_color="transparent")
        log_header.pack(fill="x", padx=10)
        ctk.CTkLabel(
            log_header,
            text="📋  Event Log",
            font=ctk.CTkFont(size=14, weight="bold"),
        ).pack(side="left", pady=(4, 2))
        self._clear_log_btn = ctk.CTkButton(
            log_header,
            text="Clear",
            width=60,
            height=24,
            font=ctk.CTkFont(size=11),
            command=self._clear_log,
        )
        self._clear_log_btn.pack(side="right", padx=4, pady=(4, 2))

        self._log_box = ctk.CTkTextbox(
            main,
            height=200,
            corner_radius=8,
            font=ctk.CTkFont(family="Consolas", size=12),
            state="disabled",
            wrap="word",
        )
        self._log_box.pack(fill="both", expand=False, padx=10, pady=(0, 10))

        # Register text tags for colour coding
        self._log_box.tag_config("alert", foreground=_CLR_ALERT_RED)
        self._log_box.tag_config("safe", foreground=_CLR_ALERT_GREEN)
        self._log_box.tag_config("info", foreground=_CLR_TEXT_DIM)

    # ==================================================================
    #  Plot helpers
    # ==================================================================
    def _setup_plot_axes(self) -> None:
        """Configure Matplotlib axes styling (dark theme)."""
        ax = self._ax
        ax.set_facecolor("#0e1117")
        ax.set_xlabel("Time (s)", color=_CLR_TEXT, fontsize=9)
        ax.set_ylabel("RMS Energy", color=_CLR_TEXT, fontsize=9)
        ax.set_title(
            "Live RMS Energy  vs  Dynamic Threshold",
            color=_CLR_TEXT,
            fontsize=11,
            fontweight="bold",
        )
        ax.tick_params(colors=_CLR_TEXT_DIM, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.set_xlim(0, _PLOT_HISTORY_SEC)
        ax.set_ylim(0, 0.05)
        # Create empty line objects for blitting-style update
        (self._line_rms,) = ax.plot([], [], color=_CLR_PLOT_RMS, linewidth=1.0,
                                    label="RMS Energy")
        (self._line_thr,) = ax.plot([], [], color=_CLR_PLOT_THR, linewidth=1.0,
                                    linestyle="--", label="Threshold")
        ax.legend(loc="upper right", fontsize=8, facecolor="#1a1a2e",
                  labelcolor=_CLR_TEXT, edgecolor="#333")
        self._fig.tight_layout()

    def _animate_plot(self) -> None:
        """Redraw the scrolling RMS / threshold lines (~12 FPS).

        Scheduled via ``root.after()`` so it only touches the canvas
        on the main thread.
        """
        if not self._is_streaming:
            return

        if self._time_buf:
            self._line_rms.set_data(self._time_buf, self._rms_buf)
            self._line_thr.set_data(self._time_buf, self._thr_buf)

            t_max = self._time_buf[-1]
            t_min = max(0, t_max - _PLOT_HISTORY_SEC)
            self._ax.set_xlim(t_min, t_min + _PLOT_HISTORY_SEC)

            # Auto-scale y
            visible_rms = [
                v
                for t, v in zip(self._time_buf, self._rms_buf)
                if t >= t_min
            ]
            visible_thr = [
                v
                for t, v in zip(self._time_buf, self._thr_buf)
                if t >= t_min
            ]
            if visible_rms:
                y_max = max(max(visible_rms), max(visible_thr)) * 1.3
                y_max = max(y_max, 0.01)
                self._ax.set_ylim(0, y_max)

            self._canvas.draw_idle()

        # Update elapsed
        elapsed = time.monotonic() - self._stream_start
        self._elapsed_label.configure(text=f"Elapsed: {elapsed:.1f} s")

        # Re-schedule
        self.after(_PLOT_INTERVAL_MS, self._animate_plot)

    # ==================================================================
    #  GUI-queue poller (inference thread → main thread)
    # ==================================================================
    def _poll_gui_queue(self) -> None:
        """Drain the gui_queue and insert results into the log box.

        Scheduled via ``root.after()`` – safe to update Tk widgets.
        """
        while True:
            try:
                result: ClassificationResult = self._gui_queue.get_nowait()
            except queue.Empty:
                break
            self._insert_log_entry(result)

        if self._is_streaming:
            self.after(_QUEUE_POLL_MS, self._poll_gui_queue)

    def _insert_log_entry(self, result: ClassificationResult) -> None:
        """Format and insert a ClassificationResult into the log box.

        Parameters
        ----------
        result : ClassificationResult
            Detection to display.
        """
        self._total_alerts += 1
        if result.is_suspicious:
            self._suspicious_alerts += 1
            tag = "alert"
            prefix = "🚨 ALERT"
        else:
            tag = "safe"
            prefix = "✅ SAFE "

        top3 = "  ".join(
            f"{e['class']}({e['score']:.2f})" for e in result.top_k[:3]
        )
        line = (
            f"[{result.timestamp:7.2f}s]  {prefix}  "
            f"{result.label}  conf={result.confidence:.3f}  "
            f"| {top3}\n"
        )

        self._log_box.configure(state="normal")
        self._log_box.insert("end", line, tag)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

        # Update counters
        self._status_label.configure(
            text=(
                f"Alerts: {self._total_alerts}  |  "
                f"Suspicious: {self._suspicious_alerts}"
            )
        )

        # Persist to JSONL
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(result.to_json() + "\n")

    # ==================================================================
    #  Sidebar callbacks
    # ==================================================================
    def _on_input_mode_changed(self, value: str) -> None:
        """Show / hide the WAV file chooser based on the dropdown.

        Parameters
        ----------
        value : str
            Selected dropdown value.
        """
        if value == "WAV File Playback":
            self._wav_frame.pack(padx=14, pady=(0, 4), fill="x", after=self._input_dropdown)
        else:
            self._wav_frame.pack_forget()

    def _browse_wav(self) -> None:
        """Open a file dialog for choosing a WAV file."""
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self._wav_path_var.set(path)

    def _on_mult_changed(self, value: float) -> None:
        """Update the energy multiplier on the live StreamMonitor.

        Parameters
        ----------
        value : float
            New multiplier value from the slider.
        """
        self._mult_label.configure(text=f"{value:.1f}×")
        if self._monitor is not None:
            self._monitor.energy_multiplier = value

    def _on_dead_changed(self, value: float) -> None:
        """Update the dead-time on the live StreamMonitor.

        Parameters
        ----------
        value : float
            New dead-time value in seconds.
        """
        self._dead_label.configure(text=f"{value:.2f} s")
        if self._monitor is not None:
            self._monitor.min_retrigger_sec = value

    # ==================================================================
    #  Start / Stop
    # ==================================================================
    def _on_start(self) -> None:
        """Start the audio stream and inference worker."""
        if self._is_streaming:
            return

        mode = self._input_var.get()

        # ── Build fresh pipeline components ───────────────────────────
        self._monitor = StreamMonitor(
            energy_multiplier=self._mult_var.get(),
            min_retrigger_sec=self._dead_var.get(),
        )
        self._classifier = YAMNetClassifier()

        # Reset buffers and counters
        self._rms_buf.clear()
        self._thr_buf.clear()
        self._time_buf.clear()
        self._total_alerts = 0
        self._suspicious_alerts = 0
        self._status_label.configure(text="Alerts: 0  |  Suspicious: 0")

        # Pre-load YAMNet (show a temporary status)
        self._log_info("Loading YAMNet model …")
        self.update_idletasks()
        self._classifier._ensure_model()
        self._log_info("YAMNet loaded ✓")

        # Start inference worker thread
        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._inference_loop,
            name="gui-yamnet-worker",
            daemon=True,
        )
        self._worker_thread.start()

        # Start audio source
        self._stream_start = time.monotonic()
        self._is_streaming = True

        if mode == "Live Microphone":
            self._start_mic_stream()
        else:
            wav_path = self._wav_path_var.get().strip()
            if not wav_path or not Path(wav_path).exists():
                self._log_info("⚠ WAV file not found – aborting.")
                self._is_streaming = False
                return
            self._start_wav_playback(Path(wav_path))

        # Toggle button states
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._input_dropdown.configure(state="disabled")

        # Start scheduled loops
        self.after(_QUEUE_POLL_MS, self._poll_gui_queue)
        self.after(_PLOT_INTERVAL_MS, self._animate_plot)

    def _on_stop(self) -> None:
        """Stop all threads and the audio stream."""
        self._is_streaming = False

        # Stop sounddevice stream
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

        # Stop output playback stream
        if self._sd_output is not None:
            try:
                self._sd_output.stop()
                self._sd_output.close()
            except Exception:
                pass
            self._sd_output = None

        # Stop WAV playback thread
        if self._wav_thread is not None:
            self._wav_thread.join(timeout=2.0)
            self._wav_thread = None

        # Stop inference worker
        self._worker_stop.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        # Toggle buttons
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._input_dropdown.configure(state="normal")

        elapsed = time.monotonic() - self._stream_start
        self._log_info(
            f"Stream stopped.  Duration: {elapsed:.1f}s  |  "
            f"Alerts: {self._total_alerts}  |  "
            f"Suspicious: {self._suspicious_alerts}"
        )

    # ==================================================================
    #  Audio sources
    # ==================================================================
    def _start_mic_stream(self) -> None:
        """Open a sounddevice InputStream for the default microphone."""
        if sd is None:
            self._log_info("⚠ sounddevice not installed – cannot open mic.")
            self._on_stop()
            return

        self._sd_stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            blocksize=1024,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._sd_stream.start()
        self._log_info("🎙  Microphone stream started")

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info,
        status,
    ) -> None:
        """sounddevice callback – runs on the PortAudio thread.

        Feeds audio into the StreamMonitor and updates the plot data
        buffers.  Must be fast – no Tk widget access here.

        Parameters
        ----------
        indata : np.ndarray
            Audio block ``(frames, 1)``.
        frames : int
            Number of frames.
        time_info
            PortAudio timing info (unused).
        status
            PortAudio status flags.
        """
        if status:
            logger.warning("sounddevice status: %s", status)
        chunk = indata[:, 0].astype(np.float32)
        self._feed_and_record(chunk)

    def _start_wav_playback(self, wav_path: Path) -> None:
        """Load a WAV file and stream it in a background thread.

        Parameters
        ----------
        wav_path : Path
            Path to the ``.wav`` file.
        """
        # Open an output stream so the user can hear the WAV file
        if sd is not None:
            try:
                self._sd_output = sd.OutputStream(
                    samplerate=config.SAMPLE_RATE,
                    channels=1,
                    dtype="float32",
                )
                self._sd_output.start()
            except Exception as exc:
                logger.warning("Cannot open audio output: %s", exc)
                self._sd_output = None

        self._wav_thread = threading.Thread(
            target=self._wav_feeder,
            args=(wav_path,),
            name="wav-feeder",
            daemon=True,
        )
        self._wav_thread.start()
        self._log_info(f"📂  Playing {wav_path.name}")

    def _wav_feeder(self, wav_path: Path) -> None:
        """Background thread: read WAV file and feed in real-time chunks.

        Parameters
        ----------
        wav_path : Path
            Path to the WAV file.
        """
        try:
            waveform, sr = load_wav(wav_path)
        except Exception as exc:
            logger.error("Failed to load %s: %s", wav_path, exc)
            return

        chunk_size = 1024
        # Simulate real-time playback speed
        chunk_duration = chunk_size / sr
        offset = 0
        while offset < len(waveform) and self._is_streaming:
            end = min(offset + chunk_size, len(waveform))
            chunk = waveform[offset:end]
            self._feed_and_record(chunk)
            # Play audio through speakers
            if self._sd_output is not None:
                try:
                    self._sd_output.write(chunk.reshape(-1, 1))
                except Exception:
                    pass  # output closed during stop
            else:
                time.sleep(chunk_duration)
            offset = end

        # Signal end of file on the GUI thread via queue
        if self._is_streaming:
            # Schedule stop from main thread
            self.after(100, self._on_stop)

    def _feed_and_record(self, chunk: np.ndarray) -> None:
        """Feed a chunk into StreamMonitor and record plot data.

        Thread-safe – called from audio or WAV-feeder threads.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 audio.
        """
        if self._monitor is None:
            return
        self._monitor.feed(chunk)

        # Snapshot latest RMS / baseline for the plot
        if self._monitor._rms_history:
            rms = self._monitor._rms_history[-1]
        else:
            rms = 0.0
        baseline = self._monitor._rolling_mean()
        threshold = baseline * self._monitor.energy_multiplier
        elapsed = time.monotonic() - self._stream_start

        self._rms_buf.append(rms)
        self._thr_buf.append(threshold)
        self._time_buf.append(elapsed)

        # Trim to keep memory bounded
        if len(self._rms_buf) > self._plot_maxlen:
            trim = len(self._rms_buf) - self._plot_maxlen
            del self._rms_buf[:trim]
            del self._thr_buf[:trim]
            del self._time_buf[:trim]

    # ==================================================================
    #  Inference worker (background thread)
    # ==================================================================
    def _inference_loop(self) -> None:
        """Drain trigger_queue, run YAMNet, push results to gui_queue.

        Runs on a daemon thread.  Never touches Tk widgets directly;
        results are sent via ``self._gui_queue`` and picked up by
        ``_poll_gui_queue`` on the main thread.
        """
        logger.info("GUI inference loop started")
        while not self._worker_stop.is_set():
            if self._monitor is None:
                time.sleep(0.05)
                continue
            try:
                trigger = self._monitor.trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                result = self._classifier.classify(
                    waveform=trigger.window,
                    timestamp=trigger.timestamp_sec,
                    onset_index=trigger.onset_index,
                )
                self._gui_queue.put(result)
            except Exception:
                logger.exception("Inference failed for trigger at t=%.2f",
                                 trigger.timestamp_sec)

        # Drain remaining
        if self._monitor is not None:
            while not self._monitor.trigger_queue.empty():
                try:
                    trigger = self._monitor.trigger_queue.get_nowait()
                except queue.Empty:
                    break
                try:
                    result = self._classifier.classify(
                        waveform=trigger.window,
                        timestamp=trigger.timestamp_sec,
                        onset_index=trigger.onset_index,
                    )
                    self._gui_queue.put(result)
                except Exception:
                    pass

        logger.info("GUI inference loop exiting")

    # ==================================================================
    #  Utility helpers
    # ==================================================================
    def _log_info(self, text: str) -> None:
        """Insert an informational line into the log box.

        Parameters
        ----------
        text : str
            Message to display.
        """
        self._log_box.configure(state="normal")
        self._log_box.insert("end", f"  ℹ  {text}\n", "info")
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _clear_log(self) -> None:
        """Clear all text in the event log box."""
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")

    def _on_close(self) -> None:
        """Handle the window-close event gracefully."""
        if self._is_streaming:
            self._on_stop()
        self.destroy()


# ======================================================================
#  Module-level launcher (called from main.py)
# ======================================================================
def launch_gui(log_path: Optional[str] = None) -> None:
    """Create and run the detection GUI.

    Parameters
    ----------
    log_path : str | None
        Optional JSONL log file path.
    """
    app = DetectionGUI(log_path=log_path)
    app.mainloop()
