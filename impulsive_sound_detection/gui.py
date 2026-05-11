"""Modern desktop dashboard for live and replayed detection."""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from typing import Dict, List, Optional

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from . import config
from .classifier import (
    CNNClassifier,
    ClassificationResult,
    YAMNetClassifier,
    YAMNetHeadClassifier,
)
from .data_loader import load_wav
from .stream_monitor import StreamMonitor

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None  # type: ignore[assignment]


PLOT_HISTORY_SEC = 18.0
PLOT_INTERVAL_MS = 80
QUEUE_POLL_MS = 70
MAX_LOG_LINES = 160
MAX_MARKERS = 24

BG = "#0B1020"
PANEL = "#111827"
PANEL_ALT = "#162033"
BORDER = "#24324D"
TEXT = "#E7EEF8"
MUTED = "#8EA1BD"
ACCENT = "#4F8CFF"
ACCENT_2 = "#53D1BB"
SAFE = "#44D17A"
WARN = "#F5B942"
DANGER = "#FF6B6B"
RMS = "#66D9EF"
THRESH = "#FF8A65"


def _fmt_time(value: float) -> str:
    minutes = int(value // 60)
    seconds = int(value % 60)
    return f"{minutes:02d}:{seconds:02d}"


def _severity_color(name: str) -> str:
    if name == "HIGH":
        return DANGER
    if name == "MEDIUM":
        return WARN
    return SAFE


class DetectionGUI(ctk.CTk):
    """Professional real-time detection console."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__()

        self.title("Impulse Detection Console")
        self.geometry("1520x920")
        self.minsize(1320, 820)
        self.configure(fg_color=BG)

        self._log_path = Path(log_path) if log_path else None
        self._monitor: Optional[StreamMonitor] = None
        self._classifier: object = None
        self._sd_stream: object = None
        self._sd_output: object = None
        self._wav_thread: Optional[threading.Thread] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._worker_stop = threading.Event()
        self._gui_queue: queue.Queue[ClassificationResult] = queue.Queue()

        self._plot_maxlen = int(
            PLOT_HISTORY_SEC * config.SAMPLE_RATE / config.RMS_FRAME_SIZE
        )
        self._rms_buf: List[float] = []
        self._thr_buf: List[float] = []
        self._time_buf: List[float] = []
        self._marker_times: List[float] = []
        self._marker_levels: List[float] = []

        self._stream_start = 0.0
        self._source_duration_sec = 0.0
        self._source_position_sec = 0.0
        self._source_name = "Microphone"
        self._is_streaming = False
        self._total_alerts = 0
        self._suspicious_alerts = 0
        self._last_result: Optional[ClassificationResult] = None
        self._metric_labels: Dict[str, ctk.CTkLabel] = {}

        # Manual accuracy-testing state
        self._user_marks: List[float] = []
        self._detection_timestamps: List[float] = []

        # Waveform sync view (file playback mode)
        self._waveform_mode: bool = False
        self._playhead_line = None
        self._det_lines: list = []

        self.grid_columnconfigure(0, minsize=320)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()
        self._build_inspector()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind("<m>", lambda e: self._on_mark_gunshot())
        self.bind("<M>", lambda e: self._on_mark_gunshot())
        self._set_status("Idle", MUTED)
        self._set_metric("alerts", "0")
        self._set_metric("suspicious", "0")
        self._set_metric("source", "MIC")
        self._set_metric("classifier", config.CLASSIFIER_MODE.upper())
        self._refresh_transport()

    def _card(self, parent, fg_color: str = PANEL, radius: int = 18):
        return ctk.CTkFrame(
            parent,
            fg_color=fg_color,
            border_width=1,
            border_color=BORDER,
            corner_radius=radius,
        )

    def _build_sidebar(self) -> None:
        sidebar = self._card(self, fg_color="#0F172A", radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            sidebar,
            text="Impulse Detection",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
            text_color=TEXT,
        ).grid(row=0, column=0, sticky="w", padx=22, pady=(24, 6))
        ctk.CTkLabel(
            sidebar,
            text="Modern live console for microphone monitoring and WAV replay.",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color=MUTED,
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=22)

        controls = self._card(sidebar)
        controls.grid(row=2, column=0, sticky="ew", padx=18, pady=14)
        controls.grid_columnconfigure(0, weight=1)

        self._source_var = tk.StringVar(value="Live")
        ctk.CTkLabel(
            controls, text="Input source", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(16, 8))
        self._source_switch = ctk.CTkSegmentedButton(
            controls,
            values=["Live", "File"],
            variable=self._source_var,
            command=self._on_source_changed,
            selected_color=ACCENT,
            selected_hover_color="#6A9EFF",
            unselected_color=PANEL_ALT,
            unselected_hover_color="#1D2942",
            text_color=TEXT,
        )
        self._source_switch.grid(row=1, column=0, sticky="ew", padx=16)

        self._file_frame = ctk.CTkFrame(controls, fg_color="transparent")
        self._file_frame.grid_columnconfigure(0, weight=1)
        self._file_path_var = tk.StringVar()
        self._file_entry = ctk.CTkEntry(
            self._file_frame,
            textvariable=self._file_path_var,
            placeholder_text="Select a WAV file",
            height=40,
            fg_color=PANEL_ALT,
            border_color=BORDER,
            text_color=TEXT,
        )
        self._file_entry.grid(row=0, column=0, sticky="ew")
        self._file_button = ctk.CTkButton(
            self._file_frame, text="Browse", width=84, height=40,
            fg_color=ACCENT, hover_color="#6A9EFF", command=self._browse_wav,
        )
        self._file_button.grid(row=0, column=1, padx=(10, 0))

        ctk.CTkLabel(
            controls, text="Classifier", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(16, 8))
        self._classifier_var = tk.StringVar(value=config.CLASSIFIER_MODE)
        self._classifier_menu = ctk.CTkOptionMenu(
            controls,
            variable=self._classifier_var,
            values=["cnn", "yamnet", "yamnet_head", "ensemble", "ensemble_head"],
            fg_color=PANEL_ALT,
            button_color=PANEL_ALT,
            button_hover_color="#1D2942",
            dropdown_fg_color=PANEL_ALT,
            text_color=TEXT,
            dropdown_text_color=TEXT,
            command=lambda _: self._set_metric("classifier", self._classifier_var.get().upper()),
        )
        self._classifier_menu.grid(row=4, column=0, sticky="ew", padx=16)

        ctk.CTkLabel(
            controls,
            text=(
                f"Model: {Path(config.CNN_MODEL_PATH).name}\n"
                f"Feature: {config.CNN_FEATURE_TYPE}\n"
                f"Threshold: {config.CNN_DECISION_THRESHOLD:.2f}"
            ),
            justify="left",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).grid(row=5, column=0, sticky="w", padx=16, pady=(14, 12))

        tuning = self._card(sidebar)
        tuning.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 14))
        tuning.grid_columnconfigure(0, weight=1)

        self._mult_var = tk.DoubleVar(value=config.ENERGY_MULTIPLIER)
        self._dead_var = tk.DoubleVar(value=config.MIN_RETRIGGER_SEC)
        self._mult_label = ctk.CTkLabel(tuning, text="", text_color=TEXT)
        self._dead_label = ctk.CTkLabel(tuning, text="", text_color=TEXT)
        self._slider_block(tuning, 0, "Trigger multiplier", self._mult_label)
        self._mult_slider = ctk.CTkSlider(
            tuning, from_=0.5, to=10.0, number_of_steps=95,
            variable=self._mult_var, progress_color=ACCENT_2,
            button_color=ACCENT, button_hover_color="#6A9EFF",
            command=self._on_mult_changed,
        )
        self._mult_slider.grid(row=2, column=0, sticky="ew", padx=16, pady=(4, 10))
        self._slider_block(tuning, 3, "Re-trigger guard", self._dead_label)
        self._dead_slider = ctk.CTkSlider(
            tuning, from_=0.1, to=3.0, number_of_steps=58,
            variable=self._dead_var, progress_color=ACCENT_2,
            button_color=ACCENT, button_hover_color="#6A9EFF",
            command=self._on_dead_changed,
        )
        self._dead_slider.grid(row=5, column=0, sticky="ew", padx=16, pady=(4, 8))
        ctk.CTkFrame(tuning, fg_color=BORDER, height=1).grid(
            row=6, column=0, sticky="ew", padx=16, pady=(4, 4)
        )
        self._bypass_var = tk.BooleanVar(value=False)
        self._bypass_check = ctk.CTkCheckBox(
            tuning,
            text="Bypass Stage 1  (classify all windows)",
            variable=self._bypass_var,
            command=self._on_bypass_changed,
            text_color=TEXT,
            fg_color=ACCENT,
            hover_color="#6A9EFF",
            checkmark_color="#08101E",
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        self._bypass_check.grid(row=7, column=0, sticky="w", padx=16, pady=(0, 14))
        self._on_mult_changed(self._mult_var.get())
        self._on_dead_changed(self._dead_var.get())

        transport = self._card(sidebar)
        transport.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 18))
        transport.grid_columnconfigure(0, weight=1)
        button_row = ctk.CTkFrame(transport, fg_color="transparent")
        button_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        button_row.grid_columnconfigure((0, 1), weight=1)
        self._start_btn = ctk.CTkButton(
            button_row, text="Start", fg_color=SAFE, hover_color="#39BF68",
            text_color="#081208", command=self._on_start,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._stop_btn = ctk.CTkButton(
            button_row, text="Stop", fg_color=DANGER, hover_color="#FF5959",
            text_color="#220808", state="disabled", command=self._on_stop,
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self._transport_caption = ctk.CTkLabel(
            transport, text="", justify="left", text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self._transport_caption.grid(row=1, column=0, sticky="w", padx=16, pady=(4, 8))
        self._transport_bar = ctk.CTkProgressBar(
            transport, fg_color=PANEL_ALT, progress_color=ACCENT, height=14,
        )
        self._transport_bar.grid(row=2, column=0, sticky="ew", padx=16)
        self._transport_bar.set(0)
        self._transport_value = ctk.CTkLabel(
            transport, text="00:00 / 00:00", text_color=MUTED,
            font=ctk.CTkFont(family="Cascadia Code", size=12),
        )
        self._transport_value.grid(row=3, column=0, sticky="w", padx=16, pady=(8, 6))
        self._mark_btn = ctk.CTkButton(
            transport,
            text="▶  Mark Gunshot  [M]",
            fg_color="#7B2FBE",
            hover_color="#9B4FDE",
            text_color=TEXT,
            state="disabled",
            height=38,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            command=self._on_mark_gunshot,
        )
        self._mark_btn.grid(row=4, column=0, sticky="ew", padx=16, pady=(4, 4))
        self._marks_label = ctk.CTkLabel(
            transport,
            text="Marks: 0  |  Press M during file playback",
            text_color=MUTED,
            font=ctk.CTkFont(family="Cascadia Code", size=11),
        )
        self._marks_label.grid(row=5, column=0, sticky="w", padx=16, pady=(0, 14))
        self._on_source_changed("Live")

    def _slider_block(self, parent, row: int, title: str, value_label) -> None:
        ctk.CTkLabel(
            parent, text=title, text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).grid(row=row, column=0, sticky="w", padx=16, pady=(16, 4))
        value_label.grid(row=row + 1, column=0, sticky="e", padx=16)

    def _build_main(self) -> None:
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)
        main.grid_rowconfigure(3, weight=1)

        header = self._card(main)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            header,
            text="Detection Operations Console",
            text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=24, pady=(20, 4))
        ctk.CTkLabel(
            header,
            text="Premade files replay through the exact same live trigger and inference path.",
            text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 20))
        self._status_chip = ctk.CTkLabel(
            header, text="Idle", width=120, height=32, corner_radius=999,
            fg_color=PANEL_ALT, text_color="#08101E",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        self._status_chip.grid(row=0, column=1, rowspan=2, padx=24, pady=20, sticky="e")

        metrics = ctk.CTkFrame(main, fg_color="transparent")
        metrics.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        for idx in range(4):
            metrics.grid_columnconfigure(idx, weight=1)
        self._metric_card(metrics, 0, "Alerts", "alerts")
        self._metric_card(metrics, 1, "Suspicious", "suspicious")
        self._metric_card(metrics, 2, "Source", "source")
        self._metric_card(metrics, 3, "Classifier", "classifier")

        plot_panel = self._card(main)
        plot_panel.grid(row=2, column=0, sticky="nsew", pady=(0, 14))
        plot_panel.grid_columnconfigure(0, weight=1)
        plot_panel.grid_rowconfigure(1, weight=1)
        self._plot_title_label = ctk.CTkLabel(
            plot_panel, text="Energy timeline", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        )
        self._plot_title_label.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 8))
        self._fig = Figure(figsize=(11, 4.2), dpi=100, facecolor=PANEL)
        self._ax = self._fig.add_subplot(111)
        self._setup_plot()
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_panel)
        self._canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 14))

        activity = self._card(main)
        activity.grid(row=3, column=0, sticky="nsew")
        activity.grid_columnconfigure(0, weight=1)
        activity.grid_rowconfigure(1, weight=1)
        top = ctk.CTkFrame(activity, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        top.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            top, text="Activity stream", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            top, text="Clear", width=72, fg_color=PANEL_ALT,
            hover_color="#1D2942", command=self._clear_activity,
        ).grid(row=0, column=1, sticky="e")
        self._activity_box = ctk.CTkTextbox(
            activity, fg_color=PANEL_ALT, border_width=1, border_color=BORDER,
            text_color=TEXT, font=ctk.CTkFont(family="Cascadia Code", size=12),
        )
        self._activity_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self._activity_box.tag_config("danger", foreground=DANGER)
        self._activity_box.tag_config("safe", foreground=SAFE)
        self._activity_box.tag_config("info", foreground=MUTED)

    def _build_inspector(self) -> None:
        inspector = self._card(self, fg_color="#0F172A", radius=0)
        inspector.grid(row=0, column=2, sticky="nsew")
        inspector.grid_columnconfigure(0, weight=1)
        card = self._card(inspector)
        card.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            card, text="Detection inspector", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 4))
        ctk.CTkLabel(
            card, text="Latest model decision and confidence trace.",
            text_color=MUTED, font=ctk.CTkFont(family="Segoe UI", size=12),
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))
        self._latest_badge = ctk.CTkLabel(
            card, text="NO DETECTION", height=42, corner_radius=14,
            fg_color=PANEL_ALT, text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        )
        self._latest_badge.grid(row=2, column=0, sticky="ew", padx=18)
        self._confidence_bar = ctk.CTkProgressBar(
            card, fg_color=PANEL_ALT, progress_color=ACCENT, height=16,
        )
        self._confidence_bar.grid(row=3, column=0, sticky="ew", padx=18, pady=(18, 8))
        self._confidence_bar.set(0)
        self._confidence_value = ctk.CTkLabel(
            card, text="Confidence 0.000", text_color=TEXT,
            font=ctk.CTkFont(family="Cascadia Code", size=13),
        )
        self._confidence_value.grid(row=4, column=0, sticky="w", padx=18)
        self._detail_label = ctk.CTkLabel(
            card, text="Waiting for a trigger", justify="left", text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        )
        self._detail_label.grid(row=5, column=0, sticky="w", padx=18, pady=(12, 16))
        self._topk_box = ctk.CTkTextbox(
            card, fg_color=PANEL_ALT, border_width=1, border_color=BORDER,
            text_color=TEXT, font=ctk.CTkFont(family="Cascadia Code", size=12),
            height=220,
        )
        self._topk_box.grid(row=6, column=0, sticky="nsew", padx=18, pady=(0, 18))

    def _metric_card(self, parent, column: int, title: str, key: str) -> None:
        card = self._card(parent)
        card.grid(row=0, column=column, sticky="nsew", padx=(0 if column == 0 else 8, 8 if column < 3 else 0))
        ctk.CTkLabel(
            card, text=title, text_color=MUTED,
            font=ctk.CTkFont(family="Segoe UI", size=12),
        ).pack(anchor="w", padx=18, pady=(16, 6))
        value = ctk.CTkLabel(
            card, text="-", text_color=TEXT,
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
        )
        value.pack(anchor="w", padx=18, pady=(0, 16))
        self._metric_labels[key] = value

    def _setup_plot(self) -> None:
        self._ax.set_facecolor(PANEL_ALT)
        for spine in self._ax.spines.values():
            spine.set_color(BORDER)
        self._ax.tick_params(colors=MUTED, labelsize=9)
        self._ax.set_xlabel("Time (s)", color=TEXT)
        self._ax.set_ylabel("Energy", color=TEXT)
        self._ax.grid(color="#20304A", alpha=0.45, linewidth=0.8)
        self._ax.set_xlim(0, PLOT_HISTORY_SEC)
        self._ax.set_ylim(0, 0.05)
        (self._rms_line,) = self._ax.plot([], [], color=RMS, linewidth=1.8)
        (self._thr_line,) = self._ax.plot([], [], color=THRESH, linewidth=1.5, linestyle="--")
        self._markers = self._ax.scatter([], [], s=64, color=DANGER, edgecolors="#FFD7D7", linewidths=1.0, zorder=5)
        self._fig.tight_layout()

    def _set_metric(self, key: str, value: str) -> None:
        if key in self._metric_labels:
            self._metric_labels[key].configure(text=value)

    def _set_status(self, text: str, color: str) -> None:
        self._status_chip.configure(text=text, fg_color=color)

    def _on_source_changed(self, value: str) -> None:
        if value == "File":
            self._file_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(12, 0))
            self._set_metric("source", "FILE")
        else:
            self._file_frame.grid_forget()
            self._set_metric("source", "MIC")
        self._refresh_transport()

    def _browse_wav(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(
            title="Select WAV file",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
        )
        if path:
            self._file_path_var.set(path)
            self._source_name = Path(path).name
            self._refresh_transport()

    def _on_mult_changed(self, value: float) -> None:
        self._mult_label.configure(text=f"{value:.1f}x")
        if self._monitor is not None:
            self._monitor.energy_multiplier = value

    def _on_dead_changed(self, value: float) -> None:
        self._dead_label.configure(text=f"{value:.2f}s")
        if self._monitor is not None:
            self._monitor.min_retrigger_sec = value

    def _build_classifier(self) -> object:
        mode = self._classifier_var.get()
        self._set_metric("classifier", mode.upper())

        def build_yamnet_family(*, prefer_head: bool) -> object:
            if prefer_head:
                head_path = Path(config.YAMNET_HEAD_MODEL_PATH)
                if head_path.exists():
                    model = YAMNetHeadClassifier(
                        head_model_path=str(head_path),
                        decision_threshold=config.YAMNET_HEAD_DECISION_THRESHOLD,
                    )
                    model._ensure_models()
                    return model
                self._log_system(
                    "YAMNet head model not found; falling back to regular YAMNet. "
                    "Train it with: python -m train.train_yamnet_head"
                )

            model = YAMNetClassifier()
            model._ensure_model()
            return model

        if mode == "cnn":
            model = CNNClassifier(
                model_path=str(config.CNN_MODEL_PATH),
                decision_threshold=config.CNN_DECISION_THRESHOLD,
                feature_type=config.CNN_FEATURE_TYPE,
            )
            model._ensure_model()
            return model
        if mode == "yamnet":
            return build_yamnet_family(prefer_head=False)
        if mode == "yamnet_head":
            return build_yamnet_family(prefer_head=True)

        if mode == "ensemble_head":
            yamnet_model = build_yamnet_family(prefer_head=True)
        else:
            yamnet_model = build_yamnet_family(prefer_head=False)
        cnn = CNNClassifier(
            model_path=str(config.CNN_MODEL_PATH),
            decision_threshold=config.CNN_DECISION_THRESHOLD,
            feature_type=config.CNN_FEATURE_TYPE,
        )
        # Keep the YAMNet-family classifier first so ENSEMBLE_WEIGHTS maps to
        # [YAMNet/YAMNet-head, CNN], matching the non-GUI pipeline.
        yamnet = yamnet_model
        cnn._ensure_model()
        return [yamnet, cnn]

    def _on_start(self) -> None:
        if self._is_streaming:
            return

        self._reset_runtime()
        source_mode = self._source_var.get()
        wav_path: Optional[Path] = None
        if source_mode == "File":
            wav_path = Path(self._file_path_var.get().strip())
            if not wav_path.exists():
                self._log_system("Choose a valid WAV file before starting.")
                self._set_status("No File", WARN)
                return

        self._monitor = StreamMonitor(
            energy_multiplier=self._mult_var.get(),
            min_retrigger_sec=self._dead_var.get(),
        )

        try:
            self._log_system("Loading classifier stack...")
            self._classifier = self._build_classifier()
        except Exception as exc:
            logger.exception("Failed to initialize classifier")
            self._log_system(f"Classifier load failed: {exc}")
            self._set_status("Load Failed", WARN)
            return

        self._worker_stop.clear()
        self._worker_thread = threading.Thread(
            target=self._inference_loop,
            name="gui-inference-worker",
            daemon=True,
        )
        self._worker_thread.start()

        self._stream_start = time.monotonic()
        self._is_streaming = True
        self._toggle_controls(running=True)
        self._set_status("Armed", SAFE)

        try:
            if source_mode == "Live":
                self._source_name = "Microphone"
                self._source_duration_sec = 0.0
                self._start_mic_stream()
            else:
                self._source_name = wav_path.name
                self._start_wav_playback(wav_path)
        except Exception as exc:
            logger.exception("Failed to start source")
            self._log_system(f"Could not start source: {exc}")
            self._on_stop()
            self._set_status("Source Error", WARN)
            return

        if source_mode == "Live":
            self._source_name = "Microphone"
        else:
            self._source_name = wav_path.name

        self.after(QUEUE_POLL_MS, self._poll_gui_queue)
        self.after(PLOT_INTERVAL_MS, self._animate_plot)
        self._log_system(f"Stream started on {self._source_name}.")

    def _on_stop(self) -> None:
        if not self._is_streaming and self._sd_stream is None and self._sd_output is None:
            return

        self._is_streaming = False
        self._worker_stop.set()

        for attr in ("_sd_stream", "_sd_output"):
            stream = getattr(self, attr)
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._wav_thread is not None:
            self._wav_thread.join(timeout=2.0)
            self._wav_thread = None

        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            self._worker_thread = None

        self._toggle_controls(running=False)
        self._set_status("Idle", MUTED)
        self._refresh_transport(finalize=True)
        self._log_system(
            f"Stream stopped. Alerts={self._total_alerts}, Suspicious={self._suspicious_alerts}"
        )

    def _toggle_controls(self, *, running: bool) -> None:
        self._start_btn.configure(state="disabled" if running else "normal")
        self._stop_btn.configure(state="normal" if running else "disabled")
        self._classifier_menu.configure(state="disabled" if running else "normal")
        self._source_switch.configure(state="disabled" if running else "normal")
        entry_state = "disabled" if running else "normal"
        self._file_entry.configure(state=entry_state)
        self._file_button.configure(state=entry_state)
        mark_state = "normal" if (running and self._source_var.get() == "File") else "disabled"
        self._mark_btn.configure(state=mark_state)

    def _reset_runtime(self) -> None:
        self._rms_buf.clear()
        self._thr_buf.clear()
        self._time_buf.clear()
        self._marker_times.clear()
        self._marker_levels.clear()
        self._source_duration_sec = 0.0
        self._source_position_sec = 0.0
        self._total_alerts = 0
        self._suspicious_alerts = 0
        self._last_result = None
        self._user_marks.clear()
        self._detection_timestamps.clear()
        self._marks_label.configure(text="Marks: 0  |  Press M during file playback")
        if self._waveform_mode:
            self._waveform_mode = False
            self._playhead_line = None
            self._det_lines.clear()
            self._ax.cla()
            self._setup_plot()
            self._plot_title_label.configure(text="Energy timeline")
            self._canvas.draw_idle()
        self._det_lines.clear()
        self._set_metric("alerts", "0")
        self._set_metric("suspicious", "0")
        self._latest_badge.configure(text="NO DETECTION", fg_color=PANEL_ALT, text_color=TEXT)
        self._confidence_bar.set(0)
        self._confidence_value.configure(text="Confidence 0.000")
        self._detail_label.configure(text="Waiting for a trigger")
        self._topk_box.delete("1.0", "end")

    def _start_mic_stream(self) -> None:
        if sd is None:
            self._log_system("sounddevice is not installed. Microphone mode is unavailable.")
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

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        del frames, time_info
        if status:
            logger.warning("sounddevice status: %s", status)
        self._feed_and_record(indata[:, 0].astype(np.float32))

    def _setup_file_waveform_plot(self, waveform: np.ndarray, sr: int) -> None:
        """Replace the rolling energy plot with a full-file waveform + playhead."""
        self._ax.cla()
        self._ax.set_facecolor(PANEL_ALT)
        for spine in self._ax.spines.values():
            spine.set_color(BORDER)
        self._ax.tick_params(colors=MUTED, labelsize=9)
        self._ax.set_xlabel("Time (s)", color=TEXT)
        self._ax.set_ylabel("Amplitude", color=TEXT)
        self._ax.grid(color="#20304A", alpha=0.45, linewidth=0.8)

        # Downsample to ~5000 points so the plot stays fast
        n_pts = 5000
        step = max(1, len(waveform) // n_pts)
        t_ds = np.arange(0, len(waveform), step) / float(sr)
        wav_ds = waveform[::step]

        duration = len(waveform) / float(sr)
        self._ax.set_xlim(0, duration)
        peak = float(np.max(np.abs(wav_ds))) if len(wav_ds) else 0.05
        self._ax.set_ylim(-max(peak * 1.15, 0.05), max(peak * 1.15, 0.05))
        self._ax.plot(t_ds, wav_ds, color=RMS, linewidth=0.5, alpha=0.85)

        # Moving playhead cursor
        self._playhead_line = self._ax.axvline(
            0, color=DANGER, linewidth=2.2, alpha=0.9, zorder=6, label="Playhead"
        )

        self._det_lines.clear()
        self._fig.tight_layout()
        self._canvas.draw_idle()
        self._waveform_mode = True
        self._plot_title_label.configure(text="Waveform + Detections  (red = suspicious · green = clear)")

    def _start_wav_playback(self, wav_path: Path) -> None:
        waveform, sr = load_wav(wav_path)
        self._source_duration_sec = len(waveform) / float(sr)
        self._source_position_sec = 0.0

        # Switch the main plot to synchronized waveform view
        self._setup_file_waveform_plot(waveform, sr)

        if sd is not None:
            try:
                self._sd_output = sd.OutputStream(
                    samplerate=sr,
                    channels=1,
                    dtype="float32",
                )
                self._sd_output.start()
            except Exception as exc:
                logger.warning("Could not open playback device: %s", exc)
                self._sd_output = None

        self._wav_thread = threading.Thread(
            target=self._wav_feeder,
            args=(waveform, sr),
            name="wav-replay-thread",
            daemon=True,
        )
        self._wav_thread.start()

    def _wav_feeder(self, waveform: np.ndarray, sr: int) -> None:
        chunk_size = 1024
        chunk_duration = chunk_size / float(sr)
        offset = 0
        while offset < len(waveform) and self._is_streaming:
            end = min(offset + chunk_size, len(waveform))
            chunk = waveform[offset:end]
            self._source_position_sec = end / float(sr)
            self._feed_and_record(chunk)
            if self._sd_output is not None:
                try:
                    self._sd_output.write(chunk.reshape(-1, 1))
                except Exception:
                    pass
            else:
                time.sleep(chunk_duration)
            offset = end

        self._source_position_sec = self._source_duration_sec
        if self._is_streaming:
            self.after(100, self._on_stop)
            if self._user_marks:
                self.after(800, self._show_accuracy_report)

    def _feed_and_record(self, chunk: np.ndarray) -> None:
        if self._monitor is None:
            return

        self._monitor.feed(chunk)
        rms = self._monitor._rms_history[-1] if self._monitor._rms_history else 0.0
        baseline = self._monitor._percentile_baseline()
        threshold = baseline * self._monitor.energy_multiplier
        elapsed = time.monotonic() - self._stream_start if self._stream_start else 0.0

        self._rms_buf.append(rms)
        self._thr_buf.append(threshold)
        self._time_buf.append(elapsed)
        if len(self._rms_buf) > self._plot_maxlen:
            trim = len(self._rms_buf) - self._plot_maxlen
            del self._rms_buf[:trim]
            del self._thr_buf[:trim]
            del self._time_buf[:trim]

    def _classify_trigger(self, trigger) -> ClassificationResult:
        if isinstance(self._classifier, list):
            results = [
                model.classify(
                    waveform=trigger.window,
                    timestamp=trigger.timestamp_sec,
                    onset_index=trigger.onset_index,
                )
                for model in self._classifier
            ]
            # _build_classifier returns [YAMNet-family, CNN], matching config's
            # ensemble weights.
            yamnet_result, cnn_result = results
            weighted_conf = sum(
                weight * result.confidence
                for weight, result in zip(config.ENSEMBLE_WEIGHTS, results)
            )
            primary = yamnet_result
            primary.is_suspicious = weighted_conf >= config.ENSEMBLE_THRESHOLD
            primary.confidence = float(weighted_conf)
            primary.label = "GUNSHOT" if primary.is_suspicious else "NOGUN"
            if primary.confidence >= 0.85:
                primary.severity = "HIGH"
            elif primary.confidence >= 0.6:
                primary.severity = "MEDIUM"
            else:
                primary.severity = "LOW"
            primary.top_k = [
                {"class": "Ensemble weighted", "score": float(weighted_conf)},
                {"class": "YAMNet score", "score": float(yamnet_result.confidence)},
                {"class": "CNN score", "score": float(cnn_result.confidence)},
            ]
            return primary

        return self._classifier.classify(
            waveform=trigger.window,
            timestamp=trigger.timestamp_sec,
            onset_index=trigger.onset_index,
        )

    def _inference_loop(self) -> None:
        while not self._worker_stop.is_set():
            if self._monitor is None:
                time.sleep(0.05)
                continue
            try:
                trigger = self._monitor.trigger_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._gui_queue.put(self._classify_trigger(trigger))
            except Exception:
                logger.exception("Inference failed for trigger at %.3fs", trigger.timestamp_sec)

    def _poll_gui_queue(self) -> None:
        while True:
            try:
                result = self._gui_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_result(result)

        if self._is_streaming:
            self.after(QUEUE_POLL_MS, self._poll_gui_queue)

    def _handle_result(self, result: ClassificationResult) -> None:
        self._last_result = result
        self._total_alerts += 1
        if result.is_suspicious:
            self._suspicious_alerts += 1
            self._detection_timestamps.append(result.timestamp)

        self._set_metric("alerts", str(self._total_alerts))
        self._set_metric("suspicious", str(self._suspicious_alerts))

        level = (self._thr_buf[-1] if self._thr_buf else max(result.confidence, 0.01)) * 1.05
        self._marker_times.append(result.timestamp)
        self._marker_levels.append(level)
        self._marker_times = self._marker_times[-MAX_MARKERS:]
        self._marker_levels = self._marker_levels[-MAX_MARKERS:]

        badge_text = "SUSPICIOUS EVENT" if result.is_suspicious else "NON-THREAT EVENT"
        badge_color = DANGER if result.is_suspicious else SAFE
        self._latest_badge.configure(text=badge_text, fg_color=badge_color, text_color="#08101E")
        self._confidence_bar.set(min(max(result.confidence, 0.0), 1.0))
        self._confidence_bar.configure(progress_color=_severity_color(result.severity))
        self._confidence_value.configure(text=f"Confidence {result.confidence:.3f}")
        self._detail_label.configure(
            text=(
                f"Label: {result.label}\n"
                f"Severity: {result.severity}\n"
                f"Timestamp: {result.timestamp:.3f}s\n"
                f"Source: {self._source_name}"
            )
        )

        self._topk_box.delete("1.0", "end")
        if result.top_k:
            lines = [
                f"{idx + 1}. {entry['class']:<24} {entry['score']:.4f}"
                for idx, entry in enumerate(result.top_k[:5])
            ]
            self._topk_box.insert("end", "\n".join(lines))
        else:
            self._topk_box.insert("end", "No alternate class scores available.")

        # Draw a vertical detection line on the waveform (file mode only)
        if self._waveform_mode:
            color = DANGER if result.is_suspicious else SAFE
            lw = 2.0 if result.is_suspicious else 1.2
            ls = "-" if result.is_suspicious else "--"
            line = self._ax.axvline(
                result.timestamp, color=color, linewidth=lw,
                linestyle=ls, alpha=0.85, zorder=5,
            )
            # Small label above the line
            ylim = self._ax.get_ylim()
            label_y = ylim[1] * 0.88
            self._ax.text(
                result.timestamp, label_y,
                f"{'⚠' if result.is_suspicious else '✓'} {result.confidence:.2f}",
                color=color, fontsize=7.5, ha="center", va="top",
                bbox=dict(boxstyle="round,pad=0.15", fc=PANEL, ec=color, alpha=0.85),
                zorder=6,
            )
            self._det_lines.append(line)
            self._canvas.draw_idle()

        self._log_result(result)
        if self._log_path:
            with self._log_path.open("a", encoding="utf-8") as fh:
                fh.write(result.to_json() + "\n")

    def _log_result(self, result: ClassificationResult) -> None:
        tag = "danger" if result.is_suspicious else "safe"
        kind = "SUSPICIOUS" if result.is_suspicious else "SAFE"
        line = (
            f"[{result.timestamp:7.2f}s] {kind:<10} "
            f"{result.label:<12} conf={result.confidence:.3f} "
            f"severity={result.severity}\n"
        )
        self._activity_box.insert("end", line, tag)
        if result.top_k:
            summary = " | ".join(
                f"{entry['class']} {entry['score']:.2f}" for entry in result.top_k[:3]
            )
            self._activity_box.insert("end", f"           top: {summary}\n", "info")
        self._activity_box.see("end")
        line_count = int(self._activity_box.index("end-1c").split(".")[0])
        if line_count > MAX_LOG_LINES:
            self._activity_box.delete("1.0", f"{line_count - MAX_LOG_LINES}.0")

    def _log_system(self, text: str) -> None:
        self._activity_box.insert("end", f"[system] {text}\n", "info")
        self._activity_box.see("end")

    def _refresh_transport(self, finalize: bool = False) -> None:
        if self._is_streaming and self._source_var.get() == "Live":
            elapsed = time.monotonic() - self._stream_start if self._stream_start else 0.0
            self._transport_caption.configure(
                text="Live microphone\nMonitoring and plotting the trigger stream in real time"
            )
            self._transport_value.configure(text=f"{_fmt_time(elapsed)} / LIVE")
            self._transport_bar.set(1.0)
            self._transport_bar.configure(progress_color=ACCENT_2)
            return

        if self._source_var.get() == "File" and self._source_duration_sec > 0:
            position = self._source_duration_sec if finalize else min(
                self._source_position_sec, self._source_duration_sec
            )
            progress = position / self._source_duration_sec if self._source_duration_sec else 0.0
            self._transport_caption.configure(
                text=f"{self._source_name}\nReplaying the file through the same live detector path"
            )
            self._transport_value.configure(
                text=f"{_fmt_time(position)} / {_fmt_time(self._source_duration_sec)}"
            )
            self._transport_bar.set(progress)
            self._transport_bar.configure(progress_color=ACCENT)
            return

        source_label = "Microphone" if self._source_var.get() == "Live" else (self._source_name or "WAV replay")
        self._transport_caption.configure(
            text=f"{source_label}\nReady for live monitoring or real-time file replay"
        )
        self._transport_value.configure(text="00:00 / 00:00")
        self._transport_bar.set(0)
        self._transport_bar.configure(progress_color=ACCENT)

    def _animate_plot(self) -> None:
        if not self._is_streaming:
            return

        if self._waveform_mode:
            # Waveform sync view: just move the playhead cursor
            if self._playhead_line is not None:
                pos = self._source_position_sec
                self._playhead_line.set_xdata([pos, pos])
            self._canvas.draw_idle()
        else:
            # Rolling energy timeline (live mic or energy-only view)
            if self._time_buf:
                self._rms_line.set_data(self._time_buf, self._rms_buf)
                self._thr_line.set_data(self._time_buf, self._thr_buf)

                current_max = self._time_buf[-1]
                x_min = max(0.0, current_max - PLOT_HISTORY_SEC)
                self._ax.set_xlim(x_min, x_min + PLOT_HISTORY_SEC)

                visible = [
                    value
                    for time_list, value_list in ((self._time_buf, self._rms_buf), (self._time_buf, self._thr_buf))
                    for time_value, value in zip(time_list, value_list)
                    if time_value >= x_min
                ]
                self._ax.set_ylim(0, max((max(visible) * 1.25) if visible else 0.05, 0.01))

                markers = [
                    (time_value, level)
                    for time_value, level in zip(self._marker_times, self._marker_levels)
                    if time_value >= x_min
                ]
                self._markers.set_offsets(np.array(markers) if markers else np.empty((0, 2)))
                self._canvas.draw_idle()

        if self._last_result is None:
            self._set_status("Armed", SAFE)
        elif self._last_result.is_suspicious:
            self._set_status("Threat", DANGER)
        else:
            self._set_status("Monitoring", SAFE)

        self._refresh_transport()
        self.after(PLOT_INTERVAL_MS, self._animate_plot)

    def _clear_activity(self) -> None:
        self._activity_box.delete("1.0", "end")

    def _on_bypass_changed(self) -> None:
        bypass = self._bypass_var.get()
        if bypass:
            self._mult_slider.configure(state="disabled")
            if self._monitor is not None:
                self._monitor.energy_multiplier = 0.01
            self._log_system(
                "Stage 1 bypass ON — every window sent to classifier (~1 per dead-time interval)."
            )
        else:
            self._mult_slider.configure(state="normal")
            restored = self._mult_var.get()
            if self._monitor is not None:
                self._monitor.energy_multiplier = restored
            self._log_system(f"Stage 1 bypass OFF — energy trigger restored to {restored:.1f}x.")

    def _on_mark_gunshot(self) -> None:
        if not self._is_streaming or self._source_var.get() != "File":
            return
        ts = self._source_position_sec
        self._user_marks.append(ts)
        n = len(self._user_marks)
        self._marks_label.configure(text=f"Marks: {n}  |  Last at {ts:.2f}s")
        self._log_system(f"[MARK] Gunshot marked at {ts:.3f}s  ({n} total)")
        self._mark_btn.configure(fg_color=DANGER)
        self.after(220, lambda: self._mark_btn.configure(fg_color="#7B2FBE"))

    def _show_accuracy_report(self) -> None:
        TOLERANCE = 1.0  # seconds — window for matching a mark to a detection
        user_marks = sorted(self._user_marks)
        detections = sorted(self._detection_timestamps)
        if not user_marks:
            return

        matched_marks: set = set()
        matched_dets: set = set()
        for i, t_user in enumerate(user_marks):
            for j, t_det in enumerate(detections):
                if j not in matched_dets and abs(t_user - t_det) <= TOLERANCE:
                    matched_marks.add(i)
                    matched_dets.add(j)
                    break

        tp = len(matched_marks)
        fn = len(user_marks) - tp
        fp = len(detections) - len(matched_dets)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        lines = [
            "═" * 30,
            "  MANUAL ACCURACY REPORT",
            "═" * 30,
            f"  User marks:     {len(user_marks):>3}",
            f"  Model fires:    {len(detections):>3}  (suspicious)",
            "─" * 30,
            f"  True Positives:  {tp:>3}  ✓ caught",
            f"  False Negatives: {fn:>3}  ✗ missed",
            f"  False Positives: {fp:>3}  ? spurious",
            "─" * 30,
            f"  Recall:    {recall:>6.1%}",
            f"  Precision: {precision:>6.1%}",
            f"  F1 Score:  {f1:>6.1%}",
            "─" * 30,
            f"  Match window: ±{TOLERANCE:.1f}s",
            "═" * 30,
        ]
        self._topk_box.delete("1.0", "end")
        self._topk_box.insert("end", "\n".join(lines))
        self._detail_label.configure(
            text=(
                f"Recall:    {recall:.1%}  ({tp}/{tp + fn} marks caught)\n"
                f"Precision: {precision:.1%}  ({tp}/{tp + fp} fires correct)\n"
                f"F1 Score:  {f1:.1%}\n"
                f"Tolerance: ±{TOLERANCE:.1f}s"
            )
        )
        badge_color = SAFE if recall >= 0.70 else (WARN if recall >= 0.40 else DANGER)
        self._latest_badge.configure(
            text=f"RECALL  {recall:.0%}",
            fg_color=badge_color,
            text_color="#08101E",
        )
        self._log_system(
            f"Accuracy: Recall={recall:.1%}  Precision={precision:.1%}  F1={f1:.1%}"
            f"  (TP={tp} FN={fn} FP={fp})"
        )

    def _on_close(self) -> None:
        if self._is_streaming or self._sd_stream is not None or self._sd_output is not None:
            self._on_stop()
        self.destroy()


def launch_gui(log_path: Optional[str] = None) -> None:
    """Create and run the graphical dashboard."""
    DetectionGUI(log_path=log_path).mainloop()
