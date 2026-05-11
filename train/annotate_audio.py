"""
annotate_audio.py – VOICe annotation verification tool.

Two modes
---------
[DEFAULT]  Active-learning mode
    Scores every annotated gunshot with the YAMNet Head and queues only the
    segments the model got wrong (false negatives) or is uncertain about.
    Good for finding model blind spots after a training run.

[--audit-all]  Full ground-truth audit
    Shows every annotated "gunshot" segment in file order so you can verify
    whether each one is truly a gunshot.  Maintains a persistent state file
    (clean/annotation_audit/audit_state.json) so sessions stack: stop any
    time and the next run picks up where you left off.

Usage
-----
# Active-learning pass (scores with YAMNet Head):
    python -m train.annotate_audio [--max-segments 100] [--threshold 0.45]

# Ground-truth audit (no model scoring, fast startup):
    python -m train.annotate_audio --audit-all [--session-segments 200]
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")

import customtkinter as ctk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    import sounddevice as sd
    _HAVE_AUDIO = True
except ImportError:
    sd = None  # type: ignore[assignment]
    _HAVE_AUDIO = False

from impulsive_sound_detection import config
from impulsive_sound_detection.data_loader import (
    AnnotationEntry,
    load_wav,
    parse_annotation,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Visual constants (matches gui.py palette)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#0B1020"
PANEL    = "#111827"
BORDER   = "#24324D"
TEXT     = "#E7EEF8"
MUTED    = "#8EA1BD"
ACCENT   = "#4F8CFF"
SAFE     = "#44D17A"
WARN     = "#F5B942"
DANGER   = "#FF6B6B"
WAVE_CLR = "#66D9EF"

CONTEXT_SEC = 1.2   # audio before/after each event
OUTPUT_DIR  = config.VOICE_DATASET_DIR / "annotation_corrected"
AUDIT_STATE_PATH = config.VOICE_DATASET_DIR / "annotation_audit" / "audit_state.json"

_AUDIT_POSITIVE_LABELS = frozenset({"gunshot"})  # only audit gunshot annotations


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReviewSegment:
    """One annotated event queued for user review."""
    wav_path:           Path
    center_sec:         float
    ann_start_sec:      float
    ann_end_sec:        float
    ann_label:          str
    model_score:        float   # -1.0 = not scored (audit mode)
    threshold:          float
    segment_type:       str     # "fn" | "uncertain" | "audit"
    context_waveform:   np.ndarray = field(repr=False)
    sample_rate:        int   = 16_000
    context_start_sec:  float = 0.0


@dataclass
class AnnotationDecision:
    wav_path:         Path
    event_start_sec:  float
    event_end_sec:    float
    user_label:       str     # "gunshot" | "not_gunshot" | "skip"
    model_score:      float
    original_label:   str
    segment_type:     str


# ─────────────────────────────────────────────────────────────────────────────
# Persistent audit state (cross-session)
# ─────────────────────────────────────────────────────────────────────────────

class AuditState:
    """
    JSON-backed state file that tracks which VOICe segments have been reviewed
    across sessions so you never see the same segment twice.

    Stored at: clean/annotation_audit/audit_state.json
    """

    def __init__(self) -> None:
        AUDIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._reviewed: dict = {}   # key → {"label": str, "original": str}
        self._load()

    # ── serialisation ────────────────────────────────────────────────────────

    @staticmethod
    def _key(wav_name: str, start: float, end: float) -> str:
        return f"{wav_name}__{start:.4f}__{end:.4f}"

    def _load(self) -> None:
        if AUDIT_STATE_PATH.exists():
            with AUDIT_STATE_PATH.open(encoding="utf-8") as fh:
                data = json.load(fh)
            self._reviewed = data.get("reviewed", {})
            logger.info(
                "Audit state loaded: %d segments reviewed (confirmed=%d rejected=%d)",
                self.n_total, self.n_confirmed, self.n_rejected,
            )

    def save(self) -> None:
        data = {"schema_version": 1, "reviewed": self._reviewed}
        with AUDIT_STATE_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        logger.info("Audit state saved: %d total reviewed", self.n_total)

    # ── public API ────────────────────────────────────────────────────────────

    def is_reviewed(self, wav_name: str, start: float, end: float) -> bool:
        return self._key(wav_name, start, end) in self._reviewed

    def record(self, wav_name: str, start: float, end: float,
               label: str, original: str) -> None:
        self._reviewed[self._key(wav_name, start, end)] = {
            "label": label,
            "original": original,
        }

    @property
    def n_total(self) -> int:
        return len(self._reviewed)

    @property
    def n_confirmed(self) -> int:
        return sum(1 for v in self._reviewed.values() if v["label"] == "gunshot")

    @property
    def n_rejected(self) -> int:
        return sum(1 for v in self._reviewed.values() if v["label"] == "not_gunshot")

    @property
    def n_skipped(self) -> int:
        return sum(1 for v in self._reviewed.values() if v["label"] == "skip")


# ─────────────────────────────────────────────────────────────────────────────
# Session management
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationSession:
    """Holds the review queue and accumulates decisions for one GUI session."""

    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        audit_state: Optional[AuditState] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue:       List[ReviewSegment]     = []
        self.decisions:   List[AnnotationDecision] = []
        self.current_idx: int = 0
        self.audit_state: Optional[AuditState] = audit_state

    @property
    def total(self) -> int:
        return len(self.queue)

    @property
    def current(self) -> Optional[ReviewSegment]:
        if self.current_idx < len(self.queue):
            return self.queue[self.current_idx]
        return None

    def record(self, label: str) -> None:
        seg = self.current
        if seg is None:
            return
        self.decisions.append(AnnotationDecision(
            wav_path=seg.wav_path,
            event_start_sec=seg.ann_start_sec,
            event_end_sec=seg.ann_end_sec,
            user_label=label,
            model_score=seg.model_score,
            original_label=seg.ann_label,
            segment_type=seg.segment_type,
        ))
        # Update cross-session state (skip = reviewed but tentative; still saved)
        if self.audit_state is not None:
            self.audit_state.record(
                seg.wav_path.name, seg.ann_start_sec, seg.ann_end_sec,
                label, seg.ann_label,
            )
        self.current_idx += 1

    def save(self) -> Dict[Path, int]:
        """
        Write annotation_corrected TSV files for every file that has at least
        one 'not_gunshot' decision (the only case where corrections matter).

        Files where everything was confirmed stay identical to the original, so
        we skip them — the training fallback picks them up automatically.
        """
        by_file: Dict[Path, List[AnnotationDecision]] = {}
        for dec in self.decisions:
            by_file.setdefault(dec.wav_path, []).append(dec)

        saved: Dict[Path, int] = {}

        for wav_path, decisions in by_file.items():
            has_rejections = any(d.user_label == "not_gunshot" for d in decisions)
            if not has_rejections:
                # Nothing changed for this file — skip writing to avoid noise
                continue

            ann_path = config.VOICE_ANNOTATION_DIR / wav_path.with_suffix(".txt").name
            originals: List[AnnotationEntry] = []
            if ann_path.exists():
                originals = parse_annotation(ann_path)

            merged = list(originals)
            for dec in decisions:
                if dec.user_label == "gunshot":
                    # Confirmed — already present, nothing to add
                    already = any(
                        abs(a.start_sec - dec.event_start_sec) < 0.05
                        and a.label == "gunshot"
                        for a in originals
                    )
                    if not already:
                        merged.append(AnnotationEntry(
                            dec.event_start_sec, dec.event_end_sec, "gunshot"
                        ))
                elif dec.user_label == "not_gunshot":
                    # Remove — strip both "gunshot" and "glassbreak" labels
                    merged = [
                        a for a in merged
                        if not (
                            abs(a.start_sec - dec.event_start_sec) < 0.05
                            and a.label in ("gunshot", "glassbreak")
                        )
                    ]

            merged.sort(key=lambda a: a.start_sec)
            out_path = self.output_dir / wav_path.with_suffix(".txt").name
            with out_path.open("w", encoding="utf-8", newline="") as fh:
                for ann in merged:
                    fh.write(f"{ann.start_sec:.4f}\t{ann.end_sec:.4f}\t{ann.label}\n")

            n_gun = sum(1 for a in merged if a.label == "gunshot")
            saved[wav_path] = n_gun
            logger.info("Saved %s → %d gunshot events", out_path.name, n_gun)

        # Always persist the cross-session audit state
        if self.audit_state is not None:
            self.audit_state.save()

        return saved


# ─────────────────────────────────────────────────────────────────────────────
# Queue builders
# ─────────────────────────────────────────────────────────────────────────────

def build_audit_queue(
    audit_state: AuditState,
    max_segments: int = 200,
) -> List[ReviewSegment]:
    """
    Queue unreviewed annotated 'gunshot' segments in file order for ground-truth
    verification.  Does NOT load YAMNet — starts up in seconds.
    """
    wav_files = sorted(config.VOICE_AUDIO_DIR.glob("*.wav"))
    queue: List[ReviewSegment] = []
    n_already = 0

    for wav_path in wav_files:
        if len(queue) >= max_segments:
            break

        ann_path = config.VOICE_ANNOTATION_DIR / wav_path.with_suffix(".txt").name
        if not ann_path.exists():
            continue

        try:
            waveform, sr = load_wav(wav_path)
        except Exception as exc:
            logger.warning("Cannot load %s: %s", wav_path.name, exc)
            continue

        annotations = parse_annotation(ann_path)
        n_samples = len(waveform)

        for ann in annotations:
            if ann.label not in _AUDIT_POSITIVE_LABELS:
                continue

            if audit_state.is_reviewed(wav_path.name, ann.start_sec, ann.end_sec):
                n_already += 1
                continue

            center_sec = (ann.start_sec + ann.end_sec) / 2.0
            ctx_s = max(0.0, center_sec - CONTEXT_SEC)
            ctx_e = min(n_samples / sr, center_sec + CONTEXT_SEC)
            ctx_wav = waveform[int(ctx_s * sr): int(ctx_e * sr)].copy()

            queue.append(ReviewSegment(
                wav_path=wav_path,
                center_sec=center_sec,
                ann_start_sec=ann.start_sec,
                ann_end_sec=ann.end_sec,
                ann_label=ann.label,
                model_score=-1.0,   # not scored in audit mode
                threshold=config.YAMNET_HEAD_DECISION_THRESHOLD,
                segment_type="audit",
                context_waveform=ctx_wav,
                sample_rate=sr,
                context_start_sec=ctx_s,
            ))

            if len(queue) >= max_segments:
                break

    logger.info(
        "Audit queue ready: %d queued, %d already reviewed (%d total in state)",
        len(queue), n_already, audit_state.n_total,
    )
    return queue


def build_review_queue(
    threshold: float = config.YAMNET_HEAD_DECISION_THRESHOLD,
    uncertain_margin: float = 0.15,
    max_segments: int = 200,
    include_uncertain: bool = True,
) -> List[ReviewSegment]:
    """Score every annotated gunshot with YAMNet Head; queue FNs and uncertain."""
    from impulsive_sound_detection.classifier import YAMNetHeadClassifier

    logger.info("Loading YAMNet Head …")
    clf = YAMNetHeadClassifier(decision_threshold=threshold)
    clf._ensure_models()

    wav_files = sorted(config.VOICE_AUDIO_DIR.glob("*.wav"))
    logger.info("Scanning %d VOICe audio files …", len(wav_files))

    queue: List[ReviewSegment] = []

    for wav_path in wav_files:
        if len(queue) >= max_segments:
            break

        ann_path = config.VOICE_ANNOTATION_DIR / wav_path.with_suffix(".txt").name
        if not ann_path.exists():
            continue

        try:
            waveform, sr = load_wav(wav_path)
        except Exception as exc:
            logger.warning("Cannot load %s: %s", wav_path.name, exc)
            continue

        annotations = parse_annotation(ann_path)
        n_samples = len(waveform)

        for ann in annotations:
            if ann.label not in _AUDIT_POSITIVE_LABELS:
                continue

            center_sec = (ann.start_sec + ann.end_sec) / 2.0
            half = config.YAMNET_WINDOW_SEC / 2.0
            ws = max(0.0, center_sec - half)
            we = min(n_samples / sr, center_sec + half)
            win = waveform[int(ws * sr): int(we * sr)].copy()

            if len(win) < int(sr * 0.2):
                continue

            if len(win) < config.YAMNET_WINDOW_SAMPLES:
                win = np.pad(win, (0, config.YAMNET_WINDOW_SAMPLES - len(win)))

            try:
                result = clf.classify(win)
                score = result.confidence
            except Exception as exc:
                logger.warning("Scoring failed %s @ %.1fs: %s", wav_path.name, center_sec, exc)
                continue

            is_fn = score < threshold
            is_uncertain = (
                include_uncertain
                and (threshold - uncertain_margin) <= score <= (threshold + uncertain_margin)
            )

            if not (is_fn or is_uncertain):
                continue

            ctx_s = max(0.0, center_sec - CONTEXT_SEC)
            ctx_e = min(n_samples / sr, center_sec + CONTEXT_SEC)
            ctx_wav = waveform[int(ctx_s * sr): int(ctx_e * sr)].copy()

            queue.append(ReviewSegment(
                wav_path=wav_path,
                center_sec=center_sec,
                ann_start_sec=ann.start_sec,
                ann_end_sec=ann.end_sec,
                ann_label=ann.label,
                model_score=score,
                threshold=threshold,
                segment_type="fn" if is_fn else "uncertain",
                context_waveform=ctx_wav,
                sample_rate=sr,
                context_start_sec=ctx_s,
            ))

            if len(queue) >= max_segments:
                break

    queue.sort(key=lambda s: (0 if s.segment_type == "fn" else 1, s.model_score))
    n_fn = sum(1 for s in queue if s.segment_type == "fn")
    n_uc = sum(1 for s in queue if s.segment_type == "uncertain")
    logger.info("Review queue ready: %d false-negatives, %d uncertain", n_fn, n_uc)
    return queue


# ─────────────────────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationApp(ctk.CTk):
    """Annotation GUI — works for both active-learning and audit modes."""

    def __init__(self, session: AnnotationSession, audit_mode: bool = False) -> None:
        super().__init__()
        self.session    = session
        self.audit_mode = audit_mode

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        title = (
            "VOICe Ground-Truth Audit  [--audit-all]"
            if audit_mode
            else "VOICe Annotation Tool – Active Learning"
        )
        self.title(title)
        self.geometry("980x700")
        self.configure(fg_color=BG)

        self._build_ui()
        self._refresh()

        for key in ("<space>", "<g>", "<G>"):
            self.bind(key, lambda _e: self._on_label("gunshot"))
        for key in ("<n>", "<N>"):
            self.bind(key, lambda _e: self._on_label("not_gunshot"))
        for key in ("<s>", "<S>"):
            self.bind(key, lambda _e: self._on_label("skip"))
        for key in ("<p>", "<P>"):
            self.bind(key, lambda _e: self._play())
        self.bind("<Escape>", lambda _e: self._on_save_exit())

    # ── Layout ──────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        hdr = ctk.CTkFrame(self, fg_color=PANEL, corner_radius=0, height=52)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        hdr_label = (
            "Ground-Truth Audit — verify every annotated gunshot"
            if self.audit_mode
            else "Active-Learning Annotation — review model errors"
        )
        ctk.CTkLabel(
            hdr, text=hdr_label,
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT,
        ).pack(side="left", padx=20, pady=14)

        self._progress_lbl = ctk.CTkLabel(
            hdr, text="", font=ctk.CTkFont(size=12), text_color=MUTED,
        )
        self._progress_lbl.pack(side="right", padx=20)

        body = ctk.CTkFrame(self, fg_color=BG)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # ── Left: waveform + label buttons ───────────────────────────────
        left = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        self._type_lbl = ctk.CTkLabel(
            left, text="", font=ctk.CTkFont(size=13, weight="bold"), text_color=WARN,
        )
        self._type_lbl.pack(anchor="w", padx=16, pady=(12, 2))

        self._info_lbl = ctk.CTkLabel(
            left, text="", font=ctk.CTkFont(size=11),
            text_color=MUTED, wraplength=600, justify="left",
        )
        self._info_lbl.pack(anchor="w", padx=16, pady=(0, 6))

        self._fig = Figure(figsize=(7.5, 2.6), dpi=96, facecolor=PANEL)
        self._ax  = self._fig.add_subplot(111)
        self._ax.set_facecolor(PANEL)
        self._fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.22)
        self._canvas = FigureCanvasTkAgg(self._fig, master=left)
        self._canvas.get_tk_widget().pack(fill="x", padx=12, pady=4)

        btn_row = ctk.CTkFrame(left, fg_color=PANEL)
        btn_row.pack(padx=16, pady=(8, 14), fill="x")

        ctk.CTkButton(
            btn_row, text="▶ Play  [P]",
            fg_color=ACCENT, hover_color="#3a70e0",
            command=self._play, width=110,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            btn_row, text="✓ Gunshot  [G / Space]",
            fg_color="#1a6b2e", hover_color="#14522a", text_color=SAFE,
            command=lambda: self._on_label("gunshot"), width=170,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row, text="✗ Not Gunshot  [N]",
            fg_color="#6b1a1a", hover_color="#521414", text_color=DANGER,
            command=lambda: self._on_label("not_gunshot"), width=155,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btn_row, text="Skip  [S]",
            fg_color="#2a2a2a", hover_color="#3a3a3a", text_color=MUTED,
            command=lambda: self._on_label("skip"), width=80,
        ).pack(side="left", padx=(4, 0))

        # ── Right: stats + shortcuts ──────────────────────────────────────
        right = ctk.CTkFrame(body, fg_color=PANEL, corner_radius=8, width=210)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        ctk.CTkLabel(
            right, text="Session", font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT,
        ).pack(padx=12, pady=(16, 6))

        self._stat_gun  = ctk.CTkLabel(right, text="Gunshot: 0",     text_color=SAFE,   font=ctk.CTkFont(size=12))
        self._stat_not  = ctk.CTkLabel(right, text="Not gunshot: 0", text_color=DANGER, font=ctk.CTkFont(size=12))
        self._stat_skip = ctk.CTkLabel(right, text="Skipped: 0",     text_color=MUTED,  font=ctk.CTkFont(size=12))
        for lbl in (self._stat_gun, self._stat_not, self._stat_skip):
            lbl.pack(anchor="w", padx=16, pady=3)

        if self.audit_mode and self.session.audit_state:
            ctk.CTkLabel(right, text="─" * 24, text_color=BORDER).pack(padx=12, pady=(8, 4))
            ctk.CTkLabel(right, text="All sessions", text_color=MUTED,
                         font=ctk.CTkFont(size=11)).pack(padx=12, pady=(0, 2))
            self._stat_total_rev = ctk.CTkLabel(
                right, text="Reviewed: 0", text_color=TEXT, font=ctk.CTkFont(size=11),
            )
            self._stat_total_conf = ctk.CTkLabel(
                right, text="Confirmed: 0", text_color=SAFE, font=ctk.CTkFont(size=11),
            )
            self._stat_total_rej = ctk.CTkLabel(
                right, text="Rejected: 0", text_color=DANGER, font=ctk.CTkFont(size=11),
            )
            for lbl in (self._stat_total_rev, self._stat_total_conf, self._stat_total_rej):
                lbl.pack(anchor="w", padx=16, pady=2)
        else:
            self._stat_total_rev = self._stat_total_conf = self._stat_total_rej = None  # type: ignore[assignment]

        ctk.CTkLabel(right, text="─" * 24, text_color=BORDER).pack(padx=12, pady=8)
        ctk.CTkLabel(right, text="Shortcuts", text_color=MUTED, font=ctk.CTkFont(size=11)).pack(padx=12, pady=(0, 4))
        for hint in ["G / Space → Gunshot", "N → Not gunshot", "S → Skip", "P → Play again", "Esc → Save & exit"]:
            ctk.CTkLabel(right, text=hint, text_color="#555", font=ctk.CTkFont(size=10)).pack(anchor="w", padx=18, pady=1)

        ctk.CTkButton(
            right, text="Save & Exit  [Esc]",
            fg_color="#5c0000", hover_color="#3d0000", text_color=DANGER,
            command=self._on_save_exit,
        ).pack(padx=12, pady=16, side="bottom")

    # ── State updates ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        idx   = self.session.current_idx
        total = self.session.total

        if self.audit_mode and self.session.audit_state:
            all_rev = self.session.audit_state.n_total
            self._progress_lbl.configure(
                text=f"Session  {idx + 1} / {total}   |   All-time: {all_rev} reviewed"
            )
        else:
            self._progress_lbl.configure(text=f"Segment  {idx + 1} / {total}")

        seg = self.session.current
        if seg is None:
            self._show_done()
            return

        if seg.segment_type == "audit":
            stype_str   = "ANNOTATED GUNSHOT — Listen and confirm or reject"
            stype_color = ACCENT
        elif seg.segment_type == "fn":
            stype_str   = "FALSE NEGATIVE — model missed this event"
            stype_color = DANGER
        else:
            stype_str   = "UNCERTAIN PREDICTION — near decision threshold"
            stype_color = WARN

        self._type_lbl.configure(text=f"⚠  {stype_str}", text_color=stype_color)

        score_text = (
            f"{seg.model_score:.3f}  (threshold {seg.threshold:.2f})"
            if seg.model_score >= 0
            else "N/A  (fast audit mode — no model scoring)"
        )
        self._info_lbl.configure(
            text=(
                f"File: {seg.wav_path.name}   |   "
                f"Event: {seg.ann_start_sec:.2f} – {seg.ann_end_sec:.2f} s   |   "
                f"Label: {seg.ann_label}   |   "
                f"Score: {score_text}"
            )
        )
        self._draw_waveform(seg)

        n_gun  = sum(1 for d in self.session.decisions if d.user_label == "gunshot")
        n_not  = sum(1 for d in self.session.decisions if d.user_label == "not_gunshot")
        n_skip = sum(1 for d in self.session.decisions if d.user_label == "skip")
        self._stat_gun.configure(text=f"Gunshot: {n_gun}")
        self._stat_not.configure(text=f"Not gunshot: {n_not}")
        self._stat_skip.configure(text=f"Skipped: {n_skip}")

        if self.audit_mode and self.session.audit_state:
            st = self.session.audit_state
            self._stat_total_rev.configure(text=f"Reviewed: {st.n_total}")
            self._stat_total_conf.configure(text=f"Confirmed: {st.n_confirmed}")
            self._stat_total_rej.configure(text=f"Rejected: {st.n_rejected}")

        self._play()

    def _draw_waveform(self, seg: ReviewSegment) -> None:
        ax = self._ax
        ax.clear()
        ax.set_facecolor(PANEL)

        sr = seg.sample_rate
        n  = len(seg.context_waveform)
        t  = np.linspace(seg.context_start_sec,
                         seg.context_start_sec + n / sr, n)

        ax.plot(t, seg.context_waveform, color=WAVE_CLR, linewidth=0.5, alpha=0.9)
        ax.axvspan(seg.ann_start_sec, seg.ann_end_sec, alpha=0.2, color=SAFE)
        ax.axvline(seg.center_sec, color=DANGER, linewidth=1.4, linestyle="--")

        title_color = ACCENT if seg.segment_type == "audit" else (
            DANGER if seg.model_score >= 0 and seg.model_score < seg.threshold else WARN
        )
        if seg.model_score >= 0:
            title = f"Score: {seg.model_score:.3f}   threshold: {seg.threshold:.2f}"
        else:
            title = f"Duration: {seg.ann_end_sec - seg.ann_start_sec:.2f} s   File: {seg.wav_path.name}"
        ax.set_title(title, color=title_color, fontsize=9, pad=3)
        ax.set_xlabel("Time (s)", color=MUTED, fontsize=8)
        ax.tick_params(colors="#555", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.set_xlim(t[0], t[-1])
        self._canvas.draw_idle()

    def _show_done(self) -> None:
        self._type_lbl.configure(text="All segments reviewed!", text_color=SAFE)
        if self.audit_mode and self.session.audit_state:
            st = self.session.audit_state
            msg = (
                f"Session complete — press Esc to save.\n"
                f"All-time: {st.n_total} reviewed, "
                f"{st.n_confirmed} confirmed, {st.n_rejected} rejected."
            )
        else:
            msg = "Press  Esc  or click  Save & Exit  to write corrected annotations."
        self._info_lbl.configure(text=msg)
        ax = self._ax
        ax.clear()
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.5, "Session complete!", transform=ax.transAxes,
                ha="center", va="center", color=SAFE, fontsize=14)
        self._canvas.draw_idle()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _play(self) -> None:
        if not _HAVE_AUDIO:
            return
        seg = self.session.current
        if seg is None:
            return
        wav = seg.context_waveform.astype(np.float32)

        def _do() -> None:
            try:
                sd.stop()
                sd.play(wav, samplerate=seg.sample_rate)
            except Exception as exc:
                logger.warning("Playback error: %s", exc)

        threading.Thread(target=_do, daemon=True).start()

    def _on_label(self, label: str) -> None:
        if _HAVE_AUDIO:
            try:
                sd.stop()
            except Exception:
                pass
        self.session.record(label)
        self._refresh()

    def _on_save_exit(self) -> None:
        if _HAVE_AUDIO:
            try:
                sd.stop()
            except Exception:
                pass

        n = len(self.session.decisions)
        if n == 0 and (self.session.audit_state is None or self.session.audit_state.n_total == 0):
            if messagebox.askyesno("Exit", "No segments reviewed. Exit without saving?"):
                self.destroy()
            return

        saved = self.session.save()   # writes annotation_corrected + saves audit state

        n_gun  = sum(1 for d in self.session.decisions if d.user_label == "gunshot")
        n_not  = sum(1 for d in self.session.decisions if d.user_label == "not_gunshot")
        n_skip = sum(1 for d in self.session.decisions if d.user_label == "skip")

        if self.audit_mode and self.session.audit_state:
            st = self.session.audit_state
            files_corrected = len(saved)
            msg = (
                f"Session saved\n"
                f"═══════════════════════════════════\n"
                f"This session:\n"
                f"  Reviewed:          {n}\n"
                f"  Confirmed gunshot: {n_gun}\n"
                f"  Rejected:          {n_not}\n"
                f"  Skipped:           {n_skip}\n"
                f"  Corrected files:   {files_corrected}\n"
                f"\n"
                f"All sessions combined:\n"
                f"  Total reviewed:  {st.n_total}\n"
                f"  Confirmed:       {st.n_confirmed}\n"
                f"  Rejected:        {st.n_rejected}\n"
                f"═══════════════════════════════════\n"
            )
            if n_not > 0:
                msg += (
                    f"Retrain with corrected annotations:\n"
                    f"  python -m train.train_yamnet_head \\\n"
                    f"    --annotation-dir clean/annotation_corrected\n"
                )
            else:
                msg += "No rejections this session — run more sessions to find bad labels.\n"
        else:
            msg = (
                f"Session summary\n"
                f"───────────────────────────────\n"
                f"Reviewed:          {n}\n"
                f"Confirmed gunshot: {n_gun}\n"
                f"Not gunshot:       {n_not}\n"
                f"Skipped:           {n_skip}\n\n"
                f"Corrected annotations saved to:\n"
                f"  {self.session.output_dir}\n\n"
                f"Retrain YAMNet Head with:\n"
                f"  python -m train.train_yamnet_head "
                f"--annotation-dir clean/annotation_corrected"
            )

        messagebox.showinfo("Saved", msg)
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="VOICe annotation verification tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Ground-truth audit (recommended first step):\n"
            "  python -m train.annotate_audio --audit-all\n\n"
            "  # Longer session (40-50 min):\n"
            "  python -m train.annotate_audio --audit-all --session-segments 300\n\n"
            "  # Active-learning pass after a training run:\n"
            "  python -m train.annotate_audio --max-segments 100\n"
        ),
    )

    # Audit-all mode
    parser.add_argument("--audit-all", action="store_true",
                        help="Full audit: verify EVERY annotated gunshot segment "
                             "(no model scoring, fast startup, incremental progress saved)")
    parser.add_argument("--session-segments", type=int, default=200,
                        help="Segments per audit session (default: 200, ≈ 30 min)")

    # Active-learning mode
    parser.add_argument("--max-segments", type=int, default=100,
                        help="Max segments for active-learning mode (default: 100)")
    parser.add_argument("--threshold", type=float,
                        default=config.YAMNET_HEAD_DECISION_THRESHOLD,
                        help="YAMNet Head threshold for finding FNs")
    parser.add_argument("--no-uncertain", action="store_true",
                        help="Only queue false negatives, skip near-threshold segments")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    if not _HAVE_AUDIO:
        print("WARNING: sounddevice not installed — audio playback disabled.")
        print("         pip install sounddevice")

    if args.audit_all:
        # ── Ground-truth audit mode ──────────────────────────────────────
        print("\nLoading audit state …")
        audit_state = AuditState()

        total_in_dataset = 4235  # cached count of VOICe gunshot annotations
        remaining = total_in_dataset - audit_state.n_total
        sessions_left = max(1, remaining) // args.session_segments

        print(f"\n  All-time reviewed:  {audit_state.n_total} / {total_in_dataset}")
        print(f"  Confirmed gunshot:  {audit_state.n_confirmed}")
        print(f"  Rejected:           {audit_state.n_rejected}")
        print(f"  Remaining:          {remaining}  (~{sessions_left} more sessions)")
        print(f"\nQueueing next {args.session_segments} unreviewed segments …")
        print("(No model loading — starts immediately)")

        queue = build_audit_queue(audit_state=audit_state, max_segments=args.session_segments)

        if not queue:
            print("\nAll annotated gunshot segments have been reviewed!")
            print(f"  Total: {audit_state.n_total} reviewed, "
                  f"{audit_state.n_confirmed} confirmed, "
                  f"{audit_state.n_rejected} rejected.")
            if audit_state.n_rejected > 0:
                print("\nRetrain with corrected annotations:")
                print("  python -m train.train_yamnet_head --annotation-dir clean/annotation_corrected")
            return

        est_min = len(queue) * 8 / 60
        print(f"\nQueued {len(queue)} segments  (~{est_min:.0f} min at 8 s/segment)")
        print("Launching annotation GUI …\n")

        session = AnnotationSession(output_dir=OUTPUT_DIR, audit_state=audit_state)
        session.queue = queue
        app = AnnotationApp(session, audit_mode=True)

    else:
        # ── Active-learning mode ─────────────────────────────────────────
        print("\nScanning VOICe dataset and scoring annotated gunshot events …")
        print("(YAMNet Head loads on first call — may take ~30 s)\n")

        queue = build_review_queue(
            threshold=args.threshold,
            max_segments=args.max_segments,
            include_uncertain=not args.no_uncertain,
        )

        if not queue:
            print("\nNo problematic segments found — model may already perform well.")
            return

        n_fn = sum(1 for s in queue if s.segment_type == "fn")
        n_uc = sum(1 for s in queue if s.segment_type == "uncertain")
        print(f"\nReview queue: {len(queue)} segments  ({n_fn} FN, {n_uc} uncertain)")
        print("Launching annotation GUI …\n")

        session = AnnotationSession(output_dir=OUTPUT_DIR, audit_state=None)
        session.queue = queue
        app = AnnotationApp(session, audit_mode=False)

    app.mainloop()


if __name__ == "__main__":
    main()
