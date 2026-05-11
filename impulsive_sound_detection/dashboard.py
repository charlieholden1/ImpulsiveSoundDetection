"""
dashboard.py – Colour-rich terminal dashboard for live-mode presentations.

Uses standard ANSI escape codes (supported on Windows 10+ and all
modern terminal emulators) to render:

* A continuously updating RMS energy bar meter.
* Colour-coded alerts when YAMNet classifies a trigger window.

All output is written via ``sys.stdout`` so that it can be redirected
or piped if needed.

Colour palette
~~~~~~~~~~~~~~
- **Red**   – suspicious detection (gunshot, glass, explosion …)
- **Green** – non-suspicious detection
- **Cyan**  – RMS energy meter
- **Yellow** – warnings / info headers

Usage
-----
>>> from impulsive_sound_detection.dashboard import LiveDashboard
>>> dash = LiveDashboard()
>>> dash.update_meter(rms=0.045, baseline=0.012)
>>> dash.show_alert(result)       # ClassificationResult
"""

from __future__ import annotations

import sys
import time
from typing import Optional

from .classifier import ClassificationResult

# Ensure the terminal accepts Unicode box-drawing characters and emoji.
# On Windows the default console encoding is cp1252; reconfigure to UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# ANSI escape helpers
# ──────────────────────────────────────────────────────────────────────
class _Ansi:
    """Container for ANSI SGR escape sequences."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    MAGENTA = "\033[95m"

    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

    CLEAR_LINE = "\033[2K"
    CURSOR_UP = "\033[1A"
    HIDE_CURSOR = "\033[?25l"
    SHOW_CURSOR = "\033[?25h"


# ──────────────────────────────────────────────────────────────────────
# Dashboard
# ──────────────────────────────────────────────────────────────────────
class LiveDashboard:
    """Terminal UI for real-time impulsive-sound monitoring.

    Parameters
    ----------
    meter_width : int
        Character width of the RMS bar (default ``40``).
    max_rms : float
        RMS value that maps to a full bar (auto-scales if exceeded).
    enable_colour : bool
        If ``False``, all ANSI codes are suppressed (plain text mode).
    """

    def __init__(
        self,
        meter_width: int = 40,
        max_rms: float = 0.15,
        enable_colour: bool = True,
    ) -> None:
        self._meter_width = meter_width
        self._max_rms = max_rms
        self._colour = enable_colour
        self._alert_count = 0
        self._suspicious_count = 0
        self._last_meter_time: float = 0.0
        self._meter_interval: float = 0.05   # 20 Hz refresh cap
        self._started = False

    # ── colour helper ─────────────────────────────────────────────────
    def _c(self, code: str) -> str:
        """Return *code* if colour is enabled, else empty string.

        Parameters
        ----------
        code : str
            ANSI escape sequence.

        Returns
        -------
        str
        """
        return code if self._colour else ""

    # ── public API ────────────────────────────────────────────────────
    def print_banner(self) -> None:
        """Print the startup banner once.

        Displays the system name, mode, and keyboard shortcut hint.
        """
        c = self._c
        banner = (
            f"\n"
            f"{c(_Ansi.BOLD)}{c(_Ansi.CYAN)}"
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║   🎙  IMPULSIVE SOUND DETECTION  –  LIVE MODE          ║\n"
            f"╚══════════════════════════════════════════════════════════╝"
            f"{c(_Ansi.RESET)}\n"
            f"{c(_Ansi.DIM)}  Press Ctrl+C to stop.{c(_Ansi.RESET)}\n"
        )
        sys.stdout.write(banner)
        sys.stdout.flush()
        self._started = True

    def update_meter(
        self,
        rms: float,
        baseline: float,
        threshold: float,
    ) -> None:
        """Redraw the RMS energy meter on the current terminal line.

        Rate-limited to ``self._meter_interval`` to avoid flooding
        the terminal.

        Parameters
        ----------
        rms : float
            Current frame RMS energy.
        baseline : float
            Rolling-average baseline.
        threshold : float
            Dynamic trigger threshold (``baseline × multiplier``).
        """
        now = time.monotonic()
        if now - self._last_meter_time < self._meter_interval:
            return
        self._last_meter_time = now

        c = self._c

        # Auto-scale
        cap = max(self._max_rms, rms * 1.2) if rms > self._max_rms else self._max_rms
        fill = int(min(rms / cap, 1.0) * self._meter_width)
        empty = self._meter_width - fill

        # Colour the bar: green normally, yellow approaching threshold, red above
        if rms > threshold:
            bar_colour = c(_Ansi.RED) + c(_Ansi.BOLD)
        elif rms > threshold * 0.6:
            bar_colour = c(_Ansi.YELLOW)
        else:
            bar_colour = c(_Ansi.GREEN)

        bar = (
            f"{c(_Ansi.CLEAR_LINE)}\r"
            f"  {c(_Ansi.CYAN)}RMS{c(_Ansi.RESET)} "
            f"[{bar_colour}{'█' * fill}{c(_Ansi.DIM)}{'░' * empty}"
            f"{c(_Ansi.RESET)}] "
            f"{c(_Ansi.WHITE)}{rms:.4f}{c(_Ansi.RESET)}  "
            f"{c(_Ansi.DIM)}baseline={baseline:.4f}  "
            f"thr={threshold:.4f}{c(_Ansi.RESET)}"
        )
        sys.stdout.write(bar)
        sys.stdout.flush()

    def show_alert(self, result: ClassificationResult) -> None:
        """Print a prominent colour-coded detection alert.

        Parameters
        ----------
        result : ClassificationResult
            The classification output to display.
        """
        c = self._c
        self._alert_count += 1

        if result.is_suspicious:
            self._suspicious_count += 1
            header_bg = c(_Ansi.BG_RED) + c(_Ansi.WHITE) + c(_Ansi.BOLD)
            label_colour = c(_Ansi.RED) + c(_Ansi.BOLD)
            icon = "⚠ "
            tag = "SUSPICIOUS"
        else:
            header_bg = c(_Ansi.BG_GREEN) + c(_Ansi.WHITE) + c(_Ansi.BOLD)
            label_colour = c(_Ansi.GREEN)
            icon = "✓ "
            tag = "SAFE"

        # Build the top-K detail string
        top_k_str = "  ".join(
            f"{e['class']}({e['score']:.2f})" for e in result.top_k[:3]
        )

        alert = (
            f"\n"
            f"  {header_bg} {icon}{tag} {c(_Ansi.RESET)}  "
            f"{label_colour}{result.label}{c(_Ansi.RESET)}  "
            f"confidence={c(_Ansi.BOLD)}{result.confidence:.3f}"
            f"{c(_Ansi.RESET)}  "
            f"t={result.timestamp:.2f}s\n"
            f"  {c(_Ansi.DIM)}Top-3: {top_k_str}{c(_Ansi.RESET)}\n"
        )
        sys.stdout.write(alert)
        sys.stdout.flush()

    def show_status(self, elapsed_sec: float) -> None:
        """Print a periodic status summary line.

        Parameters
        ----------
        elapsed_sec : float
            Seconds elapsed since the stream started.
        """
        c = self._c
        status = (
            f"{c(_Ansi.CLEAR_LINE)}\r"
            f"  {c(_Ansi.DIM)}⏱ {elapsed_sec:6.1f}s  │  "
            f"alerts={self._alert_count}  "
            f"suspicious={self._suspicious_count}"
            f"{c(_Ansi.RESET)}"
        )
        sys.stdout.write(status + "\n")
        sys.stdout.flush()

    def print_shutdown(self, elapsed_sec: float) -> None:
        """Print the shutdown summary when the stream ends.

        Parameters
        ----------
        elapsed_sec : float
            Total runtime in seconds.
        """
        c = self._c
        summary = (
            f"\n"
            f"{c(_Ansi.BOLD)}{c(_Ansi.CYAN)}"
            f"╔══════════════════════════════════════════════════════════╗\n"
            f"║                    SESSION SUMMARY                     ║\n"
            f"╚══════════════════════════════════════════════════════════╝"
            f"{c(_Ansi.RESET)}\n"
            f"  Duration         : {elapsed_sec:.1f} s\n"
            f"  Total alerts     : {self._alert_count}\n"
            f"  Suspicious       : "
            f"{c(_Ansi.RED)}{self._suspicious_count}{c(_Ansi.RESET)}\n"
            f"  Non-suspicious   : "
            f"{c(_Ansi.GREEN)}"
            f"{self._alert_count - self._suspicious_count}"
            f"{c(_Ansi.RESET)}\n"
        )
        sys.stdout.write(summary)
        sys.stdout.flush()
