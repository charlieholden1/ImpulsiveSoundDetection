#!/usr/bin/env python
"""
main.py – CLI entry point for the Robust Impulsive Sound Detection System.

Modes
-----
``detect``
    Run the two-stage detection pipeline on one or more ``.wav`` files.

``prepare``
    Discover, segment, and optionally augment the VOICe dataset for
    downstream training.

``demo``
    Process the first VOICe training file and display the detection
    visualisation.

``live``
    Open the default microphone and run real-time two-stage detection
    with a colour-coded terminal dashboard.

``gui``
    Launch the graphical dashboard (customtkinter) with live plotting,
    event log, and parameter-tuning sliders.

Usage
-----
    python -m impulsive_sound_detection.main detect  --wav path/to/file.wav
    python -m impulsive_sound_detection.main prepare --augment
    python -m impulsive_sound_detection.main demo
    python -m impulsive_sound_detection.main live --threshold-multiplier 4.0
    python -m impulsive_sound_detection.main gui
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from impulsive_sound_detection import config
from impulsive_sound_detection.augmentor import RobustAugmentor
from impulsive_sound_detection.classifier import ClassificationResult, YAMNetClassifier
from impulsive_sound_detection.data_loader import (
    DatasetBundle,
    discover_voice_dataset,
    load_wav,
)
from impulsive_sound_detection.pipeline import DetectionPipeline
from impulsive_sound_detection.stream_monitor import StreamMonitor
from impulsive_sound_detection.visualizer import plot_detections

# ──────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# CLI actions
# ──────────────────────────────────────────────────────────────────────
def action_detect(args: argparse.Namespace) -> None:
    """Run the two-stage detection pipeline on given WAV files.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (expects ``wav``, ``log``, ``no_viz``).
    """
    log_path = Path(args.log) if args.log else None
    pipeline = DetectionPipeline(log_path=log_path)

    for wav_str in args.wav:
        wav_path = Path(wav_str)
        if not wav_path.exists():
            logger.error("File not found: %s", wav_path)
            continue
        results = pipeline.run_on_file(
            wav_path,
            visualize=not args.no_viz,
        )
        logger.info(
            "%s → %d detections (%d suspicious)",
            wav_path.name,
            len(results),
            sum(1 for r in results if r.is_suspicious),
        )


def action_prepare(args: argparse.Namespace) -> None:
    """Discover and optionally augment the VOICe dataset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (expects ``augment``, ``n_aug``).
    """
    logger.info("Discovering VOICe dataset …")
    file_list = config.VOICE_SOURCE_DIR / "synthetic_source_training.txt"
    bundle: DatasetBundle = discover_voice_dataset(file_list=file_list)
    logger.info(
        "Loaded %d positive + %d negative segments",
        len(bundle.positive),
        len(bundle.negative),
    )

    if args.augment:
        augmentor = RobustAugmentor()
        pos_waves = [seg.waveform for seg in bundle.positive]
        aug_waves = augmentor.augment_batch(pos_waves, n_augmentations=args.n_aug)
        logger.info(
            "Augmented %d → %d positive waveforms",
            len(pos_waves),
            len(aug_waves),
        )

    # Save a summary
    summary = {
        "positive_count": len(bundle.positive),
        "negative_count": len(bundle.negative),
        "augmented": args.augment,
        "augmentation_factor": args.n_aug if args.augment else 0,
    }
    print(json.dumps(summary, indent=2))


def action_demo(args: argparse.Namespace) -> None:
    """Run detection on the first VOICe training file and visualise.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (expects ``file_index``).
    """
    # Use the training file list to pick a demo file
    file_list_path = config.VOICE_SOURCE_DIR / "synthetic_source_training.txt"
    with file_list_path.open("r", encoding="utf-8") as fh:
        wav_names = [l.strip() for l in fh if l.strip()]

    idx = args.file_index
    if idx >= len(wav_names):
        logger.error(
            "File index %d out of range (max %d)", idx, len(wav_names) - 1
        )
        sys.exit(1)

    wav_path = config.VOICE_AUDIO_DIR / wav_names[idx]
    logger.info("Demo file: %s", wav_path)

    pipeline = DetectionPipeline()
    results = pipeline.run_on_file(wav_path, visualize=True)

    print(f"\n{'='*60}")
    print(f"  DETECTION SUMMARY – {wav_path.name}")
    print(f"{'='*60}")
    print(f"  Total detections : {len(results)}")
    print(
        f"  Suspicious       : "
        f"{sum(1 for r in results if r.is_suspicious)}"
    )
    print(
        f"  Non-suspicious   : "
        f"{sum(1 for r in results if not r.is_suspicious)}"
    )
    print(f"{'='*60}\n")

    for i, r in enumerate(results, 1):
        print(f"  [{i:3d}] {r.to_json()}")


def action_live(args: argparse.Namespace) -> None:
    """Start live microphone monitoring with a terminal dashboard.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (expects ``threshold_multiplier``,
        ``device``, ``log``, ``no_colour``).
    """
    from impulsive_sound_detection.live_stream import LiveMicStreamer

    if args.list_devices:
        LiveMicStreamer.list_devices()
        return

    device = args.device
    # Try to interpret as int (device index); fall back to string name
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # keep as string

    streamer = LiveMicStreamer(
        energy_multiplier=args.threshold_multiplier,
        device=device,
        log_path=args.log,
        enable_colour=not args.no_colour,
    )
    streamer.start()


def action_gui(args: argparse.Namespace) -> None:
    """Launch the graphical detection dashboard.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments (expects ``log``).
    """
    from impulsive_sound_detection.gui import launch_gui

    launch_gui(log_path=args.log)


# ──────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="impulsive_sound_detection",
        description="Robust Impulsive Sound Detection System for school environments.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── detect --------------------------------------------------------
    p_detect = sub.add_parser("detect", help="Run detection on WAV file(s).")
    p_detect.add_argument(
        "--wav",
        nargs="+",
        required=True,
        help="Path(s) to .wav file(s) to process.",
    )
    p_detect.add_argument(
        "--log",
        default=None,
        help="Path to a JSONL log file for detections.",
    )
    p_detect.add_argument(
        "--no-viz",
        action="store_true",
        help="Suppress matplotlib visualisation.",
    )

    # ── prepare -------------------------------------------------------
    p_prep = sub.add_parser("prepare", help="Prepare / augment training data.")
    p_prep.add_argument(
        "--augment",
        action="store_true",
        help="Apply audio augmentation to positive segments.",
    )
    p_prep.add_argument(
        "--n-aug",
        type=int,
        default=3,
        help="Number of augmented copies per original (default: 3).",
    )

    # ── demo ----------------------------------------------------------
    p_demo = sub.add_parser("demo", help="Run a quick demo on a VOICe file.")
    p_demo.add_argument(
        "--file-index",
        type=int,
        default=0,
        help="0-based index into the training file list (default: 0).",
    )

    # ── live -----------------------------------------------------------
    p_live = sub.add_parser(
        "live",
        help="Real-time microphone monitoring with terminal dashboard.",
    )
    p_live.add_argument(
        "--threshold-multiplier",
        type=float,
        default=config.ENERGY_MULTIPLIER,
        help=(
            "Dynamic energy threshold multiplier.  Raise this in a "
            "loud room to reduce false triggers (default: "
            f"{config.ENERGY_MULTIPLIER})."
        ),
    )
    p_live.add_argument(
        "--device",
        default=None,
        help=(
            "Audio input device index or name.  Use --list-devices to "
            "see available devices."
        ),
    )
    p_live.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available audio devices and exit.",
    )
    p_live.add_argument(
        "--log",
        default=None,
        help="Path to a JSONL log file for detections.",
    )
    p_live.add_argument(
        "--no-colour",
        action="store_true",
        help="Disable ANSI colour codes (plain text output).",
    )

    # ── gui ------------------------------------------------------------
    p_gui = sub.add_parser(
        "gui",
        help="Launch the graphical dashboard (customtkinter).",
    )
    p_gui.add_argument(
        "--log",
        default=None,
        help="Path to a JSONL log file for detections.",
    )

    return parser


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """Parse arguments and dispatch to the selected action.

    Raises
    ------
    SystemExit
        On invalid arguments.
    """
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "detect": action_detect,
        "prepare": action_prepare,
        "demo": action_demo,
        "live": action_live,
        "gui": action_gui,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
