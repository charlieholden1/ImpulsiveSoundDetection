"""
Run a comparable training/evaluation sweep across spectrogram feature types.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .train import main as train_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_feature_sweep(
    dataset_dir: Path,
    output_root: Path,
    feature_types: list[str],
    batch_size: int,
    epochs_stage_a: int,
    epochs_stage_b: int,
    skip_stage_b: bool,
) -> dict:
    """
    Train one model per feature type and collect each training summary.
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for feature_type in feature_types:
        feature_output_dir = output_root / feature_type.lower()
        logger.info("=== Feature Sweep: %s ===", feature_type)
        train_main(
            dataset_dir=dataset_dir,
            feature_type=feature_type,
            batch_size=batch_size,
            output_dir=feature_output_dir,
            epochs_stage_a=epochs_stage_a,
            epochs_stage_b=epochs_stage_b,
            do_threshold_sweep=True,
            skip_stage_b=skip_stage_b,
        )

        summary_path = feature_output_dir / "training_summary.json"
        with open(summary_path, "r", encoding="utf-8") as handle:
            results[feature_type] = json.load(handle)

    best_feature = max(
        results.items(),
        key=lambda item: item[1]["selected_stage_validation_metrics"]["auprc"],
    )[0]
    results["best_feature"] = best_feature

    sweep_path = output_root / "feature_sweep_summary.json"
    with open(sweep_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    logger.info("Feature sweep summary saved to: %s", sweep_path)
    logger.info("Best feature: %s", best_feature)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare FFT, LogMel, and MFCC training runs")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
        help="Root directory of the spectrogram dataset",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("models/feature_sweep"),
        help="Directory for per-feature outputs",
    )
    parser.add_argument(
        "--feature-types",
        nargs="+",
        default=["LogMel", "MFCC", "FFT"],
        choices=["LogMel", "MFCC", "FFT"],
        help="Feature types to compare",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs-a", type=int, default=8, help="Stage A epochs per feature")
    parser.add_argument("--epochs-b", type=int, default=0, help="Stage B epochs per feature")
    parser.add_argument(
        "--skip-stage-b",
        action="store_true",
        help="Skip fine-tuning during the feature sweep",
    )

    args = parser.parse_args()

    run_feature_sweep(
        dataset_dir=args.dataset_dir,
        output_root=args.output_root,
        feature_types=args.feature_types,
        batch_size=args.batch_size,
        epochs_stage_a=args.epochs_a,
        epochs_stage_b=args.epochs_b,
        skip_stage_b=args.skip_stage_b or args.epochs_b <= 0,
    )
