"""
hyperparam_search.py — Systematic grid search over CNN hyperparameters.

Rubric requirement (Model Development): "Tune hyperparameters systematically
(e.g., grid search, random search)."

Search space (9 combinations):
  dropout_rate ∈ {0.3, 0.4, 0.5}
  stage_a_lr   ∈ {5e-4, 1e-3, 2e-3}

For each combination, a fresh EfficientNetB0 model is built and trained
through Stage A only (10 epochs max, early stopping on val_auprc).
The winning combination is the one that achieves the highest val_auprc
on the validation set.

Results are saved as:
  reports/hyperparam_search_results.json  — per-combo metrics table
  reports/hyperparam_search_heatmap.png   — val_auprc heatmap (3×3 grid)

The winning hyperparameters should then be used when running train.py:
  python -m train.train --feature-type FFT

Usage:
  python -m train.hyperparam_search [--feature-type FFT] [--epochs 10]
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .dataset import load_spectrogram_dataset
from .model import build_cnn_model, build_metrics
from .train import evaluate_model_on_dataset, train_stage_a

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Grid search space
DROPOUT_RATES = [0.3, 0.4, 0.5]
LEARNING_RATES = [5e-4, 1e-3, 2e-3]


def run_grid_search(
    dataset_dir: Path,
    feature_type: str = "FFT",
    batch_size: int = 32,
    search_epochs: int = 10,
    output_dir: Path = Path("reports"),
) -> dict:
    """
    Perform a grid search over dropout_rate × stage_a_lr.

    Parameters
    ----------
    dataset_dir : Path
        Root of the spectrogram dataset (contains FFT/, LogMel/, MFCC/).
    feature_type : str
        Which spectrogram feature type to use during the search.
    batch_size : int
        Batch size for all combos (kept constant to isolate hyperparameter effects).
    search_epochs : int
        Maximum Stage A epochs per combo.  Early stopping may terminate earlier.
    output_dir : Path
        Where to save the results JSON and heatmap PNG.

    Returns
    -------
    dict
        Full results with per-combo metrics and the winning configuration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("HYPERPARAMETER GRID SEARCH")
    logger.info("=" * 70)
    logger.info("Feature type : %s", feature_type)
    logger.info("Search epochs: %d (+ early stopping)", search_epochs)
    logger.info("Grid: dropout ∈ %s × lr ∈ %s", DROPOUT_RATES, LEARNING_RATES)
    logger.info("Total combos : %d", len(DROPOUT_RATES) * len(LEARNING_RATES))

    # Load dataset once — all combos share the same data
    logger.info("Loading dataset …")
    train_ds, val_ds, _, class_weights = load_spectrogram_dataset(
        dataset_dir,
        feature_type=feature_type,
        batch_size=batch_size,
    )

    results = []
    combo_id = 0

    for dropout, lr in product(DROPOUT_RATES, LEARNING_RATES):
        combo_id += 1
        logger.info(
            "\n[%d/%d] dropout=%.1f  lr=%.0e",
            combo_id, len(DROPOUT_RATES) * len(LEARNING_RATES),
            dropout, lr,
        )

        # Build a fresh model for this combo
        model, _ = build_cnn_model(dropout_rate=dropout)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=build_metrics(),
        )

        # Temporary checkpoint — overwritten each combo
        ckpt = output_dir / f"_hs_ckpt_d{int(dropout*10)}_lr{combo_id}.keras"
        try:
            train_stage_a(
                model,
                train_ds,
                val_ds,
                class_weights,
                epochs=search_epochs,
                checkpoint_path=ckpt,
            )
            # Load the best checkpoint (early stopping may have restored it already)
            if ckpt.exists():
                model = tf.keras.models.load_model(str(ckpt))
        except Exception as exc:
            logger.error("Combo (%.1f, %.0e) failed: %s", dropout, lr, exc)
            results.append({
                "dropout_rate": float(dropout),
                "learning_rate": float(lr),
                "val_auprc": 0.0,
                "val_auc": 0.0,
                "val_f1": 0.0,
                "val_recall": 0.0,
                "error": str(exc),
            })
            continue

        metrics = evaluate_model_on_dataset(model, val_ds, threshold=0.5)

        logger.info(
            "  → val_auprc=%.4f  val_auc=%.4f  val_f1=%.4f  val_recall=%.4f",
            metrics["auprc"], metrics["auc"], metrics["f1"], metrics["recall"],
        )

        results.append({
            "dropout_rate": float(dropout),
            "learning_rate": float(lr),
            "val_auprc": float(metrics["auprc"]),
            "val_auc": float(metrics["auc"]),
            "val_f1": float(metrics["f1"]),
            "val_recall": float(metrics["recall"]),
            "val_precision": float(metrics["precision"]),
        })

        # Clean up temp checkpoint
        if ckpt.exists():
            ckpt.unlink()

        # Free GPU memory between combos
        del model
        tf.keras.backend.clear_session()

    # Find winner
    best = max(results, key=lambda r: r.get("val_auprc", 0.0))
    logger.info(
        "\nBEST COMBINATION: dropout=%.1f, lr=%.0e → val_auprc=%.4f",
        best["dropout_rate"], best["learning_rate"], best["val_auprc"],
    )

    output = {
        "feature_type": feature_type,
        "search_epochs_per_combo": search_epochs,
        "dropout_rates_searched": DROPOUT_RATES,
        "learning_rates_searched": LEARNING_RATES,
        "results": results,
        "best_combo": best,
    }

    results_path = output_dir / "hyperparam_search_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to: %s", results_path)

    _plot_heatmap(results, output_dir)

    return output


def _plot_heatmap(results: list, output_dir: Path) -> None:
    """Generate a 3×3 heatmap of val_auprc for each (dropout, lr) combo."""
    grid = np.zeros((len(DROPOUT_RATES), len(LEARNING_RATES)))

    for entry in results:
        i = DROPOUT_RATES.index(entry["dropout_rate"])
        j = LEARNING_RATES.index(entry["learning_rate"])
        grid[i, j] = entry.get("val_auprc", 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(grid, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(LEARNING_RATES)))
    ax.set_yticks(range(len(DROPOUT_RATES)))
    ax.set_xticklabels([f"{lr:.0e}" for lr in LEARNING_RATES])
    ax.set_yticklabels([f"{d:.1f}" for d in DROPOUT_RATES])
    ax.set_xlabel("Stage A Learning Rate")
    ax.set_ylabel("Dropout Rate")
    ax.set_title("Hyperparameter Grid Search — Val AUPRC")

    # Annotate cells
    for i in range(len(DROPOUT_RATES)):
        for j in range(len(LEARNING_RATES)):
            val = grid[i, j]
            colour = "white" if val > 0.7 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    color=colour, fontsize=9, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Val AUPRC")
    fig.tight_layout()

    heatmap_path = output_dir / "hyperparam_search_heatmap.png"
    plt.savefig(heatmap_path, dpi=120, bbox_inches="tight")
    logger.info("Heatmap saved to: %s", heatmap_path)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grid search over CNN hyperparameters")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"),
        help="Root directory of the spectrogram dataset",
    )
    parser.add_argument(
        "--feature-type",
        default="FFT",
        choices=["FFT", "LogMel", "MFCC"],
        help="Spectrogram feature type to use during the search",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Max Stage A epochs per combo (early stopping may terminate sooner)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Where to save results JSON and heatmap PNG",
    )

    args = parser.parse_args()
    run_grid_search(
        dataset_dir=args.dataset_dir,
        feature_type=args.feature_type,
        batch_size=args.batch_size,
        search_epochs=args.epochs,
        output_dir=args.output_dir,
    )
