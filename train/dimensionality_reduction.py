"""
dimensionality_reduction.py — PCA and feature selection analysis.

Rubric requirement (Data Preparation): "Apply methods like PCA, LDA, or
feature selection algorithms if suitable for the problem. Justify the method
used and interpret results (e.g., amount of variance explained, improvement
in class separability)."

Two analyses are performed:

Part A — PCA on EfficientNetB0 embeddings
  The GlobalAveragePooling2D layer of the trained model produces a 1280-d
  feature vector per spectrogram.  PCA is applied to these embeddings to:
    - Show how many components explain 95% of the variance
    - Visualise the 2D projection coloured by class (GUN vs NOGUN)
  Strong class separability in 2D confirms the CNN has learned discriminative
  acoustic features.

Part B — Feature selection on MFCC coefficients (SelectKBest / ANOVA F-test)
  Each MFCC spectrogram is reduced to a 40-d feature vector (mean of each
  MFCC coefficient across time frames).  sklearn SelectKBest with the ANOVA
  F-statistic ranks which of the 40 MFCC bands are most discriminative
  between gunshot and non-gunshot audio.

Outputs (all written to reports/):
  pca_embeddings_2d.png         — 2D PCA scatter coloured by class
  pca_variance_explained.png    — cumulative variance explained curve
  pca_results.json              — n_components for 95% variance + top-5 loadings
  mfcc_feature_selection.png    — F-score bar chart per MFCC coefficient
  mfcc_feature_selection.json   — F-scores and selected feature indices

Usage:
  python -m train.dimensionality_reduction [--model-path ...] [--feature-type FFT]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from impulsive_sound_detection import config
from impulsive_sound_detection.spectrogram_utils import compute_feature_image

from .dataset import load_spectrogram_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Part A — PCA on EfficientNetB0 embeddings
# ─────────────────────────────────────────────────────────────────────────────

def _build_embedding_extractor(full_model: tf.keras.Model) -> tf.keras.Model:
    """
    Build a sub-model that outputs at the GlobalAveragePooling2D layer.

    The full model architecture is:
      Input → preprocess_input → EfficientNetB0 → GlobalAveragePooling2D
           → Dropout → Dense(128) → Dropout → Dense(1 logit)

    We tap the output right after GlobalAveragePooling2D to get the 1280-d
    embedding vector that encodes the acoustic content of each spectrogram.
    """
    # Locate the GlobalAveragePooling2D layer by type
    gap_layer = next(
        (layer for layer in full_model.layers
         if isinstance(layer, tf.keras.layers.GlobalAveragePooling2D)),
        None,
    )
    if gap_layer is None:
        raise RuntimeError(
            "Could not find GlobalAveragePooling2D in the model. "
            "Ensure the model was built with build_cnn_model()."
        )

    extractor = tf.keras.Model(
        inputs=full_model.input,
        outputs=gap_layer.output,
        name="embedding_extractor",
    )
    logger.info(
        "Embedding extractor built: output shape %s from layer '%s'",
        extractor.output_shape,
        gap_layer.name,
    )
    return extractor


def extract_embeddings(
    extractor: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run all images in a tf.data.Dataset through the extractor.

    Returns
    -------
    embeddings : np.ndarray  shape (N, embedding_dim)
    labels     : np.ndarray  shape (N,)
    """
    all_embeddings = []
    all_labels = []

    for images, labels in dataset:
        batch_emb = extractor.predict_on_batch(images)
        all_embeddings.append(np.asarray(batch_emb))
        all_labels.extend(np.asarray(labels).flatten().tolist())

    return np.vstack(all_embeddings), np.array(all_labels, dtype=np.int32)


def run_pca_analysis(
    model_path: Path,
    dataset_dir: Path,
    feature_type: str = "FFT",
    reports_dir: Path = Path("reports"),
    batch_size: int = 32,
) -> dict:
    """
    Apply PCA to EfficientNetB0 val-set embeddings and generate visualisations.

    Parameters
    ----------
    model_path : Path
        Path to the trained .keras model.
    dataset_dir : Path
        Root of the spectrogram dataset.
    feature_type : str
        Which spectrogram type the model was trained on.
    reports_dir : Path
        Where to save plots and JSON.
    batch_size : int
        Batch size for inference.

    Returns
    -------
    dict with PCA results (n_components_95pct, explained_variance_ratio, etc.)
    """
    logger.info("=== Part A: PCA on EfficientNetB0 Embeddings ===")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load model and build extractor
    model = tf.keras.models.load_model(str(model_path))
    extractor = _build_embedding_extractor(model)

    # Load val set (we apply PCA on the held-out validation split)
    _, val_ds, _, _ = load_spectrogram_dataset(
        dataset_dir, feature_type=feature_type, batch_size=batch_size
    )

    logger.info("Extracting embeddings from validation set …")
    embeddings, labels = extract_embeddings(extractor, val_ds)
    logger.info("Embeddings shape: %s  (classes: %d GUN, %d NOGUN)",
                embeddings.shape, int(labels.sum()), int((labels == 0).sum()))

    # Fit PCA (full — all components) to get the variance explained curve
    pca_full = PCA(random_state=42)
    pca_full.fit(embeddings)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    n_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    logger.info(
        "PCA: %d components explain 95%% variance, %d for 99%% (of %d total dims)",
        n_95, n_99, embeddings.shape[1],
    )

    # --- Plot 1: Cumulative variance explained ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cumvar) + 1), cumvar, color="steelblue", lw=1.5)
    ax.axhline(0.95, color="red", linestyle="--", label=f"95% variance ({n_95} components)")
    ax.axhline(0.99, color="orange", linestyle="--", label=f"99% variance ({n_99} components)")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA — Cumulative Variance Explained\n(EfficientNetB0 GlobalAveragePooling embeddings)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim([1, min(200, len(cumvar))])
    plt.tight_layout()
    var_path = reports_dir / "pca_variance_explained.png"
    plt.savefig(var_path, dpi=120, bbox_inches="tight")
    logger.info("Saved: %s", var_path)
    plt.close()

    # --- Plot 2: 2D PCA scatter ---
    pca_2d = PCA(n_components=2, random_state=42)
    coords = pca_2d.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {0: ("royalblue", "NOGUN"), 1: ("crimson", "GUN")}
    for cls, (color, name) in colors.items():
        mask = labels == cls
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, alpha=0.5, s=12, edgecolors="none",
        )
    ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title("PCA 2D Projection of EfficientNetB0 Embeddings\n(GUN vs NOGUN — Val Set)")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    scatter_path = reports_dir / "pca_embeddings_2d.png"
    plt.savefig(scatter_path, dpi=120, bbox_inches="tight")
    logger.info("Saved: %s", scatter_path)
    plt.close()

    pca_results = {
        "embedding_dim": int(embeddings.shape[1]),
        "n_val_samples": int(len(labels)),
        "n_gun": int(labels.sum()),
        "n_nogun": int((labels == 0).sum()),
        "n_components_for_95pct_variance": int(n_95),
        "n_components_for_99pct_variance": int(n_99),
        "pc1_variance_ratio": float(pca_2d.explained_variance_ratio_[0]),
        "pc2_variance_ratio": float(pca_2d.explained_variance_ratio_[1]),
        "top_10_explained_variance": cumvar[:10].tolist(),
        "feature_type": feature_type,
        "model_path": str(model_path),
    }

    pca_json_path = reports_dir / "pca_results.json"
    with open(pca_json_path, "w", encoding="utf-8") as f:
        json.dump(pca_results, f, indent=2)
    logger.info("PCA results saved to: %s", pca_json_path)

    return pca_results


# ─────────────────────────────────────────────────────────────────────────────
# Part B — Feature selection on MFCC coefficients
# ─────────────────────────────────────────────────────────────────────────────

def run_mfcc_feature_selection(
    dataset_dir: Path,
    reports_dir: Path = Path("reports"),
    k: int = 20,
    sample_limit: int | None = 2000,
) -> dict:
    """
    Apply ANOVA F-test (SelectKBest) to 40 MFCC coefficients.

    Each spectrogram image is replaced by a 40-d feature vector: the mean
    value of each MFCC coefficient averaged over time frames.  The ANOVA
    F-statistic then measures which coefficients best separate the GUN from
    the NOGUN class.

    Parameters
    ----------
    dataset_dir : Path
        Root of the spectrogram dataset (must contain MFCC/ subdirectory).
    reports_dir : Path
        Where to save the bar chart and JSON.
    k : int
        Number of top features to select and highlight.
    sample_limit : int | None
        Cap on images processed (to keep runtime under a minute).

    Returns
    -------
    dict with F-scores, p-values, and selected feature indices.
    """
    logger.info("=== Part B: MFCC Feature Selection (SelectKBest / ANOVA F-test) ===")
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    mfcc_gun_dir = dataset_dir / "MFCC" / "GUN"
    mfcc_nogun_dir = dataset_dir / "MFCC" / "NOGUN"

    if not mfcc_gun_dir.exists() or not mfcc_nogun_dir.exists():
        raise FileNotFoundError(
            f"MFCC dataset not found under {dataset_dir / 'MFCC'}. "
            "Run the feature sweep first or confirm the dataset path."
        )

    from PIL import Image as PILImage

    def _load_mfcc_mean(png_path: Path) -> np.ndarray:
        """Load a MFCC spectrogram PNG and return the mean value per row (coefficient)."""
        arr = np.asarray(PILImage.open(png_path).convert("L"), dtype=np.float32)
        # arr shape: (height, width) — height = n_mfcc padded to 224, width = time
        # We average over time (axis=1) to get one value per frequency bin
        return arr.mean(axis=1)  # shape: (224,) — we'll take first 40

    gun_paths = sorted(mfcc_gun_dir.glob("*.png"))
    nogun_paths = sorted(mfcc_nogun_dir.glob("*.png"))

    if sample_limit is not None:
        import random
        rng = random.Random(42)
        gun_paths = rng.sample(gun_paths, min(sample_limit // 2, len(gun_paths)))
        nogun_paths = rng.sample(nogun_paths, min(sample_limit // 2, len(nogun_paths)))

    logger.info("Loading MFCC features: %d GUN + %d NOGUN images …",
                len(gun_paths), len(nogun_paths))

    X_list = []
    y_list = []

    for path in gun_paths:
        try:
            feat = _load_mfcc_mean(path)[:40]  # first 40 rows = 40 MFCC coefficients
            X_list.append(feat)
            y_list.append(1)
        except Exception as exc:
            logger.debug("Skipping %s: %s", path.name, exc)

    for path in nogun_paths:
        try:
            feat = _load_mfcc_mean(path)[:40]
            X_list.append(feat)
            y_list.append(0)
        except Exception as exc:
            logger.debug("Skipping %s: %s", path.name, exc)

    X = np.stack(X_list, axis=0)  # shape: (N, 40)
    y = np.array(y_list, dtype=np.int32)
    n_features = X.shape[1]

    logger.info("Feature matrix: %s  (GUN=%d, NOGUN=%d)", X.shape, y.sum(), (y==0).sum())

    # Apply SelectKBest with ANOVA F-test
    selector = SelectKBest(f_classif, k=min(k, n_features))
    selector.fit(X, y)

    f_scores = selector.scores_
    p_values = selector.pvalues_
    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0].tolist()

    logger.info(
        "Top %d MFCC coefficients by F-score: %s",
        k, sorted(selected_indices),
    )

    # --- Plot: F-score bar chart ---
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["crimson" if sel else "steelblue" for sel in selected_mask]
    ax.bar(np.arange(n_features), f_scores, color=colors, edgecolor="none")
    ax.set_xlabel("MFCC Coefficient Index (0–39)")
    ax.set_ylabel("ANOVA F-score")
    ax.set_title(
        f"MFCC Feature Selection — ANOVA F-score per Coefficient\n"
        f"(Red = top {k} selected, Blue = not selected)"
    )
    ax.set_xticks(np.arange(0, n_features, 5))
    ax.grid(axis="y", alpha=0.3)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="crimson", label=f"Top {k} selected features"),
        Patch(facecolor="steelblue", label="Not selected"),
    ]
    ax.legend(handles=legend_elements)
    plt.tight_layout()

    bar_path = reports_dir / "mfcc_feature_selection.png"
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    logger.info("Saved: %s", bar_path)
    plt.close()

    result = {
        "n_samples": int(len(y)),
        "n_features": int(n_features),
        "k_selected": int(k),
        "selected_feature_indices": selected_indices,
        "f_scores": f_scores.tolist(),
        "p_values": p_values.tolist(),
        "top_5_features": sorted(
            range(n_features), key=lambda i: -f_scores[i]
        )[:5],
        "top_5_f_scores": sorted(f_scores, reverse=True)[:5],
    }

    json_path = reports_dir / "mfcc_feature_selection.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("MFCC selection results saved to: %s", json_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main(
    model_path: Path = Path(
        r"C:\Users\holde\Documents\MLProject\models\feature_sweep_stagea6\fft\cnn_gunshot_classifier.keras"
    ),
    dataset_dir: Path = Path(
        r"C:\Users\holde\Documents\MLProject\Gunshot Audio Spectrogram Dataset for Binary Class"
    ),
    feature_type: str = "FFT",
    reports_dir: Path = Path("reports"),
    batch_size: int = 32,
    k_mfcc: int = 20,
    skip_pca: bool = False,
    skip_mfcc: bool = False,
) -> None:
    logger.info("=" * 70)
    logger.info("DIMENSIONALITY REDUCTION & FEATURE SELECTION")
    logger.info("=" * 70)

    if not skip_pca:
        pca_results = run_pca_analysis(
            model_path=model_path,
            dataset_dir=dataset_dir,
            feature_type=feature_type,
            reports_dir=reports_dir,
            batch_size=batch_size,
        )
        logger.info(
            "PCA summary: %d components → 95%% variance (embedding dim=%d)",
            pca_results["n_components_for_95pct_variance"],
            pca_results["embedding_dim"],
        )
    else:
        logger.info("PCA analysis skipped.")

    if not skip_mfcc:
        mfcc_results = run_mfcc_feature_selection(
            dataset_dir=dataset_dir,
            reports_dir=reports_dir,
            k=k_mfcc,
        )
        logger.info(
            "MFCC selection: top %d features are indices %s",
            k_mfcc, mfcc_results["top_5_features"],
        )
    else:
        logger.info("MFCC feature selection skipped.")

    logger.info("=" * 70)
    logger.info("DIMENSIONALITY REDUCTION COMPLETE — outputs in %s", reports_dir)
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PCA + MFCC feature selection for rubric compliance"
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(r"C:\Users\holde\Documents\MLProject\models\feature_sweep_stagea6\fft\cnn_gunshot_classifier.keras"),
        help="Path to the trained .keras model (for PCA embedding extraction)",
    )
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
        help="Spectrogram type used by the model (for PCA val-set loading)",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Where to save all output plots and JSON files",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--k-mfcc",
        type=int,
        default=20,
        help="Number of top MFCC features to select",
    )
    parser.add_argument("--skip-pca", action="store_true", help="Skip PCA analysis")
    parser.add_argument("--skip-mfcc", action="store_true", help="Skip MFCC selection")

    args = parser.parse_args()
    main(
        model_path=args.model_path,
        dataset_dir=args.dataset_dir,
        feature_type=args.feature_type,
        reports_dir=args.reports_dir,
        batch_size=args.batch_size,
        k_mfcc=args.k_mfcc,
        skip_pca=args.skip_pca,
        skip_mfcc=args.skip_mfcc,
    )
