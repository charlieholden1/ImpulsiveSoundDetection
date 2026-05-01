"""
dataset.py – TensorFlow data pipeline for spectrogram PNG images.

Loads LogMel spectrogram images from the Gunshot Audio Spectrogram Dataset,
applies stratified train/val/test split, and returns tf.data.Dataset objects
with on-the-fly augmentation.
"""

from pathlib import Path
from typing import Tuple
import hashlib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


def load_spectrogram_dataset(
    dataset_dir: Path,
    feature_type: str = "LogMel",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    random_state: int = 42,
    deduplicate: bool = True,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    """
    Load spectrogram PNG images and return stratified train/val/test tf.data.Dataset objects.

    Parameters
    ----------
    dataset_dir : Path
        Root directory containing feature_type subdirectories (FFT, LogMel, MFCC).
    feature_type : str
        Spectrogram type to load: "LogMel", "FFT", or "MFCC".
    image_size : tuple
        Target image size (height, width) for model input.
    batch_size : int
        Batch size for tf.data.Dataset.
    random_state : int
        Seed for stratified split reproducibility.
    deduplicate : bool
        If True, remove exact duplicate spectrogram PNGs before splitting.

    Returns
    -------
    train_dataset, val_dataset, test_dataset : tf.data.Dataset
        Batched TF datasets with augmentation applied to train split only.
    class_weights : dict
        Dictionary {0: weight_nogun, 1: weight_gun} for weighted training.

    Raises
    ------
    FileNotFoundError
        If the feature_type subdirectories don't exist.
    """
    logger.info("Loading %s spectrogram dataset from %s", feature_type, dataset_dir)

    feature_dir = dataset_dir / feature_type
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

    # Discover all images: GUN → label 1, NOGUN → label 0
    gun_dir = feature_dir / "GUN"
    nogun_dir = feature_dir / "NOGUN"

    image_paths = []
    labels = []

    if gun_dir.exists():
        gun_images = sorted(gun_dir.glob("*.png"))
        image_paths.extend(gun_images)
        labels.extend([1] * len(gun_images))
        logger.info("Found %d GUN images", len(gun_images))
    else:
        logger.warning("GUN directory not found: %s", gun_dir)

    if nogun_dir.exists():
        nogun_images = sorted(nogun_dir.glob("*.png"))
        image_paths.extend(nogun_images)
        labels.extend([0] * len(nogun_images))
        logger.info("Found %d NOGUN images", len(nogun_images))
    else:
        logger.warning("NOGUN directory not found: %s", nogun_dir)

    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {feature_dir}")

    image_paths = np.array(image_paths)
    labels = np.array(labels)

    if deduplicate:
        image_paths, labels = _deduplicate_exact_images(image_paths, labels)

    # Stratified 70/15/15 split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )

    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.5,  # 0.5 of 0.30 = 0.15 each
        stratify=temp_labels,
        random_state=random_state,
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(train_paths),
        len(val_paths),
        len(test_paths),
    )

    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(train_labels), y=train_labels
    )
    class_weights_dict = {i: w for i, w in enumerate(class_weights)}
    logger.info("Class weights: %s", class_weights_dict)

    # Create tf.data.Dataset objects
    train_ds = _create_dataset(
        train_paths, train_labels, image_size, batch_size, augment=True
    )
    val_ds = _create_dataset(
        val_paths, val_labels, image_size, batch_size, augment=False
    )
    test_ds = _create_dataset(
        test_paths, test_labels, image_size, batch_size, augment=False
    )

    return train_ds, val_ds, test_ds, class_weights_dict


def _deduplicate_exact_images(
    image_paths: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove exact duplicate images that share the same label.

    Exact duplicates can leak near-identical samples across train/val/test and
    distort validation metrics, especially in pre-rendered spectrogram datasets.
    """
    kept_paths = []
    kept_labels = []
    seen_hashes: dict[str, Tuple[Path, int]] = {}
    removed = 0
    conflicting = 0

    for path, label in zip(image_paths, labels):
        digest = hashlib.md5(Path(path).read_bytes()).hexdigest()
        previous = seen_hashes.get(digest)
        if previous is None:
            seen_hashes[digest] = (Path(path), int(label))
            kept_paths.append(path)
            kept_labels.append(label)
            continue

        previous_path, previous_label = previous
        if previous_label != int(label):
            conflicting += 1
            logger.warning(
                "Found identical images with conflicting labels: %s vs %s. Keeping both.",
                previous_path,
                path,
            )
            kept_paths.append(path)
            kept_labels.append(label)
            continue

        removed += 1

    if removed > 0:
        logger.info("Removed %d exact duplicate spectrograms before splitting.", removed)
    if conflicting > 0:
        logger.warning("Found %d conflicting duplicate hashes across classes.", conflicting)

    return np.array(kept_paths), np.array(kept_labels)


def _create_dataset(
    image_paths: np.ndarray,
    labels: np.ndarray,
    image_size: Tuple[int, int],
    batch_size: int,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from image paths and labels.

    Parameters
    ----------
    image_paths : np.ndarray
        Array of Path objects or strings pointing to PNG files.
    labels : np.ndarray
        Array of integer labels (0 or 1).
    image_size : tuple
        Target (height, width) for resize.
    batch_size : int
        Batch size.
    augment : bool
        If True, apply training augmentations.

    Returns
    -------
    tf.data.Dataset
        Batched dataset yielding (image_tensor, label_tensor) tuples.
    """
    # Convert to list of strings for tf.data
    paths_list = [str(p) for p in image_paths]
    labels_list = labels.tolist()

    ds = tf.data.Dataset.from_tensor_slices((paths_list, labels_list))

    # Shuffle training data
    if augment:
        ds = ds.shuffle(buffer_size=len(paths_list), reshuffle_each_iteration=True)

    # Load and preprocess images
    ds = ds.map(
        lambda path, label: _load_and_preprocess_image(path, label, image_size, augment),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds


def _load_and_preprocess_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int],
    augment: bool = False,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load PNG image, resize, and optionally augment.

    IMPORTANT: Do NOT normalize here! EfficientNetB0's preprocess_input()
    handles normalization (expects [0, 255] uint8 OR float32 [0, 1]).
    We pass uint8 directly and let preprocess_input() standardize to [-1, 1].

    Parameters
    ----------
    image_path : tf.Tensor
        String tensor pointing to PNG file.
    label : tf.Tensor
        Integer label (0 or 1).
    image_size : tuple
        Target (height, width).
    augment : bool
        If True, apply random augmentations.

    Returns
    -------
    image, label : (tf.Tensor, tf.Tensor)
        Preprocessed image (uint8 or float32) and label. Will be normalized by model's preprocess_input().
    """
    # Load image (stays as uint8)
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Load as RGB, uint8

    # Resize (keeps uint8 dtype)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.uint8)  # Ensure uint8 for consistency

    # Augmentation (training only) - operate on uint8
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.cast(image, tf.float32)  # Convert to float for augmentation
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.cast(image, tf.uint8)  # Convert back to uint8

    return image, label
