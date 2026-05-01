"""
Helpers for external raw-audio datasets used in benchmarking.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator
from urllib.request import urlretrieve
import zipfile

logger = logging.getLogger(__name__)

REALISED_URL = "https://zenodo.org/records/6488321/files/ReaLISED_Dataset.zip?download=1"


def ensure_realised_dataset(dataset_root: Path) -> Path:
    """
    Download and extract the ReaLISED dataset if it is not already present.
    """
    dataset_root = Path(dataset_root)
    extract_dir = dataset_root / "ReaLISED"
    archive_path = dataset_root / "ReaLISED_Dataset.zip"

    if any(extract_dir.rglob("*.wav")) or any(extract_dir.rglob("*.flac")):
        logger.info("ReaLISED dataset already present at %s", extract_dir)
        return extract_dir

    dataset_root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading ReaLISED dataset from %s", REALISED_URL)
    urlretrieve(REALISED_URL, archive_path)

    logger.info("Extracting %s", archive_path)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(extract_dir)

    return extract_dir


def iter_audio_files(root_dir: Path) -> Iterator[Path]:
    """
    Yield audio files under a directory.
    """
    root_dir = Path(root_dir)
    for suffix in ("*.wav", "*.flac", "*.mp3"):
        yield from root_dir.rglob(suffix)


def collect_audio_files(root_dir: Path, limit: int | None = None) -> list[Path]:
    """
    Collect audio files from a directory, optionally truncating the list.
    """
    files = sorted(iter_audio_files(root_dir))
    if limit is not None:
        files = files[:limit]
    return files
