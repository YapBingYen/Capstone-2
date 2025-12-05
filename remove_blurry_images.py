"""
Remove Blurry Cat Images
========================

Utility script that scans the individual cat dataset and deletes blurry
images using the variance of Laplacian focus measure.

Dataset root:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset

Images with variance of Laplacian < BLUR_THRESHOLD (default 80) are removed.
"""

import os
from pathlib import Path
from typing import Dict, Tuple

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset")
BLUR_THRESHOLD = 80.0  # Variance of Laplacian value below which the image is considered blurry
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Return True if the file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def laplacian_variance(image_path: Path) -> float:
    """Compute the variance of the Laplacian of the image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 0.0
    lap = cv2.Laplacian(image, cv2.CV_64F)
    return lap.var()


def remove_blurry_images(dataset_root: Path, threshold: float) -> Tuple[int, int]:
    """
    Remove blurry images across all subdirectories.

    Returns:
        total_removed: Number of images deleted
        total_remaining: Number of images left after deletion
    """
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    total_removed = 0
    total_remaining = 0

    print("=" * 80)
    print(f"Scanning dataset for blurry images (threshold={threshold})")
    print("=" * 80)

    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue

        removed_in_folder = 0
        images = [p for p in cat_dir.iterdir() if p.is_file() and is_image_file(p)]

        for image_path in images:
            variance = laplacian_variance(image_path)
            if variance < threshold:
                try:
                    image_path.unlink()
                    removed_in_folder += 1
                except OSError as exc:
                    print(f"⚠️  Could not delete {image_path}: {exc}")

        remaining_in_folder = len(images) - removed_in_folder
        total_removed += removed_in_folder
        total_remaining += remaining_in_folder

        print(f"{cat_dir.name}: removed {removed_in_folder}, remaining {remaining_in_folder}")

    print("=" * 80)
    print(f"Blur filtering complete: {total_removed} images removed, {total_remaining} images remain.")
    print("=" * 80)

    return total_removed, total_remaining


if __name__ == "__main__":
    remove_blurry_images(DATASET_ROOT, BLUR_THRESHOLD)

