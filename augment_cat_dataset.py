"""
Cat Dataset Augmentation Script
===============================

Ensures every cat (subfolder) has at least TARGET_IMAGES_PER_CAT images by
creating augmented copies using TensorFlow/Keras ImageDataGenerator.

Dataset root:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset

Augmented images are saved in-place with names like "aug_1.jpg".
"""

import os
from itertools import cycle
from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_ROOT = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset")
TARGET_IMAGES_PER_CAT = 10
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Image shape (dataset already 224x224, keep consistent)
IMG_SIZE = (224, 224)

# Data augmentation parameters
DATA_AUGMENTOR = keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2,
    brightness_range=(0.8, 1.2),
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode="nearest",
)


def is_image_file(path: Path) -> bool:
    """Return True if file is a supported image."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def next_aug_index(cat_dir: Path) -> int:
    """Determine next augmentation index based on existing aug_ files."""
    max_index = 0
    for file in cat_dir.iterdir():
        if file.is_file() and file.stem.startswith("aug_"):
            try:
                idx = int(file.stem.split("_")[1])
                max_index = max(max_index, idx)
            except (IndexError, ValueError):
                continue
    return max_index + 1


def augment_cat_folder(cat_dir: Path) -> int:
    """
    Augment images within a single cat directory until it has enough images.

    Returns:
        Number of new images created.
    """
    images = sorted([p for p in cat_dir.iterdir() if p.is_file() and is_image_file(p)])

    # Separate originals (non-augmented) from already augmented
    original_images = [p for p in images if not p.name.startswith("aug_")]
    total_images = len(images)

    if total_images >= TARGET_IMAGES_PER_CAT or not original_images:
        return 0

    images_needed = TARGET_IMAGES_PER_CAT - total_images
    aug_index = next_aug_index(cat_dir)
    created = 0

    originals_cycle = cycle(original_images)

    for _ in range(images_needed):
        source_path = next(originals_cycle)
        img = keras.utils.load_img(source_path, target_size=IMG_SIZE)
        img_array = keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Generate one augmented sample
        aug_iter = DATA_AUGMENTOR.flow(img_array, batch_size=1, shuffle=False)
        batch = next(aug_iter)[0]
        aug_image = keras.utils.array_to_img(batch)

        save_path = cat_dir / f"aug_{aug_index}.jpg"
        aug_image.save(save_path, quality=95)
        aug_index += 1
        created += 1

    return created


def augment_dataset(dataset_root: Path) -> None:
    """Augment dataset so each cat has at least TARGET_IMAGES_PER_CAT images."""
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    total_created = 0
    cats_augmented = 0

    print("=" * 80)
    print(f"Augmenting cat dataset to ensure at least {TARGET_IMAGES_PER_CAT} images per cat")
    print("=" * 80)

    for cat_dir in sorted(dataset_root.iterdir()):
        if not cat_dir.is_dir():
            continue

        created = augment_cat_folder(cat_dir)
        if created > 0:
            total_created += created
            cats_augmented += 1
            print(f"{cat_dir.name}: created {created} new images")
        else:
            print(f"{cat_dir.name}: no augmentation needed")

    print("=" * 80)
    print(f"Augmentation complete: {total_created} new images created across {cats_augmented} cats.")
    print("=" * 80)


if __name__ == "__main__":
    augment_dataset(DATASET_ROOT)

