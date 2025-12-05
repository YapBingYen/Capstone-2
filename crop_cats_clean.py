"""
Clean Cat-Face Cropping Script for Individual Dataset
=====================================================

Detects and crops cat faces for each image in the individual cats dataset.
Saves cropped images (224x224) while preserving the folder structure.

Uses a two-pass approach:
  - Pass 1: Strict blur threshold (120.0) for high-quality images
  - Pass 2: Moderate blur threshold (80.0) for empty folders to ensure coverage

Input directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals\cat_individuals_dataset

Output directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v3\cat_individuals_dataset

Requirements:
  - OpenCV (cv2)
  - tqdm (for progress bar)

Behavior:
  - Uses cv2's built-in Haar Cascade: cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
  - For each image: detect faces; if found, crop the largest face, resize to 224x224; save
  - If no face is detected, the image is skipped (no blank outputs)
  - Preserves subfolder structure
  - Ensures each folder has at least one quality image (suitable for augmentation)
  - Prints a processing summary at the end
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals\cat_individuals_dataset")
OUTPUT_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v3\cat_individuals_dataset")

# Target crop size
TARGET_SIZE: Tuple[int, int] = (224, 224)
# Blur thresholds (variance of Laplacian)
STRICT_BLUR_THRESHOLD: float = 120.0  # First pass: high quality
MODERATE_BLUR_THRESHOLD: float = 80.0  # Second pass: moderate quality for empty folders


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_haar_cascade() -> cv2.CascadeClassifier:
    """Load OpenCV's cat face cascade from the local installation."""
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface.xml")
    if not os.path.exists(cascade_path):
        print(f"❌ Haar cascade not found at: {cascade_path}")
        print("   Please ensure OpenCV is installed correctly.")
        sys.exit(1)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        print("❌ Failed to load Haar cascade for cat faces.")
        sys.exit(1)
    return cascade


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def crop_largest_face(img_bgr: np.ndarray, face_cascade: cv2.CascadeClassifier) -> np.ndarray:
    """Detect cat faces and return a cropped region for the largest detected face.

    Returns the cropped BGR image if a face is found; otherwise returns None.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    if len(faces) == 0:
        return None
    # Pick largest face by area
    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    return img_bgr[y:y + h, x:x + w]


def folder_has_images(folder_path: Path) -> bool:
    """Check if a folder contains any image files."""
    if not folder_path.exists() or not folder_path.is_dir():
        return False
    return any(item.is_file() and is_image_file(item) for item in folder_path.iterdir())


def process_images_pass(image_paths: list, input_dir: Path, output_dir: Path, 
                        face_cascade: cv2.CascadeClassifier, blur_threshold: float,
                        pass_name: str, skip_existing: bool = False) -> Tuple[int, int, int]:
    """
    Process images with a given blur threshold.
    
    Args:
        image_paths: List of image paths to process
        input_dir: Input directory root
        output_dir: Output directory root
        face_cascade: Loaded Haar cascade classifier
        blur_threshold: Blur threshold to use
        pass_name: Name of the pass (for progress bar)
        skip_existing: If True, skip images that already exist in output
    
    Returns:
        Tuple of (processed_count, cropped_count, blurry_count)
    """
    processed = 0
    cropped = 0
    blurry = 0
    
    for src_path in tqdm(image_paths, desc=pass_name, unit="img"):
        # Compute destination path, preserving folder structure relative to input_dir
        rel = src_path.relative_to(input_dir)
        dst_dir = output_dir / rel.parent
        ensure_dir(dst_dir)
        dst_path = dst_dir / src_path.name
        
        # Skip if image already exists and skip_existing is True
        if skip_existing and dst_path.exists():
            continue
        
        processed += 1
        
        # Read image
        img = cv2.imread(str(src_path))
        if img is None:
            tqdm.write(f"⚠️  Skipping unreadable image: {src_path}")
            continue
        
        # Detect and crop
        cropped_img = crop_largest_face(img, face_cascade)
        if cropped_img is None or cropped_img.size == 0:
            tqdm.write(f"⚠️  No cat face detected; skipped: {src_path}")
            continue
        
        # Resize
        resized = cv2.resize(cropped_img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
        
        # Blur detection (variance of Laplacian)
        laplacian_var = cv2.Laplacian(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if laplacian_var < blur_threshold:
            blurry += 1
            tqdm.write(f"⚠️  Skipping blurry crop ({laplacian_var:.1f} < {blur_threshold}): {src_path}")
            continue
        
        # Save (keep JPEG/PNG as-is by extension)
        ok = cv2.imwrite(str(dst_path), resized)
        if ok:
            cropped += 1
        else:
            tqdm.write(f"⚠️  Failed to save: {dst_path}")
    
    return processed, cropped, blurry


def process_dataset(input_dir: Path, output_dir: Path) -> None:
    """Process all images under input_dir using a two-pass approach."""
    face_cascade = load_haar_cascade()
    ensure_dir(output_dir)
    
    # Collect all image files under input_dir
    all_image_paths = [p for p in input_dir.rglob('*') if p.is_file() and is_image_file(p)]
    
    if not all_image_paths:
        print(f"❌ No images found under: {input_dir}")
        return
    
    print("="*80)
    print("PASS 1: Strict Quality Threshold (120.0)")
    print("="*80)
    
    # Pass 1: Strict threshold
    p1_processed, p1_cropped, p1_blurry = process_images_pass(
        all_image_paths, input_dir, output_dir, face_cascade,
        STRICT_BLUR_THRESHOLD, "Pass 1: Strict quality", skip_existing=False
    )
    
    print(f"\n✅ Pass 1 Complete!")
    print(f"   Processed: {p1_processed} images")
    print(f"   Cropped and saved: {p1_cropped} images")
    print(f"   Skipped (blurry): {p1_blurry} images")
    
    # Identify empty folders
    print("\n" + "="*80)
    print("Identifying empty folders...")
    print("="*80)
    
    empty_folders = set()
    for cat_folder in input_dir.iterdir():
        if not cat_folder.is_dir():
            continue
        
        cat_id = cat_folder.name
        output_cat_folder = output_dir / cat_id
        
        if not folder_has_images(output_cat_folder):
            empty_folders.add(cat_id)
    
    print(f"Found {len(empty_folders)} empty folders after Pass 1")
    
    if empty_folders:
        print("\n" + "="*80)
        print(f"PASS 2: Moderate Quality Threshold (80.0) for {len(empty_folders)} empty folders")
        print("="*80)
        
        # Pass 2: Process only images from empty folders with moderate threshold
        empty_folder_paths = []
        for img_path in all_image_paths:
            rel = img_path.relative_to(input_dir)
            cat_id = rel.parts[0] if len(rel.parts) > 0 else None
            if cat_id in empty_folders:
                empty_folder_paths.append(img_path)
        
        p2_processed, p2_cropped, p2_blurry = process_images_pass(
            empty_folder_paths, input_dir, output_dir, face_cascade,
            MODERATE_BLUR_THRESHOLD, "Pass 2: Moderate quality (empty folders)", skip_existing=True
        )
        
        print(f"\n✅ Pass 2 Complete!")
        print(f"   Processed: {p2_processed} images")
        print(f"   Cropped and saved: {p2_cropped} images")
        print(f"   Skipped (blurry): {p2_blurry} images")
        
        # Check remaining empty folders
        still_empty = []
        for cat_id in empty_folders:
            output_cat_folder = output_dir / cat_id
            if not folder_has_images(output_cat_folder):
                still_empty.append(cat_id)
        
        if still_empty:
            print(f"\n⚠️  Warning: {len(still_empty)} folders are still empty after Pass 2")
            print("   These folders may have:")
            print("   - No detectable cat faces in any images")
            print("   - All images too blurry even with relaxed threshold")
            print("   - Corrupted or unreadable images")
    else:
        print("\n✅ All folders have at least one image! No Pass 2 needed.")
        p2_processed, p2_cropped, p2_blurry = 0, 0, 0
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total images processed: {p1_processed + p2_processed}")
    print(f"Total cropped and saved: {p1_cropped + p2_cropped}")
    print(f"Total skipped for blur: {p1_blurry + p2_blurry}")
    print(f"\nPass 1 (strict, 120.0): {p1_cropped} images")
    print(f"Pass 2 (moderate, 80.0): {p2_cropped} images")
    print(f"\nCropped dataset saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    process_dataset(INPUT_DIR, OUTPUT_DIR)


