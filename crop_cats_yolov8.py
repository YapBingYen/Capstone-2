"""
YOLOv8 Cat Face Detection and Cropping Script
============================================

Uses YOLOv8 for more accurate cat detection, then focuses on the cat's face region
for better face recognition. More accurate than Haar Cascade.

Input directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals\cat_individuals_dataset

Output directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v4\cat_individuals_dataset

Requirements:
  - ultralytics (YOLOv8)
  - opencv-python
  - tqdm
  - numpy

Installation:
    pip install ultralytics opencv-python tqdm numpy

Behavior:
  - Uses YOLOv8 for cat detection (more accurate than Haar Cascade)
  - TIGHT cropping: Focuses on cat's face region (upper 40% of detection)
  - Uses only 65% of width (centered) to minimize background
  - Very minimal padding (5%) for tight face-focused crops
  - Reduces background noise for better model accuracy
  - For each image: detect cat; crop tight face region; resize to 224x224; save
  - If no cat is detected, the image is skipped (no blank outputs)
  - Preserves subfolder structure
  - Uses two-pass approach with blur filtering
  - Ensures each folder has at least one quality image
"""

import os
import sys
from pathlib import Path
from typing import Tuple, Optional

try:
    from ultralytics import YOLO
    import cv2
    import numpy as np
    from tqdm import tqdm
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\nPlease install required packages:")
    print("  pip install ultralytics opencv-python tqdm numpy")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals\cat_individuals_dataset")
OUTPUT_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v4\cat_individuals_dataset")

# Target crop size
TARGET_SIZE: Tuple[int, int] = (224, 224)

# Blur thresholds (variance of Laplacian)
STRICT_BLUR_THRESHOLD: float = 120.0  # First pass: high quality
MODERATE_BLUR_THRESHOLD: float = 80.0  # Second pass: moderate quality for empty folders

# YOLOv8 model configuration
# Using YOLOv8n (nano) for speed, can use YOLOv8s/m/l for better accuracy
YOLO_MODEL_SIZE = "n"  # Options: n (nano), s (small), m (medium), l (large), x (xlarge)
YOLO_CONFIDENCE_THRESHOLD = 0.25  # Detection confidence threshold
YOLO_IOU_THRESHOLD = 0.45  # NMS IoU threshold

# Face-focused cropping configuration (tight cropping for better accuracy)
# Cat faces are typically in the upper portion of the cat's body
FACE_REGION_HEIGHT_RATIO = 0.4  # Focus on upper 40% of cat detection (face + ears region)
FACE_WIDTH_RATIO = 0.65  # Use 65% of width (tighter, less background)
FACE_PADDING_RATIO = 0.05  # Add only 5% padding (very minimal background)
MIN_FACE_SIZE = 50  # Minimum face crop size (pixels)
# Tighter cropping reduces background noise and improves model accuracy


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yolo_model() -> YOLO:
    """Load YOLOv8 model for object detection."""
    model_name = f"yolov8{YOLO_MODEL_SIZE}.pt"
    print(f"\nLoading YOLOv8 model: {model_name}")
    print("Note: First run will download the model (~6-25 MB depending on size)")
    
    try:
        model = YOLO(model_name)
        print("✅ YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load YOLOv8 model: {e}")
        print("\nTroubleshooting:")
        print("  1. Check internet connection (first download)")
        print("  2. Try: pip install --upgrade ultralytics")
        print("  3. Check available disk space")
        sys.exit(1)


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def detect_cat_yolo(img_bgr: np.ndarray, yolo_model: YOLO) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect cat using YOLOv8 and return face-focused bounding box.
    
    Args:
        img_bgr: BGR image array
        yolo_model: Loaded YOLOv8 model
    
    Returns:
        Tuple of (x, y, w, h) bounding box focused on cat's face region, or None
    """
    # YOLOv8 expects RGB, but cv2 gives BGR
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Run inference
    results = yolo_model.predict(
        img_rgb,
        conf=YOLO_CONFIDENCE_THRESHOLD,
        iou=YOLO_IOU_THRESHOLD,
        verbose=False,
        classes=[15]  # COCO class 15 = "cat"
    )
    
    if len(results) == 0 or len(results[0].boxes) == 0:
        return None
    
    # Get all cat detections
    boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 format
    
    if len(boxes) == 0:
        return None
    
    # Find largest detection by area
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
    largest_idx = np.argmax(areas)
    largest_box = boxes[largest_idx]
    
    # Get full cat bounding box
    cat_x1 = int(largest_box[0])
    cat_y1 = int(largest_box[1])
    cat_x2 = int(largest_box[2])
    cat_y2 = int(largest_box[3])
    cat_w = cat_x2 - cat_x1
    cat_h = cat_y2 - cat_y1
    
    # Focus on upper portion where face is located (tighter region)
    face_height = int(cat_h * FACE_REGION_HEIGHT_RATIO)
    face_y1 = cat_y1
    face_y2 = cat_y1 + face_height
    
    # Focus on center portion horizontally (not full width - reduces background)
    # Face is typically centered, so use configurable percentage of width
    face_center_x = (cat_x1 + cat_x2) // 2
    face_width = int(cat_w * FACE_WIDTH_RATIO)  # Use configurable width ratio (tighter, less background)
    
    # Calculate centered face bounding box
    face_x1 = face_center_x - face_width // 2
    face_x2 = face_center_x + face_width // 2
    
    # Ensure face box is within cat detection bounds
    face_x1 = max(cat_x1, face_x1)
    face_x2 = min(cat_x2, face_x2)
    face_w = face_x2 - face_x1
    
    # Add minimal padding around face region (reduced padding)
    padding_x = int(face_w * FACE_PADDING_RATIO)
    padding_y = int(face_height * FACE_PADDING_RATIO)
    
    # Expand bounding box with minimal padding
    face_x1 = max(0, face_x1 - padding_x)
    face_y1 = max(0, face_y1 - padding_y)
    face_x2 = min(img_bgr.shape[1], face_x2 + padding_x)
    face_y2 = min(img_bgr.shape[0], face_y2 + padding_y)
    
    # Calculate dimensions
    face_w = face_x2 - face_x1
    face_h = face_y2 - face_y1
    
    # Ensure minimum size (only if really needed)
    if face_w < MIN_FACE_SIZE:
        diff = MIN_FACE_SIZE - face_w
        face_x1 = max(0, face_x1 - diff // 2)
        face_x2 = min(img_bgr.shape[1], face_x2 + diff // 2)
        face_w = face_x2 - face_x1
    
    if face_h < MIN_FACE_SIZE:
        diff = MIN_FACE_SIZE - face_h
        face_y1 = max(0, face_y1 - diff // 2)
        face_y2 = min(img_bgr.shape[0], face_y2 + diff // 2)
        face_h = face_y2 - face_y1
    
    # Make slightly square (but don't over-expand - keep it tight)
    # Only adjust if aspect ratio is very different (>50% difference)
    aspect_diff = abs(face_w - face_h) / min(face_w, face_h) if min(face_w, face_h) > 0 else 1.0
    if aspect_diff > 0.5:  # Only if >50% difference
        if face_w < face_h:
            # Slightly expand width (but limit expansion)
            diff = min(face_h - face_w, int(face_w * 0.2))  # Max 20% expansion
            face_x1 = max(0, face_x1 - diff // 2)
            face_x2 = min(img_bgr.shape[1], face_x2 + diff // 2)
        else:
            # Slightly expand height (but limit expansion)
            diff = min(face_w - face_h, int(face_h * 0.2))  # Max 20% expansion
            face_y1 = max(0, face_y1 - diff // 2)
            face_y2 = min(img_bgr.shape[0], face_y2 + diff // 2)
    
    # Final dimensions
    x = face_x1
    y = face_y1
    w = face_x2 - face_x1
    h = face_y2 - face_y1
    
    return (x, y, w, h)


def crop_largest_cat(img_bgr: np.ndarray, yolo_model: YOLO) -> Optional[np.ndarray]:
    """
    Detect cat using YOLOv8 and return TIGHT face-focused cropped region.
    
    The function uses tight cropping to minimize background:
    - Focuses on upper 40% of cat detection (face + ears region)
    - Uses only 65% of width (centered) to reduce side background
    - Very minimal padding (5%) for tight face crops
    - Optimized for better model accuracy by reducing background noise
    
    Returns the tightly cropped BGR image focused on cat's face if a cat is found; 
    otherwise returns None.
    """
    bbox = detect_cat_yolo(img_bgr, yolo_model)
    if bbox is None:
        return None
    
    x, y, w, h = bbox
    
    # Ensure coordinates are within image bounds
    h_img, w_img = img_bgr.shape[:2]
    x = max(0, min(x, w_img - 1))
    y = max(0, min(y, h_img - 1))
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    
    if w <= 0 or h <= 0:
        return None
    
    return img_bgr[y:y + h, x:x + w]


def folder_has_images(folder_path: Path) -> bool:
    """Check if a folder contains any image files."""
    if not folder_path.exists() or not folder_path.is_dir():
        return False
    return any(item.is_file() and is_image_file(item) for item in folder_path.iterdir())


def process_images_pass(image_paths: list, input_dir: Path, output_dir: Path,
                        yolo_model: YOLO, blur_threshold: float,
                        pass_name: str, skip_existing: bool = False) -> Tuple[int, int, int]:
    """
    Process images with a given blur threshold using YOLOv8.
    
    Args:
        image_paths: List of image paths to process
        input_dir: Input directory root
        output_dir: Output directory root
        yolo_model: Loaded YOLOv8 model
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
        
        # Detect and crop cat's face using YOLOv8
        cropped_img = crop_largest_cat(img, yolo_model)
        if cropped_img is None or cropped_img.size == 0:
            tqdm.write(f"⚠️  No cat detected; skipped: {src_path}")
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
    """Process all images under input_dir using YOLOv8 and a two-pass approach."""
    yolo_model = load_yolo_model()
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
        all_image_paths, input_dir, output_dir, yolo_model,
        STRICT_BLUR_THRESHOLD, "Pass 1: Strict quality (YOLOv8)", skip_existing=False
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
            empty_folder_paths, input_dir, output_dir, yolo_model,
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
            print("   - No detectable cats in any images")
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

