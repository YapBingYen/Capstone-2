"""
Check Empty Folders in Cropped Dataset
=====================================

Scans the cropped dataset directory and identifies which cat folders are empty
(contain no images). This helps identify cats whose images were all filtered out
due to blur detection or face detection failures.

Output directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v3\cat_individuals_dataset

Requirements:
  - pathlib (standard library)
  - tqdm (for progress bar, optional)
"""

import os
from pathlib import Path
from typing import List, Tuple

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("WARNING: tqdm not installed. Progress bar disabled. Install with: pip install tqdm")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped_v3\cat_individuals_dataset")

# Supported image extensions
SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXT


def count_images_in_folder(folder_path: Path) -> int:
    """Count the number of image files in a folder."""
    if not folder_path.is_dir():
        return 0
    
    count = 0
    for item in folder_path.iterdir():
        if item.is_file() and is_image_file(item):
            count += 1
    return count


def scan_dataset(dataset_dir: Path) -> Tuple[List[str], List[Tuple[str, int]], int, int]:
    """
    Scan the dataset directory and identify empty folders.
    
    Returns:
        empty_folders: List of folder names that are empty
        non_empty_folders: List of (folder_name, image_count) tuples for non-empty folders
        total_folders: Total number of cat folders found
        total_images: Total number of images across all folders
    """
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        return [], [], 0, 0
    
    if not dataset_dir.is_dir():
        print(f"ERROR: Path is not a directory: {dataset_dir}")
        return [], [], 0, 0
    
    empty_folders = []
    non_empty_folders = []
    total_images = 0
    
    # Get all subdirectories (cat folders)
    cat_folders = [f for f in dataset_dir.iterdir() if f.is_dir()]
    cat_folders.sort(key=lambda x: x.name)
    
    total_folders = len(cat_folders)
    
    if HAS_TQDM:
        iterator = tqdm(cat_folders, desc="Scanning folders", unit="folder")
    else:
        iterator = cat_folders
        print(f"Scanning {total_folders} folders...")
    
    for folder in iterator:
        folder_name = folder.name
        image_count = count_images_in_folder(folder)
        
        if image_count == 0:
            empty_folders.append(folder_name)
        else:
            non_empty_folders.append((folder_name, image_count))
            total_images += image_count
    
    return empty_folders, non_empty_folders, total_folders, total_images


def print_report(empty_folders: List[str], non_empty_folders: List[Tuple[str, int]], 
                 total_folders: int, total_images: int, dataset_dir: Path):
    """Print a detailed report of empty and non-empty folders."""
    print("\n" + "="*80)
    print("EMPTY FOLDERS REPORT")
    print("="*80)
    print(f"\nDataset directory: {dataset_dir}")
    print(f"\nSummary:")
    print(f"   Total cat folders: {total_folders}")
    print(f"   Empty folders: {len(empty_folders)}")
    print(f"   Non-empty folders: {len(non_empty_folders)}")
    print(f"   Total images: {total_images}")
    
    if len(non_empty_folders) > 0:
        avg_images = total_images / len(non_empty_folders)
        print(f"   Average images per non-empty folder: {avg_images:.1f}")
    
    # Show empty folders
    if empty_folders:
        print(f"\nEmpty Folders ({len(empty_folders)}):")
        print("-" * 80)
        
        # Group by ranges if there are many
        if len(empty_folders) > 50:
            print("   (Showing first 50, saving full list to file...)")
            empty_list_path = dataset_dir.parent / "empty_folders_list.txt"
            with open(empty_list_path, 'w') as f:
                for folder in empty_folders:
                    f.write(f"{folder}\n")
            print(f"   Full list saved to: {empty_list_path}")
            
            # Show first 50
            for i, folder in enumerate(empty_folders[:50], 1):
                print(f"   {i:4d}. {folder}")
            print(f"   ... and {len(empty_folders) - 50} more (see {empty_list_path})")
        else:
            for i, folder in enumerate(empty_folders, 1):
                print(f"   {i:4d}. {folder}")
    else:
        print("\nNo empty folders found! All cat folders contain images.")
    
    # Show statistics for non-empty folders
    if non_empty_folders:
        print(f"\nNon-Empty Folders ({len(non_empty_folders)}):")
        print("-" * 80)
        
        # Sort by image count (descending)
        sorted_non_empty = sorted(non_empty_folders, key=lambda x: x[1], reverse=True)
        
        # Show top 10 and bottom 10
        print("\n   Top 10 folders (most images):")
        for i, (folder, count) in enumerate(sorted_non_empty[:10], 1):
            print(f"   {i:2d}. {folder}: {count} images")
        
        if len(sorted_non_empty) > 20:
            print(f"\n   ... ({len(sorted_non_empty) - 20} folders in between) ...")
            print("\n   Bottom 10 folders (fewest images):")
            for i, (folder, count) in enumerate(sorted_non_empty[-10:], 1):
                print(f"   {i:2d}. {folder}: {count} images")
        elif len(sorted_non_empty) > 10:
            print("\n   Remaining folders:")
            for i, (folder, count) in enumerate(sorted_non_empty[10:], 11):
                print(f"   {i:2d}. {folder}: {count} images")
    
    print("\n" + "="*80)
    
    # Recommendations
    if empty_folders:
        print("\nRecommendations:")
        print("   - Empty folders may indicate:")
        print("     • All images were too blurry (below BLUR_THRESHOLD)")
        print("     • No cat faces were detected in any images")
        print("     • Images failed to load or process")
        print("   - Consider:")
        print("     • Lowering BLUR_THRESHOLD in crop_cats_clean.py")
        print("     • Checking the original images in the input dataset")
        print("     • Manually reviewing failed images")
        print("="*80)


def main():
    """Main function."""
    print("="*80)
    print("EMPTY FOLDERS CHECKER")
    print("="*80)
    
    empty_folders, non_empty_folders, total_folders, total_images = scan_dataset(DATASET_DIR)
    
    print_report(empty_folders, non_empty_folders, total_folders, total_images, DATASET_DIR)


if __name__ == "__main__":
    main()

