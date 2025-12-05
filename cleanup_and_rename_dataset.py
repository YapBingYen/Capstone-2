"""
Cleanup and Rename Dataset Script
=================================

Deletes empty folders and renames remaining folders sequentially (0001, 0002, 0003, ...)
to ensure continuous numbering without gaps.

Dataset directory:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset

Requirements:
  - pathlib (standard library)
  - tqdm (for progress bar, optional)
"""

import os
import shutil
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

DATASET_DIR = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset")

# Supported image extensions
SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    """Return True if path has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_IMAGE_EXT


def folder_has_images(folder_path: Path) -> bool:
    """Check if a folder contains any image files."""
    if not folder_path.exists() or not folder_path.is_dir():
        return False
    return any(item.is_file() and is_image_file(item) for item in folder_path.iterdir())


def get_non_empty_folders(dataset_dir: Path) -> List[Tuple[Path, str]]:
    """
    Get all non-empty folders with their original names.
    
    Returns:
        List of (folder_path, original_name) tuples
    """
    non_empty = []
    
    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir():
            if folder_has_images(folder):
                non_empty.append((folder, folder.name))
            else:
                print(f"Found empty folder: {folder.name}")
    
    return non_empty


def delete_empty_folders(dataset_dir: Path) -> int:
    """Delete all empty folders from the dataset."""
    deleted_count = 0
    
    folders_to_delete = []
    for folder in dataset_dir.iterdir():
        if folder.is_dir() and not folder_has_images(folder):
            folders_to_delete.append(folder)
    
    if folders_to_delete:
        print(f"\nDeleting {len(folders_to_delete)} empty folders...")
        for folder in folders_to_delete:
            try:
                shutil.rmtree(folder)
                deleted_count += 1
                print(f"  Deleted: {folder.name}")
            except Exception as e:
                print(f"  ERROR: Failed to delete {folder.name}: {e}")
    
    return deleted_count


def rename_folders_sequentially(dataset_dir: Path, non_empty_folders: List[Tuple[Path, str]]) -> dict:
    """
    Rename folders sequentially (0001, 0002, 0003, ...).
    
    Args:
        dataset_dir: Root dataset directory
        non_empty_folders: List of (folder_path, original_name) tuples
    
    Returns:
        Dictionary mapping old names to new names
    """
    rename_map = {}
    temp_renames = []
    
    # First pass: Rename to temporary names to avoid conflicts
    print("\nStep 1: Renaming to temporary names...")
    for i, (folder_path, original_name) in enumerate(non_empty_folders, 1):
        temp_name = f"__temp_{i:04d}__"
        temp_path = dataset_dir / temp_name
        
        if folder_path.name != temp_name:
            try:
                folder_path.rename(temp_path)
                temp_renames.append((temp_path, original_name, i))
                rename_map[original_name] = f"{i:04d}"
            except Exception as e:
                print(f"  ERROR: Failed to rename {original_name} to {temp_name}: {e}")
    
    # Second pass: Rename from temporary names to final sequential names
    print("\nStep 2: Renaming to final sequential names...")
    for temp_path, original_name, new_num in temp_renames:
        final_name = f"{new_num:04d}"
        final_path = dataset_dir / final_name
        
        try:
            temp_path.rename(final_path)
            print(f"  {original_name} -> {final_name}")
        except Exception as e:
            print(f"  ERROR: Failed to rename {temp_path.name} to {final_name}: {e}")
    
    return rename_map


def main():
    """Main function."""
    print("="*80)
    print("DATASET CLEANUP AND RENAME")
    print("="*80)
    
    if not DATASET_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATASET_DIR}")
        return
    
    if not DATASET_DIR.is_dir():
        print(f"ERROR: Path is not a directory: {DATASET_DIR}")
        return
    
    print(f"\nDataset directory: {DATASET_DIR}")
    
    # Step 1: Identify non-empty folders
    print("\n" + "="*80)
    print("STEP 1: Identifying non-empty folders...")
    print("="*80)
    
    non_empty_folders = get_non_empty_folders(DATASET_DIR)
    print(f"\nFound {len(non_empty_folders)} non-empty folders")
    
    # Count total images
    total_images = 0
    for folder_path, _ in non_empty_folders:
        image_count = sum(1 for item in folder_path.iterdir() 
                         if item.is_file() and is_image_file(item))
        total_images += image_count
    
    print(f"Total images: {total_images}")
    
    # Confirm before proceeding
    print("\n" + "="*80)
    print("WARNING: This will:")
    print("  1. DELETE all empty folders")
    print("  2. RENAME all remaining folders sequentially (0001, 0002, 0003, ...)")
    print("="*80)
    
    response = input("\nDo you want to proceed? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    # Step 2: Delete empty folders
    print("\n" + "="*80)
    print("STEP 2: Deleting empty folders...")
    print("="*80)
    
    deleted_count = delete_empty_folders(DATASET_DIR)
    print(f"\nDeleted {deleted_count} empty folders")
    
    # Step 3: Rename folders sequentially
    print("\n" + "="*80)
    print("STEP 3: Renaming folders sequentially...")
    print("="*80)
    
    # Re-scan after deletion
    non_empty_folders = get_non_empty_folders(DATASET_DIR)
    
    if not non_empty_folders:
        print("ERROR: No non-empty folders found after deletion!")
        return
    
    rename_map = rename_folders_sequentially(DATASET_DIR, non_empty_folders)
    
    # Final summary
    print("\n" + "="*80)
    print("CLEANUP COMPLETE!")
    print("="*80)
    print(f"Deleted empty folders: {deleted_count}")
    print(f"Renamed folders: {len(rename_map)}")
    print(f"Final folder count: {len(rename_map)}")
    print(f"Total images: {total_images}")
    
    # Save rename mapping to file
    mapping_file = DATASET_DIR.parent / "folder_rename_mapping.txt"
    try:
        with open(mapping_file, 'w') as f:
            f.write("Folder Rename Mapping\n")
            f.write("="*80 + "\n")
            f.write(f"Old Name -> New Name\n")
            f.write("-"*80 + "\n")
            for old_name, new_name in sorted(rename_map.items(), key=lambda x: int(x[1])):
                f.write(f"{old_name} -> {new_name}\n")
        print(f"\nRename mapping saved to: {mapping_file}")
    except Exception as e:
        print(f"WARNING: Could not save rename mapping: {e}")
    
    print("\n" + "="*80)
    print("Dataset is now cleaned and sequentially numbered!")
    print("="*80)


if __name__ == "__main__":
    main()

