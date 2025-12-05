"""
Simple smoke tests for YOLO Cat Face Cropper
===========================================

Run with: python test_cropper.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

# Import the cropper module
try:
    from yolo_cat_face_cropper import CatFaceCropper, parse_args, is_image_file
except ImportError:
    print("❌ Failed to import yolo_cat_face_cropper")
    print("Make sure yolo_cat_face_cropper.py is in the same directory")
    exit(1)


def create_test_image(width=640, height=480, color=(100, 150, 200)) -> np.ndarray:
    """Create a simple test image."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    return img


def create_test_dataset(temp_dir: Path):
    """Create a minimal test dataset structure."""
    # Create test cat folders
    cat1_dir = temp_dir / "CAT001"
    cat2_dir = temp_dir / "CAT002"
    cat1_dir.mkdir(parents=True)
    cat2_dir.mkdir(parents=True)
    
    # Create test images
    img1 = create_test_image()
    img2 = create_test_image(color=(200, 100, 150))
    
    cv2.imwrite(str(cat1_dir / "test1.jpg"), img1)
    cv2.imwrite(str(cat1_dir / "test2.jpg"), img2)
    cv2.imwrite(str(cat2_dir / "test1.jpg"), img1)
    
    return temp_dir


def test_basic_functionality():
    """Test basic cropper functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Functionality")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        # Create test dataset
        create_test_dataset(input_dir)
        
        # Create args
        class Args:
            input_dir = str(input_dir)
            output_dir = str(output_dir)
            weights = "yolov8n"
            conf = 0.25
            iou = 0.45
            padding = 0.20
            img_size = 224
            workers = 2  # Use fewer workers for test
            min_face_area = 1024
            save_json = str(output_dir / "metadata.json")
            fallback_haarcascade = True
            blur_threshold = 60.0
            remove_blurry = False
            min_images = 2
            verbose = True
        
        args = Args()
        
        try:
            # Run cropper
            cropper = CatFaceCropper(args)
            cropper.process_all()
            
            # Check outputs
            metadata_path = Path(args.save_json)
            assert metadata_path.exists(), "Metadata JSON not created"
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            assert "images" in metadata, "Metadata missing 'images' key"
            assert "summary" in metadata, "Metadata missing 'summary' key"
            assert metadata["summary"]["total_input"] > 0, "No images processed"
            
            print("✅ Test 1 PASSED: Basic functionality works")
            print(f"   Processed {metadata['summary']['total_input']} images")
            print(f"   Generated {metadata['summary']['total_output']} outputs")
            
            return True
            
        except Exception as e:
            print(f"❌ Test 1 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_utility_functions():
    """Test utility functions."""
    print("\n" + "="*80)
    print("TEST 2: Utility Functions")
    print("="*80)
    
    try:
        # Test is_image_file
        assert is_image_file(Path("test.jpg")) == True
        assert is_image_file(Path("test.png")) == True
        assert is_image_file(Path("test.txt")) == False
        
        # Test image creation
        img = create_test_image()
        assert img.shape == (480, 640, 3), "Image shape incorrect"
        
        print("✅ Test 2 PASSED: Utility functions work")
        return True
        
    except Exception as e:
        print(f"❌ Test 2 FAILED: {e}")
        return False


def test_metadata_structure():
    """Test metadata JSON structure."""
    print("\n" + "="*80)
    print("TEST 3: Metadata Structure")
    print("="*80)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        
        create_test_dataset(input_dir)
        
        class Args:
            input_dir = str(input_dir)
            output_dir = str(output_dir)
            weights = "yolov8n"
            conf = 0.25
            iou = 0.45
            padding = 0.20
            img_size = 224
            workers = 2
            min_face_area = 1024
            save_json = str(output_dir / "metadata.json")
            fallback_haarcascade = True
            blur_threshold = 60.0
            remove_blurry = False
            min_images = 2
            verbose = False
        
        args = Args()
        
        try:
            cropper = CatFaceCropper(args)
            cropper.process_all()
            
            with open(args.save_json, 'r') as f:
                metadata = json.load(f)
            
            # Validate structure
            assert isinstance(metadata["images"], list), "images should be list"
            assert isinstance(metadata["summary"], dict), "summary should be dict"
            
            # Check required summary fields
            required_fields = ["total_input", "total_output", "fallback_count", "blur_removed"]
            for field in required_fields:
                assert field in metadata["summary"], f"Missing field: {field}"
            
            # Check image entry structure (if any)
            if metadata["images"]:
                entry = metadata["images"][0]
                assert "original_path" in entry, "Missing original_path"
                assert "cat_id" in entry, "Missing cat_id"
                assert "timestamp" in entry, "Missing timestamp"
            
            print("✅ Test 3 PASSED: Metadata structure is correct")
            return True
            
        except Exception as e:
            print(f"❌ Test 3 FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("="*80)
    print("YOLO Cat Face Cropper - Smoke Tests")
    print("="*80)
    
    tests = [
        test_utility_functions,
        test_basic_functionality,
        test_metadata_structure,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    exit(main())

