"""
YOLO-Based Cat Face Detection and Cropping Pipeline
===================================================

Production-ready script for detecting and cropping cat faces from images.
Uses YOLOv8 (ultralytics) with fallback to Haar Cascade for robustness.

Author: AI Assistant
Python: 3.9+
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: ultralytics not available. Will use Haar Cascade fallback only.")


# ============================================================================
# Configuration and Constants
# ============================================================================

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_PADDING = 0.20
DEFAULT_IMG_SIZE = 224
DEFAULT_WORKERS = 8
DEFAULT_MIN_FACE_AREA = 1024  # 32x32 pixels minimum
DEFAULT_BLUR_THRESHOLD = 60.0
DEFAULT_MIN_IMAGES = 2
JPEG_QUALITY = 95


# ============================================================================
# Utility Functions
# ============================================================================

def is_image_file(path: Path) -> bool:
    """Check if file is a supported image format."""
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_gpu_info() -> Dict[str, str]:
    """Detect GPU availability and return device info."""
    info = {"device": "cpu", "device_name": "CPU", "diagnostics": []}
    
    if ULTRALYTICS_AVAILABLE:
        try:
            import torch
            
            # Diagnostic: Check PyTorch CUDA support
            has_cuda_build = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
            info["diagnostics"].append(f"PyTorch CUDA available: {has_cuda_build}")
            
            # Check for CUDA (NVIDIA GPU)
            if has_cuda_build:
                try:
                    device_count = torch.cuda.device_count()
                    if device_count > 0:
                        info["device"] = "cuda"
                        info["device_name"] = torch.cuda.get_device_name(0)
                        info["device_count"] = device_count
                        info["diagnostics"].append(f"Found {device_count} CUDA device(s)")
                        return info
                except Exception as e:
                    info["diagnostics"].append(f"CUDA check error: {e}")
            
            # Check for MPS (Apple Silicon)
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                info["device"] = "mps"
                info["device_name"] = "Apple Silicon GPU"
                info["diagnostics"].append("MPS (Apple Silicon) available")
                return info
            
            # Check for DirectML (Windows GPU) - try to detect
            try:
                import importlib
                torch_directml = importlib.import_module("torch_directml")
                if torch_directml.is_available():
                    info["device"] = "dml"
                    info["device_name"] = "DirectML GPU"
                    info["diagnostics"].append("DirectML available")
                    return info
            except (ImportError, ModuleNotFoundError, AttributeError):
                info["diagnostics"].append("DirectML not available")
            
            # Even if torch.cuda.is_available() is False, YOLOv8 might still use GPU
            # Let YOLOv8 try to auto-detect (it's smarter about GPU detection)
            info["diagnostics"].append("Will let YOLOv8 auto-detect device (may use GPU even if PyTorch reports CPU)")
            info["device"] = "auto"
            info["device_name"] = "Auto-detect (YOLOv8 will try GPU first)"
                
        except Exception as e:
            info["diagnostics"].append(f"GPU detection error: {e}")
    
    return info


def compute_blur_score(image: np.ndarray) -> float:
    """Compute variance of Laplacian as blur score (higher = sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def expand_bbox(x1: int, y1: int, x2: int, y2: int, padding: float,
                img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Expand bounding box by padding fraction and clip to image bounds."""
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * padding)
    pad_h = int(h * padding)
    
    x1_new = max(0, x1 - pad_w)
    y1_new = max(0, y1 - pad_h)
    x2_new = min(img_w, x2 + pad_w)
    y2_new = min(img_h, y2 + pad_h)
    
    return x1_new, y1_new, x2_new, y2_new


def resize_image(image: np.ndarray, target_size: int) -> np.ndarray:
    """Resize image to target_size x target_size (square, preserving aspect ratio with center crop)."""
    h, w = image.shape[:2]
    
    # Calculate scale to fit
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Center crop to square
    if new_w != new_h:
        if new_w > new_h:
            start_x = (new_w - target_size) // 2
            resized = resized[:, start_x:start_x + target_size]
        else:
            start_y = (new_h - target_size) // 2
            resized = resized[start_y:start_y + target_size, :]
    
    return resized


def augment_image(image: np.ndarray, aug_type: str) -> np.ndarray:
    """Apply deterministic augmentation to image."""
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if aug_type == "flip":
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    elif aug_type == "rotate":
        pil_img = pil_img.rotate(5, fillcolor=(255, 255, 255))
    elif aug_type == "brightness":
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.2)
    elif aug_type == "contrast":
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.1)
    
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ============================================================================
# Detection Classes
# ============================================================================

class YOLODetector:
    """YOLOv8-based cat detector using ultralytics."""
    
    def __init__(self, weights: str, conf: float, iou: float, device=None):
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("ultralytics package not available")
        
        # Load model - YOLOv8 will auto-detect GPU if device=None
        self.model = YOLO(weights)
        self.conf = conf
        self.iou = iou
        # If device is None, YOLOv8 will auto-select (prefers GPU)
        # Otherwise use the specified device
        self.device = device if device is not None else None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect cats in image. Returns list of detections with bbox and confidence."""
        # Convert BGR to RGB for YOLO
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Build predict kwargs
        predict_kwargs = {
            "conf": self.conf,
            "iou": self.iou,
            "verbose": False,
            "classes": [15]  # COCO class 15 = "cat"
        }
        
        # Only add device if specified (None = auto-detect, which prefers GPU)
        if self.device is not None:
            predict_kwargs["device"] = self.device
        
        results = self.model.predict(img_rgb, **predict_kwargs)
        
        detections = []
        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "detector": "yolov8"
                })
        
        return detections


class HaarCascadeDetector:
    """Haar Cascade-based cat face detector (fallback)."""
    
    def __init__(self):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalcatface.xml")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade not found: {cascade_path}")
        
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError("Failed to load Haar cascade")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect cat faces using Haar Cascade."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": 0.8,  # Haar doesn't provide confidence, use default
                "detector": "haarcascade"
            })
        
        return detections


# ============================================================================
# Main Processing Pipeline
# ============================================================================

class CatFaceCropper:
    """Main processing pipeline for cat face detection and cropping."""
    
    def __init__(self, args):
        self.args = args
        self.metadata = {"images": [], "summary": {
            "total_input": 0,
            "total_output": 0,
            "fallback_count": 0,
            "blur_removed": 0,
            "skipped_small": 0,
            "errors": 0
        }}
        
        # Initialize detectors
        self.yolo_detector = None
        self.haar_detector = None
        
        if ULTRALYTICS_AVAILABLE:
            try:
                gpu_info = get_gpu_info()
                
                # Print diagnostics
                print(f"\n🔍 GPU Detection Diagnostics:")
                for diag in gpu_info.get("diagnostics", []):
                    print(f"   • {diag}")
                
                # Let YOLOv8 auto-detect and use GPU if available
                # YOLOv8 automatically uses GPU when available, so we pass None to let it decide
                device = None  # None = auto-detect (prefers GPU if available)
                
                # But if we detected a specific GPU type, use it explicitly
                if gpu_info["device"] == "cuda":
                    device = 0  # Use first CUDA GPU
                    print(f"✅ CUDA GPU detected: {gpu_info.get('device_name', 'NVIDIA GPU')}")
                elif gpu_info["device"] == "mps":
                    device = "mps"
                    print(f"✅ Apple Silicon GPU detected")
                elif gpu_info["device"] == "dml":
                    device = "dml"
                    print(f"✅ DirectML GPU detected")
                elif gpu_info["device"] == "auto":
                    print(f"⚠️  PyTorch reports CPU, but letting YOLOv8 auto-detect (may still use GPU)")
                    device = None  # Let YOLOv8 try GPU anyway
                else:
                    print(f"⚠️  No GPU detected. Using CPU.")
                    print(f"   💡 Tip: Install PyTorch with CUDA support: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                    device = None  # Still let YOLOv8 try
                
                # Initialize YOLO detector (will use GPU if available)
                print(f"\n📦 Loading YOLO model...")
                self.yolo_detector = YOLODetector(args.weights, args.conf, args.iou, device)
                
                # Verify what device YOLOv8 actually used
                actual_device = "unknown"
                try:
                    # Check model's device
                    if hasattr(self.yolo_detector.model, 'device'):
                        actual_device = str(self.yolo_detector.model.device)
                    elif hasattr(self.yolo_detector.model.model, 'device'):
                        actual_device = str(self.yolo_detector.model.model.device)
                    else:
                        # Try to infer from torch
                        import torch
                        if torch.cuda.is_available():
                            actual_device = f"cuda:{torch.cuda.current_device()}"
                        else:
                            actual_device = "cpu"
                except:
                    # Try to check via YOLOv8's device property
                    try:
                        if hasattr(self.yolo_detector.model, 'overrides') and 'device' in self.yolo_detector.model.overrides:
                            actual_device = str(self.yolo_detector.model.overrides['device'])
                    except:
                        pass
                
                print(f"✅ YOLO detector loaded: {args.weights}")
                print(f"   YOLOv8 using device: {actual_device}")
                
                # Warn if GPU was detected but CPU is being used
                if gpu_info["device"] != "cpu" and "cpu" in str(actual_device).lower():
                    print(f"\n⚠️  WARNING: GPU was detected but YOLOv8 is using CPU!")
                    print(f"   Possible causes:")
                    print(f"   • PyTorch installed without CUDA support (CPU-only version)")
                    print(f"   • CUDA drivers not properly installed")
                    print(f"   • GPU not accessible to PyTorch")
                    print(f"   Solution: Install PyTorch with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
                elif "cuda" in str(actual_device).lower() or "gpu" in str(actual_device).lower():
                    print(f"✅ GPU acceleration is active!")
            except Exception as e:
                print(f"⚠️  Failed to load YOLO: {e}")
                if not args.fallback_haarcascade:
                    raise
        
        if args.fallback_haarcascade or self.yolo_detector is None:
            try:
                self.haar_detector = HaarCascadeDetector()
                print("✅ Haar Cascade detector loaded")
            except Exception as e:
                print(f"⚠️  Failed to load Haar Cascade: {e}")
    
    def process_image(self, image_path: Path, cat_id: str) -> List[Dict]:
        """Process a single image and return metadata entries."""
        try:
            # Convert output_dir to Path if it's a string
            output_dir = Path(self.args.output_dir) if isinstance(self.args.output_dir, str) else self.args.output_dir
            
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return [{
                    "original_path": str(image_path),
                    "cat_id": cat_id,
                    "error": "Failed to load image",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }]
            
            img_h, img_w = image.shape[:2]
            detections = []
            
            # Try YOLO first
            if self.yolo_detector:
                try:
                    detections = self.yolo_detector.detect(image)
                except Exception as e:
                    if self.args.verbose:
                        print(f"YOLO detection failed for {image_path}: {e}")
            
            # Fallback to Haar Cascade if no detections
            if not detections and self.haar_detector:
                try:
                    detections = self.haar_detector.detect(image)
                except Exception as e:
                    if self.args.verbose:
                        print(f"Haar detection failed for {image_path}: {e}")
            
            metadata_entries = []
            
            if detections:
                # Process each detection
                for idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det["bbox"]
                    
                    # Check minimum area
                    area = (x2 - x1) * (y2 - y1)
                    if area < self.args.min_face_area:
                        self.metadata["summary"]["skipped_small"] += 1
                        continue
                    
                    # Expand bbox
                    x1, y1, x2, y2 = expand_bbox(
                        x1, y1, x2, y2, self.args.padding, img_w, img_h
                    )
                    
                    # Crop and resize
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    crop_resized = resize_image(crop, self.args.img_size)
                    
                    # Generate output filename
                    stem = image_path.stem
                    output_filename = f"{stem}__face{idx}_{det['detector']}.jpg"
                    output_path = output_dir / cat_id / output_filename
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save crop
                    success = cv2.imwrite(str(output_path), crop_resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    if not success:
                        if self.args.verbose:
                            print(f"Failed to save image: {output_path}")
                        continue
                    
                    # Compute blur score
                    blur_score = compute_blur_score(crop_resized)
                    
                    metadata_entries.append({
                        "original_path": str(image_path),
                        "output_path": str(output_path),
                        "cat_id": cat_id,
                        "bbox": [x1, y1, x2, y2],
                        "confidence": det["confidence"],
                        "detector": det["detector"],
                        "fallback": False,
                        "blur_score": blur_score,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            else:
                # No detection - save fallback (resized original)
                stem = image_path.stem
                output_filename = f"{stem}__fallback.jpg"
                output_path = output_dir / cat_id / output_filename
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                fallback_resized = resize_image(image, self.args.img_size)
                success = cv2.imwrite(str(output_path), fallback_resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not success:
                    if self.args.verbose:
                        print(f"Failed to save fallback image: {output_path}")
                    return [{
                        "original_path": str(image_path),
                        "cat_id": cat_id,
                        "error": "Failed to save fallback image",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }]
                
                blur_score = compute_blur_score(fallback_resized)
                
                metadata_entries.append({
                    "original_path": str(image_path),
                    "output_path": str(output_path),
                    "cat_id": cat_id,
                    "bbox": None,
                    "confidence": 0.0,
                    "detector": "haarcascade" if self.haar_detector else "none",
                    "fallback": True,
                    "blur_score": blur_score,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                self.metadata["summary"]["fallback_count"] += 1
            
            return metadata_entries
            
        except Exception as e:
            self.metadata["summary"]["errors"] += 1
            error_msg = str(e)
            if self.args.verbose:
                print(f"Error processing {image_path}: {error_msg}")
            return [{
                "original_path": str(image_path),
                "cat_id": cat_id,
                "error": error_msg,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }]
    
    def process_all(self):
        """Process all images in input directory."""
        input_path = Path(self.args.input_dir)
        output_path = Path(self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Collect all images
        image_files = []
        for img_path in input_path.rglob("*"):
            if img_path.is_file() and is_image_file(img_path):
                # Extract cat_id from folder structure
                rel_path = img_path.relative_to(input_path)
                cat_id = rel_path.parts[0] if len(rel_path.parts) > 1 else "unknown"
                image_files.append((img_path, cat_id))
        
        self.metadata["summary"]["total_input"] = len(image_files)
        
        if not image_files:
            print("❌ No images found in input directory")
            return
        
        print(f"\nProcessing {len(image_files)} images with {self.args.workers} workers...")
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=self.args.workers) as executor:
            futures = {executor.submit(self.process_image, img_path, cat_id): (img_path, cat_id)
                      for img_path, cat_id in image_files}
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    entries = future.result()
                    self.metadata["images"].extend(entries)
                    self.metadata["summary"]["total_output"] += len([e for e in entries if "error" not in e])
                except Exception as e:
                    if self.args.verbose:
                        print(f"Error processing {futures[future]}: {e}")
        
        # Post-processing: blur removal and augmentation
        self._post_process()
        
        # Save metadata
        if self.args.save_json:
            with open(self.args.save_json, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            print(f"\n✅ Metadata saved to: {self.args.save_json}")
        
        # Print summary
        self._print_summary()
        
        # If there are errors, show a sample error message
        if self.metadata["summary"]["errors"] > 0:
            error_entries = [e for e in self.metadata["images"] if "error" in e]
            if error_entries:
                print(f"\n⚠️  Sample error (showing first of {len(error_entries)} errors):")
                print(f"   Image: {error_entries[0].get('original_path', 'unknown')}")
                print(f"   Error: {error_entries[0].get('error', 'unknown')}")
                print(f"\n   Run with --verbose to see all errors")
    
    def _post_process(self):
        """Post-process: remove blurry images and augment if needed."""
        # Convert output_dir to Path if needed
        output_dir = Path(self.args.output_dir) if isinstance(self.args.output_dir, str) else self.args.output_dir
        
        # Group by cat_id
        by_cat = {}
        for entry in self.metadata["images"]:
            if "error" in entry or entry.get("fallback"):
                continue
            cat_id = entry["cat_id"]
            if cat_id not in by_cat:
                by_cat[cat_id] = []
            by_cat[cat_id].append(entry)
        
        removed_dir = output_dir / "_removed_blur"
        removed_dir.mkdir(exist_ok=True)
        
        # Remove blurry images (if enabled)
        if self.args.remove_blurry:
            for cat_id, entries in by_cat.items():
                blurry_entries = [e for e in entries if e.get("blur_score", 0) < self.args.blur_threshold]
                
                # Don't remove if it would leave folder empty
                if len(blurry_entries) >= len(entries):
                    if self.args.verbose:
                        print(f"⚠️  Skipping blur removal for {cat_id}: would leave folder empty")
                    continue
                
                for entry in blurry_entries:
                    src_path = Path(entry["output_path"])
                    if src_path.exists():
                        dst_path = removed_dir / src_path.name
                        src_path.rename(dst_path)
                        self.metadata["summary"]["blur_removed"] += 1
                        entry["removed_blurry"] = True
        
        # Augment folders with < min_images
        if self.args.min_images > 0:
            aug_types = ["flip", "rotate", "brightness", "contrast"]
            for cat_id, entries in by_cat.items():
                valid_entries = [e for e in entries if not e.get("removed_blurry", False)]
                if len(valid_entries) < self.args.min_images:
                    needed = self.args.min_images - len(valid_entries)
                    for i in range(needed):
                        # Use first valid entry for augmentation
                        if valid_entries:
                            source_entry = valid_entries[i % len(valid_entries)]
                            source_path = Path(source_entry["output_path"])
                            
                            if source_path.exists():
                                source_img = cv2.imread(str(source_path))
                                aug_type = aug_types[i % len(aug_types)]
                                aug_img = augment_image(source_img, aug_type)
                                
                                aug_filename = f"{source_path.stem}_aug{i}.jpg"
                                aug_path = output_dir / cat_id / aug_filename
                                cv2.imwrite(str(aug_path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                                
                                self.metadata["images"].append({
                                    "original_path": source_entry["original_path"],
                                    "output_path": str(aug_path),
                                    "cat_id": cat_id,
                                    "bbox": source_entry.get("bbox"),
                                    "confidence": source_entry.get("confidence", 0.0),
                                    "detector": source_entry.get("detector", "none"),
                                    "fallback": False,
                                    "augmented": True,
                                    "augmentation_type": aug_type,
                                    "blur_score": compute_blur_score(aug_img),
                                    "timestamp": datetime.now(timezone.utc).isoformat()
                                })
                                self.metadata["summary"]["total_output"] += 1
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "="*80)
        print("PROCESSING SUMMARY")
        print("="*80)
        s = self.metadata["summary"]
        print(f"Total input images: {s['total_input']}")
        print(f"Total output images: {s['total_output']}")
        print(f"Fallback saves: {s['fallback_count']}")
        print(f"Blur removed: {s['blur_removed']}")
        print(f"Skipped (small area): {s['skipped_small']}")
        print(f"Errors: {s['errors']}")
        print("="*80)


def visualize_detections(image_path: str, json_path: str, out_vis_path: str):
    """Visualize detections by drawing bounding boxes on image."""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Find entries for this image
    for entry in metadata["images"]:
        if entry.get("original_path") == image_path and entry.get("bbox"):
            x1, y1, x2, y2 = entry["bbox"]
            conf = entry.get("confidence", 0.0)
            detector = entry.get("detector", "unknown")
            
            # Draw box
            color = (0, 255, 0) if detector == "yolov8" else (255, 0, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detector} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    cv2.imwrite(out_vis_path, image)
    print(f"Visualization saved to: {out_vis_path}")


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    # Get script directory for default paths
    script_dir = Path(__file__).parent.absolute()
    default_input = str(script_dir / "dataset_individuals" / "cat_individuals_dataset")
    default_output = str(script_dir / "dataset_individuals_cropped_v5" / "cat_individuals_dataset")
    
    parser = argparse.ArgumentParser(
        description="YOLO-based cat face detection and cropping pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--input_dir", type=str, default=default_input,
                       help=f"Input directory with cat images (default: {default_input})")
    parser.add_argument("--output_dir", type=str, default=default_output,
                       help=f"Output directory for cropped faces (default: {default_output})")
    parser.add_argument("--weights", type=str, default="yolov8n",
                       help="YOLO weights path or model name (e.g., yolov8n, yolov8s)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                       help="Confidence threshold for detections")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                       help="NMS IoU threshold")
    parser.add_argument("--padding", type=float, default=DEFAULT_PADDING,
                       help="Padding fraction (0.2 = 20%% padding around bbox)")
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMG_SIZE,
                       help="Output image size (square)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                       help="Number of parallel workers")
    parser.add_argument("--min_face_area", type=int, default=DEFAULT_MIN_FACE_AREA,
                       help="Minimum face area in pixels to keep detection")
    parser.add_argument("--save_json", type=str, default=None,
                       help="Path to save metadata JSON file")
    parser.add_argument("--fallback_haarcascade", action="store_true",
                       help="Use Haar Cascade as fallback when YOLO fails")
    parser.add_argument("--blur_threshold", type=float, default=DEFAULT_BLUR_THRESHOLD,
                       help="Blur score threshold (variance of Laplacian)")
    parser.add_argument("--remove_blurry", action="store_true",
                       help="Remove blurry images (moves to _removed_blur folder)")
    parser.add_argument("--min_images", type=int, default=DEFAULT_MIN_IMAGES,
                       help="Minimum images per cat folder (augment if below)")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.input_dir):
        print(f"❌ Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Set default JSON path if not provided
    if not args.save_json:
        args.save_json = str(Path(args.output_dir) / "metadata.json")
    
    # Print configuration
    print("="*80)
    print("YOLO Cat Face Cropper")
    print("="*80)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Weights: {args.weights}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Workers: {args.workers}")
    gpu_info = get_gpu_info()
    print(f"Device: {gpu_info['device_name']}")
    print("="*80)
    
    # Run processing
    cropper = CatFaceCropper(args)
    cropper.process_all()


if __name__ == "__main__":
    main()

