# YOLO Cat Face Cropper

Production-ready pipeline for detecting and cropping cat faces from images using YOLOv8 with fallback to Haar Cascade.

## Features

- **YOLOv8 Detection**: Uses ultralytics YOLOv8 for accurate cat detection
- **Haar Cascade Fallback**: Automatic fallback when YOLO fails
- **GPU Support**: Automatically detects and uses GPU if available
- **Parallel Processing**: Multi-threaded processing for speed
- **Robust Error Handling**: Continues processing even if individual images fail
- **Blur Detection**: Optional blur removal with safeguards
- **Auto-Augmentation**: Augments folders with too few images
- **Metadata Export**: JSON metadata with detection details
- **No Empty Folders**: Ensures every cat folder has at least minimum images

## Installation

```bash
pip install -r requirements_yolo_cropper.txt
```

## Quick Start

### Basic Usage (CPU)

```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights yolov8n \
    --workers 4
```

### GPU Usage

```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights yolov8n \
    --workers 8
```

The script automatically detects GPU and uses it if available.

### With Custom YOLO Weights

If you've trained a custom YOLO model specifically for cat faces:

```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights /path/to/catface_yolov8.pt \
    --conf 0.3 \
    --workers 8
```

### Advanced: Remove Blurry Images

```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights yolov8n \
    --blur_threshold 80 \
    --remove_blurry \
    --min_images 5 \
    --workers 8
```

### With Haar Cascade Fallback

```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights yolov8n \
    --fallback_haarcascade \
    --workers 4
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input_dir` | *required* | Input directory with cat images (preserves folder structure) |
| `--output_dir` | *required* | Output directory for cropped faces |
| `--weights` | `yolov8n` | YOLO weights path or model name (yolov8n/s/m/l/x) |
| `--conf` | `0.25` | Confidence threshold for detections |
| `--iou` | `0.45` | NMS IoU threshold |
| `--padding` | `0.20` | Padding fraction (20% padding around bbox) |
| `--img_size` | `224` | Output image size (square) |
| `--workers` | `8` | Number of parallel workers |
| `--min_face_area` | `1024` | Minimum face area in pixels (32x32) |
| `--save_json` | `output_dir/metadata.json` | Path to save metadata JSON |
| `--fallback_haarcascade` | `False` | Use Haar Cascade as fallback |
| `--blur_threshold` | `60.0` | Blur score threshold (variance of Laplacian) |
| `--remove_blurry` | `False` | Remove blurry images |
| `--min_images` | `2` | Minimum images per cat folder (augment if below) |
| `--verbose` | `False` | Verbose output |

## Input Directory Structure

The script expects images organized by cat ID:

```
dataset_raw/
├── CAT001/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── CAT002/
│   ├── img1.jpg
│   └── ...
└── ...
```

## Output Structure

```
dataset_individuals_cropped_v5/
├── CAT001/
│   ├── img1__face0_yolov8.jpg
│   ├── img1__face1_yolov8.jpg
│   ├── img2__face0_yolov8.jpg
│   └── ...
├── CAT002/
│   └── ...
├── _removed_blur/  (if --remove_blurry enabled)
│   └── ...
└── metadata.json
```

## Metadata JSON Format

```json
{
  "images": [
    {
      "original_path": "/path/to/original.jpg",
      "output_path": "/path/to/cropped.jpg",
      "cat_id": "CAT001",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.98,
      "detector": "yolov8",
      "fallback": false,
      "blur_score": 123.4,
      "timestamp": "2025-01-19T12:34:56Z"
    }
  ],
  "summary": {
    "total_input": 1234,
    "total_output": 2345,
    "fallback_count": 12,
    "blur_removed": 34,
    "skipped_small": 5,
    "errors": 0
  }
}
```

## Using Custom YOLO Weights

### Training Your Own Cat Face Model

1. **Prepare Dataset**: Annotate cat faces in YOLO format
2. **Train Model**: Use ultralytics YOLOv8 training
3. **Use Custom Weights**: Point `--weights` to your `.pt` file

Example training command (reference):
```bash
yolo detect train data=cat_faces.yaml model=yolov8n.pt epochs=100
```

Then use your trained model:
```bash
python yolo_cat_face_cropper.py \
    --input_dir ./dataset_raw \
    --output_dir ./dataset_individuals_cropped_v5 \
    --weights ./runs/detect/train/weights/best.pt \
    --conf 0.3
```

## Visualization

To visualize detections on an image:

```python
from yolo_cat_face_cropper import visualize_detections

visualize_detections(
    image_path="path/to/image.jpg",
    json_path="metadata.json",
    out_vis_path="visualization.jpg"
)
```

## Performance Tips

1. **GPU**: Use GPU for 5-10x speedup (automatically detected)
2. **Workers**: Increase `--workers` for CPU (up to CPU cores)
3. **Model Size**: Use `yolov8n` for speed, `yolov8s/m` for accuracy
4. **Batch Processing**: Process large datasets in batches

## Troubleshooting

### YOLO Model Not Found
- First run downloads model automatically (~6-25 MB)
- Check internet connection
- Or download manually: `yolo export model=yolov8n format=onnx`

### Out of Memory
- Reduce `--workers` (try 2-4)
- Use smaller model (`yolov8n` instead of `yolov8m`)
- Process in smaller batches

### No Detections
- Lower `--conf` threshold (try 0.15-0.2)
- Enable `--fallback_haarcascade`
- Check image quality and cat visibility

### Empty Folders
- Script automatically saves fallback crops
- Enable `--min_images` with augmentation
- Check blur threshold if using `--remove_blurry`

## Why YOLOv8 is Better Than Haar Cascade

**YOLOv8 Advantages:**
- **Higher Accuracy**: Deep learning model trained on millions of images
- **Better Generalization**: Handles various angles, lighting, occlusions
- **Confidence Scores**: Provides detection confidence for filtering
- **Speed**: GPU acceleration provides real-time performance
- **Robustness**: Fewer false positives and negatives

**Haar Cascade Limitations:**
- **Feature-Based**: Relies on predefined features (less flexible)
- **False Negatives**: Misses faces in challenging conditions
- **No Confidence**: Binary detection (detected/not detected)
- **Limited Training**: Trained on smaller, less diverse dataset

**Best Practice**: Use YOLOv8 as primary detector with Haar Cascade as fallback for maximum robustness.

## License

MIT License - Use freely for your projects.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review metadata.json for error details
3. Run with `--verbose` for detailed logs

