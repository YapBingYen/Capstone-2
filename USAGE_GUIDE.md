# Cat Identity Recognition - Complete Usage Guide

## 📋 Table of Contents
1. [Quick Start](#quick-start)
2. [Step-by-Step Training](#step-by-step-training)
3. [Testing the Model](#testing-the-model)
4. [Fine-Tuning for Individual Cats](#fine-tuning-for-individual-cats)
5. [Analyzing Results](#analyzing-results)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU support, ensure you have CUDA installed and use:
```bash
pip install tensorflow-gpu>=2.13.0
```

### 2. Train the Model

```bash
python cat_identity_model_trainer.py
```

This will:
- Load images from `dataset_cropped/cats/`
- Train for 20 epochs (~1-2 hours on GPU, 3-5 hours on CPU)
- Save the trained model as `cat_identity_model.h5`
- Generate embeddings and visualizations

### 3. Test the Model

```bash
# Compare two images
python test_cat_comparison.py image1.jpg image2.jpg

# Or use interactive mode
python test_cat_comparison.py
```

---

## Step-by-Step Training

### Configuration

Before training, you can customize settings in `cat_identity_model_trainer.py`:

```python
class Config:
    # Paths
    DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
    
    # Training parameters
    BATCH_SIZE = 32              # Reduce to 16 if GPU memory issues
    EPOCHS = 20                  # Increase for better accuracy
    LEARNING_RATE = 1e-4         # Lower for fine-tuning
    TRIPLETS_PER_EPOCH = 5000    # More triplets = better learning
    
    # Model parameters
    EMBEDDING_DIM = 128          # Embedding vector size
    DROPOUT_RATE = 0.3           # Regularization strength
    BACKBONE_TRAINABLE_LAYERS = 20  # Fine-tune top N layers
    
    # Inference
    SIMILARITY_THRESHOLD = 0.8   # Adjust based on use case
```

### Training Process

1. **Data Loading**: Loads all images and organizes by breed
2. **Triplet Generation**: Creates anchor-positive-negative triplets
   - Anchor: Random cat image
   - Positive: Different image, same breed
   - Negative: Image from different breed
3. **Model Training**: Trains EfficientNet-B0 with triplet loss
4. **Evaluation**: Tests on breed-level similarity

### Expected Output

```
================================================================================
CAT IDENTITY RECOGNITION MODEL TRAINER
Using Triplet Loss with EfficientNet-B0
================================================================================

================================================================================
STEP 1: Loading Dataset
================================================================================
Loading dataset...
✅ Loaded 2358 images from 13 breeds
Breeds: ['Abyssinian', 'Bengal', 'Birman', 'Bombay', ...]

================================================================================
STEP 2: Creating Triplet Generator
================================================================================

================================================================================
STEP 3: Building Models
================================================================================
Backbone: 237 total layers, 20 trainable

Embedding Model Summary:
...

================================================================================
STEP 4: Training Model
================================================================================
Epoch 1/20
--------------------------------------------------------------------------------
Generating 5000 triplets...
Training: 100%|████████████████| 156/156 [02:45<00:00,  1.06s/it]

Epoch 1 Results:
  Loss: 0.1234
  Avg Positive Distance: 0.0567
  Avg Negative Distance: 0.3421
  Distance Margin: 0.2854
  ✅ New best loss! Saving model...

...

================================================================================
🎉 TRAINING COMPLETE!
================================================================================
```

### Generated Files

After training, you'll have:

- **cat_identity_model.h5**: Trained model (saved after each improvement)
- **cat_embeddings.npz**: Embeddings for all training images
- **cat_metadata.csv**: Image paths and breed labels
- **training_history.json**: Loss and distance metrics per epoch
- **training_history.png**: Training visualization plots

---

## Testing the Model

### Method 1: Command Line

```bash
# Compare two specific images
python test_cat_comparison.py path/to/cat1.jpg path/to/cat2.jpg

# Batch comparison of multiple images
python test_cat_comparison.py img1.jpg img2.jpg img3.jpg img4.jpg
```

**Output Example:**
```
================================================================================
IMAGE COMPARISON RESULTS
================================================================================
Image 1: dataset_cropped/cats/Abyssinian/img1.jpg
Image 2: dataset_cropped/cats/Abyssinian/img2.jpg

Cosine Similarity: 0.8734
Threshold: 0.8000
Prediction: ✅ SAME CAT
Confidence: High
================================================================================
```

### Method 2: Interactive Mode

```bash
python test_cat_comparison.py
```

Then follow the prompts:
```
Options:
1. Compare two images
2. Batch comparison of multiple images
3. Change threshold
4. Exit

Enter your choice (1-4): 1
Enter path to first image: dataset_cropped/cats/Abyssinian/img1.jpg
Enter path to second image: dataset_cropped/cats/Bengal/img1.jpg
```

### Method 3: Python API

```python
from tensorflow import keras
from cat_identity_model_trainer import compare_images

# Load model
model = keras.models.load_model('cat_identity_model.h5')

# Compare images
result = compare_images(
    model,
    'path/to/cat1.jpg',
    'path/to/cat2.jpg',
    threshold=0.8,
    visualize=True  # Shows images side-by-side
)

print(f"Similarity: {result['similarity']:.4f}")
print(f"Same cat: {result['is_same_cat']}")
print(f"Confidence: {result['confidence']}")
```

---

## Fine-Tuning for Individual Cats

Once you have images of specific individual cats, you can fine-tune the model.

### Step 1: Organize Your Data

Create a folder structure like this:

```
individual_cats/
├── Snowy/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   ├── photo3.jpg
│   └── photo4.jpg
├── Luna/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── Whiskers/
│   ├── pic1.jpg
│   └── pic2.jpg
└── Mittens/
    ├── image1.jpg
    ├── image2.jpg
    └── image3.jpg
```

**Important**: 
- Each folder = one individual cat
- Minimum 2 images per cat (more is better, 5-10 recommended)
- Images should be cropped to 224x224 (use `cat_face_cropper.py`)

### Step 2: Fine-Tune

```python
from cat_identity_model_trainer import fine_tune_on_new_data

history = fine_tune_on_new_data(
    model_path='cat_identity_model.h5',
    new_data_path='path/to/individual_cats',
    output_model_path='cat_identity_model_finetuned.h5',
    epochs=10  # Fewer epochs for fine-tuning
)
```

### Step 3: Use Fine-Tuned Model

```python
from tensorflow import keras
from cat_identity_model_trainer import compare_images

# Load fine-tuned model
model = keras.models.load_model('cat_identity_model_finetuned.h5')

# Now it recognizes individual cats!
result = compare_images(
    model,
    'new_photos/snowy_1.jpg',
    'new_photos/snowy_2.jpg',
    threshold=0.85  # May need higher threshold for individuals
)

# Should show high similarity if both are Snowy
print(f"Same cat: {result['is_same_cat']}")
```

---

## Analyzing Results

Use the analysis tool to understand your model's performance:

```bash
python analyze_embeddings.py
```

### Available Analyses

1. **t-SNE Visualization**: Projects 128D embeddings to 2D
   - Good clusters = model learned to separate breeds well
   - Mixed clusters = needs more training

2. **PCA Visualization**: Linear projection to 2D
   - Shows variance explained
   - Faster than t-SNE

3. **Similarity Statistics**: 
   - Distribution of same-breed vs different-breed similarities
   - Suggests optimal threshold

4. **Breed Classification**: k-NN accuracy on breed prediction
   - Tests if embeddings capture breed information

5. **Interactive Search**: Find similar cats to a query image

### Example Output

```
================================================================================
SIMILARITY STATISTICS
================================================================================

📊 Same Breed (Positive Pairs):
   Mean: 0.7823
   Std:  0.1245
   Min:  0.4521
   Max:  0.9876

📊 Different Breeds (Negative Pairs):
   Mean: 0.3421
   Std:  0.1123
   Min:  0.0234
   Max:  0.7654

📊 Separation:
   Mean Difference: 0.4402

💡 Recommended Threshold: 0.742
   Expected Accuracy: 87.34%
```

---

## Troubleshooting

### Issue: Out of Memory Error

**Solution 1**: Reduce batch size
```python
Config.BATCH_SIZE = 16  # or even 8
```

**Solution 2**: Enable memory growth (GPU)
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Solution 3**: Generate fewer triplets
```python
Config.TRIPLETS_PER_EPOCH = 2000  # instead of 5000
```

### Issue: Low Accuracy / Poor Separation

**Solution 1**: Train longer
```python
Config.EPOCHS = 50  # instead of 20
```

**Solution 2**: Fine-tune more layers
```python
Config.BACKBONE_TRAINABLE_LAYERS = 50  # instead of 20
```

**Solution 3**: Generate more triplets
```python
Config.TRIPLETS_PER_EPOCH = 10000  # instead of 5000
```

**Solution 4**: Lower learning rate
```python
Config.LEARNING_RATE = 5e-5  # instead of 1e-4
```

### Issue: Model Too Conservative (Always Says "Different")

**Solution**: Lower the threshold
```python
Config.SIMILARITY_THRESHOLD = 0.7  # instead of 0.8
```

Or when comparing:
```python
compare_images(model, img1, img2, threshold=0.7)
```

### Issue: Model Too Permissive (Always Says "Same")

**Solution**: Raise the threshold
```python
Config.SIMILARITY_THRESHOLD = 0.85  # instead of 0.8
```

### Issue: Dataset Not Found

**Error**: `❌ ERROR: Dataset not found at ...`

**Solution**: Check the path in Config
```python
Config.DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
```

Make sure:
1. Path exists
2. Contains breed subfolders
3. Subfolders contain .jpg/.jpeg/.png images

---

## API Reference

### Core Functions

#### `compare_images(model, img_path1, img_path2, threshold=0.8, visualize=True)`
Compare two cat images.

**Parameters:**
- `model`: Trained Keras model
- `img_path1`: Path to first image
- `img_path2`: Path to second image
- `threshold`: Similarity threshold (default: 0.8)
- `visualize`: Show images side-by-side (default: True)

**Returns:**
```python
{
    'similarity': 0.8734,     # Cosine similarity (0-1)
    'is_same_cat': True,      # Prediction
    'confidence': 'High'      # 'High' or 'Low'
}
```

#### `fine_tune_on_new_data(model_path, new_data_path, output_model_path, epochs=10)`
Fine-tune model on individual cat data.

**Parameters:**
- `model_path`: Path to pretrained model
- `new_data_path`: Path to folder with individual cat subfolders
- `output_model_path`: Where to save fine-tuned model
- `epochs`: Number of fine-tuning epochs

**Returns:** Training history dictionary

#### `extract_embeddings(model, image_paths, batch_size=32)`
Extract embeddings for a list of images.

**Parameters:**
- `model`: Trained model
- `image_paths`: List of image paths
- `batch_size`: Batch size for processing

**Returns:** 
- `embeddings`: NumPy array (N, 128)
- `valid_paths`: List of successfully processed paths

### Configuration Options

---

## Web App Usage (EfficientNetV2‑S)

### Start the Server

1. Activate venv (Python 3.9)
```
cd "D:/Cursor AI projects/Capstone2.1"
py -3.9 -m venv .venv39
".venv39/ Scripts/Activate.ps1"
```
2. Install runtime deps
```
python -m pip install -U pip
python -m pip install tensorflow==2.10.0 numpy==1.23.5 opencv-python==4.7.0.72 pillow flask tqdm matplotlib scikit-learn
```
3. Run
```
python "D:/Cursor AI projects/Capstone2.1/120 transfer now/120 transfer now/app.py"
```

### Endpoints
- `GET /api/health` — { model_loaded, embeddings_count, dataset_path, img_size }
- `POST /api/reindex` — rebuild centroid store
- `POST /api/add` — form fields: `file`, `cat_id`

### Dataset & Caches
- Dataset path used by the app:
  `D:/Cursor AI projects/Capstone2.1/120 transfer now/120 transfer now/cat_individuals_dataset/dataset_individuals_cropped/cat_individuals_dataset`
- Caches at project root: `cat_embeddings_cache.npy`, `cat_metadata_cache.json`

### UI
- Drag‑and‑drop upload, responsive results, cosine‑based confidence badges, loading spinner

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 20 | Training epochs |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `EMBEDDING_DIM` | 128 | Embedding vector size |
| `DROPOUT_RATE` | 0.3 | Dropout rate |
| `TRIPLET_MARGIN` | 0.5 | Margin for triplet loss |
| `SIMILARITY_THRESHOLD` | 0.8 | Threshold for same cat |

---

## Tips and Best Practices

### Data Collection

1. **Quality over quantity**: 5 good images > 20 poor images
2. **Variety**: Different angles, lighting, expressions
3. **Consistency**: Same image size (224x224)
4. **Clean data**: Remove blurry/corrupted images

### Training

1. **Start with breed-level**: Train on breeds first
2. **Fine-tune for individuals**: Then adapt to specific cats
3. **Monitor metrics**: Watch distance margin increase
4. **Early stopping**: Model saves best version automatically

### Inference

1. **Adjust threshold**: Based on your use case
   - High threshold (0.9): Fewer false positives, ID verification
   - Low threshold (0.7): More permissive, ID in group photos
2. **Batch processing**: More efficient than one-by-one
3. **Cache embeddings**: Compute once, compare many times

### Production Deployment

1. **Use embeddings**: Faster than full model inference
2. **Approximate nearest neighbors**: For large-scale search
3. **Regular retraining**: As you collect more data
4. **Version control**: Keep track of model versions

---

## Additional Resources

- **EfficientNet Paper**: [Tan & Le, 2019](https://arxiv.org/abs/1905.11946)
- **Triplet Loss (FaceNet)**: [Schroff et al., 2015](https://arxiv.org/abs/1503.03832)
- **TensorFlow Documentation**: [tensorflow.org](https://www.tensorflow.org/)

---

**Questions or Issues?**
Check the troubleshooting section or review the generated logs and visualizations.

Happy Cat Recognition! 🐱✨

