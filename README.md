# Cat Identity Recognition Model 🐱

An AI-powered system for recognizing individual cats using deep learning embeddings. This project uses **EfficientNet-B0** with **Triplet Loss** to learn unique facial features that can distinguish between individual cats.

## 🎯 Project Overview

This model learns to create unique "fingerprints" (embeddings) for each cat, enabling:
- **Individual cat recognition** (e.g., distinguishing "Snowy" from "Luna")
- **Similarity comparison** between two cat images
- **Scalable to new cats** through fine-tuning

### How It Works

1. **Pre-training Phase**: Train on breed-level data (same breed = similar, different breed = different)
2. **Embedding Extraction**: Model learns to create 128-dimensional embeddings for each cat face
3. **Fine-tuning Phase**: Adapt the model to recognize individual cats with their specific images
4. **Inference**: Compare any two cat images to determine if they're the same cat

## 📁 Project Structure

```
Capstone2.1/
├── dataset/                              # Original cat images
│   └── cats/
│       ├── Abyssinian/
│       ├── Bengal/
│       └── ...
├── dataset_cropped/                      # Cropped cat faces (224x224)
│   └── cats/
│       ├── Abyssinian/
│       ├── Bengal/
│       └── ...
├── cat_face_cropper.py                   # Script to crop cat faces
├── cat_identity_model_trainer.py         # Main training script
├── haarcascade_frontalcatface.xml        # Haar cascade for cat face detection
├── requirements.txt                      # Python dependencies
└── README.md                             # This file

# Generated after training:
├── cat_identity_model.h5                 # Trained model
├── cat_embeddings.npz                    # Extracted embeddings
├── cat_metadata.csv                      # Image metadata
├── training_history.json                 # Training metrics
└── training_history.png                  # Training visualization
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd "D:/Cursor AI projects/Capstone2.1"

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

If you haven't already cropped the cat faces:

```python
python cat_face_cropper.py
```

This will create cropped 224x224 images in `dataset_cropped/cats/`.

### 3. Train the Model

```bash
python cat_identity_model_trainer.py
```

**Training Process:**
- Loads images from `dataset_cropped/cats/`
- Generates triplets (anchor, positive, negative)
- Trains EfficientNet-B0 with triplet loss
- Saves the trained model and embeddings
- Creates visualization of training progress

**Expected Training Time:**
- CPU: ~2-4 hours for 20 epochs
- GPU: ~30-60 minutes for 20 epochs

### 4. Use the Trained Model

```python
from tensorflow import keras
from cat_identity_model_trainer import compare_images

# Load the trained model
model = keras.models.load_model('cat_identity_model.h5')

# Compare two images
result = compare_images(
    model,
    'path/to/cat1.jpg',
    'path/to/cat2.jpg',
    threshold=0.8,
    visualize=True
)

print(f"Similarity: {result['similarity']:.4f}")
print(f"Same cat: {result['is_same_cat']}")
```

## 🔧 Configuration

Edit the `Config` class in `cat_identity_model_trainer.py`:

```python
class Config:
    # Paths
    DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
    
    # Model parameters
    EMBEDDING_DIM = 128           # Size of embedding vector
    DROPOUT_RATE = 0.3            # Dropout for regularization
    BACKBONE_TRAINABLE_LAYERS = 20  # Fine-tune top 20 layers
    
    # Training parameters
    BATCH_SIZE = 32               # Batch size
    EPOCHS = 20                   # Number of epochs
    LEARNING_RATE = 1e-4          # Learning rate
    TRIPLET_MARGIN = 0.5          # Triplet loss margin
    TRIPLETS_PER_EPOCH = 5000     # Triplets per epoch
    
    # Inference
    SIMILARITY_THRESHOLD = 0.8    # Cosine similarity threshold
```

## 📊 Model Architecture

```
Input (224x224x3)
    ↓
EfficientNet-B0 (pretrained on ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, activation='relu')
    ↓
L2 Normalization
    ↓
Output: 128-dim embedding
```

### Triplet Loss

The model uses triplet loss to learn discriminative embeddings:

```
L = max(0, ||f(anchor) - f(positive)||² - ||f(anchor) - f(negative)||² + margin)
```

- **Anchor**: Reference cat image
- **Positive**: Different image of the same cat/breed
- **Negative**: Image of a different cat/breed
- **Margin**: Minimum distance between positive and negative pairs (0.5)

## 🎓 Fine-Tuning for Individual Cats

Once you have images of individual cats, fine-tune the model:

### 1. Organize Your Data

```
individual_cats/
├── Snowy/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
├── Luna/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── img3.jpg
└── Whiskers/
    ├── img1.jpg
    └── img2.jpg
```

### 2. Fine-Tune

```python
from cat_identity_model_trainer import fine_tune_on_new_data

history = fine_tune_on_new_data(
    model_path='cat_identity_model.h5',
    new_data_path='path/to/individual_cats',
    output_model_path='cat_identity_model_finetuned.h5',
    epochs=10
)
```

### 3. Use Fine-Tuned Model

```python
model = keras.models.load_model('cat_identity_model_finetuned.h5')

# Now it can recognize individual cats!
result = compare_images(
    model,
    'snowy_photo1.jpg',
    'snowy_photo2.jpg'
)
# Should return high similarity if both are Snowy
```

## 📈 Monitoring Training

The script automatically generates:

1. **training_history.json**: Numerical training metrics
2. **training_history.png**: Visualization showing:
   - Training loss over epochs
   - Positive vs negative distances

**Good training indicators:**
- Loss decreases over time
- Negative distance > Positive distance (with increasing gap)
- Positive distance approaches 0
- Negative distance approaches or exceeds margin (0.5)

## 🔍 Understanding Results

### Cosine Similarity Score

- **0.9 - 1.0**: Very likely the same cat
- **0.8 - 0.9**: Probably the same cat
- **0.6 - 0.8**: Uncertain (could be same breed)
- **0.0 - 0.6**: Different cats

### Adjusting Threshold

- **Higher threshold (0.9)**: More conservative, fewer false positives
- **Lower threshold (0.7)**: More permissive, more false positives

```python
# Adjust threshold based on your needs
result = compare_images(model, img1, img2, threshold=0.85)
```

## 🧪 Advanced Usage

### Extract Embeddings for All Images

```python
from cat_identity_model_trainer import extract_embeddings
import numpy as np

# Load model
model = keras.models.load_model('cat_identity_model.h5')

# Load saved embeddings
data = np.load('cat_embeddings.npz')
embeddings = data['embeddings']
image_paths = data['image_paths']

# Find most similar images to a query image
query_embedding = model.predict(np.expand_dims(preprocess_image('query.jpg'), 0))[0]

similarities = np.dot(embeddings, query_embedding)
top_5_indices = np.argsort(similarities)[-5:][::-1]

print("Top 5 most similar images:")
for idx in top_5_indices:
    print(f"{image_paths[idx]}: {similarities[idx]:.4f}")
```

### Batch Comparison

```python
def find_all_images_of_cat(model, query_image_path, all_image_paths, threshold=0.8):
    """Find all images that likely contain the same cat"""
    query_img = preprocess_image(query_image_path)
    query_embedding = model.predict(np.expand_dims(query_img, 0))[0]
    
    matches = []
    for img_path in all_image_paths:
        img = preprocess_image(img_path)
        embedding = model.predict(np.expand_dims(img, 0))[0]
        similarity = np.dot(query_embedding, embedding)
        
        if similarity >= threshold:
            matches.append((img_path, similarity))
    
    return sorted(matches, key=lambda x: x[1], reverse=True)
```

## 🐛 Troubleshooting

### GPU Memory Issues

If you encounter GPU memory errors:

```python
# In cat_identity_model_trainer.py, add at the top:
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

Or reduce batch size:

```python
Config.BATCH_SIZE = 16  # Reduce from 32
```

### Low Accuracy

- **Increase training epochs**: Set `Config.EPOCHS = 50`
- **More triplets**: Set `Config.TRIPLETS_PER_EPOCH = 10000`
- **Fine-tune more layers**: Set `Config.BACKBONE_TRAINABLE_LAYERS = 50`
- **Adjust learning rate**: Try `Config.LEARNING_RATE = 5e-5`

### Dataset Not Found Error

Ensure the path is correct:

```python
Config.DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
```

## 📚 Technical Details

### Dependencies

- **TensorFlow 2.13+**: Deep learning framework
- **EfficientNet-B0**: Efficient CNN architecture (5.3M parameters)
- **Keras**: High-level API for model building
- **OpenCV**: Image processing
- **NumPy/Pandas**: Data manipulation

### Model Size

- **Trainable parameters**: ~4.8M
- **Model file size**: ~20 MB
- **Embedding size**: 128 floats (512 bytes per image)

### Performance Metrics

Expected performance on breed-level data:
- **Same breed comparison**: 0.7-0.9 similarity
- **Different breed comparison**: 0.2-0.5 similarity

After fine-tuning on individual cats:
- **Same cat comparison**: 0.85-0.98 similarity
- **Different cat (same breed)**: 0.4-0.7 similarity

## 🤝 Contributing

This is a capstone project. Feel free to extend it with:
- Data augmentation for better generalization
- Hard negative mining for improved triplet selection
- Integration with mobile apps
- Real-time webcam recognition
- Database for storing cat profiles

## 📄 License

This project is for educational purposes.

## 🙏 Acknowledgments

- EfficientNet paper: [Tan & Le, 2019]
- Triplet Loss: [Schroff et al., 2015 - FaceNet]
- Cat dataset: Various sources
- Haar Cascade: OpenCV contributors

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the training logs
3. Verify dataset structure
4. Ensure all dependencies are installed

---

**Happy Cat Recognition! 🐱✨**

---

## Project Update (2025‑11) — Web App + EfficientNetV2‑S

### Overview
- Added a Flask web application that serves the triplet‑loss embedding model based on EfficientNetV2‑S (512‑dim embeddings).
- Supports uploads, Top‑K matching via L2 distance on normalized embeddings, and maintenance APIs.

### How to Run (Web App)
1. Activate Python 3.9 venv
```
cd "D:/Cursor AI projects/Capstone2.1"
py -3.9 -m venv .venv39
".venv39/ Scripts/Activate.ps1"
```
2. Install runtime deps (TF 2.10 + compatible set)
```
python -m pip install -U pip
python -m pip install tensorflow==2.10.0 numpy==1.23.5 opencv-python==4.7.0.72 pillow flask tqdm matplotlib scikit-learn
```
3. Start the server
```
python "D:/Cursor AI projects/Capstone2.1/120 transfer now/120 transfer now/app.py"
```
4. Open
- App: `http://localhost:5000/`
- Health: `http://localhost:5000/api/health`

### Backend Architecture
- File: `120 transfer now/120 transfer now/app.py`
  - Model loading priority: SavedModel → .keras → .h5 → weights (from `D:/Cursor AI projects/Capstone2.1/models`)
  - Preprocessing: `keras.applications.efficientnet_v2.preprocess_input` with 224×224 RGB inputs
  - Matching: L2 distance on normalized 512‑dim embeddings; cosine also reported
  - Embedding store: per‑cat centroid; caches at repo root (`cat_embeddings_cache.npy`, `cat_metadata_cache.json`)
  - APIs:
    - `GET /api/health` — status, embeddings count
    - `POST /api/reindex` — rebuild centroids from dataset
    - `POST /api/add` — add a new cat embedding from uploaded image

### Frontend
- Templates: `120 transfer now/120 transfer now/templates/`
- Responsive results cards, confidence badges, loading spinner on upload

### Changes Summary
- Migrated inference to EfficientNetV2‑S (512‑dim)
- Implemented Flask web app and maintenance endpoints
- Built centroid‑based embedding DB with caching and reindex
- Aligned test script to V2 backbone; replaced augmentation with TF‑compatible ops
- Pinned environment: Python 3.9, TensorFlow 2.10, NumPy 1.23.5, OpenCV 4.7.0.72
- Resolved IDE diagnostics by guarding shape logging, using safe CV2 attribute access, and secure filename defaults
