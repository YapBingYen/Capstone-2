# Cat Identity Recognition - Project Summary

## 📋 Overview

This project implements a **deep learning-based cat identity recognition system** using **triplet loss** and **EfficientNet-B0**. The model learns unique embeddings for each cat, enabling individual cat recognition (e.g., distinguishing "Snowy" from "Luna").

---

## 🎯 Key Features

✅ **EfficientNet-B0 Backbone**: Pretrained on ImageNet for robust feature extraction  
✅ **Triplet Loss Training**: Learns discriminative embeddings via anchor-positive-negative triplets  
✅ **Breed-Level Pre-training**: Initial training on cat breeds (Abyssinian, Bengal, etc.)  
✅ **Individual Fine-Tuning**: Adaptable to specific cats (Snowy, Luna, etc.)  
✅ **Fast Inference**: L2-normalized embeddings enable efficient cosine similarity comparison  
✅ **Comprehensive Tooling**: Training, testing, analysis, and visualization scripts  
✅ **Production-Ready**: Modular, well-documented, and extensible codebase  

---

## 📁 Project Structure

```
Capstone2.1/
│
├── 📊 Dataset
│   ├── dataset/cats/                    # Original images (13 breeds)
│   └── dataset_cropped/cats/            # Cropped faces (224x224)
│
├── 🧠 Core Training
│   ├── cat_identity_model_trainer.py    # Main training script (850+ lines)
│   ├── requirements.txt                 # Python dependencies
│   └── haarcascade_frontalcatface.xml   # Face detection cascade
│
├── 🧪 Testing & Inference
│   ├── test_cat_comparison.py           # Image comparison tool
│   ├── analyze_embeddings.py            # Embedding analysis & visualization
│   └── example_workflow.py              # Complete workflow examples
│
├── 🚀 User Interface
│   ├── quick_start.py                   # Interactive menu system
│   └── cat_face_cropper.py              # Face cropping utility
│
├── 📚 Documentation
│   ├── README.md                        # Project overview
│   ├── USAGE_GUIDE.md                   # Step-by-step guide
│   └── PROJECT_SUMMARY.md               # This file
│
└── 📈 Generated (after training)
    ├── cat_identity_model.h5            # Trained model (~20 MB)
    ├── cat_embeddings.npz               # Image embeddings
    ├── cat_metadata.csv                 # Image metadata
    ├── training_history.json            # Training metrics
    └── training_history.png             # Training plots
```

---

## 🏗️ Architecture

### Model Pipeline

```
Input Image (224x224x3)
        ↓
EfficientNet-B0 (pretrained)
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

### Training Strategy

**Triplet Loss Formula:**
```
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
```

- **Anchor**: Reference cat image
- **Positive**: Different image, same cat/breed
- **Negative**: Image of different cat/breed
- **Margin**: 0.5 (minimum distance between positive and negative)

**Training Process:**
1. Generate 5,000 triplets per epoch
2. Train for 20 epochs with Adam optimizer (lr=1e-4)
3. Fine-tune top 20 layers of EfficientNet-B0
4. Use EarlyStopping and ModelCheckpoint callbacks

---

## 🚀 Quick Start

### Installation

```bash
cd "D:/Cursor AI projects/Capstone2.1"
pip install -r requirements.txt
```

### Option 1: Interactive Menu (Recommended)

```bash
python quick_start.py
```

This launches an interactive menu with options for:
- Training (full or quick)
- Testing
- Analysis
- Examples
- System check

### Option 2: Direct Training

```bash
python cat_identity_model_trainer.py
```

### Option 3: Direct Testing

```bash
python test_cat_comparison.py image1.jpg image2.jpg
```

---

## 📊 Expected Results

### Breed-Level Performance

After 20 epochs of training:

| Metric | Value |
|--------|-------|
| Training Loss | ~0.05-0.15 |
| Same Breed Similarity | 0.70-0.90 |
| Different Breed Similarity | 0.20-0.50 |
| Distance Margin | 0.30-0.50 |
| k-NN Classification Accuracy | 85-95% |

### Individual Cat Performance (After Fine-Tuning)

| Comparison Type | Similarity Range |
|----------------|------------------|
| Same Cat | 0.85-0.98 |
| Different Cat (Same Breed) | 0.40-0.70 |
| Different Cat (Different Breed) | 0.20-0.50 |

---

## 🛠️ Usage Examples

### Example 1: Compare Two Images

```python
from tensorflow import keras
from cat_identity_model_trainer import compare_images

model = keras.models.load_model('cat_identity_model.h5')

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

### Example 2: Find Similar Cats

```python
import numpy as np

# Load embeddings
data = np.load('cat_embeddings.npz')
embeddings = data['embeddings']
image_paths = data['image_paths']

# Get query embedding
query_embedding = model.predict(np.expand_dims(query_image, 0))[0]

# Compute similarities
similarities = np.dot(embeddings, query_embedding)

# Get top 5
top_5_indices = np.argsort(similarities)[-5:][::-1]
```

### Example 3: Fine-Tune for Individual Cats

```python
from cat_identity_model_trainer import fine_tune_on_new_data

history = fine_tune_on_new_data(
    model_path='cat_identity_model.h5',
    new_data_path='path/to/individual_cats',
    output_model_path='cat_identity_model_finetuned.h5',
    epochs=10
)
```

---

## 🎨 Visualization Tools

### Training History

Automatically generated after training:
- Loss curve over epochs
- Positive vs negative distance evolution
- Saved as `training_history.png`

### Embedding Visualization

```bash
python analyze_embeddings.py
```

Provides:
1. **t-SNE Plot**: 2D projection of embeddings
2. **PCA Plot**: Principal component analysis
3. **Similarity Distribution**: Histogram of similarities
4. **Confusion Matrix**: Breed classification performance

---

## ⚙️ Configuration

### Key Parameters (in `cat_identity_model_trainer.py`)

```python
class Config:
    # Training
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    TRIPLETS_PER_EPOCH = 5000
    
    # Model
    EMBEDDING_DIM = 128
    DROPOUT_RATE = 0.3
    BACKBONE_TRAINABLE_LAYERS = 20
    
    # Loss
    TRIPLET_MARGIN = 0.5
    
    # Inference
    SIMILARITY_THRESHOLD = 0.8
```

### Performance Tuning

**For Better Accuracy:**
- Increase `EPOCHS` to 50
- Increase `TRIPLETS_PER_EPOCH` to 10,000
- Increase `BACKBONE_TRAINABLE_LAYERS` to 50

**For Faster Training:**
- Reduce `EPOCHS` to 10
- Reduce `TRIPLETS_PER_EPOCH` to 2,000
- Reduce `BATCH_SIZE` to 16

**For Memory Issues:**
- Reduce `BATCH_SIZE` to 16 or 8
- Enable GPU memory growth

---

## 📈 Training Timeline

| Hardware | Time (20 epochs) |
|----------|------------------|
| GPU (NVIDIA RTX 3060+) | 30-60 minutes |
| GPU (NVIDIA GTX 1060) | 1-2 hours |
| CPU (Modern) | 3-5 hours |
| CPU (Older) | 6-10 hours |

**Quick Training (5 epochs):** ~20-30 minutes on GPU

---

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `BATCH_SIZE` to 16 or 8 |
| Low accuracy | Increase `EPOCHS` to 50, more triplets |
| Model too conservative | Lower `SIMILARITY_THRESHOLD` to 0.7 |
| Model too permissive | Raise `SIMILARITY_THRESHOLD` to 0.85 |
| Slow training | Use GPU, reduce triplets |
| Dataset not found | Check path in `Config.DATASET_PATH` |

See `USAGE_GUIDE.md` for detailed troubleshooting.

---

## 📚 Scripts Reference

### 1. cat_identity_model_trainer.py
**Main training script** (850+ lines)
- Dataset loading
- Triplet generation
- Model building
- Training loop
- Embedding extraction
- Inference functions

### 2. test_cat_comparison.py
**Testing interface**
- Single image comparison
- Batch comparison
- Interactive mode
- Similarity matrix

### 3. analyze_embeddings.py
**Analysis tools**
- t-SNE visualization
- PCA visualization
- Similarity statistics
- k-NN classification
- Interactive search

### 4. example_workflow.py
**Workflow demonstrations**
- 6 complete examples
- Training, inference, fine-tuning
- Best practices
- Common patterns

### 5. quick_start.py
**User-friendly interface**
- Interactive menu
- System check
- Guided training
- Help system

### 6. cat_face_cropper.py
**Preprocessing utility**
- Detects cat faces
- Crops to 224x224
- Preserves folder structure

---

## 🎯 Use Cases

### 1. Pet Recognition App
- Identify individual pets in photos
- Track pet health over time
- Match lost pets with owners

### 2. Animal Shelter Management
- Catalog animals with unique IDs
- Prevent duplicate entries
- Track adoption history

### 3. Wildlife Research
- Identify individual animals
- Track movement patterns
- Monitor populations

### 4. Smart Pet Door
- Allow only authorized pets
- Log entry/exit times
- Security monitoring

---

## 🔬 Technical Details

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| TensorFlow | 2.13+ | Deep learning framework |
| Keras | 2.13+ | High-level API |
| NumPy | 1.23+ | Numerical operations |
| Pandas | 1.5+ | Data management |
| Scikit-learn | 1.2+ | ML utilities |
| OpenCV | 4.7+ | Image processing |
| Matplotlib | 3.6+ | Visualization |

### Model Specifications

- **Backbone**: EfficientNet-B0 (5.3M parameters)
- **Total Parameters**: ~4.8M trainable

---

## Update (2025‑11): Web Application + EfficientNetV2‑S

### Overview
- Added a Flask web app (`120 transfer now/120 transfer now/app.py`) that serves a triplet‑loss embedding model based on **EfficientNetV2‑S** with 512‑dim embeddings.
- Supports upload, Top‑K matching via L2 distance on normalized embeddings, and maintenance APIs.

### Environment
- Python 3.9
- TensorFlow 2.10.0
- NumPy 1.23.5
- OpenCV 4.7.0.72

### APIs
- `GET /api/health` — model status and embeddings count
- `POST /api/reindex` — rebuild centroid embeddings
- `POST /api/add` — add a cat embedding from an uploaded image

### Frontend
- Bootstrap 5 templates with responsive results cards, confidence badges, and loading states.

### Scripts Alignment
- `train_cat_identifier_v2.py` — EfficientNetV2‑S training
- `test_cat_identifier_v2.py` — EfficientNetV2‑S testing; TTA via `tf.image.adjust_*`; model guards added

### Notes
- Embedding caches stored at project root (`cat_embeddings_cache.npy`, `cat_metadata_cache.json`).
- Diagnostics resolved via guarded shape logging and safe CV2 attribute access.
- **Input Size**: 224×224×3
- **Output Size**: 128-dimensional embedding
- **Model File Size**: ~20 MB
- **Inference Time**: ~20-50ms per image (GPU)

---

## 🚀 Future Enhancements

### Potential Improvements

1. **Data Augmentation**
   - Random crops, flips, rotations
   - Color jittering
   - Mixup/Cutmix

2. **Advanced Triplet Mining**
   - Hard negative mining
   - Semi-hard negative mining
   - Online triplet mining

3. **Model Optimization**
   - Model quantization
   - TensorFlow Lite conversion
   - ONNX export

4. **Multi-Modal Learning**
   - Combine with audio (meow recognition)
   - Add behavioral features
   - Include metadata (size, age)

5. **Web Interface**
   - Flask/FastAPI backend
   - React frontend
   - REST API

6. **Mobile Deployment**
   - iOS CoreML
   - Android TFLite
   - Edge TPU support

---

## 📝 Citation

If you use this code in your research or project, please cite:

```bibtex
@software{cat_identity_recognition,
  title = {Cat Identity Recognition using Triplet Loss},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/cat-identity-recognition}
}
```

### References

1. **EfficientNet**: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML.

2. **Triplet Loss**: Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A Unified Embedding for Face Recognition and Clustering. CVPR.

3. **Metric Learning**: Kaya, M., & Bilge, H. Ş. (2019). Deep Metric Learning: A Survey. Symmetry, 11(9), 1066.

---

## 🤝 Contributing

This is a capstone project, but contributions are welcome:

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

---

## 📄 License

This project is for educational purposes. Feel free to use and modify for your own projects.

---

## 📞 Support

For questions or issues:
1. Check `USAGE_GUIDE.md` for detailed instructions
2. Review error messages in training logs
3. Run system check: `python quick_start.py` → option 7
4. Check generated visualizations for insights

---

## 🎉 Acknowledgments

- **TensorFlow/Keras Team**: For the excellent deep learning framework
- **EfficientNet Authors**: For the efficient CNN architecture
- **FaceNet Authors**: For pioneering triplet loss in face recognition
- **OpenCV Community**: For computer vision tools
- **Cat Dataset Contributors**: For providing training data

---

**Happy Cat Recognition! 🐱✨**

---

*Last Updated: October 28, 2025*  
*Version: 1.0.0*

