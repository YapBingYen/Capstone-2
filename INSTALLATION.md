# Installation Guide

Complete step-by-step guide to set up the Cat Identity Recognition system.

---

## 📋 Prerequisites

### System Requirements

**Minimum:**
- Python 3.8 or higher
- 8 GB RAM
- 10 GB free disk space
- Windows, macOS, or Linux

**Recommended:**
- Python 3.9-3.11
- 16 GB RAM
- NVIDIA GPU with 6+ GB VRAM (for faster training)
- 20 GB free disk space

### Software Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (optional, for version control)
- CUDA 11.8+ (optional, for GPU support)

---

## 🚀 Installation Steps

### Step 1: Verify Python Installation

Open a terminal/command prompt and run:

```bash
python --version
```

You should see Python 3.8 or higher. If not, download from [python.org](https://www.python.org/downloads/).

### Step 2: Navigate to Project Directory

```bash
cd "D:/Cursor AI projects/Capstone2.1"
```

Or wherever your project is located.

### Step 3: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Your terminal should now show `(venv)` prefix.

### Step 4: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.13+
- Keras
- NumPy, Pandas
- OpenCV
- Matplotlib
- scikit-learn
- tqdm

**Installation time:** ~5-10 minutes depending on internet speed.

### Step 5: Verify Installation

Run the system check:

```bash
python quick_start.py
```

Select option 7 (System Check) to verify everything is installed correctly.

---

## 🎮 GPU Support (Optional but Recommended)

### For NVIDIA GPU Users

1. **Check GPU Compatibility:**
   - NVIDIA GPU with CUDA Compute Capability 3.5+
   - Check at: [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus)

2. **Install CUDA Toolkit 11.8:**
   - Download from: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
   - Follow installation wizard

3. **Install cuDNN 8.6:**
   - Download from: [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn) (requires free account)
   - Extract and copy files to CUDA installation directory

4. **Verify GPU Setup:**

```python
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should show your GPU. If empty, TensorFlow will use CPU (slower but works).

### For AMD/Intel GPU Users

TensorFlow doesn't officially support AMD/Intel GPUs on Windows. Use CPU or try:
- **DirectML**: [tensorflow-directml](https://github.com/microsoft/tensorflow-directml)
- **ROCm** (Linux only): [tensorflow-rocm](https://github.com/ROCmSoftwarePlatform/tensorflow-upstream)

---

## 📁 Dataset Setup

### Option 1: Using Existing Dataset

If you already have cat images:

1. **Organize your raw images:**
```
dataset/cats/
├── Abyssinian/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Bengal/
│   ├── img1.jpg
│   └── ...
└── ...
```

2. **Run the face cropper:**
```bash
python cat_face_cropper.py
```

This will create:
```
dataset_cropped/cats/
├── Abyssinian/
│   ├── img1.jpg (224x224)
│   └── ...
├── Bengal/
│   └── ...
└── ...
```

### Option 2: Download Public Cat Dataset

You can use datasets from:
- **Kaggle**: [Cat Breeds Dataset](https://www.kaggle.com/datasets)
- **Oxford-IIIT Pet Dataset**: [Link](https://www.robots.ox.ac.uk/~vgg/data/pets/)

After downloading:
1. Extract to `dataset/cats/`
2. Run face cropper as above

### Minimum Dataset Requirements

- **Minimum breeds**: 2 (but 5+ recommended)
- **Images per breed**: 50+ (100+ recommended)
- **Total images**: 500+ (1000+ recommended)
- **Image format**: JPG, JPEG, or PNG
- **Image quality**: Clear, well-lit cat faces

---

## ✅ Verification Steps

### 1. Check File Structure

Ensure you have these files:

```
✅ cat_identity_model_trainer.py
✅ test_cat_comparison.py
✅ analyze_embeddings.py
✅ example_workflow.py
✅ quick_start.py
✅ cat_face_cropper.py
✅ requirements.txt
✅ README.md
✅ USAGE_GUIDE.md
✅ haarcascade_frontalcatface.xml
```

### 2. Test Import

```python
python -c "from cat_identity_model_trainer import Config; print('✅ Import successful!')"
```

### 3. Quick Training Test

Run a very quick training test (won't produce a useful model, just tests the pipeline):

```python
python -c "from cat_identity_model_trainer import Config; Config.EPOCHS = 1; Config.TRIPLETS_PER_EPOCH = 100"
```

Then run:
```bash
python cat_identity_model_trainer.py
```

If it runs for ~2-5 minutes and completes without errors, installation is successful!

---

## 🔧 Troubleshooting Installation

### Issue: "No module named 'tensorflow'"

**Solution:**
```bash
pip install tensorflow>=2.13.0
```

### Issue: "Could not install packages due to an EnvironmentError"

**Solution:** Run as administrator (Windows) or use `sudo` (macOS/Linux):
```bash
pip install -r requirements.txt --user
```

### Issue: "ERROR: Could not find a version that satisfies the requirement tensorflow"

**Solution:** Update pip and setuptools:
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue: OpenCV Import Error

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

### Issue: NumPy/SciPy Compatibility Errors

**Error 1:** `np.complex_` was removed in the NumPy 2.0 release  
**Error 2:** `All ufuncs must have type numpy.ufunc` (SciPy 1.12+ issue)

**Quick Fix (Recommended):**
```bash
python fix_all_compatibility.py
```

**Manual Fix:**
```bash
pip uninstall numpy scipy -y
pip install 'numpy>=1.21.0,<2.0.0'
pip install 'scipy>=1.9.0,<1.12.0'
pip install -r requirements.txt --force-reinstall
```

**Why this happens:** 
- TensorFlow 2.13 doesn't support NumPy 2.0 yet
- SciPy 1.12+ has compatibility issues with NumPy 1.x
- These are known compatibility issues that will be fixed in future versions

### Issue: TensorFlow Not Using GPU

**Check CUDA/cuDNN versions:**
```bash
nvidia-smi  # Should show GPU
nvcc --version  # Should show CUDA version
```

If CUDA is not found, reinstall CUDA Toolkit 11.8.

**Enable memory growth:**
Add to your script:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### Issue: Out of Memory During Installation

**Solution:** Install packages one by one:
```bash
pip install tensorflow
pip install numpy pandas
pip install opencv-python
pip install matplotlib scikit-learn
pip install tqdm
```

---

## 🐍 Alternative Installation Methods

### Using Conda (Alternative to pip)

```bash
# Create conda environment
conda create -n cat-recognition python=3.9
conda activate cat-recognition

# Install packages
conda install tensorflow-gpu  # or tensorflow for CPU
conda install numpy pandas matplotlib scikit-learn
pip install opencv-python tqdm  # Some packages only via pip
```

### Using Docker (Advanced)

Create `Dockerfile`:

```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "quick_start.py"]
```

Build and run:
```bash
docker build -t cat-recognition .
docker run -it --gpus all cat-recognition
```

---

## 📦 Optional Dependencies

### For Development

```bash
pip install jupyter notebook ipython
pip install tensorboard
pip install seaborn
```

### For Model Optimization

```bash
pip install tensorflow-model-optimization
pip install onnx tf2onnx
```

### For API Deployment

```bash
pip install flask fastapi uvicorn
pip install pillow
```

---

## 🎯 Next Steps

After successful installation:

1. **Run System Check:**
   ```bash
   python quick_start.py
   ```
   Select option 7

2. **Prepare Dataset:**
   - Place images in `dataset/cats/`
   - Run `cat_face_cropper.py`

3. **Start Training:**
   ```bash
   python quick_start.py
   ```
   Select option 1 or 2

---

## Compatibility Notes (2025‑11)

- Web app runtime stack:
  - Python 3.9
  - TensorFlow 2.10.0
  - NumPy 1.23.5 (do not use NumPy 2.x with TF 2.10)
  - OpenCV 4.7.0.72

- If you encounter `_ARRAY_API not found` or NumPy import errors:
  - `python -m pip install "numpy==1.23.5"`
  - `python -m pip install "opencv-python==4.7.0.72"`

- To run the web app:
```
".venv39/ Scripts/Activate.ps1"
python "D:/Cursor AI projects/Capstone2.1/120 transfer now/120 transfer now/app.py"
```

4. **Read Documentation:**
   - `README.md` - Overview
   - `USAGE_GUIDE.md` - Detailed usage
   - `PROJECT_SUMMARY.md` - Technical details

---

## 📞 Getting Help

If you encounter issues:

1. **Check the error message** - Most errors are self-explanatory
2. **Review this guide** - Common issues covered above
3. **Check system requirements** - Ensure Python 3.8+
4. **Try virtual environment** - Isolates dependencies
5. **Update packages** - `pip install --upgrade -r requirements.txt`

---

## 🔄 Updating

To update the project to the latest version:

```bash
# Pull latest code (if using git)
git pull origin main

# Update dependencies
pip install --upgrade -r requirements.txt

# Verify installation
python quick_start.py  # Select option 7
```

---

## 🗑️ Uninstallation

To remove the project:

1. **Deactivate virtual environment:**
   ```bash
   deactivate
   ```

2. **Delete project folder:**
   - Windows: Right-click → Delete
   - macOS/Linux: `rm -rf "path/to/Capstone2.1"`

3. **Remove virtual environment:**
   ```bash
   rm -rf venv/  # or rmdir /s venv on Windows
   ```

---

## ✅ Installation Complete!

You're all set! The Cat Identity Recognition system is ready to use.

**Quick Start:**
```bash
python quick_start.py
```

**Happy Cat Recognition! 🐱✨**

---

*Last Updated: October 28, 2025*

