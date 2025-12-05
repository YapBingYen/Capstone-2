# GPU Setup Guide for TensorFlow on Windows

## Current Situation
- **Your GPU**: NVIDIA RTX 3050 ✅
- **CUDA Driver**: 13.0 (installed)
- **TensorFlow**: 2.15.1 (CPU-only)
- **Issue**: TensorFlow 2.15.1 requires CUDA 12.x, not 13.0

## Solution: Install CUDA 12.x Libraries (Recommended)

You can have both CUDA 13.0 (driver) and CUDA 12.x (libraries) installed simultaneously.

### Step 1: Install CUDA 12.5 Libraries

**Option A: Via pip (Easiest)**
```bash
# In your virtual environment
.venv\Scripts\python.exe -m pip install nvidia-cudnn-cu12==8.9.4.25
```

**Option B: Manual Installation**
1. Download CUDA 12.5 from: https://developer.nvidia.com/cuda-12-5-0-download-archive
2. Download cuDNN 8.9.4 for CUDA 12.5 from: https://developer.nvidia.com/cudnn
3. Extract and copy cuDNN files to CUDA installation

### Step 2: Verify GPU Detection

```python
import tensorflow as tf
print("GPUs:", tf.config.list_physical_devices('GPU'))
```

If GPU is detected, you're ready! If not, continue to Step 3.

## Alternative: Use DirectML (Windows Native)

For Windows, you can use DirectML which works with any GPU:

```bash
pip install tensorflow-directml
```

Then in your code, use:
```python
import tensorflow as tf
tf.config.experimental.enable_mlir_bridge()
```

## Quick Test Script

Save this as `test_gpu.py`:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))

for gpu in tf.config.list_physical_devices('GPU'):
    print(f"  {gpu}")

# Test GPU computation
if tf.config.list_physical_devices('GPU'):
    print("\n✅ GPU is available!")
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print("GPU computation test:", c.numpy())
else:
    print("\n⚠️  No GPU detected - using CPU")
```

Run: `python test_gpu.py`

## If GPU Still Not Detected

1. **Check CUDA_PATH environment variable**
   ```powershell
   $env:CUDA_PATH
   ```

2. **Verify NVIDIA driver**
   ```powershell
   nvidia-smi
   ```

3. **Try restarting** after installing CUDA libraries

## Recommended: Train on CPU for Now

If GPU setup is taking too long, you can:
- Train on CPU (slower but works)
- Reduce epochs/batch size for faster iteration
- Train overnight on CPU

Your RTX 3050 will work once CUDA 12.x libraries are properly installed!


