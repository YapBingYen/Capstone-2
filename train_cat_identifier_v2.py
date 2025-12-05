"""
Cat Identifier Training Script (EfficientNetV2-S + Triplet Loss)
================================================================

Trains a metric-learning model to distinguish individual cats using
triplet loss with EfficientNetV2-S embeddings (upgraded from EfficientNet-B0).

Dataset:
    D:\\Cursor AI projects\\Capstone2.1\\dataset_individuals_cropped\\cat_individuals_dataset

Each subfolder (0001, 0002, ...) corresponds to a single cat.

Improvements over EfficientNet-B0:
    - Better accuracy (~2-5% improvement)
    - Improved architecture with better feature extraction
    - Similar training speed and model size
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import random
import json
import time

import importlib
import numpy as np
try:
    tensorflow_module = importlib.import_module("tensorflow")
    tf = tensorflow_module
    keras = tensorflow_module.keras
except ImportError as exc:
    raise ImportError("TensorFlow is required. Install it with 'pip install tensorflow tensorflow-addons'.") from exc

try:
    plt = importlib.import_module("matplotlib.pyplot")
except ImportError as exc:
    raise ImportError("Matplotlib is required. Install it with 'pip install matplotlib'.") from exc

try:
    tqdm = importlib.import_module("tqdm").tqdm
except ImportError as exc:
    raise ImportError("tqdm is required. Install it with 'pip install tqdm'.") from exc

# ---------------------------------------------------------------------------
# GPU Configuration
# ---------------------------------------------------------------------------

def setup_gpu():
    """Configure TensorFlow to use GPU if available."""
    print("\n" + "="*80)
    print("GPU Configuration")
    print("="*80)
    
    # List all available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
                # Get GPU details if possible
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details:
                        print(f"      Details: {gpu_details}")
                except:
                    pass
            
            # Set default device to GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"✅ Using GPU: {gpus[0].name}")
            return True
        except RuntimeError as e:
            print(f"⚠️  GPU configuration error: {e}")
            print("   Falling back to CPU")
            return False
    else:
        print("⚠️  No GPU detected. Training will use CPU.")
        print("   For GPU support, install TensorFlow with CUDA:")
        print("   pip install tensorflow[and-cuda]")
        return False

# Setup GPU before setting seed
gpu_available = setup_gpu()

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Config:
    DATASET_PATH = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset")
    MODEL_SAVE_PATH = Path(r"D:\Cursor AI projects\Capstone2.1\models\cat_identifier_efficientnet_v2.keras")
    HISTORY_PATH = Path("training_history_v2.json")
    PLOT_PATH = Path("training_curves_v2.png")

    IMG_SIZE = 224
    EMBEDDING_DIM = 512

    BATCH_SIZE = 16
    EPOCHS = 30
    LEARNING_RATE = 5e-5
    TRIPLETS_PER_EPOCH = 6000  # adjust as needed

    TRIPLET_MARGIN = 1.0

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXT

def compute_mean_color(img_array: np.ndarray) -> np.ndarray:
    """Compute mean HSV color for an image array."""
    hsv = tf.image.rgb_to_hsv(img_array / 255.0)
    return tf.reduce_mean(hsv, axis=(0, 1)).numpy()

# ---------------------------------------------------------------------------
# Dataset Loader and Triplet Generator
# ---------------------------------------------------------------------------

class TripletGenerator:
    """
    Generates triplets (anchor, positive, hard-negative) for training.
    Hard negatives chosen based on similar mean color features.
    """

    def __init__(self, dataset_path: Path, batch_size: int = 16):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.cat_to_images: Dict[str, List[str]] = {}
        self.cat_ids: List[str] = []
        self.cat_color_feature: Dict[str, np.ndarray] = {}
        self._load_dataset()

    def _load_dataset(self):
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")

        print("\n" + "="*80)
        print("Loading dataset and computing color features...")
        print("="*80)

        for cat_dir in sorted(self.dataset_path.iterdir()):
            if not cat_dir.is_dir():
                continue
            images = [str(p) for p in cat_dir.iterdir() if p.is_file() and is_image_file(p)]
            if len(images) < 2:
                continue  # need at least two images per cat
            self.cat_to_images[cat_dir.name] = images
            self.cat_ids.append(cat_dir.name)

            # Compute mean color using first image
            img = keras.utils.load_img(images[0], target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
            img_array = keras.utils.img_to_array(img)
            mean_color = compute_mean_color(img_array)
            self.cat_color_feature[cat_dir.name] = mean_color

        if not self.cat_ids:
            raise ValueError("No cats with sufficient images found.")

        print(f"Total cats: {len(self.cat_ids)}")
        total_images = sum(len(imgs) for imgs in self.cat_to_images.values())
        print(f"Total images: {total_images}")

    def _load_image(self, path: str) -> np.ndarray:
        img = keras.utils.load_img(path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        arr = keras.utils.img_to_array(img)
        arr = keras.applications.efficientnet_v2.preprocess_input(arr)
        return arr

    def _augment(self, image: np.ndarray) -> np.ndarray:
        aug = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.08),
            keras.layers.RandomZoom(0.1),
            keras.layers.RandomTranslation(0.05, 0.05),
            keras.layers.Lambda(lambda x: tf.image.adjust_brightness(x, 0.05)),
            keras.layers.Lambda(lambda x: tf.image.adjust_contrast(x, 1.05)),
        ])
        return aug(image, training=True).numpy()

    def _pick_hard_negative(self, anchor_cat: str, anchor_color: np.ndarray) -> str:
        candidates = []
        for cat_id in self.cat_ids:
            if cat_id == anchor_cat:
                continue
            color = self.cat_color_feature.get(cat_id)
            if color is None:
                continue
            distance = np.linalg.norm(anchor_color - color)
            candidates.append((distance, cat_id))

        candidates.sort(key=lambda x: x[0])
        # Try to pick most similar but not identical
        for dist, cat_id in candidates:
            if dist < 0.3:  # threshold for "similar color"
                return cat_id
        # Fallback to moderately similar
        if candidates:
            return candidates[min(len(candidates)-1, 3)][1]
        # Final fallback random
        return random.choice([cid for cid in self.cat_ids if cid != anchor_cat])

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        anchors, positives, negatives = [], [], []

        max_attempts = self.batch_size * 10
        attempts = 0

        while len(anchors) < self.batch_size and attempts < max_attempts:
            attempts += 1
            anchor_cat = random.choice(self.cat_ids)
            imgs = self.cat_to_images[anchor_cat]
            if len(imgs) < 2:
                continue

            anchor_path, positive_path = random.sample(imgs, 2)
            anchor_color = self.cat_color_feature[anchor_cat]
            negative_cat = self._pick_hard_negative(anchor_cat, anchor_color)
            negative_imgs = self.cat_to_images[negative_cat]
            negative_path = random.choice(negative_imgs)

            try:
                anchor_img = self._load_image(anchor_path)
                positive_img = self._load_image(positive_path)
                negative_img = self._load_image(negative_path)

                # Light augmentation
                anchor_img = self._augment(anchor_img)
                positive_img = self._augment(positive_img)
                negative_img = self._augment(negative_img)

                anchors.append(anchor_img)
                positives.append(positive_img)
                negatives.append(negative_img)
            except Exception:
                continue

        if len(anchors) == 0:
            raise ValueError("Unable to generate triplets. Check dataset integrity.")

        return (np.array(anchors, dtype=np.float32),
                np.array(positives, dtype=np.float32),
                np.array(negatives, dtype=np.float32))

# ---------------------------------------------------------------------------
# Model Architecture
# ---------------------------------------------------------------------------

def build_embedding_model() -> keras.Model:
    """Build EfficientNetV2-S embedding model (upgraded from EfficientNet-B0)."""
    inputs = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    
    # Use EfficientNetV2-S (Small) - better accuracy than EfficientNet-B0
    # Options: EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    #          EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    
    backbone.trainable = True
    # Unfreeze last 30 layers (EfficientNetV2-S has more layers than B0)
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    x = backbone.output
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(Config.EMBEDDING_DIM, activation="relu")(x)
    # Use a custom layer instead of Lambda for better serialization
    outputs = keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1), name="l2_normalize")(x)

    model = keras.Model(inputs, outputs, name="cat_embedding_model_efficientnetv2s")
    
    print("\n" + "="*80)
    print("Model Architecture")
    print("="*80)
    model.summary()
    print("="*80)
    
    return model

class TripletLoss(keras.losses.Loss):
    """Label-free triplet loss with mixed precision support."""
    def __init__(self, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin

    def call(self, y_true, y_pred):
        emb_dim = tf.shape(y_pred)[1] // 3
        anchor = y_pred[:, :emb_dim]
        positive = y_pred[:, emb_dim:2*emb_dim]
        negative = y_pred[:, 2*emb_dim:]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        # Cast margin and constants to match the dtype of distances (for mixed precision)
        margin_tensor = tf.cast(self.margin, dtype=pos_dist.dtype)
        zero_tensor = tf.cast(0.0, dtype=pos_dist.dtype)
        
        loss = tf.maximum(zero_tensor, pos_dist - neg_dist + margin_tensor)
        # Cast loss to float32 for proper loss computation (mixed precision requirement)
        loss = tf.cast(tf.reduce_mean(loss), dtype=tf.float32)
        return loss

def build_triplet_model(embedding_model: keras.Model) -> keras.Model:
    """Wrap embedding model to output concatenated triplets."""
    anchor_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name="anchor")
    positive_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name="positive")
    negative_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name="negative")

    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)

    concatenated = keras.layers.Concatenate(axis=1)([anchor_emb, positive_emb, negative_emb])
    triplet_model = keras.Model(inputs=[anchor_input, positive_input, negative_input],
                                outputs=concatenated,
                                name="triplet_model")
    return triplet_model

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_model(embedding_model: keras.Model, generator: TripletGenerator) -> Dict[str, List[float]]:
    # Enable mixed precision training for GPU (faster training, lower memory)
    if gpu_available:
        try:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision training enabled (float16) for faster GPU training")
        except Exception as e:
            print(f"⚠️  Could not enable mixed precision: {e}")
            print("   Continuing with float32")
    
    triplet_model = build_triplet_model(embedding_model)
    triplet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                          loss=TripletLoss(margin=Config.TRIPLET_MARGIN))

    history = {
        "loss": [],
        "pos_dist": [],
        "neg_dist": [],
        "margin": []
    }

    best_loss = float("inf")
    
    # Early stopping variables
    early_stop_patience = 3
    early_stop_min_delta = 1e-4
    no_improvement_count = 0
    previous_loss = float("inf")

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    print(f"Early stopping: Will stop if loss improvement < {early_stop_min_delta} for {early_stop_patience} consecutive epochs")
    print("="*80)

    for epoch in range(Config.EPOCHS):
        epoch_start_time = time.time()
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"{'='*80}")
        
        batch_losses, pos_dists, neg_dists = [], [], []
        batch_times = []

        num_batches = Config.TRIPLETS_PER_EPOCH // Config.BATCH_SIZE

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch+1}"):
            batch_start = time.time()
            
            anchors, positives, negatives = generator.generate_batch()
            loss = triplet_model.train_on_batch([anchors, positives, negatives],
                                                np.zeros((len(anchors), 1)))

            anchor_emb = embedding_model.predict(anchors, verbose=0)
            positive_emb = embedding_model.predict(positives, verbose=0)
            negative_emb = embedding_model.predict(negatives, verbose=0)

            pos_dist = np.mean(np.sum((anchor_emb - positive_emb)**2, axis=1))
            neg_dist = np.mean(np.sum((anchor_emb - negative_emb)**2, axis=1))

            batch_losses.append(loss)
            pos_dists.append(pos_dist)
            neg_dists.append(neg_dist)
            batch_times.append(time.time() - batch_start)

        epoch_loss = float(np.mean(batch_losses))
        epoch_pos = float(np.mean(pos_dists))
        epoch_neg = float(np.mean(neg_dists))
        epoch_margin = epoch_neg - epoch_pos
        avg_batch_time = float(np.mean(batch_times))
        epoch_time = time.time() - epoch_start_time

        history["loss"].append(epoch_loss)
        history["pos_dist"].append(epoch_pos)
        history["neg_dist"].append(epoch_neg)
        history["margin"].append(epoch_margin)

        # Calculate loss improvement (skip check on first epoch)
        if epoch > 0:
            loss_improvement = previous_loss - epoch_loss
        else:
            loss_improvement = float("inf")  # First epoch, no comparison
        
        # Enhanced logging with epoch number, batch time, and early stopping info
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Loss: {epoch_loss:.6f} | PosDist: {epoch_pos:.4f} | NegDist: {epoch_neg:.4f} | Margin: {epoch_margin:.4f}")
        if epoch > 0:
            print(f"   Loss Improvement: {loss_improvement:.6f} (previous: {previous_loss:.6f})")
        else:
            print(f"   Loss Improvement: N/A (first epoch)")
        print(f"   Avg Batch Time: {avg_batch_time:.3f}s | Epoch Time: {epoch_time:.1f}s")
        
        # Check for early stopping (skip on first epoch)
        if epoch > 0 and loss_improvement < early_stop_min_delta:
            no_improvement_count += 1
            print(f"   ⚠️  No significant improvement (count: {no_improvement_count}/{early_stop_patience})")
            
            if no_improvement_count >= early_stop_patience:
                print(f"\n{'='*80}")
                print(f"🛑 EARLY STOPPING TRIGGERED")
                print(f"{'='*80}")
                print(f"Loss improvement ({loss_improvement:.6f}) < threshold ({early_stop_min_delta})")
                print(f"for {early_stop_patience} consecutive epochs.")
                print(f"Stopping training at epoch {epoch+1}.")
                print(f"{'='*80}\n")
                break
        else:
            no_improvement_count = 0  # Reset counter on improvement

        # Save best model (multiple formats for compatibility)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            os.makedirs(Config.MODEL_SAVE_PATH.parent, exist_ok=True)
            
            saved_files = []
            
            # Method 1: Save weights separately (most reliable)
            weights_path = Config.MODEL_SAVE_PATH.parent / f"{Config.MODEL_SAVE_PATH.stem}_weights.h5"
            try:
                embedding_model.save_weights(str(weights_path))
                saved_files.append(("weights", weights_path))
                print(f"   ✅ Saved model weights: {weights_path}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not save weights: {e}")
            
            # Method 2: Save as SavedModel format (most robust, avoids Lambda layer issues)
            savedmodel_path = Config.MODEL_SAVE_PATH.parent / f"{Config.MODEL_SAVE_PATH.stem}_savedmodel"
            try:
                embedding_model.save(str(savedmodel_path), save_format='tf')
                saved_files.append(("savedmodel", savedmodel_path))
                print(f"   ✅ Saved SavedModel format: {savedmodel_path}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not save SavedModel format: {e}")
            
            # Method 3: Try saving as .h5 format
            h5_path = Config.MODEL_SAVE_PATH.with_suffix('.h5')
            try:
                embedding_model.save(str(h5_path), save_format='h5')
                # Verify file size is reasonable (> 1 MB)
                if h5_path.exists() and h5_path.stat().st_size > 1024 * 1024:
                    saved_files.append(("h5", h5_path))
                    print(f"   ✅ Saved .h5 format: {h5_path}")
                else:
                    print(f"   ⚠️  Warning: .h5 file too small, may be corrupted")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not save .h5 format: {e}")
            
            # Method 4: Try saving as .keras format (use SavedModel format internally)
            try:
                # Save using SavedModel format but with .keras extension
                embedding_model.save(str(Config.MODEL_SAVE_PATH), save_format='tf')
                # Verify file/directory exists
                if Config.MODEL_SAVE_PATH.exists():
                    if Config.MODEL_SAVE_PATH.is_dir():
                        # It's a directory (SavedModel format)
                        total_size = sum(f.stat().st_size for f in Config.MODEL_SAVE_PATH.rglob('*') if f.is_file())
                    else:
                        total_size = Config.MODEL_SAVE_PATH.stat().st_size
                    
                    if total_size > 1024 * 1024:  # > 1 MB
                        saved_files.append(("keras", Config.MODEL_SAVE_PATH))
                        print(f"   ✅ Saved .keras format: {Config.MODEL_SAVE_PATH}")
                    else:
                        print(f"   ⚠️  Warning: .keras file too small, may be corrupted")
                else:
                    print(f"   ⚠️  Warning: .keras file was not created")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not save .keras format: {e}")
            
            if saved_files:
                print(f"\n   ✅ Successfully saved model in {len(saved_files)} format(s):")
                for fmt, path in saved_files:
                    if path.is_dir():
                        # Calculate directory size
                        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        size_mb = total_size / (1024 * 1024)
                    else:
                        size_mb = path.stat().st_size / (1024 * 1024)
                    print(f"      - {fmt}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"   ❌ Error: Failed to save model in any format!")
        
        previous_loss = epoch_loss

    return history

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_history(history: Dict[str, List[float]]):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history["loss"], label="Triplet Loss", color="blue")
    axes[0].set_title("Triplet Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["pos_dist"], label="Positive Distance", color="green")
    axes[1].plot(epochs, history["neg_dist"], label="Negative Distance", color="red")
    axes[1].set_title("Embedding Distances")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Squared Distance")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(Config.PLOT_PATH, dpi=150)
    plt.close()
    print(f"📈 Training curves saved to {Config.PLOT_PATH}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("="*80)
    print("Cat Identifier Training - EfficientNetV2-S Triplet Model")
    print("="*80)
    print("Upgraded from EfficientNet-B0 for better accuracy")
    print("="*80)
    
    # Validate dataset path
    if not Config.DATASET_PATH.exists():
        print(f"\n❌ ERROR: Dataset path not found: {Config.DATASET_PATH}")
        print(f"   Please check the path in Config.DATASET_PATH")
        sys.exit(1)
    
    if not Config.DATASET_PATH.is_dir():
        print(f"\n❌ ERROR: Dataset path is not a directory: {Config.DATASET_PATH}")
        sys.exit(1)
    
    print(f"\n✅ Dataset path validated: {Config.DATASET_PATH}")
    
    # Print configuration summary
    print("\n" + "="*80)
    print("Training Configuration")
    print("="*80)
    print(f"Dataset: {Config.DATASET_PATH}")
    print(f"Model save path: {Config.MODEL_SAVE_PATH}")
    print(f"Image size: {Config.IMG_SIZE}x{Config.IMG_SIZE}")
    print(f"Embedding dimension: {Config.EMBEDDING_DIM}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Triplet margin: {Config.TRIPLET_MARGIN}")
    print(f"Triplets per epoch: {Config.TRIPLETS_PER_EPOCH}")
    print(f"GPU available: {'Yes' if gpu_available else 'No (CPU mode)'}")
    print("="*80)

    # Load dataset
    generator = TripletGenerator(Config.DATASET_PATH, Config.BATCH_SIZE)

    # Build model
    embedding_model = build_embedding_model()
    
    # Verify model is on correct device
    print("\n" + "="*80)
    print("Model Device Placement")
    print("="*80)
    try:
        sample_input = tf.zeros((1, Config.IMG_SIZE, Config.IMG_SIZE, 3))
        _ = embedding_model(sample_input)
        print("✅ Model built successfully")
        if gpu_available:
            print("✅ Model will run on GPU (TensorFlow will automatically place operations on GPU)")
        else:
            print("⚠️  Model will run on CPU")
    except Exception as e:
        print(f"⚠️  Warning: Could not verify model device placement: {e}")
    print("="*80)

    # Train
    history = train_model(embedding_model, generator)

    # Save history
    with open(Config.HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)
    print(f"📝 Training history saved to {Config.HISTORY_PATH}")

    # Plot
    plot_history(history)

    # Final save (ensure model is saved in working formats)
    print("\n" + "="*80)
    print("Performing final model save...")
    print("="*80)
    
    os.makedirs(Config.MODEL_SAVE_PATH.parent, exist_ok=True)
    saved_files = []
    
    # Save weights (most reliable)
    weights_path = Config.MODEL_SAVE_PATH.parent / f"{Config.MODEL_SAVE_PATH.stem}_weights.h5"
    try:
        embedding_model.save_weights(str(weights_path))
        size_mb = weights_path.stat().st_size / (1024 * 1024)
        saved_files.append(("weights", weights_path, size_mb))
        print(f"✅ Saved model weights: {weights_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to save weights: {e}")
    
    # Save SavedModel format (most robust)
    savedmodel_path = Config.MODEL_SAVE_PATH.parent / f"{Config.MODEL_SAVE_PATH.stem}_savedmodel"
    try:
        embedding_model.save(str(savedmodel_path), save_format='tf')
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in savedmodel_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        saved_files.append(("savedmodel", savedmodel_path, size_mb))
        print(f"✅ Saved SavedModel format: {savedmodel_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to save SavedModel: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("Training complete!")
    print("="*80)
    
    if saved_files:
        print(f"\n✅ Model successfully saved in {len(saved_files)} format(s):")
        for fmt, path, size_mb in saved_files:
            print(f"   - {fmt}: {path} ({size_mb:.2f} MB)")
        print(f"\n💡 Recommended: Use the 'weights' or 'savedmodel' format for loading.")
        print(f"   The test script will automatically detect and use these formats.")
    else:
        print(f"\n❌ WARNING: Failed to save model in any reliable format!")
        print(f"   The .h5 and .keras files may be corrupted due to Lambda layer serialization issues.")
        print(f"   Please check the error messages above.")
    
    print(f"\nTraining curves: {Config.PLOT_PATH}")
    print(f"History JSON: {Config.HISTORY_PATH}")

if __name__ == "__main__":
    main()

