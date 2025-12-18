"""
Test Script for Individual Cat Identifier (v2)
===============================================
Tests the trained EfficientNet-B0 based cat identification model.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pickle
from typing import List, Tuple, Optional
import math

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("❌ Error: TensorFlow not found. Please install it with: pip install tensorflow")
    sys.exit(1)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except ImportError:
    print("⚠️  Warning: Matplotlib not found. Visualization will be disabled.")
    plt = None

def ensure_gui_backend():
    if plt is None:
        return
    try:
        import matplotlib as _mpl
        backend = _mpl.get_backend()
        if 'agg' in backend.lower():
            for candidate in ['TkAgg', 'Qt5Agg', 'WXAgg']:
                try:
                    _mpl.use(candidate, force=True)
                    break
                except Exception:
                    continue
        try:
            plt.ion()
        except Exception:
            pass
    except Exception:
        pass



# Model configuration (should match training script)
MODEL_BASE_PATH = Path(r"D:\Cursor AI projects\Capstone2.1\models\cat_identifier_efficientnet_v2")
MODEL_PATH = MODEL_BASE_PATH.with_suffix('.keras')
WEIGHTS_PATH = MODEL_BASE_PATH.parent / f"{MODEL_BASE_PATH.name}_weights.h5"
SAVEDMODEL_PATH = MODEL_BASE_PATH.parent / f"{MODEL_BASE_PATH.name}_savedmodel"
DATASET_PATH = Path(r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset")
EMBEDDINGS_CACHE_PATH = Path(r"D:\Cursor AI projects\Capstone2.1\models\dataset_embeddings_cache.pkl")
IMG_SIZE = 224
EMBEDDING_DIM = 512
DEFAULT_THRESHOLD = 0.8
SUPPORTED_IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BATCH_SIZE = 32  # Batch size for embedding extraction
BLUR_SHARPNESS_THRESHOLD = 0.002


def build_embedding_model():
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    backbone = keras.applications.EfficientNetV2S(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg"
    )
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    x = backbone.output
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(EMBEDDING_DIM, activation="relu")(x)
    outputs = keras.layers.Lambda(lambda t: tf.nn.l2_normalize(t, axis=1))(x)
    model = keras.Model(inputs, outputs, name="cat_embedding_model_efficientnetv2s")
    return model


def load_model(model_path=None):
    """Load the trained model - tries multiple formats for compatibility"""
    model = None
    last_error = None
    
    # Method 1: Try SavedModel format (most reliable)
    if SAVEDMODEL_PATH.exists():
        print(f"Found SavedModel format: {SAVEDMODEL_PATH}")
        try:
            model = keras.models.load_model(str(SAVEDMODEL_PATH), compile=False)
            if model is not None:
                print("✅ Model loaded successfully from SavedModel format!")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
            return model
        except Exception as e:
            last_error = e
            print(f"⚠️  Failed to load SavedModel: {e}")
    
    # Method 2: Try loading weights file (reconstruct architecture + load weights)
    if WEIGHTS_PATH.exists():
        print(f"\nFound weights file: {WEIGHTS_PATH}")
        print("Reconstructing model architecture and loading weights...")
        try:
            model = build_embedding_model()
            model.load_weights(str(WEIGHTS_PATH))
            if model is not None:
                print("✅ Model loaded successfully from weights file!")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
            return model
        except Exception as e:
            last_error = e
            print(f"⚠️  Failed to load weights: {e}")
    
    # Method 3: Try .h5 format
    h5_path = MODEL_PATH.with_suffix('.h5')
    if h5_path.exists():
        file_size_mb = h5_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1.0:  # Only try if file is reasonably sized
            print(f"\nFound .h5 model file: {h5_path} ({file_size_mb:.2f} MB)")
            try:
                model = keras.models.load_model(str(h5_path), compile=False)
                if model is not None:
                    print("✅ Model loaded successfully from .h5 file!")
                    print(f"   Input shape: {model.input_shape}")
                    print(f"   Output shape: {model.output_shape}")
                return model
            except Exception as e:
                last_error = e
                print(f"⚠️  Failed to load .h5 file: {e}")
        else:
            print(f"\n⚠️  .h5 file too small ({file_size_mb:.2f} MB), likely corrupted. Skipping...")
    
    # Method 4: Try .keras format
    if MODEL_PATH.exists():
        file_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if file_size_mb > 1.0:  # Only try if file is reasonably sized
            print(f"\nFound .keras model file: {MODEL_PATH} ({file_size_mb:.2f} MB)")
            try:
                model = keras.models.load_model(str(MODEL_PATH), compile=False)
                if model is not None:
                    print("✅ Model loaded successfully from .keras file!")
                    print(f"   Input shape: {model.input_shape}")
                    print(f"   Output shape: {model.output_shape}")
                return model
            except Exception as e:
                last_error = e
                print(f"⚠️  Failed to load .keras file: {e}")
        else:
            print(f"\n⚠️  .keras file too small ({file_size_mb:.2f} MB), likely corrupted. Skipping...")
    
    # Method 5: Last resort - try to load weights from any available file
    print("\n⚠️  All standard loading methods failed. Attempting last resort...")
    model = build_embedding_model()
    
    # Try loading weights from any available file
    for weight_file in [WEIGHTS_PATH, h5_path, MODEL_PATH]:
        if weight_file.exists():
            try:
                model.load_weights(str(weight_file))
                print(f"✅ Model weights loaded from: {weight_file}")
                print(f"   Input shape: {model.input_shape}")
                print(f"   Output shape: {model.output_shape}")
                return model
            except Exception:
                continue
    
    # If we get here, nothing worked
    print(f"\n❌ Failed to load model from any available format.")
    print(f"\nChecked locations:")
    print(f"   - SavedModel: {SAVEDMODEL_PATH} ({'exists' if SAVEDMODEL_PATH.exists() else 'not found'})")
    print(f"   - Weights: {WEIGHTS_PATH} ({'exists' if WEIGHTS_PATH.exists() else 'not found'})")
    print(f"   - .h5: {h5_path} ({'exists' if h5_path.exists() else 'not found'})")
    print(f"   - .keras: {MODEL_PATH} ({'exists' if MODEL_PATH.exists() else 'not found'})")
    print(f"\nLast error: {last_error}")
    print("\n⚠️  WARNING: Model is using untrained weights (ImageNet only).")
    print("   Accuracy will be significantly reduced!")
    print("\nPlease retrain the model by running: python train_cat_identifier_v2.py")
    print("   The updated training script will save in multiple formats for better compatibility.")
    
    # Return model with ImageNet weights as fallback (better than nothing)
    print("\n⚠️  Using model with ImageNet pretrained weights only (not fine-tuned)...")
    return model


def preprocess_image(image_path):
    """Preprocess image for EfficientNetV2-S (matches training)."""
    img = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras.utils.img_to_array(img)
    # Use EfficientNetV2 preprocessing to match the V2 backbone
    arr = keras.applications.efficientnet_v2.preprocess_input(arr)
    return arr

def load_raw_image(image_path):
    img = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras.utils.img_to_array(img)
    return arr

def image_sharpness(arr):
    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    s = float(np.mean(gx * gx) + np.mean(gy * gy)) / 255.0
    return s


def extract_embedding(model, image_path):
    """Extract embedding from an image"""
    img_array = preprocess_image(image_path)
    img_batch = np.expand_dims(img_array, axis=0)
    embedding = model.predict(img_batch, verbose=0)[0]
    return embedding

def extract_embedding_tta(model, image_path):
    base = keras.utils.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras.utils.img_to_array(base)
    variants = []
    variants.append(arr)
    variants.append(np.flip(arr, axis=1))
    b_adj = tf.image.adjust_brightness(arr, 0.05)
    c_adj = tf.image.adjust_contrast(arr, 1.05)
    variants.append(np.array(b_adj))
    variants.append(np.array(c_adj))
    inputs = [keras.applications.efficientnet_v2.preprocess_input(v) for v in variants]
    batch = np.stack(inputs, axis=0)
    if model is None:
        raise RuntimeError("model is not loaded")
    embs = model.predict(batch)
    mean_emb = np.mean(embs, axis=0)
    mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
    return mean_emb


def extract_embeddings_batch(model, image_paths: List[str], batch_size: int = BATCH_SIZE):
    """
    Extract embeddings for multiple images in batches (much faster).
    
    Args:
        model: The embedding model
        image_paths: List of image paths
        batch_size: Batch size for processing
    
    Returns:
        embeddings: NumPy array of embeddings
        valid_paths: List of successfully processed image paths
    """
    embeddings = []
    valid_paths = []
    
    print(f"Processing {len(image_paths)} images in batches of {batch_size}...")
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        # Load and preprocess batch
        for img_path in batch_paths:
            try:
                raw = load_raw_image(img_path)
                sharp = image_sharpness(raw)
                if sharp < BLUR_SHARPNESS_THRESHOLD:
                    continue
                img_array = preprocess_image(img_path)
                batch_images.append(img_array)
                batch_valid_paths.append(img_path)
            except Exception as e:
                print(f"⚠️  Warning: Failed to load {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Predict on batch
        batch_array = np.array(batch_images)
        if model is None:
            raise RuntimeError("model is not loaded")
        batch_embeddings = model.predict(batch_array)
        
        embeddings.extend(batch_embeddings)
        valid_paths.extend(batch_valid_paths)
        
        if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(image_paths):
            print(f"  Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images...", end='\r')
    
    print(f"\n✅ Processed {len(valid_paths)} images successfully")
    return np.array(embeddings), valid_paths


def cosine_similarity(emb1, emb2):
    """Compute cosine similarity between two embeddings"""
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    return float(np.dot(emb1, emb2))


def compare_images(model, img1_path, img2_path, threshold=DEFAULT_THRESHOLD, visualize=True):
    """Compare two cat images and determine if they're the same cat"""
    print("\n" + "="*80)
    print("CAT COMPARISON TEST")
    print("="*80)
    
    if not os.path.exists(img1_path):
        print(f"❌ Error: Image 1 not found at {img1_path}")
        return None
    
    if not os.path.exists(img2_path):
        print(f"❌ Error: Image 2 not found at {img2_path}")
        return None
    
    print(f"\nImage 1: {img1_path}")
    print(f"Image 2: {img2_path}")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    emb1 = extract_embedding_tta(model, img1_path)
    emb2 = extract_embedding_tta(model, img2_path)
    
    # Compute similarity
    similarity = cosine_similarity(emb1, emb2)
    
    # Determine result
    is_same = similarity >= threshold
    
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Cosine Similarity: {similarity:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Verdict: {'✅ SAME CAT' if is_same else '❌ DIFFERENT CATS'}")
    print(f"{'='*80}")
    
    # Visualize if matplotlib is available
    if visualize and plt is not None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            # Load and display images
            img1_display = mpimg.imread(img1_path)
            img2_display = mpimg.imread(img2_path)
            
            axes[0].imshow(img1_display)
            axes[0].set_title(f"Image 1\n{os.path.basename(img1_path)}", fontsize=10)
            axes[0].axis('off')
            
            axes[1].imshow(img2_display)
            axes[1].set_title(f"Image 2\n{os.path.basename(img2_path)}", fontsize=10)
            axes[1].axis('off')
            
            # Add similarity score
            fig.suptitle(
                f"Similarity: {similarity:.4f} | {'SAME CAT' if is_same else 'DIFFERENT CATS'}",
                fontsize=14,
                fontweight='bold',
                color='green' if is_same else 'red'
            )
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"⚠️  Warning: Could not display images: {e}")
    
    return {
        'similarity': similarity,
        'is_same': is_same,
        'threshold': threshold
    }


def test_multiple_comparisons(model, image_paths, threshold=DEFAULT_THRESHOLD):
    """Test comparing multiple images against each other"""
    print("\n" + "="*80)
    print("BATCH COMPARISON TEST")
    print("="*80)
    
    n = len(image_paths)
    print(f"Comparing {n} images with each other...")
    
    # Extract all embeddings first
    print("\nExtracting embeddings for all images...")
    embeddings = []
    for i, img_path in enumerate(image_paths):
        print(f"  Processing {i+1}/{n}: {os.path.basename(img_path)}")
        emb = extract_embedding(model, img_path)
        embeddings.append(emb)
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    print("\nComputing similarity matrix...")
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                similarity = cosine_similarity(embeddings[i], embeddings[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    
    # Print results
    print("\n" + "="*80)
    print("SIMILARITY MATRIX")
    print("="*80)
    
    # Print header
    print(f"{'Image':30}", end='')
    for i in range(n):
        print(f"{i+1:8}", end='')
    print()
    print("-" * (30 + 8 * n))
    
    # Print matrix
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)[:28]
        print(f"{filename:30}", end='')
        for j in range(n):
            sim = similarity_matrix[i][j]
            if i == j:
                marker = "  "
            elif sim >= threshold:
                marker = "✅"
            else:
                marker = "❌"
            print(f"{sim:6.3f} {marker:2}", end='')
        print()
    
    print("\n" + "="*80)
    print("Legend:")
    print("✅ = Same cat (similarity >= threshold)")
    print("❌ = Different cats (similarity < threshold)")
    print("="*80)


def load_dataset_images(dataset_path=None):
    """
    Load all images from the dataset directory.
    Returns a list of (image_path, cat_id) tuples.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        return []
    
    print(f"Scanning dataset: {dataset_path}")
    image_list = []
    
    # Each subfolder represents a cat (e.g., 0001, 0002, etc.)
    for cat_folder in sorted(os.listdir(dataset_path)):
        cat_path = os.path.join(dataset_path, cat_folder)
        if not os.path.isdir(cat_path):
            continue
        
        cat_id = cat_folder
        # Find all images in this cat's folder
        for filename in os.listdir(cat_path):
            file_path = os.path.join(cat_path, filename)
            if os.path.isfile(file_path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in SUPPORTED_IMAGE_EXT:
                    image_list.append((file_path, cat_id))
    
    print(f"✅ Found {len(image_list)} images from {len(set(cat_id for _, cat_id in image_list))} cats")
    return image_list


def load_cached_embeddings() -> Optional[Tuple[np.ndarray, List[str], List[str]]]:
    """
    Load cached embeddings from disk.
    
    Returns:
        (embeddings, image_paths, cat_ids) if cache exists, None otherwise
    """
    if not EMBEDDINGS_CACHE_PATH.exists():
        return None
    
    try:
        print(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}...")
        with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
            cache_data = pickle.load(f)
        
        embeddings = cache_data['embeddings']
        image_paths = cache_data['image_paths']
        cat_ids = cache_data['cat_ids']
        
        print(f"✅ Loaded {len(embeddings)} cached embeddings")
        return embeddings, image_paths, cat_ids
    except Exception as e:
        print(f"⚠️  Warning: Failed to load cache: {e}")
        return None


def save_cached_embeddings(embeddings: np.ndarray, image_paths: List[str], cat_ids: List[str]):
    """
    Save embeddings to cache file.
    
    Args:
        embeddings: NumPy array of embeddings
        image_paths: List of image paths
        cat_ids: List of cat IDs
    """
    try:
        # Ensure parent directory exists
        EMBEDDINGS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'embeddings': embeddings,
            'image_paths': image_paths,
            'cat_ids': cat_ids
        }
        
        print(f"Saving embeddings cache to {EMBEDDINGS_CACHE_PATH}...")
        with open(EMBEDDINGS_CACHE_PATH, 'wb') as f:
            pickle.dump(cache_data, f)
        
        cache_size_mb = EMBEDDINGS_CACHE_PATH.stat().st_size / (1024 * 1024)
        print(f"✅ Cache saved ({cache_size_mb:.2f} MB)")
    except Exception as e:
        print(f"⚠️  Warning: Failed to save cache: {e}")


def compute_dataset_embeddings(model, dataset_path=None, force_recompute=False):
    """
    Compute embeddings for all dataset images, using cache if available.
    
    Args:
        model: The embedding model
        dataset_path: Path to dataset (default: DATASET_PATH)
        force_recompute: If True, recompute even if cache exists
    
    Returns:
        (embeddings, image_paths, cat_ids)
    """
    # Try to load from cache first
    if not force_recompute:
        cached = load_cached_embeddings()
        if cached is not None:
            return cached
    
    # Load dataset images
    dataset_images = load_dataset_images(dataset_path)
    if not dataset_images:
        return None, [], []
    
    # Separate paths and cat IDs
    image_paths = [path for path, _ in dataset_images]
    cat_ids = [cat_id for _, cat_id in dataset_images]
    
    # Extract embeddings in batches (much faster)
    print("\nComputing embeddings for all dataset images...")
    print("This will take a few minutes, but only needs to be done once!")
    embeddings, valid_paths = extract_embeddings_batch(model, image_paths, batch_size=BATCH_SIZE)
    
    # Match valid paths with cat IDs
    valid_cat_ids = []
    path_to_cat = dict(dataset_images)
    for path in valid_paths:
        valid_cat_ids.append(path_to_cat[path])
    
    # Save to cache
    save_cached_embeddings(embeddings, valid_paths, valid_cat_ids)
    
    return embeddings, valid_paths, valid_cat_ids


def find_similar_cats(model, query_image_path, dataset_path=None, top_k=10, threshold=None, 
                     visualize=True, force_recompute=False):
    """
    Find the most similar cats in the dataset for a given query image.
    Uses cached embeddings for fast searches!
    
    Args:
        model: The trained embedding model
        query_image_path: Path to the query image
        dataset_path: Path to the dataset directory (default: DATASET_PATH)
        top_k: Number of top similar cats to return
        threshold: Optional similarity threshold to filter results
        visualize: Whether to display visualization
        force_recompute: If True, recompute dataset embeddings even if cache exists
    
    Returns:
        List of tuples: (image_path, cat_id, similarity_score)
    """
    print("\n" + "="*80)
    print("DATASET SEARCH: Finding Most Similar Cats")
    print("="*80)
    
    if not os.path.exists(query_image_path):
        print(f"❌ Error: Query image not found at {query_image_path}")
        return []
    
    print(f"\nQuery Image: {query_image_path}")
    
    # Load or compute dataset embeddings (uses cache if available)
    embeddings, image_paths, cat_ids = compute_dataset_embeddings(
        model, dataset_path, force_recompute=force_recompute
    )
    
    if embeddings is None or len(embeddings) == 0:
        print("❌ No embeddings available!")
        return []
    
    # Extract embedding for query image (fast - just one image)
    print("\nExtracting embedding for query image...")
    try:
        query_embedding = extract_embedding_tta(model, query_image_path)
        print("✅ Query embedding extracted")
    except Exception as e:
        print(f"❌ Error extracting query embedding: {e}")
        return []
    
    # Compute similarities (very fast - just matrix multiplication)
    print(f"\nComputing similarities with {len(embeddings)} dataset images...")
    # Embeddings are L2-normalized, so dot product = cosine similarity
    similarities = np.dot(embeddings, query_embedding)
    
    # Create list of (path, cat_id, similarity) tuples
    results = [(path, cat_id, float(sim)) for path, cat_id, sim in 
               zip(image_paths, cat_ids, similarities)]
    
    # Sort by similarity (descending)
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Filter by threshold if provided
    if threshold is not None:
        results = [(path, cat_id, sim) for path, cat_id, sim in results if sim >= threshold]
    
    # Get top K results
    top_results = results[:top_k]
    
    # Print results
    print("\n" + "="*80)
    print(f"TOP {len(top_results)} MOST SIMILAR CATS")
    print("="*80)
    print(f"{'Rank':<6} {'Cat ID':<12} {'Similarity':<12} {'Image Path'}")
    print("-" * 80)
    
    for rank, (img_path, cat_id, similarity) in enumerate(top_results, 1):
        marker = "✅" if threshold is None or similarity >= threshold else "❌"
        print(f"{rank:<6} {cat_id:<12} {similarity:.4f} {marker:<2} {img_path}")
    
    print("="*80)
    
    # Visualize if matplotlib is available
    if visualize and plt is not None and top_results:
        try:
            ensure_gui_backend()
            plt.close('all')
            n_display = min(len(top_results), top_k)
            cols = 5
            rows = math.ceil((n_display + 1) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)

            try:
                mgr = getattr(fig.canvas, "manager", None)
                if mgr is not None:
                    mgr.set_window_title("Cat Search Results")
            except Exception:
                pass

            if mpimg is None:
                raise RuntimeError("matplotlib.image not available")
            query_img = mpimg.imread(query_image_path)
            axes[0].imshow(query_img)
            axes[0].set_title("Query Image", fontsize=10, fontweight='bold')
            axes[0].axis('off')

            for i, (img_path, cat_id, similarity) in enumerate(top_results[:n_display], start=1):
                try:
                    if mpimg is None:
                        raise RuntimeError("matplotlib.image not available")
                    result_img = mpimg.imread(img_path)
                    axes[i].imshow(result_img)
                except Exception:
                    blank = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                    axes[i].imshow(blank)
                    axes[i].text(0.5, 0.5, "missing", ha='center', va='center', fontsize=10, color='red')
                axes[i].set_title(
                    f"#{i}: Cat {cat_id}\nSim: {similarity:.4f}",
                    fontsize=9,
                    color='green' if similarity >= (threshold or DEFAULT_THRESHOLD) else 'orange'
                )
                axes[i].axis('off')

            for j in range(n_display + 1, rows * cols):
                axes[j].axis('off')

            fig.suptitle(
                f"Top {n_display} Most Similar Cats",
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show(block=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not display visualization: {e}")
    
    return top_results


def find_similar_cats_by_centroid(model, query_image_path, dataset_path=None, top_k=10, threshold=None,
                                  visualize=True, force_recompute=False):
    print("\n" + "="*80)
    print("DATASET SEARCH: Finding Most Similar Cats (Centroid)")
    print("="*80)
    if not os.path.exists(query_image_path):
        print("❌ Error: Query image not found")
        return []
    print(f"\nQuery Image: {query_image_path}")
    embeddings, image_paths, cat_ids = compute_dataset_embeddings(
        model, dataset_path, force_recompute=force_recompute
    )
    if embeddings is None or len(embeddings) == 0:
        print("❌ No embeddings available!")
        return []
    query_embedding = extract_embedding_tta(model, query_image_path)
    by_cat = {}
    rep_path = {}
    for emb, p, cid in zip(embeddings, image_paths, cat_ids):
        by_cat.setdefault(cid, []).append(emb)
        if cid not in rep_path:
            rep_path[cid] = p
    centroids = {cid: np.mean(np.stack(vals, axis=0), axis=0) for cid, vals in by_cat.items() if len(vals) > 0}
    results = []
    for cid, cen in centroids.items():
        sim = cosine_similarity(query_embedding, cen)
        results.append((cid, sim, rep_path[cid]))
    results.sort(key=lambda x: x[1], reverse=True)
    if threshold is not None:
        results = [(cid, sim, p) for cid, sim, p in results if sim >= threshold]
    top = results[:top_k]
    print("\n" + "="*80)
    print(f"TOP {len(top)} MOST SIMILAR CATS (Centroid)")
    print("="*80)
    print(f"{'Rank':<6} {'Cat ID':<12} {'Similarity':<12} {'Representative'}")
    print("-" * 80)
    for i, (cid, sim, p) in enumerate(top, start=1):
        print(f"{i:<6} {cid:<12} {sim:.4f} {p}")
    if visualize and plt is not None and top:
        try:
            ensure_gui_backend()
            plt.close('all')
            n_display = min(len(top), top_k)
            cols = 5
            rows = math.ceil((n_display + 1) / cols)
            fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
            axes = np.array(axes).reshape(-1)
            try:
                mgr = getattr(fig.canvas, "manager", None)
                if mgr is not None:
                    mgr.set_window_title("Cat Search Results (Centroid)")
            except Exception:
                pass
            query_img = mpimg.imread(query_image_path)
            axes[0].imshow(query_img)
            axes[0].set_title("Query Image", fontsize=10, fontweight='bold')
            axes[0].axis('off')
            for i, (cid, sim, p) in enumerate(top[:n_display], start=1):
                try:
                    result_img = mpimg.imread(p)
                    axes[i].imshow(result_img)
                except Exception:
                    blank = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
                    axes[i].imshow(blank)
                    axes[i].text(0.5, 0.5, "missing", ha='center', va='center', fontsize=10, color='red')
                axes[i].set_title(
                    f"#{i}: Cat {cid}\nSim: {sim:.4f}",
                    fontsize=9,
                )
                axes[i].axis('off')
            for j in range(n_display + 1, rows * cols):
                axes[j].axis('off')
            fig.suptitle(
                f"Top {n_display} Most Similar Cats (Centroid)",
                fontsize=14,
                fontweight='bold'
            )
            plt.tight_layout()
            plt.show(block=True)
        except Exception as e:
            print(f"⚠️  Warning: Could not display visualization: {e}")
    return top


def interactive_mode(model):
    """Interactive mode for testing"""
    print("\n" + "="*80)
    print("INTERACTIVE CAT COMPARISON MODE")
    print("="*80)
    
    threshold = float(input("Enter similarity threshold (default 0.8): ") or str(DEFAULT_THRESHOLD))
    top_k = 10
    
    while True:
        print("\n" + "-"*80)
        print("Options:")
        print("1. Compare two images")
        print("2. Batch comparison of multiple images")
        print("3. Find most similar cat in dataset (search)")
        print("4. Change threshold")
        print("5. Change top-K results (for search)")
        print("6. Exit")
        print("7. Find most similar cats by centroid (stable)")
        print("-"*80)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == '1':
            img1 = input("Enter path to first image: ").strip().strip('"')
            img2 = input("Enter path to second image: ").strip().strip('"')
            compare_images(model, img1, img2, threshold)
        
        elif choice == '2':
            print("Enter image paths (one per line, empty line to finish):")
            paths = []
            while True:
                path = input(f"Image {len(paths)+1}: ").strip().strip('"')
                if not path:
                    break
                if os.path.exists(path):
                    paths.append(path)
                else:
                    print(f"⚠️  Warning: {path} not found, skipping...")
            
            if len(paths) >= 2:
                test_multiple_comparisons(model, paths, threshold)
            else:
                print("❌ Need at least 2 valid images for comparison")
        
        elif choice == '3':
            query_img = input("Enter path to query image: ").strip().strip('"')
            find_similar_cats(model, query_img, top_k=10, threshold=None, force_recompute=False)

        elif choice == '7':
            query_img = input("Enter path to query image: ").strip().strip('"')
            use_threshold = input(f"Filter by threshold {threshold}? (y/n, default=n): ").strip().lower()
            filter_threshold = threshold if use_threshold == 'y' else None
            recompute = input("Force recompute embeddings? (y/n, default=n): ").strip().lower() == 'y'
            find_similar_cats_by_centroid(model, query_img, top_k=10, threshold=filter_threshold, force_recompute=recompute)
        
        elif choice == '4':
            threshold = float(input("Enter new similarity threshold: ") or str(DEFAULT_THRESHOLD))
            print(f"✅ Threshold updated to {threshold}")
        
        elif choice == '5':
            top_k = int(input(f"Enter number of top results to show (current: {top_k}): ") or str(top_k))
            print(f"✅ Top-K updated to {top_k}")
        
        elif choice == '6':
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice, please enter 1-6")


def main():
    """Main function"""
    print("="*80)
    print("CAT IDENTITY COMPARISON TEST (v2 - Individual Cats)")
    print("="*80)
    
    # Load model
    model = load_model()
    
    # Check command line arguments
    if len(sys.argv) == 2:
        # Single image provided - search dataset
        query_image = sys.argv[1]
        print(f"\nSearching dataset for cats similar to: {query_image}")
        # Force recompute to avoid stale cache after retraining
        find_similar_cats(model, query_image, force_recompute=True)
    
    elif len(sys.argv) == 3:
        # Two images provided as arguments
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        compare_images(model, img1_path, img2_path)
    
    elif len(sys.argv) > 3:
        # Multiple images for batch comparison
        image_paths = sys.argv[1:]
        test_multiple_comparisons(model, image_paths)
    
    else:
        # Interactive mode
        print("\nNo images provided as arguments. Starting interactive mode...")
        print("Usage examples:")
        print("  - Compare two images: python test_cat_identifier_v2.py img1.jpg img2.jpg")
        print("  - Search dataset: python test_cat_identifier_v2.py query.jpg")
        print("  - Batch compare: python test_cat_identifier_v2.py img1.jpg img2.jpg img3.jpg ...")
        interactive_mode(model)


if __name__ == "__main__":
    main()
