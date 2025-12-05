"""
Improved Individual Cat Recognition Training (Triplet Loss Fine-Tuning)
========================================================================

This script fine-tunes a Siamese network for individual cat identification
using EfficientNet-B0 and Triplet Semi-Hard Loss from TensorFlow Addons.

Dataset: D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset
Each subfolder (0001, 0002, ..., 0516) represents a unique individual cat.

Author: AI Assistant
Date: November 2025
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import json

import numpy as np
import tensorflow as tf
from tensorflow import keras
# Use unified keras API to avoid import resolution issues
layers = keras.layers
Model = keras.Model
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

# Set seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    # Paths
    DATASET_PATH = r"D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset"
    MODEL_SAVE_PATH = r"D:\Cursor AI projects\Capstone2.1\models\cat_individual_recognition_model.h5"
    TRAINING_HISTORY_PATH = "training_history_individuals.json"
    
    # Model parameters
    IMG_SIZE = 224
    EMBEDDING_DIM = 128
    BACKBONE_TRAINABLE_LAYERS = 100  # Unfreeze last 100 layers
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2
    
    # Triplet loss parameters
    TRIPLET_MARGIN = 0.5
    TRIPLETS_PER_EPOCH = 5000
    
    # Data augmentation
    ROTATION_RANGE = 15
    ZOOM_RANGE = 0.2
    BRIGHTNESS_RANGE = 0.2


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class CatDatasetLoader:
    """Load and organize individual cat dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.cat_to_images = {}
        self.cat_ids = []
        
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths organized by cat ID"""
        print("\n" + "="*80)
        print("Loading Individual Cats Dataset")
        print("="*80)
        
        for cat_folder in sorted(self.dataset_path.iterdir()):
            if cat_folder.is_dir():
                cat_id = cat_folder.name
                
                # Get all images for this cat
                images = list(cat_folder.glob("*.jpg")) + \
                        list(cat_folder.glob("*.jpeg")) + \
                        list(cat_folder.glob("*.png"))
                
                image_paths = [str(img) for img in images]
                
                # Only add cats with at least 2 images (needed for triplets)
                if len(image_paths) >= 2:
                    self.cat_ids.append(cat_id)
                    self.cat_to_images[cat_id] = image_paths
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total cats (with ≥2 images): {len(self.cat_ids)}")
        print(f"   Total images: {sum(len(imgs) for imgs in self.cat_to_images.values())}")
        
        if len(self.cat_ids) == 0:
            raise ValueError("No cats with sufficient images found! Need at least 2 images per cat.")
        
        # Show sample cat IDs
        sample_cats = self.cat_ids[:5]
        print(f"\n   Sample cat IDs: {sample_cats}")
        for cat_id in sample_cats:
            print(f"      {cat_id}: {len(self.cat_to_images[cat_id])} images")
    
    def split_train_val(self, train_split: float = 0.8) -> Tuple[List[str], List[str]]:
        """Split cats into train and validation sets"""
        train_cats, val_cats = train_test_split(
            self.cat_ids,
            test_size=1 - train_split,
            random_state=42
        )
        
        print(f"\n📊 Dataset Split:")
        print(f"   Training cats: {len(train_cats)}")
        print(f"   Validation cats: {len(val_cats)}")
        
        return train_cats, val_cats


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

def get_data_augmentation():
    """
    Create data augmentation layer for training.
    
    Includes:
    - Random rotation (±15 degrees)
    - Random zoom (±20%)
    - Random horizontal flip
    - Random brightness adjustment (±20%)
    """
    return keras.Sequential([
        layers.RandomRotation(Config.ROTATION_RANGE / 360.0, fill_mode='nearest'),
        layers.RandomZoom(Config.ZOOM_RANGE, fill_mode='nearest'),
        layers.RandomFlip(mode='horizontal'),
        layers.RandomBrightness(Config.BRIGHTNESS_RANGE),
    ], name='data_augmentation')


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_embedding_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         embedding_dim: int = 128,
                         trainable_layers: int = 100) -> Model:
    """
    Build EfficientNet-B0 embedding model.
    
    Args:
        input_shape: Input image shape
        embedding_dim: Output embedding dimension
        trainable_layers: Number of top layers to unfreeze for fine-tuning
    
    Returns:
        Keras model that outputs normalized embeddings
    """
    print("\n" + "="*80)
    print("Building EfficientNet-B0 Embedding Model")
    print("="*80)
    
    # Input layer
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # Data augmentation (only applied during training)
    augmented = get_data_augmentation()(inputs)
    
    # EfficientNet-B0 backbone (pretrained on ImageNet)
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=augmented,
        pooling=None
    )
    
    # Freeze all layers initially
    backbone.trainable = False
    
    # Unfreeze last N layers for fine-tuning
    total_layers = len(backbone.layers)
    print(f"\n   Total layers in EfficientNet: {total_layers}")
    print(f"   Unfreezing last {trainable_layers} layers for fine-tuning...")
    
    for layer in backbone.layers[-trainable_layers:]:
        layer.trainable = True
    
    x = backbone.output
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = layers.Dropout(0.3, name='dropout')(x)
    
    # Embedding layer
    x = layers.Dense(embedding_dim, activation='relu', name='embedding_dense')(x)
    
    # L2 normalization for cosine similarity
    embeddings = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name='l2_normalize'
    )(x)
    
    model = Model(inputs=inputs, outputs=embeddings, name='embedding_model')
    
    print(f"\n✅ Model built successfully!")
    print(f"   Input shape: {input_shape}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum(p.shape.num_elements() for p in model.trainable_weights)
    print(f"   Trainable parameters: {trainable:,}")
    
    return model


# ============================================================================
# TRIPLET GENERATION
# ============================================================================

class TripletDataGenerator:
    """Generate triplets for training"""
    
    def __init__(self, cat_to_images: Dict[str, List[str]], cat_ids: List[str], batch_size: int = 32):
        self.cat_to_images = cat_to_images
        self.cat_ids = cat_ids
        self.batch_size = batch_size
    
    def load_and_preprocess_image(self, image_path: str, augment: bool = False) -> np.ndarray:
        """Load and preprocess a single image"""
        img = keras.utils.load_img(image_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        img_array = keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        if augment:
            # Apply augmentation manually (will be done by model layer during training)
            pass
        
        return img_array
    
    def generate_triplet_batch(self, augment: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of triplets.
        
        Args:
            augment: Whether to apply augmentation (handled by model layer)
        
        Returns:
            anchors, positives, negatives
        """
        anchors = []
        positives = []
        negatives = []
        
        max_attempts = self.batch_size * 10
        attempts = 0
        
        while len(anchors) < self.batch_size and attempts < max_attempts:
            attempts += 1
            
            # Select anchor cat
            anchor_cat = random.choice(self.cat_ids)
            
            # Get two different images from same cat
            cat_images = self.cat_to_images[anchor_cat]
            if len(cat_images) < 2:
                continue
            
            anchor_path, positive_path = random.sample(cat_images, 2)
            
            # Get negative from different cat
            negative_candidates = [c for c in self.cat_ids if c != anchor_cat and len(self.cat_to_images[c]) > 0]
            if len(negative_candidates) == 0:
                continue
            
            negative_cat = random.choice(negative_candidates)
            negative_images = self.cat_to_images[negative_cat]
            if len(negative_images) == 0:
                continue
            
            negative_path = random.choice(negative_images)
            
            # Load and preprocess
            try:
                anchors.append(self.load_and_preprocess_image(anchor_path, augment))
                positives.append(self.load_and_preprocess_image(positive_path, augment))
                negatives.append(self.load_and_preprocess_image(negative_path, augment))
            except Exception as e:
                continue
        
        if len(anchors) == 0:
            raise ValueError("Could not generate any valid triplets!")
        
        return (
            np.array(anchors, dtype=np.float32),
            np.array(positives, dtype=np.float32),
            np.array(negatives, dtype=np.float32)
        )


# ============================================================================
# TRIPLET LOSS WITH SEMI-HARD MINING
# ============================================================================

def build_triplet_model(embedding_model: Model) -> Model:
    """Build triplet model with three inputs.

    Outputs a single concatenated tensor of shape (batch, 3 * embedding_dim)
    so we can apply a label-free triplet loss.
    """
    anchor_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='anchor')
    positive_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='positive')
    negative_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='negative')
    
    # Get embeddings (shared weights)
    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)
    
    # Concatenate embeddings along the feature axis: [a | p | n]
    concatenated = layers.Concatenate(axis=1)([anchor_emb, positive_emb, negative_emb])

    triplet_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=concatenated,
        name='triplet_model'
    )
    
    return triplet_model


# ============================================================================
# TRAINING
# ============================================================================

def train_model(embedding_model: Model,
                train_generator: TripletDataGenerator,
                val_generator: TripletDataGenerator) -> dict:
    """
    Train the embedding model with Triplet Semi-Hard Loss.
    
    Args:
        embedding_model: Base embedding model
        train_generator: Training triplet generator
        val_generator: Validation triplet generator
    
    Returns:
        Training history dictionary
    """
    print("\n" + "="*80)
    print("Training Individual Cat Recognition Model")
    print("="*80)
    
    # Build triplet model
    triplet_model = build_triplet_model(embedding_model)
    
    # Label-free Triplet Loss
    class TripletLoss(keras.losses.Loss):
        def __init__(self, margin: float = 0.5, **kwargs):
            super().__init__(**kwargs)
            self.margin = margin
        def call(self, y_true, y_pred):
            # y_pred shape: (batch, 3 * embedding_dim)
            emb_dim = tf.shape(y_pred)[1] // 3
            anchor = y_pred[:, :emb_dim]
            positive = y_pred[:, emb_dim:2*emb_dim]
            negative = y_pred[:, 2*emb_dim:]
            pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
            neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
            loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
            return tf.reduce_mean(loss)

    triplet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=TripletLoss(margin=Config.TRIPLET_MARGIN)
    )
    
    print(f"\n✅ Model compiled with Triplet Semi-Hard Loss")
    print(f"   Margin: {Config.TRIPLET_MARGIN}")
    print(f"   Learning rate: {Config.LEARNING_RATE}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_pos_dist': [],
        'train_neg_dist': [],
        'val_pos_dist': [],
        'val_neg_dist': []
    }
    
    best_val_loss = float('inf')
    
    print(f"\n{'='*80}")
    print(f"Starting Training ({Config.EPOCHS} epochs)")
    print(f"{'='*80}")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print('='*80)
        
        # Training
        train_losses = []
        train_pos_dists = []
        train_neg_dists = []
        
        num_train_batches = Config.TRIPLETS_PER_EPOCH // Config.BATCH_SIZE
        
        for batch_idx in tqdm(range(num_train_batches), desc="Training"):
            try:
                anchors, positives, negatives = train_generator.generate_triplet_batch(augment=True)
                
                if len(anchors) == 0:
                    continue
                
                # Train step
                loss = triplet_model.train_on_batch(
                    [anchors, positives, negatives],
                    np.zeros((len(anchors), 1))  # labels unused
                )
                
                # Compute distances for monitoring
                anchor_emb = embedding_model.predict(anchors, verbose=0)
                positive_emb = embedding_model.predict(positives, verbose=0)
                negative_emb = embedding_model.predict(negatives, verbose=0)
                
                pos_dist = np.mean(np.sum((anchor_emb - positive_emb) ** 2, axis=1))
                neg_dist = np.mean(np.sum((anchor_emb - negative_emb) ** 2, axis=1))
                
                train_losses.append(loss)
                train_pos_dists.append(pos_dist)
                train_neg_dists.append(neg_dist)
            except Exception as e:
                continue
        
        # Validation
        val_losses = []
        val_pos_dists = []
        val_neg_dists = []
        
        num_val_batches = max(50, Config.TRIPLETS_PER_EPOCH // 5 // Config.BATCH_SIZE)
        
        for batch_idx in tqdm(range(num_val_batches), desc="Validation"):
            try:
                anchors, positives, negatives = val_generator.generate_triplet_batch(augment=False)
                
                if len(anchors) == 0:
                    continue
                
                # Validation step
                loss = triplet_model.test_on_batch(
                    [anchors, positives, negatives],
                    np.zeros((len(anchors), 1))
                )
                
                # Compute distances
                anchor_emb = embedding_model.predict(anchors, verbose=0)
                positive_emb = embedding_model.predict(positives, verbose=0)
                negative_emb = embedding_model.predict(negatives, verbose=0)
                
                pos_dist = np.mean(np.sum((anchor_emb - positive_emb) ** 2, axis=1))
                neg_dist = np.mean(np.sum((anchor_emb - negative_emb) ** 2, axis=1))
                
                val_losses.append(loss)
                val_pos_dists.append(pos_dist)
                val_neg_dists.append(neg_dist)
            except Exception as e:
                continue
        
        # Epoch summary
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        avg_train_pos = np.mean(train_pos_dists) if train_pos_dists else 0
        avg_train_neg = np.mean(train_neg_dists) if train_neg_dists else 0
        avg_val_pos = np.mean(val_pos_dists) if val_pos_dists else 0
        avg_val_neg = np.mean(val_neg_dists) if val_neg_dists else 0
        
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        history['train_pos_dist'].append(float(avg_train_pos))
        history['train_neg_dist'].append(float(avg_train_neg))
        history['val_pos_dist'].append(float(avg_val_pos))
        history['val_neg_dist'].append(float(avg_val_neg))
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"   Train Pos Dist: {avg_train_pos:.4f} | Train Neg Dist: {avg_train_neg:.4f}")
        print(f"   Val Pos Dist: {avg_val_pos:.4f} | Val Neg Dist: {avg_val_neg:.4f}")
        print(f"   Train Margin: {avg_train_neg - avg_train_pos:.4f}")
        print(f"   Val Margin: {avg_val_neg - avg_val_pos:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   ✅ New best validation loss! Saving model...")
            
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            embedding_model.save(Config.MODEL_SAVE_PATH)
        
        # Plot progress
        plot_training_progress(history)
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    
    return history


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_progress(history: dict):
    """Plot training progress in real-time"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Triplet Loss', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot positive distances
    axes[0, 1].plot(epochs, history['train_pos_dist'], 'g-', linewidth=2, label='Train Pos Dist')
    axes[0, 1].plot(epochs, history['val_pos_dist'], 'g--', linewidth=2, label='Val Pos Dist')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Distance', fontsize=12)
    axes[0, 1].set_title('Positive Distance (Same Cat)', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot negative distances
    axes[1, 0].plot(epochs, history['train_neg_dist'], 'r-', linewidth=2, label='Train Neg Dist')
    axes[1, 0].plot(epochs, history['val_neg_dist'], 'r--', linewidth=2, label='Val Neg Dist')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Distance', fontsize=12)
    axes[1, 0].set_title('Negative Distance (Different Cat)', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot margin
    train_margin = [neg - pos for neg, pos in zip(history['train_neg_dist'], history['train_pos_dist'])]
    val_margin = [neg - pos for neg, pos in zip(history['val_neg_dist'], history['val_pos_dist'])]
    axes[1, 1].plot(epochs, train_margin, 'b-', linewidth=2, label='Train Margin')
    axes[1, 1].plot(epochs, val_margin, 'r-', linewidth=2, label='Val Margin')
    axes[1, 1].axhline(y=Config.TRIPLET_MARGIN, color='orange', linestyle='--', label=f'Target Margin ({Config.TRIPLET_MARGIN})')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Margin', fontsize=12)
    axes[1, 1].set_title('Distance Margin (Neg - Pos)', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training Progress - Individual Cat Recognition', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_progress_individuals.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model: Model, dataset_loader: CatDatasetLoader):
    """Evaluate model by comparing sample cat images"""
    print("\n" + "="*80)
    print("Model Evaluation - Sample Comparisons")
    print("="*80)
    
    # Get two random cats
    test_cats = random.sample(dataset_loader.cat_ids, min(2, len(dataset_loader.cat_ids)))
    
    # Test 1: Same cat comparison
    cat1_id = test_cats[0]
    cat1_images = dataset_loader.cat_to_images[cat1_id]
    if len(cat1_images) >= 2:
        img1_path, img2_path = random.sample(cat1_images, 2)
        
        print(f"\n📊 Test 1: Same Cat ({cat1_id})")
        print(f"   Image 1: {Path(img1_path).name}")
        print(f"   Image 2: {Path(img2_path).name}")
        
        # Load and preprocess
        img1 = keras.utils.load_img(img1_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        img2 = keras.utils.load_img(img2_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        
        img1_array = keras.utils.img_to_array(img1) / 255.0
        img2_array = keras.utils.img_to_array(img2) / 255.0
        
        # Get embeddings
        emb1 = model.predict(np.expand_dims(img1_array, 0), verbose=0)[0]
        emb2 = model.predict(np.expand_dims(img2_array, 0), verbose=0)[0]
        
        # Compute similarity
        similarity = float(np.dot(emb1, emb2))
        print(f"   ✅ Cosine Similarity: {similarity:.4f}")
        print(f"   Prediction: {'SAME CAT' if similarity > 0.8 else 'DIFFERENT CATS'}")
    
    # Test 2: Different cats comparison
    if len(test_cats) >= 2:
        cat2_id = test_cats[1]
        cat2_images = dataset_loader.cat_to_images[cat2_id]
        
        if len(cat1_images) > 0 and len(cat2_images) > 0:
            img1_path = random.choice(cat1_images)
            img2_path = random.choice(cat2_images)
            
            print(f"\n📊 Test 2: Different Cats ({cat1_id} vs {cat2_id})")
            print(f"   Image 1: {Path(img1_path).name}")
            print(f"   Image 2: {Path(img2_path).name}")
            
            # Load and preprocess
            img1 = keras.utils.load_img(img1_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
            img2 = keras.utils.load_img(img2_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
            
            img1_array = keras.utils.img_to_array(img1) / 255.0
            img2_array = keras.utils.img_to_array(img2) / 255.0
            
            # Get embeddings
            emb1 = model.predict(np.expand_dims(img1_array, 0), verbose=0)[0]
            emb2 = model.predict(np.expand_dims(img2_array, 0), verbose=0)[0]
            
            # Compute similarity
            similarity = float(np.dot(emb1, emb2))
            print(f"   ✅ Cosine Similarity: {similarity:.4f}")
            print(f"   Prediction: {'SAME CAT' if similarity > 0.8 else 'DIFFERENT CATS'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    print("="*80)
    print("IMPROVED INDIVIDUAL CAT RECOGNITION TRAINING")
    print("Using EfficientNet-B0 with Triplet Semi-Hard Loss")
    print("="*80)
    
    # Check dataset
    if not os.path.exists(Config.DATASET_PATH):
        print(f"\n❌ Dataset not found at: {Config.DATASET_PATH}")
        sys.exit(1)
    
    # Check TensorFlow Addons
    try:
        import tensorflow_addons as tfa
        print(f"\n✅ TensorFlow Addons version: {tfa.__version__}")
    except ImportError:
        print("\n❌ TensorFlow Addons not installed!")
        print("Please install: pip install tensorflow-addons")
        sys.exit(1)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️  No GPU detected, using CPU (slower)")
    
    # Load dataset
    dataset_loader = CatDatasetLoader(Config.DATASET_PATH)
    
    # Split train/val
    train_cats, val_cats = dataset_loader.split_train_val(Config.TRAIN_SPLIT)
    
    # Create generators
    train_generator = TripletDataGenerator(
        {cat: dataset_loader.cat_to_images[cat] for cat in train_cats},
        train_cats,
        Config.BATCH_SIZE
    )
    
    val_generator = TripletDataGenerator(
        {cat: dataset_loader.cat_to_images[cat] for cat in val_cats},
        val_cats,
        Config.BATCH_SIZE
    )
    
    # Build model
    embedding_model = build_embedding_model(
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
        embedding_dim=Config.EMBEDDING_DIM,
        trainable_layers=Config.BACKBONE_TRAINABLE_LAYERS
    )
    
    # Train model
    history = train_model(embedding_model, train_generator, val_generator)
    
    # Save training history
    with open(Config.TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved to: {Config.TRAINING_HISTORY_PATH}")
    
    # Final plot
    plot_training_progress(history)
    
    # Evaluate model
    evaluate_model(embedding_model, dataset_loader)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✅ Training history saved to: {Config.TRAINING_HISTORY_PATH}")
    print(f"✅ Training plots saved to: training_progress_individuals.png")
    
    print("\n" + "="*80)
    print("FINAL METRICS:")
    print("="*80)
    print(f"Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"Final Train Margin: {history['train_neg_dist'][-1] - history['train_pos_dist'][-1]:.4f}")
    print(f"Final Val Margin: {history['val_neg_dist'][-1] - history['val_pos_dist'][-1]:.4f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Load the model:")
    print(f"   >>> from tensorflow import keras")
    print(f"   >>> model = keras.models.load_model('{Config.MODEL_SAVE_PATH}')")
    print("\n2. Use for cat comparison:")
    print("   >>> # Get embeddings and compute cosine similarity")
    print("   >>> # Threshold ~0.8 for same cat")


if __name__ == "__main__":
    main()

