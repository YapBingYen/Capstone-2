"""
Individual Cat Recognition Model Trainer
=========================================
Train an EfficientNet-B0 based Siamese/Triplet-loss model to identify
individual cats (not just breeds).

Dataset:
    D:\Cursor AI projects\Capstone2.1\dataset_individuals_cropped\cat_individuals_dataset

Each folder (0001, 0002, 0003, ...) represents a unique cat identity.
The model learns embeddings where same-cat images are close together.

Author: AI Assistant
Date: November 2025
"""

import os
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
    MODEL_SAVE_PATH = "models/efficientnetb0_catid_v1.keras"
    EMBEDDINGS_SAVE_PATH = "embeddings/individual_cat_embeddings.npy"
    METADATA_SAVE_PATH = "embeddings/individual_cat_metadata.npy"
    
    # Model parameters
    IMG_SIZE = 224
    EMBEDDING_DIM = 128
    BACKBONE = 'EfficientNetB0'
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.1
    
    # Triplet loss parameters
    TRIPLET_MARGIN = 0.3
    
    # Early stopping
    PATIENCE = 2


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_embedding_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         embedding_dim: int = 128) -> keras.Model:
    """
    Build EfficientNet-B0 embedding model.
    
    Args:
        input_shape: Input image shape
        embedding_dim: Output embedding dimension
    
    Returns:
        Keras model that outputs normalized embeddings
    """
    print("\n" + "="*80)
    print("Building EfficientNet-B0 Embedding Model")
    print("="*80)
    
    # Input
    inputs = keras.Input(shape=input_shape, name='image_input')
    
    # EfficientNet-B0 backbone (pretrained on ImageNet)
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None
    )
    
    # Fine-tune top layers
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    
    x = backbone.output
    
    # Global pooling
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    # Dropout for regularization
    x = keras.layers.Dropout(0.3)(x)
    
    # Embedding layer
    x = keras.layers.Dense(embedding_dim, activation=None, name='embedding')(x)
    
    # L2 normalization
    embeddings = keras.layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name='l2_normalize'
    )(x)
    
    model = keras.Model(inputs=inputs, outputs=embeddings, name='embedding_model')
    
    print(f"\n✅ Model built successfully!")
    print(f"   Input shape: {input_shape}")
    print(f"   Embedding dimension: {embedding_dim}")
    print(f"   Total parameters: {model.count_params():,}")
    trainable = sum(p.shape.num_elements() for p in model.trainable_weights)
    print(f"   Trainable parameters: {trainable:,}")
    
    return model


# ============================================================================
# TRIPLET LOSS
# ============================================================================

class TripletLoss(keras.losses.Loss):
    """
    Triplet loss implementation.
    
    L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
    """
    
    def __init__(self, margin: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """Compute triplet loss"""
        # y_pred shape: (batch_size, embedding_dim * 3)
        # Split into anchor, positive, negative
        embedding_dim = y_pred.shape[-1] // 3
        
        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim:2*embedding_dim]
        negative = y_pred[:, 2*embedding_dim:]
        
        # Compute squared distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Triplet loss with margin
        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
        
        return tf.reduce_mean(loss)


# ============================================================================
# DATA LOADING
# ============================================================================

class TripletDataGenerator:
    """Generate triplets for training"""
    
    def __init__(self, dataset_path: str, batch_size: int = 16):
        self.dataset_path = Path(dataset_path)
        self.batch_size = batch_size
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
                else:
                    print(f"⚠️  Skipping {cat_id}: only {len(image_paths)} image(s) (need at least 2)")
        
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
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        img = keras.utils.load_img(image_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        img_array = keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        return img_array
    
    def generate_triplet_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of triplets.
        
        Returns:
            anchors, positives, negatives
        """
        anchors = []
        positives = []
        negatives = []
        
        max_attempts = self.batch_size * 10  # Prevent infinite loops
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
            
            # Get negative from different cat (ensure it has images)
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
                anchors.append(self.load_and_preprocess_image(anchor_path))
                positives.append(self.load_and_preprocess_image(positive_path))
                negatives.append(self.load_and_preprocess_image(negative_path))
            except Exception as e:
                # Skip if image loading fails
                print(f"⚠️  Failed to load image: {e}")
                continue
        
        # If we couldn't generate enough triplets, return what we have
        if len(anchors) == 0:
            raise ValueError("Could not generate any valid triplets! Check dataset.")
        
        return (
            np.array(anchors, dtype=np.float32),
            np.array(positives, dtype=np.float32),
            np.array(negatives, dtype=np.float32)
        )


# ============================================================================
# TRAINING
# ============================================================================

def build_triplet_model(embedding_model: keras.Model) -> keras.Model:
    """Build triplet model with three inputs"""
    anchor_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='anchor')
    positive_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='positive')
    negative_input = keras.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='negative')
    
    # Get embeddings (shared weights)
    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)
    
    # Concatenate for loss computation
    outputs = keras.layers.Concatenate()([anchor_emb, positive_emb, negative_emb])
    
    triplet_model = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=outputs,
        name='triplet_model'
    )
    
    return triplet_model


def train_model(embedding_model: keras.Model, 
                triplet_generator: TripletDataGenerator) -> dict:
    """
    Train the embedding model with triplet loss.
    
    Args:
        embedding_model: Base embedding model
        triplet_generator: Triplet data generator
    
    Returns:
        Training history dictionary
    """
    print("\n" + "="*80)
    print("Training Individual Cat Recognition Model")
    print("="*80)
    
    # Build triplet model
    triplet_model = build_triplet_model(embedding_model)
    
    # Compile
    triplet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        loss=TripletLoss(margin=Config.TRIPLET_MARGIN)
    )
    
    # Training history
    history = {
        'loss': [],
        'positive_dist': [],
        'negative_dist': [],
        'margin': []
    }
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Calculate batches per epoch
    total_images = sum(len(imgs) for imgs in triplet_generator.cat_to_images.values())
    batches_per_epoch = max(50, total_images // Config.BATCH_SIZE)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Batch size: {Config.BATCH_SIZE}")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Learning rate: {Config.LEARNING_RATE}")
    print(f"  Triplet margin: {Config.TRIPLET_MARGIN}")
    print(f"  Early stopping patience: {Config.PATIENCE}")
    
    # Training loop
    for epoch in range(Config.EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{Config.EPOCHS}")
        print('='*80)
        
        epoch_losses = []
        epoch_pos_dists = []
        epoch_neg_dists = []
        
        # Train for one epoch
        for batch_idx in tqdm(range(batches_per_epoch), desc=f"Epoch {epoch+1}"):
            try:
                # Generate triplet batch
                anchors, positives, negatives = triplet_generator.generate_triplet_batch()
                
                if len(anchors) == 0:
                    continue
                
                # Train step
                loss = triplet_model.train_on_batch(
                    [anchors, positives, negatives],
                    np.zeros((len(anchors), Config.EMBEDDING_DIM * 3))
                )
            except (ValueError, IndexError) as e:
                # Skip batch if triplet generation fails
                print(f"⚠️  Skipping batch due to error: {e}")
                continue
            
            # Compute distances for monitoring
            anchor_emb = embedding_model.predict(anchors, verbose=0)
            positive_emb = embedding_model.predict(positives, verbose=0)
            negative_emb = embedding_model.predict(negatives, verbose=0)
            
            pos_dist = np.mean(np.sum((anchor_emb - positive_emb) ** 2, axis=1))
            neg_dist = np.mean(np.sum((anchor_emb - negative_emb) ** 2, axis=1))
            
            epoch_losses.append(loss)
            epoch_pos_dists.append(pos_dist)
            epoch_neg_dists.append(neg_dist)
        
        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        avg_pos_dist = np.mean(epoch_pos_dists)
        avg_neg_dist = np.mean(epoch_neg_dists)
        margin = avg_neg_dist - avg_pos_dist
        
        history['loss'].append(avg_loss)
        history['positive_dist'].append(avg_pos_dist)
        history['negative_dist'].append(avg_neg_dist)
        history['margin'].append(margin)
        
        print(f"\n📊 Epoch {epoch + 1} Summary:")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Positive Distance: {avg_pos_dist:.4f}")
        print(f"   Negative Distance: {avg_neg_dist:.4f}")
        print(f"   Margin: {margin:.4f}")
        
        # Early stopping check
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f"   ✅ New best loss! Saving model...")
            
            # Save model
            os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
            embedding_model.save(Config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            print(f"   No improvement. Patience: {patience_counter}/{Config.PATIENCE}")
            
            if patience_counter >= Config.PATIENCE:
                print(f"\n⚠️  Early stopping triggered after {epoch + 1} epochs")
                break
    
    print("\n" + "="*80)
    print("✅ Training Complete!")
    print("="*80)
    
    return history


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_all_embeddings(model: keras.Model, 
                          triplet_generator: TripletDataGenerator) -> Tuple[np.ndarray, List]:
    """
    Extract embeddings for all images in the dataset.
    
    Args:
        model: Trained embedding model
        triplet_generator: Data generator with loaded dataset
    
    Returns:
        embeddings, metadata (list of dicts with cat_id and image_path)
    """
    print("\n" + "="*80)
    print("Extracting Embeddings for All Images")
    print("="*80)
    
    all_embeddings = []
    all_metadata = []
    
    for cat_id in tqdm(triplet_generator.cat_ids, desc="Processing cats"):
        for img_path in triplet_generator.cat_to_images[cat_id]:
            # Load and preprocess
            img = triplet_generator.load_and_preprocess_image(img_path)
            img_batch = np.expand_dims(img, axis=0)
            
            # Get embedding
            embedding = model.predict(img_batch, verbose=0)[0]
            
            all_embeddings.append(embedding)
            all_metadata.append({
                'cat_id': cat_id,
                'image_path': img_path
            })
    
    embeddings_array = np.array(all_embeddings)
    
    print(f"\n✅ Extracted {len(all_embeddings)} embeddings")
    print(f"   Shape: {embeddings_array.shape}")
    
    return embeddings_array, all_metadata


def save_embeddings(embeddings: np.ndarray, metadata: List):
    """Save embeddings and metadata"""
    # Save embeddings
    os.makedirs(os.path.dirname(Config.EMBEDDINGS_SAVE_PATH), exist_ok=True)
    np.save(Config.EMBEDDINGS_SAVE_PATH, embeddings)
    print(f"✅ Embeddings saved to: {Config.EMBEDDINGS_SAVE_PATH}")
    
    # Save metadata
    np.save(Config.METADATA_SAVE_PATH, metadata)
    print(f"✅ Metadata saved to: {Config.METADATA_SAVE_PATH}")


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_random_triplets(model: keras.Model, 
                             triplet_generator: TripletDataGenerator,
                             num_triplets: int = 5):
    """Visualize random triplets with similarity scores"""
    print("\n" + "="*80)
    print(f"Visualizing {num_triplets} Random Triplets")
    print("="*80)
    
    fig, axes = plt.subplots(num_triplets, 3, figsize=(12, num_triplets * 4))
    
    if num_triplets == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_triplets):
        # Generate single triplet
        anchors, positives, negatives = triplet_generator.generate_triplet_batch()
        
        if len(anchors) == 0:
            continue
        
        # Get first triplet from batch
        anchor = anchors[0]
        positive = positives[0]
        negative = negatives[0]
        
        # Get embeddings
        anchor_emb = model.predict(np.expand_dims(anchor, 0), verbose=0)[0]
        positive_emb = model.predict(np.expand_dims(positive, 0), verbose=0)[0]
        negative_emb = model.predict(np.expand_dims(negative, 0), verbose=0)[0]
        
        # Compute similarities
        pos_sim = float(np.dot(anchor_emb, positive_emb))
        neg_sim = float(np.dot(anchor_emb, negative_emb))
        
        # Display images
        axes[i, 0].imshow(anchor)
        axes[i, 0].set_title("Anchor", fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(positive)
        axes[i, 1].set_title(f"Positive\nSim: {pos_sim:.3f}", fontsize=10, color='green')
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(negative)
        axes[i, 2].set_title(f"Negative\nSim: {neg_sim:.3f}", fontsize=10, color='red')
        axes[i, 2].axis('off')
    
    plt.suptitle("Sample Triplets with Similarity Scores", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('triplet_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to: triplet_visualization.png")
    plt.show()


def plot_training_history(history: dict):
    """Plot training metrics"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Triplet Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot distances
    axes[1].plot(epochs, history['positive_dist'], 'g-', linewidth=2, label='Positive Distance')
    axes[1].plot(epochs, history['negative_dist'], 'r-', linewidth=2, label='Negative Distance')
    axes[1].axhline(y=Config.TRIPLET_MARGIN, color='orange', linestyle='--', label=f'Margin ({Config.TRIPLET_MARGIN})')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Distance', fontsize=12)
    axes[1].set_title('Embedding Distances', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Individual Cat Recognition Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_history_individuals.png', dpi=150, bbox_inches='tight')
    print("✅ Training history plot saved to: training_history_individuals.png")
    plt.show()


# ============================================================================
# TESTING
# ============================================================================

def test_similarity(model: keras.Model, img_path1: str, img_path2: str):
    """Test similarity between two images"""
    print("\n" + "="*80)
    print("Testing Image Similarity")
    print("="*80)
    
    # Load images
    img1 = keras.utils.load_img(img_path1, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
    img2 = keras.utils.load_img(img_path2, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
    
    img1_array = keras.utils.img_to_array(img1) / 255.0
    img2_array = keras.utils.img_to_array(img2) / 255.0
    
    # Get embeddings
    emb1 = model.predict(np.expand_dims(img1_array, 0), verbose=0)[0]
    emb2 = model.predict(np.expand_dims(img2_array, 0), verbose=0)[0]
    
    # Compute cosine similarity
    similarity = float(np.dot(emb1, emb2))
    
    print(f"\nImage 1: {img_path1}")
    print(f"Image 2: {img_path2}")
    print(f"\nCosine Similarity: {similarity:.4f}")
    print(f"Same cat: {'✅ YES' if similarity > 0.8 else '❌ NO'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training pipeline"""
    print("="*80)
    print("INDIVIDUAL CAT RECOGNITION MODEL TRAINER")
    print("Using EfficientNet-B0 with Triplet Loss")
    print("="*80)
    
    # Check dataset
    if not os.path.exists(Config.DATASET_PATH):
        print(f"\n❌ Dataset not found at: {Config.DATASET_PATH}")
        print("Please ensure the cropped dataset exists.")
        sys.exit(1)
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️  No GPU detected, using CPU (slower)")
    
    # Load data
    triplet_generator = TripletDataGenerator(Config.DATASET_PATH, Config.BATCH_SIZE)
    
    # Build model
    embedding_model = build_embedding_model(
        input_shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3),
        embedding_dim=Config.EMBEDDING_DIM
    )
    
    # Train model
    history = train_model(embedding_model, triplet_generator)
    
    # Plot training history
    plot_training_history(history)
    
    # Extract embeddings
    embeddings, metadata = extract_all_embeddings(embedding_model, triplet_generator)
    save_embeddings(embeddings, metadata)
    
    # Visualize triplets
    visualize_random_triplets(embedding_model, triplet_generator, num_triplets=5)
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✅ Embeddings saved to: {Config.EMBEDDINGS_SAVE_PATH}")
    print(f"✅ Metadata saved to: {Config.METADATA_SAVE_PATH}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Test the model:")
    print("   >>> from tensorflow import keras")
    print("   >>> model = keras.models.load_model('models/efficientnetb0_catid_v1.keras')")
    print("   >>> # Use model to compare cat images")
    print("\n2. The model outputs 128-dim normalized embeddings")
    print("3. Use cosine similarity to compare cats (threshold ~0.8)")


if __name__ == "__main__":
    main()

