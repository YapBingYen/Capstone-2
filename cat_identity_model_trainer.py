"""
Cat Identity Recognition Model using Triplet Loss
==================================================
This script trains a neural network to produce unique embeddings for individual cat recognition.
Uses EfficientNet-B0 backbone with triplet loss for learning discriminative features.

Author: AI Assistant
Date: October 28, 2025
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow import keras
load_img = keras.utils.load_img
img_to_array = keras.utils.img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, List, Dict
import random
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration parameters for the model training"""
    # Paths
    DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
    MODEL_SAVE_PATH = "cat_identity_model.h5"
    EMBEDDINGS_SAVE_PATH = "cat_embeddings.npz"
    METADATA_SAVE_PATH = "cat_metadata.csv"
    TRAINING_HISTORY_PATH = "training_history.json"
    
    # Image parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    IMG_CHANNELS = 3
    
    # Model parameters
    EMBEDDING_DIM = 128
    DROPOUT_RATE = 0.3
    BACKBONE_TRAINABLE_LAYERS = 20  # Fine-tune top 20 layers of EfficientNet
    
    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    TRIPLET_MARGIN = 0.5  # Margin for triplet loss
    
    # Triplet generation
    TRIPLETS_PER_EPOCH = 5000
    
    # Inference threshold
    SIMILARITY_THRESHOLD = 0.8  # Cosine similarity threshold for same cat
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


# Set random seeds for reproducibility
np.random.seed(Config.RANDOM_SEED)
tf.random.set_seed(Config.RANDOM_SEED)
random.seed(Config.RANDOM_SEED)


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

class CatDatasetLoader:
    """Handles loading and organizing cat images from the dataset"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.breed_to_images = {}
        self.image_paths = []
        self.labels = []
        self.breed_to_idx = {}
        
    def load_dataset(self) -> Tuple[List[str], List[int], Dict[str, int]]:
        """
        Load all images from the dataset directory.
        
        Returns:
            image_paths: List of image file paths
            labels: List of breed indices (used for triplet generation)
            breed_to_idx: Mapping from breed name to index
        """
        print("Loading dataset...")
        
        # Get all breed folders
        breed_folders = [f for f in self.dataset_path.iterdir() if f.is_dir()]
        breed_folders = sorted(breed_folders)
        
        # Create breed to index mapping
        self.breed_to_idx = {breed.name: idx for idx, breed in enumerate(breed_folders)}
        
        # Load all images
        for breed_folder in breed_folders:
            breed_name = breed_folder.name
            breed_idx = self.breed_to_idx[breed_name]
            
            # Get all image files in this breed folder
            image_files = list(breed_folder.glob("*.jpg")) + \
                         list(breed_folder.glob("*.jpeg")) + \
                         list(breed_folder.glob("*.png"))
            
            self.breed_to_images[breed_idx] = []
            
            for img_path in image_files:
                self.image_paths.append(str(img_path))
                self.labels.append(breed_idx)
                self.breed_to_images[breed_idx].append(str(img_path))
        
        print(f"✅ Loaded {len(self.image_paths)} images from {len(breed_folders)} breeds")
        print(f"Breeds: {list(self.breed_to_idx.keys())}")
        
        return self.image_paths, self.labels, self.breed_to_idx


def preprocess_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load and preprocess a single image for EfficientNet input.
    
    Args:
        image_path: Path to the image file
        target_size: Target image size (height, width)
    
    Returns:
        Preprocessed image array
    """
    try:
        # Load image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        
        # Normalize to [0, 1] range (EfficientNet expects this)
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        # Return a black image as fallback
        return np.zeros((target_size[0], target_size[1], 3))


# ============================================================================
# TRIPLET GENERATION
# ============================================================================

class TripletGenerator:
    """Generates triplets (anchor, positive, negative) for triplet loss training"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 breed_to_images: Dict[int, List[str]]):
        self.image_paths = np.array(image_paths)
        self.labels = np.array(labels)
        self.breed_to_images = breed_to_images
        self.unique_labels = list(set(labels))
        
    def generate_triplets(self, num_triplets: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate triplets for training.
        
        Strategy:
        - Anchor: Random image from a breed
        - Positive: Different image from the SAME breed
        - Negative: Random image from a DIFFERENT breed
        
        Args:
            num_triplets: Number of triplets to generate
        
        Returns:
            anchors, positives, negatives: Arrays of preprocessed images
        """
        anchors = []
        positives = []
        negatives = []
        
        print(f"Generating {num_triplets} triplets...")
        
        for _ in range(num_triplets):
            # Select a random breed for anchor
            anchor_breed = random.choice(self.unique_labels)
            
            # Get two different images from the same breed (anchor and positive)
            breed_images = self.breed_to_images[anchor_breed]
            
            if len(breed_images) < 2:
                # Skip if breed has less than 2 images
                continue
            
            anchor_img_path, positive_img_path = random.sample(breed_images, 2)
            
            # Select a negative image from a different breed
            negative_breed = random.choice([b for b in self.unique_labels if b != anchor_breed])
            negative_img_path = random.choice(self.breed_to_images[negative_breed])
            
            # Preprocess images
            anchor_img = preprocess_image(anchor_img_path)
            positive_img = preprocess_image(positive_img_path)
            negative_img = preprocess_image(negative_img_path)
            
            anchors.append(anchor_img)
            positives.append(positive_img)
            negatives.append(negative_img)
        
        return (np.array(anchors, dtype=np.float32),
                np.array(positives, dtype=np.float32),
                np.array(negatives, dtype=np.float32))
    
    def generate_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a single batch of triplets (float32)"""
        return self.generate_triplets(batch_size)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

def build_embedding_model(input_shape: Tuple[int, int, int] = (224, 224, 3),
                         embedding_dim: int = 128,
                         dropout_rate: float = 0.3,
                         trainable_backbone_layers: int = 20) -> Model:
    """
    Build the embedding model using EfficientNet-B0 backbone.
    
    Architecture:
    - EfficientNet-B0 (pretrained on ImageNet, without top)
    - GlobalAveragePooling2D
    - Dropout(0.3)
    - Dense(128, activation='relu')
    - L2 Normalization
    
    Args:
        input_shape: Input image shape (height, width, channels)
        embedding_dim: Dimension of the output embedding
        dropout_rate: Dropout rate for regularization
        trainable_backbone_layers: Number of top layers to fine-tune
    
    Returns:
        Keras Model that outputs normalized embeddings
    """
    # Input layer
    input_layer = layers.Input(shape=input_shape, name='image_input')
    
    # Load EfficientNet-B0 backbone (pretrained on ImageNet)
    backbone = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer,
        pooling=None
    )
    
    # Freeze the backbone initially, then unfreeze top layers
    backbone.trainable = True
    total_layers = len(backbone.layers)
    
    # Freeze all layers except the top N layers
    for layer in backbone.layers[:-trainable_backbone_layers]:
        layer.trainable = False
    
    print(f"Backbone: {total_layers} total layers, {trainable_backbone_layers} trainable")
    
    # Get backbone output
    x = backbone.output
    
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = layers.Dropout(dropout_rate, name='dropout')(x)
    
    # Dense embedding layer
    x = layers.Dense(embedding_dim, activation='relu', name='embedding_dense')(x)
    
    # L2 normalization for cosine similarity computation
    embeddings = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1), 
                               name='l2_normalization')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=embeddings, name='cat_embedding_model')
    
    return model


def build_triplet_model(embedding_model: Model) -> Model:
    """
    Build the triplet model that takes anchor, positive, and negative inputs.
    
    Args:
        embedding_model: The base embedding model
    
    Returns:
        Triplet model with three inputs and three embedding outputs
    """
    # Define three inputs
    anchor_input = layers.Input(shape=(224, 224, 3), name='anchor_input')
    positive_input = layers.Input(shape=(224, 224, 3), name='positive_input')
    negative_input = layers.Input(shape=(224, 224, 3), name='negative_input')
    
    # Get embeddings for all three inputs (shared weights)
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)
    
    # Create triplet model
    triplet_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=[anchor_embedding, positive_embedding, negative_embedding],
        name='triplet_model'
    )
    
    return triplet_model


# ============================================================================
# TRIPLET LOSS
# ============================================================================

class TripletLoss(keras.losses.Loss):
    """
    Triplet Loss implementation.
    
    Formula: L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
    
    Where:
    - f(a): Anchor embedding
    - f(p): Positive embedding (same identity)
    - f(n): Negative embedding (different identity)
    - margin: Minimum distance between positive and negative pairs
    """
    
    def __init__(self, margin: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """
        Compute triplet loss.
        
        Args:
            y_true: Not used (required by Keras API)
            y_pred: Concatenated embeddings [anchor, positive, negative]
        
        Returns:
            Triplet loss value
        """
        # Split embeddings
        embedding_dim = y_pred.shape[-1] // 3
        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim:2*embedding_dim]
        negative = y_pred[:, 2*embedding_dim:]
        
        # Compute distances
        positive_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        negative_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Compute triplet loss
        loss = tf.maximum(0.0, positive_dist - negative_dist + self.margin)
        
        return tf.reduce_mean(loss)


# ============================================================================
# CUSTOM TRAINING LOOP WITH TRIPLET GENERATION
# ============================================================================

class TripletTrainer:
    """Custom trainer for triplet loss model"""
    
    def __init__(self, embedding_model: Model, triplet_model: Model,
                 triplet_generator: TripletGenerator, config: Config):
        self.embedding_model = embedding_model
        self.triplet_model = triplet_model
        self.triplet_generator = triplet_generator
        self.config = config
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
        
        # Loss
        self.triplet_loss = TripletLoss(margin=config.TRIPLET_MARGIN)
        
        # Metrics
        self.train_loss_metric = keras.metrics.Mean(name='train_loss')
        
        # History
        self.history = {
            'loss': [],
            'positive_dist': [],
            'negative_dist': []
        }
    
    @tf.function
    def train_step(self, anchors, positives, negatives):
        """Single training step"""
        with tf.GradientTape() as tape:
            # Get embeddings
            anchor_embeddings = self.embedding_model(anchors, training=True)
            positive_embeddings = self.embedding_model(positives, training=True)
            negative_embeddings = self.embedding_model(negatives, training=True)
            
            # Concatenate for loss computation
            embeddings = tf.concat([anchor_embeddings, positive_embeddings, negative_embeddings], axis=1)
            
            # Compute loss
            loss = self.triplet_loss(None, embeddings)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.embedding_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.embedding_model.trainable_variables))
        
        # Update metrics
        self.train_loss_metric.update_state(loss)
        
        # Compute distances for monitoring
        positive_dist = tf.reduce_mean(tf.reduce_sum(tf.square(anchor_embeddings - positive_embeddings), axis=-1))
        negative_dist = tf.reduce_mean(tf.reduce_sum(tf.square(anchor_embeddings - negative_embeddings), axis=-1))
        
        return loss, positive_dist, negative_dist
    
    def train(self, epochs: int, triplets_per_epoch: int, batch_size: int):
        """
        Train the model for specified number of epochs.
        
        Args:
            epochs: Number of training epochs
            triplets_per_epoch: Number of triplets to generate per epoch
            batch_size: Batch size for training
        """
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80)
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 80)
            
            # Reset metrics
            self.train_loss_metric.reset_states()
            
            # Number of batches (generate on-the-fly to avoid large memory use)
            num_batches = max(1, triplets_per_epoch // batch_size)
            
            epoch_losses = []
            epoch_pos_dists = []
            epoch_neg_dists = []
            
            # Training loop (generate each batch lazily)
            for batch_idx in tqdm(range(num_batches), desc="Training"):
                batch_anchors, batch_positives, batch_negatives = self.triplet_generator.generate_batch(batch_size)
                # In rare cases a breed has <2 images and a batch returns empty; skip
                if len(batch_anchors) == 0:
                    continue
                
                loss, pos_dist, neg_dist = self.train_step(
                    batch_anchors, batch_positives, batch_negatives
                )
                
                epoch_losses.append(float(loss))
                epoch_pos_dists.append(float(pos_dist))
                epoch_neg_dists.append(float(neg_dist))
            
            # Compute epoch metrics
            avg_loss = np.mean(epoch_losses)
            avg_pos_dist = np.mean(epoch_pos_dists)
            avg_neg_dist = np.mean(epoch_neg_dists)
            
            # Store history
            self.history['loss'].append(avg_loss)
            self.history['positive_dist'].append(avg_pos_dist)
            self.history['negative_dist'].append(avg_neg_dist)
            
            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Avg Positive Distance: {avg_pos_dist:.4f}")
            print(f"  Avg Negative Distance: {avg_neg_dist:.4f}")
            print(f"  Distance Margin: {avg_neg_dist - avg_pos_dist:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"  ✅ New best loss! Saving model...")
                self.embedding_model.save(self.config.MODEL_SAVE_PATH)
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        
        return self.history


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(model: Model, image_paths: List[str], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
    """
    Extract embeddings for all images in the dataset.
    
    Args:
        model: Trained embedding model
        image_paths: List of image paths
        batch_size: Batch size for processing
    
    Returns:
        embeddings: Array of embeddings (N, embedding_dim)
        image_paths: List of image paths (same order)
    """
    print("\nExtracting embeddings for all images...")
    
    embeddings = []
    valid_paths = []
    
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = np.array([preprocess_image(path) for path in batch_paths])
        
        # Get embeddings
        batch_embeddings = model.predict(batch_images, verbose=0)
        
        embeddings.append(batch_embeddings)
        valid_paths.extend(batch_paths)
    
    embeddings = np.vstack(embeddings)
    
    print(f"✅ Extracted embeddings for {len(valid_paths)} images")
    print(f"Embedding shape: {embeddings.shape}")
    
    return embeddings, valid_paths


def save_embeddings(embeddings: np.ndarray, image_paths: List[str], 
                   labels: List[int], breed_to_idx: Dict[str, int],
                   embeddings_path: str, metadata_path: str):
    """
    Save embeddings and metadata to disk.
    
    Args:
        embeddings: Array of embeddings
        image_paths: List of image paths
        labels: List of breed labels
        breed_to_idx: Breed to index mapping
        embeddings_path: Path to save embeddings
        metadata_path: Path to save metadata CSV
    """
    # Save embeddings as NPZ file
    np.savez_compressed(embeddings_path, 
                       embeddings=embeddings,
                       image_paths=np.array(image_paths))
    
    print(f"✅ Saved embeddings to {embeddings_path}")
    
    # Create metadata DataFrame
    idx_to_breed = {v: k for k, v in breed_to_idx.items()}
    
    metadata_df = pd.DataFrame({
        'image_path': image_paths,
        'breed_label': labels,
        'breed_name': [idx_to_breed[label] for label in labels]
    })
    
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✅ Saved metadata to {metadata_path}")


# ============================================================================
# INFERENCE AND COMPARISON
# ============================================================================

def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score (0 to 1)
    """
    # Embeddings are already L2-normalized, so dot product = cosine similarity
    similarity = np.dot(embedding1, embedding2)
    return float(similarity)


def compare_images(model: Model, img_path1: str, img_path2: str, 
                  threshold: float = 0.8, visualize: bool = True) -> Dict:
    """
    Compare two cat images and determine if they are the same cat.
    
    Args:
        model: Trained embedding model
        img_path1: Path to first image
        img_path2: Path to second image
        threshold: Similarity threshold for same cat (default: 0.8)
        visualize: Whether to display the images
    
    Returns:
        Dictionary with similarity score and prediction
    """
    # Load and preprocess images
    img1 = preprocess_image(img_path1)
    img2 = preprocess_image(img_path2)
    
    # Add batch dimension
    img1_batch = np.expand_dims(img1, axis=0)
    img2_batch = np.expand_dims(img2, axis=0)
    
    # Get embeddings
    embedding1 = model.predict(img1_batch, verbose=0)[0]
    embedding2 = model.predict(img2_batch, verbose=0)[0]
    
    # Compute similarity
    similarity = compute_cosine_similarity(embedding1, embedding2)
    
    # Predict if same cat
    is_same_cat = similarity >= threshold
    
    result = {
        'similarity': similarity,
        'is_same_cat': is_same_cat,
        'confidence': 'High' if abs(similarity - threshold) > 0.1 else 'Low'
    }
    
    # Print results
    print("\n" + "="*80)
    print("IMAGE COMPARISON RESULTS")
    print("="*80)
    print(f"Image 1: {img_path1}")
    print(f"Image 2: {img_path2}")
    print(f"\nCosine Similarity: {similarity:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Prediction: {'✅ SAME CAT' if is_same_cat else '❌ DIFFERENT CATS'}")
    print(f"Confidence: {result['confidence']}")
    print("="*80)
    
    # Visualize images if requested
    if visualize:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Load images for display
        img1_display = load_img(img_path1)
        img2_display = load_img(img_path2)
        
        # Extract breed from path (assumes path contains breed folder name)
        from pathlib import Path
        breed1 = Path(img_path1).parent.name if Path(img_path1).parent.name != 'cats' else 'Unknown'
        breed2 = Path(img_path2).parent.name if Path(img_path2).parent.name != 'cats' else 'Unknown'
        
        # Format breed names (replace underscores with spaces)
        breed1_display = breed1.replace('_', ' ')
        breed2_display = breed2.replace('_', ' ')
        
        axes[0].imshow(img1_display)
        axes[0].set_title(f"Image 1\nBreed: {breed1_display}", fontsize=11)
        axes[0].axis('off')
        
        axes[1].imshow(img2_display)
        axes[1].set_title(f"Image 2\nBreed: {breed2_display}", fontsize=11)
        axes[1].axis('off')
        
        plt.suptitle(f"Similarity: {similarity:.4f} | {'SAME CAT' if is_same_cat else 'DIFFERENT CATS'}", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('comparison_result.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved to comparison_result.png")
        plt.show()
    
    return result


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history: Dict, save_path: str = 'training_history.png'):
    """
    Plot training history metrics.
    
    Args:
        history: Dictionary with training metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot distances
    axes[1].plot(epochs, history['positive_dist'], 'g-', linewidth=2, label='Positive Distance')
    axes[1].plot(epochs, history['negative_dist'], 'r-', linewidth=2, label='Negative Distance')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Distance', fontsize=12)
    axes[1].set_title('Embedding Distances over Epochs', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training history plot saved to {save_path}")
    plt.show()


# ============================================================================
# FINE-TUNING UTILITIES
# ============================================================================

def fine_tune_on_new_data(model_path: str, new_data_path: str, 
                         output_model_path: str, epochs: int = 10):
    """
    Fine-tune the trained model on new individual cat data.
    
    This function demonstrates how to continue training with real cat data
    (e.g., multiple images of 'Snowy', 'Luna', etc.)
    
    Args:
        model_path: Path to the pre-trained model
        new_data_path: Path to new dataset with individual cat folders
                      Expected structure: new_data_path/cat_name/image.jpg
        output_model_path: Path to save the fine-tuned model
        epochs: Number of fine-tuning epochs
    
    Usage Example:
        # Organize your data like this:
        # new_data/
        #   ├── Snowy/
        #   │   ├── img1.jpg
        #   │   ├── img2.jpg
        #   │   └── img3.jpg
        #   ├── Luna/
        #   │   ├── img1.jpg
        #   │   └── img2.jpg
        #   └── ...
        
        fine_tune_on_new_data(
            model_path='cat_identity_model.h5',
            new_data_path='path/to/new_data',
            output_model_path='cat_identity_model_finetuned.h5',
            epochs=10
        )
    """
    print("\n" + "="*80)
    print("FINE-TUNING ON NEW DATA")
    print("="*80)
    
    # Load pre-trained model
    print(f"Loading pre-trained model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    
    # Load new dataset
    print(f"Loading new dataset from {new_data_path}...")
    loader = CatDatasetLoader(new_data_path)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Create triplet generator
    triplet_generator = TripletGenerator(image_paths, labels, loader.breed_to_images)
    
    # Create triplet model
    triplet_model = build_triplet_model(model)
    
    # Create trainer
    config = Config()
    config.EPOCHS = epochs
    config.MODEL_SAVE_PATH = output_model_path
    
    trainer = TripletTrainer(model, triplet_model, triplet_generator, config)
    
    # Fine-tune
    history = trainer.train(
        epochs=epochs,
        triplets_per_epoch=2000,  # Fewer triplets for fine-tuning
        batch_size=Config.BATCH_SIZE
    )
    
    print(f"\n✅ Fine-tuning complete! Model saved to {output_model_path}")
    
    return history


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("CAT IDENTITY RECOGNITION MODEL TRAINER")
    print("Using Triplet Loss with EfficientNet-B0")
    print("="*80)
    
    # Check if dataset exists
    if not os.path.exists(Config.DATASET_PATH):
        print(f"\n❌ ERROR: Dataset not found at {Config.DATASET_PATH}")
        print("Please ensure the cropped cat images are in the correct location.")
        return
    
    # 1. Load dataset
    print("\n" + "="*80)
    print("STEP 1: Loading Dataset")
    print("="*80)
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # 2. Create triplet generator
    print("\n" + "="*80)
    print("STEP 2: Creating Triplet Generator")
    print("="*80)
    triplet_generator = TripletGenerator(image_paths, labels, loader.breed_to_images)
    
    # 3. Build models
    print("\n" + "="*80)
    print("STEP 3: Building Models")
    print("="*80)
    
    embedding_model = build_embedding_model(
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS),
        embedding_dim=Config.EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT_RATE,
        trainable_backbone_layers=Config.BACKBONE_TRAINABLE_LAYERS
    )
    
    print("\nEmbedding Model Summary:")
    embedding_model.summary()
    
    triplet_model = build_triplet_model(embedding_model)
    
    print("\nTriplet Model Summary:")
    print(f"Inputs: {[inp.shape for inp in triplet_model.inputs]}")
    print(f"Outputs: {[out.shape for out in triplet_model.outputs]}")
    
    # 4. Train model
    print("\n" + "="*80)
    print("STEP 4: Training Model")
    print("="*80)
    
    trainer = TripletTrainer(embedding_model, triplet_model, triplet_generator, Config)
    
    history = trainer.train(
        epochs=Config.EPOCHS,
        triplets_per_epoch=Config.TRIPLETS_PER_EPOCH,
        batch_size=Config.BATCH_SIZE
    )
    
    # 5. Save training history
    with open(Config.TRAINING_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n✅ Training history saved to {Config.TRAINING_HISTORY_PATH}")
    
    # 6. Plot training history
    print("\n" + "="*80)
    print("STEP 5: Visualizing Training History")
    print("="*80)
    plot_training_history(history)
    
    # 7. Extract and save embeddings
    print("\n" + "="*80)
    print("STEP 6: Extracting and Saving Embeddings")
    print("="*80)
    
    embeddings, valid_paths = extract_embeddings(
        embedding_model, 
        image_paths, 
        batch_size=Config.BATCH_SIZE
    )
    
    save_embeddings(
        embeddings, 
        valid_paths, 
        labels, 
        breed_to_idx,
        Config.EMBEDDINGS_SAVE_PATH,
        Config.METADATA_SAVE_PATH
    )
    
    # 8. Test the model with sample comparisons
    print("\n" + "="*80)
    print("STEP 7: Testing Model with Sample Comparisons")
    print("="*80)
    
    # Compare two images from the same breed (should be similar)
    same_breed_images = [img for img in image_paths if 'Abyssinian' in img][:2]
    if len(same_breed_images) >= 2:
        print("\n--- Testing: Same Breed (Expected: Similar) ---")
        compare_images(
            embedding_model, 
            same_breed_images[0], 
            same_breed_images[1],
            threshold=Config.SIMILARITY_THRESHOLD,
            visualize=False
        )
    
    # Compare two images from different breeds (should be different)
    abyssinian_img = [img for img in image_paths if 'Abyssinian' in img][0]
    bengal_img = [img for img in image_paths if 'Bengal' in img][0]
    
    print("\n--- Testing: Different Breeds (Expected: Different) ---")
    compare_images(
        embedding_model,
        abyssinian_img,
        bengal_img,
        threshold=Config.SIMILARITY_THRESHOLD,
        visualize=False
    )
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Trained model saved to: {Config.MODEL_SAVE_PATH}")
    print(f"✅ Embeddings saved to: {Config.EMBEDDINGS_SAVE_PATH}")
    print(f"✅ Metadata saved to: {Config.METADATA_SAVE_PATH}")
    print(f"✅ Training history saved to: {Config.TRAINING_HISTORY_PATH}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. To compare two images:")
    print("   >>> model = keras.models.load_model('cat_identity_model.h5')")
    print("   >>> compare_images(model, 'path/to/img1.jpg', 'path/to/img2.jpg')")
    print("\n2. To fine-tune on individual cat data:")
    print("   >>> fine_tune_on_new_data(")
    print("   ...     model_path='cat_identity_model.h5',")
    print("   ...     new_data_path='path/to/individual_cats',")
    print("   ...     output_model_path='cat_identity_model_finetuned.h5'")
    print("   ... )")
    print("\n3. Check the saved embeddings and metadata for further analysis")
    print("="*80)


if __name__ == "__main__":
    main()

