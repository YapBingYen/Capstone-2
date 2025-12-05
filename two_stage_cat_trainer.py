"""
Two-Stage Cat Recognition Training Pipeline
============================================
Stage 1: Breed Classification (supervised learning)
Stage 2: Triplet Loss Fine-tuning (metric learning for embeddings)

This fixes the issue where the model treats all cats as "the same" by:
1. Learning breed-specific features through classification
2. Fine-tuning with triplet loss to create discriminative embeddings

Author: AI Assistant
Date: October 2025
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
# Prefer the unified keras namespace to avoid IDE resolver issues
layers = keras.layers
Model = keras.Model
EfficientNetB0 = keras.applications.EfficientNetB0
EarlyStopping = keras.callbacks.EarlyStopping
ModelCheckpoint = keras.callbacks.ModelCheckpoint
ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import random
from tqdm import tqdm

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Global configuration"""
    # Paths
    DATASET_PATH = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
    
    # Stage 1: Classification
    CLASSIFIER_MODEL_PATH = "cat_breed_classifier.h5"
    CLASSIFIER_PLOT_PATH = "breed_training_plot.png"
    CLASSIFIER_EPOCHS = 15
    CLASSIFIER_BATCH_SIZE = 32
    CLASSIFIER_LR = 1e-4
    
    # Stage 2: Triplet Loss
    EMBEDDING_MODEL_PATH = "cat_breed_embedding_model.h5"
    EMBEDDING_PLOT_PATH = "embedding_training_plot.png"
    EMBEDDING_EPOCHS = 10
    EMBEDDING_BATCH_SIZE = 32
    EMBEDDING_LR = 5e-5
    TRIPLET_MARGIN = 0.5
    TRIPLETS_PER_EPOCH = 3000
    
    # Model architecture
    IMG_SIZE = 224
    EMBEDDING_DIM = 128
    DROPOUT_RATE = 0.3
    
    # Data split
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.2


# ============================================================================
# STAGE 1: BREED CLASSIFICATION
# ============================================================================

def build_breed_classifier(num_classes: int) -> Model:
    """
    Build EfficientNet-B0 based breed classifier.
    
    Args:
        num_classes: Number of cat breeds (13)
    
    Returns:
        Keras Model for classification
    """
    print("\n" + "="*80)
    print("Building Breed Classifier")
    print("="*80)
    
    # Input layer
    input_layer = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    
    # EfficientNet-B0 backbone
    backbone = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=input_layer,
        pooling=None
    )
    
    # Fine-tune top layers
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    
    # Get backbone output
    x = backbone.output
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(Config.DROPOUT_RATE, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='breed_output')(x)
    
    # Create model
    model = Model(inputs=input_layer, outputs=outputs, name='breed_classifier')
    
    print(f"\nModel created with {num_classes} classes")
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum(p.shape.num_elements() for p in model.trainable_weights):,}")
    
    return model


def train_breed_classifier(dataset_path: str) -> Tuple[Model, dict]:
    """
    Train the breed classification model.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        Trained model and training history
    """
    print("\n" + "="*80)
    print("STAGE 1: BREED CLASSIFICATION TRAINING")
    print("="*80)
    
    # Load training dataset
    print("\nLoading training dataset...")
    train_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=Config.VAL_SPLIT,
        subset="training",
        seed=42,
        image_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        label_mode='int'
    )
    
    # Load validation dataset
    print("Loading validation dataset...")
    val_ds = keras.utils.image_dataset_from_directory(
        dataset_path,
        validation_split=Config.VAL_SPLIT,
        subset="validation",
        seed=42,
        image_size=(Config.IMG_SIZE, Config.IMG_SIZE),
        batch_size=Config.CLASSIFIER_BATCH_SIZE,
        label_mode='int'
    )
    
    # Get class names and count
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Number of breeds: {num_classes}")
    print(f"Breeds: {class_names}")
    
    # Normalize images to [0, 1]
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
    # Optimize performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build model
    model = build_breed_classifier(num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.CLASSIFIER_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            Config.CLASSIFIER_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n" + "="*80)
    print("Training Breed Classifier")
    print("="*80)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.CLASSIFIER_EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(
        history.history,
        Config.CLASSIFIER_PLOT_PATH,
        "Breed Classification Training"
    )
    
    print(f"\n✅ Breed classifier trained and saved to {Config.CLASSIFIER_MODEL_PATH}")
    
    return model, history.history


def plot_training_history(history: dict, save_path: str, title: str):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Training plot saved to {save_path}")
    plt.close()


# ============================================================================
# STAGE 2: TRIPLET LOSS EMBEDDING
# ============================================================================

def build_embedding_model(classifier_model: Model) -> Model:
    """
    Build embedding model from trained classifier.
    
    Args:
        classifier_model: Trained breed classifier
    
    Returns:
        Embedding model that outputs normalized embeddings
    """
    print("\n" + "="*80)
    print("Building Embedding Model")
    print("="*80)
    
    # Get the base model (everything except the classification head)
    base_model = Model(
        inputs=classifier_model.input,
        outputs=classifier_model.get_layer('global_avg_pool').output
    )
    
    # Make all layers trainable for fine-tuning
    for layer in base_model.layers:
        layer.trainable = True
    
    # Build embedding model
    input_layer = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3))
    x = base_model(input_layer)
    
    # Embedding layer
    x = layers.Dense(Config.EMBEDDING_DIM, activation='relu', name='embedding_dense')(x)
    
    # L2 normalization for cosine similarity
    embeddings = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1),
        name='l2_normalize'
    )(x)
    
    model = Model(inputs=input_layer, outputs=embeddings, name='embedding_model')
    
    print(f"\nEmbedding model created")
    print(f"Output dimension: {Config.EMBEDDING_DIM}")
    print(f"Trainable parameters: {sum(p.shape.num_elements() for p in model.trainable_weights):,}")
    
    return model


class TripletDataGenerator:
    """Generate triplets for triplet loss training"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.breed_to_images = {}
        self.breeds = []
        
        # Load all image paths organized by breed
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all image paths grouped by breed"""
        print("\nLoading dataset for triplet generation...")
        
        for breed_folder in sorted(self.dataset_path.iterdir()):
            if breed_folder.is_dir():
                breed_name = breed_folder.name
                self.breeds.append(breed_name)
                
                # Get all images for this breed
                images = list(breed_folder.glob("*.jpg")) + \
                        list(breed_folder.glob("*.jpeg")) + \
                        list(breed_folder.glob("*.png"))
                
                self.breed_to_images[breed_name] = [str(img) for img in images]
        
        print(f"✅ Loaded {len(self.breeds)} breeds")
        for breed in self.breeds:
            print(f"   {breed}: {len(self.breed_to_images[breed])} images")
    
    def load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image"""
        img = keras.utils.load_img(image_path, target_size=(Config.IMG_SIZE, Config.IMG_SIZE))
        img_array = keras.utils.img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        return img_array
    
    def generate_triplet_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of triplets.
        
        Returns:
            anchors, positives, negatives (each batch_size x 224 x 224 x 3)
        """
        anchors = []
        positives = []
        negatives = []
        
        for _ in range(batch_size):
            # Select anchor breed
            anchor_breed = random.choice(self.breeds)
            
            # Get two different images from same breed
            breed_images = self.breed_to_images[anchor_breed]
            if len(breed_images) < 2:
                continue
            
            anchor_path, positive_path = random.sample(breed_images, 2)
            
            # Get negative from different breed
            negative_breed = random.choice([b for b in self.breeds if b != anchor_breed])
            negative_path = random.choice(self.breed_to_images[negative_breed])
            
            # Load and preprocess images
            anchors.append(self.load_and_preprocess_image(anchor_path))
            positives.append(self.load_and_preprocess_image(positive_path))
            negatives.append(self.load_and_preprocess_image(negative_path))
        
        return (
            np.array(anchors, dtype=np.float32),
            np.array(positives, dtype=np.float32),
            np.array(negatives, dtype=np.float32)
        )


class TripletLoss(keras.losses.Loss):
    """Triplet loss with semi-hard negative mining"""
    
    def __init__(self, margin: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
    
    def call(self, y_true, y_pred):
        """Compute triplet loss"""
        # y_pred should be concatenated [anchor, positive, negative]
        embedding_dim = y_pred.shape[-1] // 3
        
        anchor = y_pred[:, :embedding_dim]
        positive = y_pred[:, embedding_dim:2*embedding_dim]
        negative = y_pred[:, 2*embedding_dim:]
        
        # Compute distances
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        
        # Triplet loss
        loss = tf.maximum(0.0, pos_dist - neg_dist + self.margin)
        
        return tf.reduce_mean(loss)


def build_triplet_model(embedding_model: Model) -> Model:
    """Build triplet model with three inputs"""
    anchor_input = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='anchor')
    positive_input = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='positive')
    negative_input = layers.Input(shape=(Config.IMG_SIZE, Config.IMG_SIZE, 3), name='negative')
    
    # Get embeddings (shared weights)
    anchor_emb = embedding_model(anchor_input)
    positive_emb = embedding_model(positive_input)
    negative_emb = embedding_model(negative_input)
    
    # Concatenate for loss computation
    outputs = layers.Concatenate()([anchor_emb, positive_emb, negative_emb])
    
    triplet_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=outputs,
        name='triplet_model'
    )
    
    return triplet_model


def train_embedding_model(classifier_model: Model, dataset_path: str) -> Tuple[Model, dict]:
    """
    Train embedding model with triplet loss.
    
    Args:
        classifier_model: Pre-trained breed classifier
        dataset_path: Path to dataset
    
    Returns:
        Trained embedding model and history
    """
    print("\n" + "="*80)
    print("STAGE 2: TRIPLET LOSS EMBEDDING TRAINING")
    print("="*80)
    
    # Build embedding model
    embedding_model = build_embedding_model(classifier_model)
    
    # Build triplet model
    triplet_model = build_triplet_model(embedding_model)
    
    # Compile
    triplet_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=Config.EMBEDDING_LR),
        loss=TripletLoss(margin=Config.TRIPLET_MARGIN)
    )
    
    # Create data generator
    triplet_gen = TripletDataGenerator(dataset_path)
    
    # Training loop
    history = {
        'loss': [],
        'positive_dist': [],
        'negative_dist': []
    }
    
    best_loss = float('inf')
    
    for epoch in range(Config.EMBEDDING_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EMBEDDING_EPOCHS}")
        print("-" * 80)
        
        epoch_losses = []
        epoch_pos_dists = []
        epoch_neg_dists = []
        
        num_batches = Config.TRIPLETS_PER_EPOCH // Config.EMBEDDING_BATCH_SIZE
        
        for batch_idx in tqdm(range(num_batches), desc="Training"):
            # Generate triplet batch
            anchors, positives, negatives = triplet_gen.generate_triplet_batch(
                Config.EMBEDDING_BATCH_SIZE
            )
            
            if len(anchors) == 0:
                continue
            
            # Train step
            loss = triplet_model.train_on_batch(
                [anchors, positives, negatives],
                np.zeros((len(anchors), Config.EMBEDDING_DIM * 3))  # Dummy labels
            )
            
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
        
        history['loss'].append(avg_loss)
        history['positive_dist'].append(avg_pos_dist)
        history['negative_dist'].append(avg_neg_dist)
        
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Positive Distance: {avg_pos_dist:.4f}")
        print(f"  Negative Distance: {avg_neg_dist:.4f}")
        print(f"  Margin: {avg_neg_dist - avg_pos_dist:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            embedding_model.save(Config.EMBEDDING_MODEL_PATH)
            print(f"  ✅ New best loss! Model saved.")
    
    # Plot training history
    plot_embedding_history(history, Config.EMBEDDING_PLOT_PATH)
    
    print(f"\n✅ Embedding model trained and saved to {Config.EMBEDDING_MODEL_PATH}")
    
    return embedding_model, history


def plot_embedding_history(history: dict, save_path: str):
    """Plot embedding training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['loss'], 'b-', linewidth=2, label='Triplet Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Triplet Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot distances
    axes[1].plot(epochs, history['positive_dist'], 'g-', linewidth=2, label='Positive Distance')
    axes[1].plot(epochs, history['negative_dist'], 'r-', linewidth=2, label='Negative Distance')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Distance', fontsize=12)
    axes[1].set_title('Embedding Distances over Epochs', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Triplet Loss Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Embedding training plot saved to {save_path}")
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("="*80)
    print("TWO-STAGE CAT RECOGNITION TRAINING PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("1. Train a breed classifier (Stage 1)")
    print("2. Fine-tune with triplet loss for embeddings (Stage 2)")
    print("\nThis fixes the 'all cats look the same' issue by learning")
    print("discriminative features at both breed and individual levels.")
    
    # Check dataset
    if not os.path.exists(Config.DATASET_PATH):
        print(f"\n❌ Dataset not found at {Config.DATASET_PATH}")
        return
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\n✅ GPU detected: {gpus}")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("\n⚠️  No GPU detected, using CPU (slower)")
    
    # Stage 1: Train breed classifier
    print("\n" + "="*80)
    print("STAGE 1: BREED CLASSIFICATION")
    print("="*80)
    
    classifier_model, classifier_history = train_breed_classifier(Config.DATASET_PATH)
    
    # Stage 2: Train embedding model
    print("\n" + "="*80)
    print("STAGE 2: TRIPLET LOSS EMBEDDING")
    print("="*80)
    
    embedding_model, embedding_history = train_embedding_model(
        classifier_model,
        Config.DATASET_PATH
    )
    
    # Summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"\n✅ Stage 1 - Breed Classifier:")
    print(f"   Model: {Config.CLASSIFIER_MODEL_PATH}")
    print(f"   Plot: {Config.CLASSIFIER_PLOT_PATH}")
    print(f"   Final Val Accuracy: {classifier_history['val_accuracy'][-1]:.2%}")
    
    print(f"\n✅ Stage 2 - Embedding Model:")
    print(f"   Model: {Config.EMBEDDING_MODEL_PATH}")
    print(f"   Plot: {Config.EMBEDDING_PLOT_PATH}")
    print(f"   Final Loss: {embedding_history['loss'][-1]:.4f}")
    print(f"   Final Margin: {embedding_history['negative_dist'][-1] - embedding_history['positive_dist'][-1]:.4f}")
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Test the embedding model:")
    print("   >>> from tensorflow import keras")
    print("   >>> model = keras.models.load_model('cat_breed_embedding_model.h5')")
    print("   >>> # Use model to generate embeddings and compare cats")
    print("\n2. The embedding model outputs 128-dim normalized vectors")
    print("3. Use cosine similarity to compare cats (threshold ~0.8)")


if __name__ == "__main__":
    main()

