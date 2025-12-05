#!/usr/bin/env python3
"""
Pet ID Malaysia - EfficientNet-B3 Cat Recognition Training
Trains EfficientNet-B3 on cat dataset for optimal accuracy

Usage: python train_efficientnet_b3.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import legacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import cv2
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATASET_PATH = 'cat_individuals_dataset'
MODEL_SAVE_PATH = 'models/cat_identifier_efficientnet_b3_trained.keras'
TRAINING_HISTORY_PATH = 'efficientnet_b3_training_history.json'
IMG_SIZE = (300, 300)  # EfficientNet-B3 input size
BATCH_SIZE = 12  # Smaller batch size for B3 due to larger model
EPOCHS = 6  # Fewer epochs with B3 for efficiency
LEARNING_RATE = 5e-5  # Lower learning rate for stable training
VALIDATION_SPLIT = 0.2

class EfficientNetB3Trainer:
    """Trains EfficientNet-B3 for cat recognition"""

    def __init__(self):
        self.dataset_path = DATASET_PATH
        self.model_save_path = MODEL_SAVE_PATH
        self.img_size = IMG_SIZE
        self.batch_size = BATCH_SIZE
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE
        self.validation_split = VALIDATION_SPLIT

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        self.label_encoder = LabelEncoder()
        self.num_classes = None
        self.model = None

    def load_dataset(self):
        """Load and prepare the cat dataset for training"""
        logger.info("🔄 Loading cat dataset for EfficientNet-B3 training...")

        image_paths = []
        labels = []

        # Walk through dataset directories
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, file)

                    # Extract cat ID from directory name
                    cat_id = os.path.basename(root)

                    image_paths.append(img_path)
                    labels.append(cat_id)

        logger.info(f"📊 Found {len(image_paths)} cat images")
        logger.info(f"📊 Found {len(set(labels))} unique cats")

        # Encode labels
        self.labels_encoded = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)

        logger.info(f"🏷️  Encoded {self.num_classes} cat classes")

        return image_paths, self.labels_encoded

    def preprocess_image(self, img_path):
        """Preprocess a single image for EfficientNet-B3 training"""
        try:
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                return None

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize to B3 input size
            img_resized = cv2.resize(img_rgb, self.img_size)

            # EfficientNet-B3 specific preprocessing
            img_array = img_resized.astype(np.float32)
            img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

            return img_array

        except Exception as e:
            logger.warning(f"⚠️ Error processing {img_path}: {e}")
            return None

    def prepare_data(self, image_paths, labels):
        """Prepare training and validation data"""
        logger.info("📊 Preparing training data...")

        # Split data
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            test_size=self.validation_split,
            random_state=42,
            stratify=labels
        )

        logger.info(f"📊 Training samples: {len(train_paths)}")
        logger.info(f"📊 Validation samples: {len(val_paths)}")

        # Process training data
        X_train = []
        y_train = []
        for img_path, label in zip(train_paths, train_labels):
            img_array = self.preprocess_image(img_path)
            if img_array is not None:
                X_train.append(img_array)
                y_train.append(label)

        # Process validation data
        X_val = []
        y_val = []
        for img_path, label in zip(val_paths, val_labels):
            img_array = self.preprocess_image(img_path)
            if img_array is not None:
                X_val.append(img_array)
                y_val.append(label)

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        logger.info(f"✅ Prepared {len(X_train)} training samples")
        logger.info(f"✅ Prepared {len(X_val)} validation samples")

        return X_train, y_train, X_val, y_val

    def build_model(self):
        """Build the EfficientNet-B3 model architecture"""
        logger.info("🏗️  Building EfficientNet-B3 model...")

        # Load EfficientNet-B3 with pre-trained ImageNet weights
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3),
            pooling='avg'
        )

        # Freeze the base model initially
        base_model.trainable = False

        # Add custom classification head
        inputs = keras.Input(shape=(*self.img_size, 3))

        # Data augmentation
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.05)(x)  # Less rotation for faces
        x = layers.RandomZoom(0.05)(x)

        # Pass through base model
        x = base_model(x, training=False)

        # Add classification layers
        x = layers.Dropout(0.4)(x)  # Higher dropout for B3
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        # Use legacy optimizer for M1/M2 Mac compatibility
        model.compile(
            optimizer=legacy.Adam(learning_rate=self.learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        logger.info("✅ EfficientNet-B3 model built successfully")
        logger.info(f"📊 Model parameters: {model.count_params():,}")

        return model

    def train_model(self):
        """Train the EfficientNet-B3 model"""
        logger.info("🚀 Starting EfficientNet-B3 training...")

        # Load dataset
        image_paths, labels = self.load_dataset()

        # Prepare data
        X_train, y_train, X_val, y_val = self.prepare_data(image_paths, labels)

        # Build model
        self.build_model()

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1,
                save_format='tf'  # Use TF format to avoid compatibility issues
            )
        ]

        logger.info("🏃‍♂️ Training EfficientNet-B3...")

        # Phase 1: Train only the top layers
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs // 2,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        logger.info("🔧 Fine-tuning EfficientNet-B3...")

        # Phase 2: Fine-tune the entire model with lower learning rate
        self.model.trainable = True
        self.model.compile(
            optimizer=legacy.Adam(learning_rate=self.learning_rate / 10),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Continue training
        history_finetune = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )

        # Save training history
        training_history = {
            'initial_training': {k: [float(x) for x in v] for k, v in history.history.items()},
            'fine_tuning': {k: [float(x) for x in v] for k, v in history_finetune.history.items()},
            'num_classes': self.num_classes,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'epochs_completed': len(history.history['loss']) + len(history_finetune.history['loss']),
            'model_type': 'EfficientNet-B3'
        }

        with open(TRAINING_HISTORY_PATH, 'w') as f:
            json.dump(training_history, f, indent=2)

        logger.info(f"💾 EfficientNet-B3 model saved to {self.model_save_path}")
        logger.info(f"📈 Training history saved to {TRAINING_HISTORY_PATH}")

        return training_history

    def evaluate_model(self):
        """Evaluate the trained EfficientNet-B3 model"""
        logger.info("📊 Evaluating EfficientNet-B3 model...")

        # Load dataset for evaluation
        image_paths, labels = self.load_dataset()
        _, test_paths, _, test_labels = train_test_split(
            image_paths, labels, test_size=0.1, random_state=42
        )

        # Prepare test data
        X_test = []
        y_test = []
        for img_path, label in zip(test_paths, test_labels):
            img_array = self.preprocess_image(img_path)
            if img_array is not None:
                X_test.append(img_array)
                y_test.append(label)

        if X_test:
            X_test = np.array(X_test)
            y_test = np.array(y_test)

            # Evaluate
            loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

            logger.info(f"📊 Test Loss: {loss:.4f}")
            logger.info(f"📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

            return {
                'test_loss': loss,
                'test_accuracy': accuracy
            }

        return None

def main():
    """Main function to run the EfficientNet-B3 training"""
    print("🐾 Pet ID Malaysia - EfficientNet-B3 Cat Recognition Training")
    print("=" * 70)

    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        logger.error(f"❌ Dataset path not found: {DATASET_PATH}")
        logger.error("Please ensure your cat dataset is available.")
        return

    # Initialize trainer
    trainer = EfficientNetB3Trainer()

    try:
        # Train the model
        training_history = trainer.train_model()

        # Evaluate the model
        evaluation_results = trainer.evaluate_model()

        if evaluation_results:
            print("\n" + "=" * 70)
            print("🎉 EFFICIENTNET-B3 TRAINING COMPLETED!")
            print("=" * 70)
            print(f"📊 Test Accuracy: {evaluation_results['test_accuracy']*100:.1f}%")
            print(f"💾 Model saved: {trainer.model_save_path}")
            print(f"📈 History saved: {TRAINING_HISTORY_PATH}")
            print("=" * 70)
            print("\n✅ The EfficientNet-B3 trained model is now ready for use!")
            print("🔄 Update your app.py MODEL_PATH to use the new model for maximum accuracy.")

    except Exception as e:
        logger.error(f"❌ Error during EfficientNet-B3 training: {e}")
        raise

if __name__ == '__main__':
    main()