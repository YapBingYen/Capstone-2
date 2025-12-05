"""
Complete Workflow Example
=========================
This script demonstrates the complete workflow from training to inference.
Run this to see how all components work together.
"""

import os
from pathlib import Path
from tensorflow import keras
from cat_identity_model_trainer import (
    Config, CatDatasetLoader, TripletGenerator, 
    build_embedding_model, build_triplet_model,
    TripletTrainer, extract_embeddings, save_embeddings,
    compare_images, fine_tune_on_new_data
)


def example_1_basic_training():
    """
    Example 1: Basic Training Workflow
    This shows the minimal code needed to train the model.
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Training")
    print("="*80)
    
    # Check dataset exists
    if not os.path.exists(Config.DATASET_PATH):
        print(f"❌ Dataset not found at {Config.DATASET_PATH}")
        print("Please ensure cropped images are in the correct location.")
        return
    
    # Load dataset
    print("\n1. Loading dataset...")
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Create triplet generator
    print("\n2. Creating triplet generator...")
    triplet_generator = TripletGenerator(image_paths, labels, loader.breed_to_images)
    
    # Build models
    print("\n3. Building embedding model...")
    embedding_model = build_embedding_model(
        embedding_dim=Config.EMBEDDING_DIM,
        dropout_rate=Config.DROPOUT_RATE
    )
    
    triplet_model = build_triplet_model(embedding_model)
    
    # Train
    print("\n4. Training model...")
    trainer = TripletTrainer(embedding_model, triplet_model, triplet_generator, Config)
    
    history = trainer.train(
        epochs=5,  # Reduced for demo
        triplets_per_epoch=1000,  # Reduced for demo
        batch_size=Config.BATCH_SIZE
    )
    
    print("\n✅ Training complete!")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")


def example_2_inference():
    """
    Example 2: Using the Trained Model for Inference
    This shows how to compare cat images.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Inference")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model not found at {Config.MODEL_SAVE_PATH}")
        print("Please train the model first (run example_1_basic_training)")
        return
    
    # Load model
    print("\n1. Loading trained model...")
    model = keras.models.load_model(Config.MODEL_SAVE_PATH, compile=False)
    print("✅ Model loaded!")
    
    # Find some example images
    print("\n2. Finding example images...")
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Get images from same breed
    breed_folders = list(Path(Config.DATASET_PATH).iterdir())
    if len(breed_folders) == 0:
        print("❌ No breed folders found")
        return
    
    first_breed = breed_folders[0]
    breed_images = list(first_breed.glob("*.jpg"))[:2]
    
    if len(breed_images) < 2:
        print("❌ Not enough images in first breed folder")
        return
    
    # Compare same breed (should be similar)
    print(f"\n3. Comparing two images from {first_breed.name} (Expected: Similar)")
    result1 = compare_images(
        model, 
        str(breed_images[0]), 
        str(breed_images[1]),
        threshold=0.8,
        visualize=False
    )
    
    # Get images from different breeds
    if len(breed_folders) >= 2:
        second_breed = breed_folders[1]
        breed2_images = list(second_breed.glob("*.jpg"))[:1]
        
        if len(breed2_images) > 0:
            print(f"\n4. Comparing {first_breed.name} vs {second_breed.name} (Expected: Different)")
            result2 = compare_images(
                model,
                str(breed_images[0]),
                str(breed2_images[0]),
                threshold=0.8,
                visualize=False
            )


def example_3_embedding_extraction():
    """
    Example 3: Extract and Use Embeddings
    This shows how to extract embeddings and use them for fast comparison.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Embedding Extraction and Fast Comparison")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model not found at {Config.MODEL_SAVE_PATH}")
        print("Please train the model first")
        return
    
    # Load model
    print("\n1. Loading model...")
    model = keras.models.load_model(Config.MODEL_SAVE_PATH, compile=False)
    
    # Load dataset
    print("\n2. Loading dataset...")
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Extract embeddings (compute once)
    print("\n3. Extracting embeddings for all images...")
    embeddings, valid_paths = extract_embeddings(
        model, 
        image_paths[:100],  # First 100 for demo
        batch_size=32
    )
    
    # Save embeddings
    print("\n4. Saving embeddings...")
    save_embeddings(
        embeddings,
        valid_paths,
        labels[:100],
        breed_to_idx,
        'demo_embeddings.npz',
        'demo_metadata.csv'
    )
    
    # Fast comparison using embeddings
    print("\n5. Fast comparison using precomputed embeddings...")
    import numpy as np
    
    # Compare first image with all others
    query_embedding = embeddings[0]
    similarities = np.dot(embeddings, query_embedding)
    
    # Find top 5 most similar
    top_5_indices = np.argsort(similarities)[-5:][::-1]
    
    print(f"\nTop 5 most similar images to {valid_paths[0]}:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"{i}. {valid_paths[idx]}: {similarities[idx]:.4f}")
    
    # Cleanup demo files
    if os.path.exists('demo_embeddings.npz'):
        os.remove('demo_embeddings.npz')
    if os.path.exists('demo_metadata.csv'):
        os.remove('demo_metadata.csv')


def example_4_batch_comparison():
    """
    Example 4: Batch Comparison
    Compare multiple images efficiently.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Batch Comparison")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model not found at {Config.MODEL_SAVE_PATH}")
        return
    
    # Load model
    print("\n1. Loading model...")
    model = keras.models.load_model(Config.MODEL_SAVE_PATH, compile=False)
    
    # Get some test images
    print("\n2. Loading test images...")
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Select 5 random images
    import random
    random.seed(42)
    test_images = random.sample(image_paths, min(5, len(image_paths)))
    
    # Extract embeddings for test images
    print("\n3. Extracting embeddings...")
    from cat_identity_model_trainer import preprocess_image
    import numpy as np
    
    test_embeddings = []
    for img_path in test_images:
        img = preprocess_image(img_path)
        embedding = model.predict(np.expand_dims(img, 0), verbose=0)[0]
        test_embeddings.append(embedding)
    
    test_embeddings = np.array(test_embeddings)
    
    # Create similarity matrix
    print("\n4. Computing pairwise similarities...")
    similarity_matrix = np.dot(test_embeddings, test_embeddings.T)
    
    # Display results
    print("\n5. Similarity Matrix:")
    print("-" * 80)
    
    # Print header
    print(f"{'Image':30}", end='')
    for i in range(len(test_images)):
        print(f"Img{i+1:2d}  ", end='')
    print()
    
    # Print matrix
    for i in range(len(test_images)):
        img_name = Path(test_images[i]).parent.name[:20]  # Breed name
        print(f"{img_name:30}", end='')
        for j in range(len(test_images)):
            sim = similarity_matrix[i, j]
            print(f"{sim:5.3f} ", end='')
        print()


def example_5_fine_tuning():
    """
    Example 5: Fine-tuning for Individual Cats
    This demonstrates how to adapt the model for specific cats.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Fine-tuning for Individual Cats")
    print("="*80)
    
    print("\n📝 Note: This is a demonstration of the fine-tuning workflow.")
    print("For actual fine-tuning, you need to provide individual cat data.")
    
    print("\nStep 1: Organize your data")
    print("-" * 80)
    print("""
Create a folder structure like this:

individual_cats/
├── Snowy/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
├── Luna/
│   ├── img1.jpg
│   └── img2.jpg
└── Whiskers/
    ├── pic1.jpg
    └── pic2.jpg
    """)
    
    print("\nStep 2: Run fine-tuning")
    print("-" * 80)
    print("""
from cat_identity_model_trainer import fine_tune_on_new_data

history = fine_tune_on_new_data(
    model_path='cat_identity_model.h5',
    new_data_path='path/to/individual_cats',
    output_model_path='cat_identity_model_finetuned.h5',
    epochs=10
)
    """)
    
    print("\nStep 3: Use the fine-tuned model")
    print("-" * 80)
    print("""
from tensorflow import keras
from cat_identity_model_trainer import compare_images

model = keras.models.load_model('cat_identity_model_finetuned.h5')

# Compare two photos of Snowy
result = compare_images(
    model,
    'snowy_photo1.jpg',
    'snowy_photo2.jpg',
    threshold=0.85
)

print(f"Same cat: {result['is_same_cat']}")
# Should return True if both are Snowy
    """)
    
    print("\n✅ Fine-tuning workflow explained!")


def example_6_custom_threshold():
    """
    Example 6: Finding the Optimal Threshold
    This shows how to find the best threshold for your use case.
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Finding Optimal Threshold")
    print("="*80)
    
    # Check if model exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"❌ Model not found at {Config.MODEL_SAVE_PATH}")
        return
    
    # Load model
    print("\n1. Loading model...")
    model = keras.models.load_model(Config.MODEL_SAVE_PATH, compile=False)
    
    # Load dataset
    print("\n2. Loading dataset...")
    loader = CatDatasetLoader(Config.DATASET_PATH)
    image_paths, labels, breed_to_idx = loader.load_dataset()
    
    # Sample some pairs
    print("\n3. Sampling image pairs...")
    from cat_identity_model_trainer import preprocess_image
    import numpy as np
    import random
    
    random.seed(42)
    
    # Same breed pairs (positive)
    same_breed_sims = []
    for _ in range(50):  # Sample 50 pairs
        breed_idx = random.choice(list(loader.breed_to_images.keys()))
        breed_images = loader.breed_to_images[breed_idx]
        if len(breed_images) >= 2:
            img1, img2 = random.sample(breed_images, 2)
            
            emb1 = model.predict(np.expand_dims(preprocess_image(img1), 0), verbose=0)[0]
            emb2 = model.predict(np.expand_dims(preprocess_image(img2), 0), verbose=0)[0]
            
            sim = float(np.dot(emb1, emb2))
            same_breed_sims.append(sim)
    
    # Different breed pairs (negative)
    different_breed_sims = []
    breeds = list(loader.breed_to_images.keys())
    for _ in range(50):
        breed1, breed2 = random.sample(breeds, 2)
        img1 = random.choice(loader.breed_to_images[breed1])
        img2 = random.choice(loader.breed_to_images[breed2])
        
        emb1 = model.predict(np.expand_dims(preprocess_image(img1), 0), verbose=0)[0]
        emb2 = model.predict(np.expand_dims(preprocess_image(img2), 0), verbose=0)[0]
        
        sim = float(np.dot(emb1, emb2))
        different_breed_sims.append(sim)
    
    # Find optimal threshold
    print("\n4. Computing statistics...")
    same_breed_sims = np.array(same_breed_sims)
    different_breed_sims = np.array(different_breed_sims)
    
    print(f"\nSame Breed Similarities:")
    print(f"  Mean: {np.mean(same_breed_sims):.4f}")
    print(f"  Min:  {np.min(same_breed_sims):.4f}")
    print(f"  Max:  {np.max(same_breed_sims):.4f}")
    
    print(f"\nDifferent Breed Similarities:")
    print(f"  Mean: {np.mean(different_breed_sims):.4f}")
    print(f"  Min:  {np.min(different_breed_sims):.4f}")
    print(f"  Max:  {np.max(different_breed_sims):.4f}")
    
    # Try different thresholds
    print("\n5. Testing different thresholds...")
    print("-" * 80)
    print(f"{'Threshold':12} {'Accuracy':12} {'TPR':12} {'TNR':12}")
    print("-" * 80)
    
    for threshold in [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
        tp = np.sum(same_breed_sims >= threshold)  # True positives
        tn = np.sum(different_breed_sims < threshold)  # True negatives
        
        accuracy = (tp + tn) / (len(same_breed_sims) + len(different_breed_sims))
        tpr = tp / len(same_breed_sims)  # True positive rate
        tnr = tn / len(different_breed_sims)  # True negative rate
        
        print(f"{threshold:12.2f} {accuracy:12.2%} {tpr:12.2%} {tnr:12.2%}")
    
    print("\n💡 Recommendation:")
    print("- Use threshold 0.8-0.85 for balanced performance")
    print("- Use threshold 0.9+ for high precision (fewer false positives)")
    print("- Use threshold 0.7- for high recall (fewer false negatives)")


def main():
    """Main menu"""
    print("="*80)
    print("CAT IDENTITY RECOGNITION - WORKFLOW EXAMPLES")
    print("="*80)
    
    examples = {
        '1': ('Basic Training', example_1_basic_training),
        '2': ('Inference', example_2_inference),
        '3': ('Embedding Extraction', example_3_embedding_extraction),
        '4': ('Batch Comparison', example_4_batch_comparison),
        '5': ('Fine-tuning (Demo)', example_5_fine_tuning),
        '6': ('Find Optimal Threshold', example_6_custom_threshold),
    }
    
    while True:
        print("\n" + "="*80)
        print("Available Examples:")
        print("="*80)
        for key, (name, _) in examples.items():
            print(f"{key}. {name}")
        print("7. Run all examples")
        print("8. Exit")
        print("="*80)
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice in examples:
            _, func = examples[choice]
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error running example: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '7':
            print("\n🚀 Running all examples...")
            for key in sorted(examples.keys()):
                _, func = examples[key]
                try:
                    func()
                except Exception as e:
                    print(f"\n❌ Error in example {key}: {e}")
        
        elif choice == '8':
            print("\nGoodbye! 👋")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-8.")


if __name__ == "__main__":
    main()

