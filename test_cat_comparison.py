"""
Quick Test Script for Cat Identity Comparison
==============================================
This script provides an easy interface to test the trained cat identity model.
"""

import sys
import os
from tensorflow import keras
from cat_identity_model_trainer import compare_images, preprocess_image
import numpy as np

def load_model(model_path='cat_identity_model.h5'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first by running: python cat_identity_model_trainer.py")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully!")
    return model


def test_single_comparison(model, img1_path, img2_path, threshold=0.8):
    """Test comparing two images"""
    print("\n" + "="*80)
    print("TESTING CAT COMPARISON")
    print("="*80)
    
    if not os.path.exists(img1_path):
        print(f"❌ Error: Image 1 not found at {img1_path}")
        return
    
    if not os.path.exists(img2_path):
        print(f"❌ Error: Image 2 not found at {img2_path}")
        return
    
    result = compare_images(model, img1_path, img2_path, threshold, visualize=True)
    return result


def test_multiple_comparisons(model, image_paths, threshold=0.8):
    """Test comparing multiple images against each other"""
    print("\n" + "="*80)
    print("BATCH COMPARISON TEST")
    print("="*80)
    
    n = len(image_paths)
    print(f"Comparing {n} images with each other...")
    
    # Create similarity matrix
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                img1 = preprocess_image(image_paths[i])
                img2 = preprocess_image(image_paths[j])
                
                emb1 = model.predict(np.expand_dims(img1, 0), verbose=0)[0]
                emb2 = model.predict(np.expand_dims(img2, 0), verbose=0)[0]
                
                similarity = float(np.dot(emb1, emb2))
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    
    # Print results
    print("\nSimilarity Matrix:")
    print("-" * 80)
    
    # Print header
    print(f"{'':25}", end='')
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)[:10]
        print(f"{filename:12}", end='')
    print()
    
    # Print matrix
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)[:20]
        print(f"{filename:25}", end='')
        for j in range(len(image_paths)):
            sim = similarity_matrix[i][j]
            color = '✅' if sim >= threshold and i != j else '❌' if i != j else '  '
            print(f"{sim:5.3f} {color:3}", end='')
        print()
    
    print("\n" + "="*80)
    print("Legend:")
    print("✅ = Same cat (similarity >= threshold)")
    print("❌ = Different cats (similarity < threshold)")
    print("="*80)


def interactive_mode(model):
    """Interactive mode for testing"""
    print("\n" + "="*80)
    print("INTERACTIVE CAT COMPARISON MODE")
    print("="*80)
    
    threshold = float(input("Enter similarity threshold (default 0.8): ") or "0.8")
    
    while True:
        print("\n" + "-"*80)
        print("Options:")
        print("1. Compare two images")
        print("2. Batch comparison of multiple images")
        print("3. Change threshold")
        print("4. Exit")
        print("-"*80)
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            img1 = input("Enter path to first image: ").strip()
            img2 = input("Enter path to second image: ").strip()
            test_single_comparison(model, img1, img2, threshold)
        
        elif choice == '2':
            print("Enter image paths (one per line, empty line to finish):")
            paths = []
            while True:
                path = input(f"Image {len(paths)+1}: ").strip()
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
            threshold = float(input("Enter new similarity threshold: ") or "0.8")
            print(f"✅ Threshold updated to {threshold}")
        
        elif choice == '4':
            print("Goodbye! 👋")
            break
        
        else:
            print("❌ Invalid choice, please enter 1-4")


def main():
    """Main function"""
    print("="*80)
    print("CAT IDENTITY COMPARISON TEST")
    print("="*80)
    
    # Load model
    model = load_model()
    
    # Check command line arguments
    if len(sys.argv) == 3:
        # Two images provided as arguments
        img1_path = sys.argv[1]
        img2_path = sys.argv[2]
        test_single_comparison(model, img1_path, img2_path)
    
    elif len(sys.argv) > 3:
        # Multiple images for batch comparison
        image_paths = sys.argv[1:]
        test_multiple_comparisons(model, image_paths)
    
    else:
        # Interactive mode
        print("\nNo images provided as arguments. Starting interactive mode...")
        print("(You can also run: python test_cat_comparison.py image1.jpg image2.jpg)")
        interactive_mode(model)


if __name__ == "__main__":
    main()

