"""
Quick Start Script
==================
Run this script to quickly start training or testing the cat identity model.
"""

import os
import sys
from pathlib import Path


def check_requirements():
    """Check if all requirements are installed"""
    print("Checking requirements...")
    try:
        import tensorflow
        import numpy
        import pandas
        import sklearn
        import cv2
        import matplotlib
        import tqdm
        
        # Check NumPy version compatibility
        numpy_version = numpy.__version__
        major_version = int(numpy_version.split('.')[0])
        if major_version >= 2:
            print(f"⚠️  Warning: NumPy {numpy_version} detected. This may cause compatibility issues with TensorFlow.")
            print("   Consider downgrading to NumPy 1.x for better compatibility:")
            print("   pip install 'numpy<2.0.0'")
        
        print("✅ All requirements are installed!")
        return True
    except ImportError as e:
        print(f"❌ Missing requirement: {e}")
        print("\nPlease install requirements:")
        print("  pip install -r requirements.txt")
        return False
    except AttributeError as e:
        if "np.complex_" in str(e):
            print("❌ NumPy 2.0 compatibility error detected!")
            print("\nThis is a known issue with TensorFlow and NumPy 2.0.")
            print("Please run the fix script:")
            print("\n  python fix_numpy_compatibility.py")
            print("\nOr manually fix:")
            print("  pip install 'numpy>=1.21.0,<2.0.0' 'scipy>=1.9.0,<1.12.0'")
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            return False
    except ValueError as e:
        if "All ufuncs must have type `numpy.ufunc`" in str(e):
            print("❌ SciPy/NumPy compatibility error detected!")
            print("\nThis is a known issue with SciPy 1.12+ and NumPy 1.x.")
            print("Please run the fix script:")
            print("\n  python fix_numpy_compatibility.py")
            print("\nOr manually fix:")
            print("  pip install 'numpy>=1.21.0,<2.0.0' 'scipy>=1.9.0,<1.12.0'")
            return False
        else:
            print(f"❌ Unexpected error: {e}")
            return False


def check_dataset():
    """Check if dataset exists"""
    dataset_path = r"D:/Cursor AI projects/Capstone2.1/dataset_cropped/cats/"
    
    print(f"\nChecking dataset at: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset not found!")
        print("\nPlease ensure:")
        print("1. Original cat images are in: dataset/cats/")
        print("2. Run the face cropper: python cat_face_cropper.py")
        print("3. Cropped images will be in: dataset_cropped/cats/")
        return False
    
    # Count breed folders and images
    breed_folders = [f for f in Path(dataset_path).iterdir() if f.is_dir()]
    
    if len(breed_folders) == 0:
        print("❌ No breed folders found in dataset!")
        return False
    
    total_images = 0
    for breed_folder in breed_folders:
        images = list(breed_folder.glob("*.jpg")) + \
                list(breed_folder.glob("*.jpeg")) + \
                list(breed_folder.glob("*.png"))
        total_images += len(images)
    
    print(f"✅ Dataset found!")
    print(f"   - {len(breed_folders)} breeds")
    print(f"   - {total_images} images")
    
    return True


def show_menu():
    """Show main menu"""
    print("\n" + "="*80)
    print("CAT IDENTITY RECOGNITION - QUICK START")
    print("="*80)
    print("\nWhat would you like to do?")
    print("\n1. 🎓 Train Model (breed-based, 20 epochs)")
    print("2. 🎓 Quick Train (breed-based, 5 epochs)")
    print("3. 🧪 Test Model (compare images)")
    print("4. 📊 Analyze Results")
    print("5. 📚 View Examples")
    print("6. ❓ Help / Documentation")
    print("7. ⚙️  System Check")
    print("8. 🚪 Exit")
    print("\nNote: For individual cat training, run: python train_cat_identifier_v2.py")
    print("="*80)
    
    return input("\nEnter your choice (1-8): ").strip()


def train_model(quick=False):
    """Train the model"""
    print("\n" + "="*80)
    print("TRAINING MODEL")
    print("="*80)
    
    if not check_dataset():
        return
    
    if quick:
        print("\n🚀 Starting quick training (5 epochs)...")
        print("Note: For better results, use full training (20 epochs)")
    else:
        print("\n🚀 Starting full training (20 epochs)...")
        print("This will take approximately 1-2 hours on GPU, 3-5 hours on CPU")
    
    confirm = input("\nContinue? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Training cancelled.")
        return
    
    if quick:
        # Modify config for quick training
        import cat_identity_model_trainer as trainer
        trainer.Config.EPOCHS = 5
        trainer.Config.TRIPLETS_PER_EPOCH = 2000
        print("\n⚙️  Quick training mode: 5 epochs, 2000 triplets per epoch")
    
    # Run training
    from cat_identity_model_trainer import main
    main()


def test_model():
    """Test the model"""
    print("\n" + "="*80)
    print("TESTING MODEL")
    print("="*80)
    
    # Check for new v2 model first (individual cats)
    v2_model_path = Path(r"D:\Cursor AI projects\Capstone2.1\models\cat_identifier_efficientnet_v2.keras")
    v2_model_h5 = v2_model_path.with_suffix('.h5')
    
    # Check for old model (breed-based)
    old_model_path = Path('cat_identity_model.h5')
    
    if v2_model_path.exists() or v2_model_h5.exists():
        print("\n✅ Found individual cat model (v2)!")
        print("Starting interactive testing...")
        import test_cat_identifier_v2
        test_cat_identifier_v2.main()
    elif old_model_path.exists():
        print("\n✅ Found breed-based model (v1)!")
        print("Starting interactive testing...")
        import test_cat_comparison
        test_cat_comparison.main()
    else:
        print("❌ No trained model found!")
        print("\nAvailable models:")
        print(f"  - Individual cats (v2): {v2_model_path}")
        print(f"  - Breed-based (v1): {old_model_path}")
        print("\nPlease train a model first:")
        print("  - For individual cats: python train_cat_identifier_v2.py")
        print("  - For breed-based: python cat_identity_model_trainer.py")
        return


def analyze_results():
    """Analyze model results"""
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    if not os.path.exists('cat_embeddings.npz'):
        print("❌ Embeddings not found!")
        print("Please train the model first (option 1 or 2)")
        return
    
    print("\nEmbeddings found! Starting analysis...")
    
    import analyze_embeddings
    analyze_embeddings.main()


def view_examples():
    """View workflow examples"""
    print("\n" + "="*80)
    print("WORKFLOW EXAMPLES")
    print("="*80)
    
    import example_workflow
    example_workflow.main()


def show_help():
    """Show help and documentation"""
    print("\n" + "="*80)
    print("HELP & DOCUMENTATION")
    print("="*80)
    
    print("\n📚 Documentation Files:")
    print("-" * 80)
    
    files = [
        ('README.md', 'Main project documentation'),
        ('USAGE_GUIDE.md', 'Detailed usage guide'),
        ('requirements.txt', 'Python dependencies'),
    ]
    
    for filename, description in files:
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"{exists} {filename:20s} - {description}")
    
    print("\n📖 Quick Reference:")
    print("-" * 80)
    print("""
1. Training Workflow:
   - Prepare dataset (crop faces)
   - Train model (option 1 or 2)
   - Wait for training to complete
   - Model saved as 'cat_identity_model.h5'

2. Testing Workflow:
   - Load trained model
   - Compare two images
   - Get similarity score
   - Interpret results

3. Fine-tuning Workflow:
   - Collect individual cat photos
   - Organize in folders (one per cat)
   - Run fine-tuning
   - Test with new model

4. Files Generated:
   - cat_identity_model.h5 (trained model)
   - cat_embeddings.npz (image embeddings)
   - cat_metadata.csv (image metadata)
   - training_history.json (metrics)
   - training_history.png (plots)
    """)
    
    print("\n🔗 For more details, read:")
    print("   - README.md for overview")
    print("   - USAGE_GUIDE.md for step-by-step instructions")


def system_check():
    """Run system check"""
    print("\n" + "="*80)
    print("SYSTEM CHECK")
    print("="*80)
    
    print("\n1. Python Version:")
    print(f"   {sys.version}")
    
    print("\n2. Requirements:")
    check_requirements()
    
    print("\n3. Dataset:")
    check_dataset()
    
    print("\n4. GPU Availability:")
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"   ✅ {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print("   ⚠️  No GPU found, training will use CPU (slower)")
    except Exception as e:
        print(f"   ❌ Error checking GPU: {e}")
    
    print("\n5. Project Files:")
    project_files = [
        'cat_face_cropper.py',
        'cat_identity_model_trainer.py',
        'train_cat_identifier_v2.py',
        'test_cat_comparison.py',
        'test_cat_identifier_v2.py',
        'analyze_embeddings.py',
        'example_workflow.py',
        'requirements.txt',
        'README.md',
        'USAGE_GUIDE.md'
    ]
    
    for filename in project_files:
        exists = "✅" if os.path.exists(filename) else "❌"
        print(f"   {exists} {filename}")
    
    print("\n6. Generated Files:")
    generated_files = [
        ('cat_identity_model.h5', 'Breed-based model (v1)'),
        ('models/cat_identifier_efficientnet_v2.keras', 'Individual cat model (v2)'),
        ('models/cat_identifier_efficientnet_v2.h5', 'Individual cat model (v2, h5)'),
        ('cat_embeddings.npz', 'Embeddings'),
        ('cat_metadata.csv', 'Metadata'),
        ('training_history.json', 'Training history'),
        ('training_history.png', 'Training plots')
    ]
    
    for filename, description in generated_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ {filename:40s} ({size:.1f} MB) - {description}")
        else:
            print(f"   ⚠️  {filename:40s} (not generated yet) - {description}")
    
    print("\n" + "="*80)


def main():
    """Main function"""
    print("="*80)
    print("🐱 CAT IDENTITY RECOGNITION - QUICK START 🐱")
    print("="*80)
    print("\nWelcome! This tool will help you train a cat identity recognition model.")
    
    # Quick system check
    if not check_requirements():
        print("\n⚠️  Please install requirements first!")
        return
    
    # Main loop
    while True:
        choice = show_menu()
        
        if choice == '1':
            train_model(quick=False)
        
        elif choice == '2':
            train_model(quick=True)
        
        elif choice == '3':
            test_model()
        
        elif choice == '4':
            analyze_results()
        
        elif choice == '5':
            view_examples()
        
        elif choice == '6':
            show_help()
        
        elif choice == '7':
            system_check()
        
        elif choice == '8':
            print("\n" + "="*80)
            print("Thank you for using Cat Identity Recognition!")
            print("Happy cat detecting! 🐱✨")
            print("="*80)
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-8.")
        
        # Pause before showing menu again
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Goodbye! 👋")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

