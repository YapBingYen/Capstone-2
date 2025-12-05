"""
Complete Compatibility Fix Script
=================================
This script fixes all known compatibility issues between TensorFlow, NumPy, and SciPy.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False

def check_versions():
    """Check current package versions"""
    packages = {}
    
    try:
        import numpy
        packages['numpy'] = numpy.__version__
    except ImportError:
        packages['numpy'] = None
    
    try:
        import scipy
        packages['scipy'] = scipy.__version__
    except ImportError:
        packages['scipy'] = None
    
    try:
        import tensorflow
        packages['tensorflow'] = tensorflow.__version__
    except ImportError:
        packages['tensorflow'] = None
    
    return packages

def fix_all_compatibility():
    """Fix all compatibility issues"""
    print("="*80)
    print("COMPLETE COMPATIBILITY FIX")
    print("="*80)
    
    # Check current versions
    print("\nChecking current package versions...")
    packages = check_versions()
    
    for pkg, version in packages.items():
        if version:
            print(f"  {pkg}: {version}")
        else:
            print(f"  {pkg}: Not installed")
    
    # Determine what needs fixing
    needs_fix = False
    
    # Check NumPy
    if packages['numpy']:
        numpy_major = int(packages['numpy'].split('.')[0])
        if numpy_major >= 2:
            print(f"\n⚠️  NumPy 2.0+ detected ({packages['numpy']})")
            print("   This is incompatible with TensorFlow 2.13")
            needs_fix = True
    else:
        print("\n⚠️  NumPy not installed")
        needs_fix = True
    
    # Check SciPy
    if packages['scipy']:
        scipy_parts = packages['scipy'].split('.')
        scipy_major = int(scipy_parts[0])
        scipy_minor = int(scipy_parts[1]) if len(scipy_parts) > 1 else 0
        if scipy_major >= 1 and scipy_minor >= 12:
            print(f"\n⚠️  SciPy 1.12+ detected ({packages['scipy']})")
            print("   This may be incompatible with NumPy 1.x")
            needs_fix = True
    else:
        print("\n⚠️  SciPy not installed")
        needs_fix = True
    
    if not needs_fix:
        print("\n✅ All packages appear to be compatible!")
        return True
    
    # Apply fixes
    print("\n" + "="*80)
    print("APPLYING FIXES")
    print("="*80)
    
    # Step 1: Uninstall problematic packages
    print("\nStep 1: Removing problematic packages...")
    run_command("pip uninstall numpy scipy -y", "Uninstalling NumPy and SciPy")
    
    # Step 2: Install compatible versions
    print("\nStep 2: Installing compatible versions...")
    if not run_command("pip install 'numpy>=1.21.0,<2.0.0'", "Installing compatible NumPy"):
        print("❌ Failed to install NumPy")
        return False
    
    if not run_command("pip install 'scipy>=1.9.0,<1.12.0'", "Installing compatible SciPy"):
        print("❌ Failed to install SciPy")
        return False
    
    # Step 3: Reinstall other packages that might be affected
    print("\nStep 3: Reinstalling related packages...")
    run_command("pip install --upgrade tensorflow", "Upgrading TensorFlow")
    run_command("pip install --upgrade scikit-learn", "Upgrading scikit-learn")
    run_command("pip install --upgrade pandas", "Upgrading pandas")
    
    # Step 4: Verify installation
    print("\nStep 4: Verifying installation...")
    packages_after = check_versions()
    
    print("\nAfter fix:")
    for pkg, version in packages_after.items():
        if version:
            print(f"  {pkg}: {version}")
        else:
            print(f"  {pkg}: Not installed")
    
    # Check if fixes worked
    numpy_ok = False
    scipy_ok = False
    
    if packages_after['numpy']:
        numpy_major = int(packages_after['numpy'].split('.')[0])
        if numpy_major < 2:
            numpy_ok = True
            print("✅ NumPy version is now compatible")
        else:
            print("❌ NumPy is still 2.0+")
    
    if packages_after['scipy']:
        scipy_parts = packages_after['scipy'].split('.')
        scipy_major = int(scipy_parts[0])
        scipy_minor = int(scipy_parts[1]) if len(scipy_parts) > 1 else 0
        if scipy_major < 1 or (scipy_major == 1 and scipy_minor < 12):
            scipy_ok = True
            print("✅ SciPy version is now compatible")
        else:
            print("❌ SciPy is still 1.12+")
    
    return numpy_ok and scipy_ok

def test_tensorflow_import():
    """Test if TensorFlow can be imported without errors"""
    print("\n" + "="*80)
    print("TESTING TENSORFLOW IMPORT")
    print("="*80)
    
    try:
        print("Importing TensorFlow...")
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__} imported successfully!")
        
        print("\nTesting basic operations...")
        # Test basic operations
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        result = c.numpy()
        print(f"✅ Basic operation test: {result}")
        
        print("\nTesting Keras import...")
        from tensorflow import keras
        print("✅ Keras imported successfully!")
        
        print("\nTesting image utilities...")
        from tensorflow import keras
        _ = keras.utils.load_img
        print("✅ Image utilities imported successfully!")
        
        print("\nTesting scipy integration...")
        from scipy import ndimage
        print("✅ SciPy integration working!")
        
        print("\n✅ All tests passed! TensorFlow is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        print("\nThis might be due to:")
        print("1. Incompatible package versions")
        print("2. Missing dependencies")
        print("3. System-specific issues")
        return False

def main():
    """Main function"""
    print("🐱 Cat Identity Recognition - Complete Compatibility Fix")
    print("="*80)
    
    # Fix compatibility issues
    if not fix_all_compatibility():
        print("\n❌ Failed to fix compatibility issues")
        print("\nManual fix instructions:")
        print("1. pip uninstall numpy scipy -y")
        print("2. pip install 'numpy>=1.21.0,<2.0.0'")
        print("3. pip install 'scipy>=1.9.0,<1.12.0'")
        print("4. pip install -r requirements.txt --force-reinstall")
        return False
    
    # Test TensorFlow
    if not test_tensorflow_import():
        print("\n❌ TensorFlow still not working after fixes")
        print("\nTry these additional steps:")
        print("1. Restart your terminal/command prompt")
        print("2. pip install --upgrade pip")
        print("3. pip install --upgrade tensorflow")
        print("4. Check for conflicting packages: pip list | findstr tensorflow")
        return False
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! All compatibility issues fixed!")
    print("="*80)
    print("\nYou can now run the cat identity recognition system:")
    print("  python quick_start.py")
    print("\nOr start training directly:")
    print("  python cat_identity_model_trainer.py")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            print("\n" + "="*80)
            print("❌ Fix failed. Please check the error messages above.")
            print("You may need to manually install compatible versions.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
