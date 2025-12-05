"""
NumPy Compatibility Fix Script
==============================
This script fixes the NumPy 2.0 compatibility issue with TensorFlow.
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
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False

def check_numpy_version():
    """Check current NumPy version"""
    try:
        import numpy
        version = numpy.__version__
        major_version = int(version.split('.')[0])
        print(f"Current NumPy version: {version}")
        return major_version
    except ImportError:
        print("NumPy not installed")
        return None

def check_scipy_version():
    """Check current SciPy version"""
    try:
        import scipy
        version = scipy.__version__
        major_version = int(version.split('.')[0])
        minor_version = int(version.split('.')[1])
        print(f"Current SciPy version: {version}")
        return major_version, minor_version
    except ImportError:
        print("SciPy not installed")
        return None, None

def fix_numpy_compatibility():
    """Fix NumPy and SciPy compatibility issues"""
    print("="*80)
    print("NUMPY & SCIPY COMPATIBILITY FIX")
    print("="*80)
    
    # Check current versions
    numpy_major = check_numpy_version()
    scipy_major, scipy_minor = check_scipy_version()
    
    # Fix NumPy if needed
    if numpy_major is None:
        print("Installing NumPy...")
        if not run_command("pip install 'numpy>=1.21.0,<2.0.0'", "Installing NumPy 1.x"):
            return False
    elif numpy_major >= 2:
        print(f"\n⚠️  NumPy 2.0+ detected (version {numpy_major})")
        print("This is incompatible with TensorFlow. Downgrading to NumPy 1.x...")
        
        # Uninstall current NumPy
        if not run_command("pip uninstall numpy -y", "Uninstalling current NumPy"):
            print("⚠️  Could not uninstall NumPy, trying to install compatible version anyway...")
        
        # Install compatible NumPy
        if not run_command("pip install 'numpy>=1.21.0,<2.0.0'", "Installing NumPy 1.x"):
            return False
    else:
        print("✅ NumPy version is compatible")
    
    # Fix SciPy if needed
    if scipy_major is None:
        print("Installing SciPy...")
        if not run_command("pip install 'scipy>=1.9.0,<1.12.0'", "Installing compatible SciPy"):
            return False
    elif scipy_major >= 1 and scipy_minor >= 12:
        print(f"\n⚠️  SciPy 1.12+ detected (version {scipy_major}.{scipy_minor})")
        print("This may be incompatible with NumPy 1.x. Downgrading SciPy...")
        
        # Uninstall current SciPy
        if not run_command("pip uninstall scipy -y", "Uninstalling current SciPy"):
            print("⚠️  Could not uninstall SciPy, trying to install compatible version anyway...")
        
        # Install compatible SciPy
        if not run_command("pip install 'scipy>=1.9.0,<1.12.0'", "Installing compatible SciPy"):
            return False
    else:
        print("✅ SciPy version is compatible")
    
    # Verify installations
    print("\nVerifying installations...")
    try:
        import numpy
        import scipy
        numpy_version = numpy.__version__
        scipy_version = scipy.__version__
        numpy_major = int(numpy_version.split('.')[0])
        scipy_major = int(scipy_version.split('.')[0])
        scipy_minor = int(scipy_version.split('.')[1])
        
        print(f"✅ NumPy {numpy_version} installed successfully")
        print(f"✅ SciPy {scipy_version} installed successfully")
        
        if numpy_major >= 2:
            print("❌ Still NumPy 2.0+, the fix didn't work")
            return False
        elif scipy_major >= 1 and scipy_minor >= 12:
            print("❌ Still SciPy 1.12+, the fix didn't work")
            return False
        else:
            print("✅ Both NumPy and SciPy versions are now compatible")
            return True
            
    except ImportError as e:
        print(f"❌ Installation failed: {e}")
        return False

def test_tensorflow_import():
    """Test if TensorFlow can be imported without errors"""
    print("\nTesting TensorFlow import...")
    try:
        import tensorflow as tf
        print("✅ TensorFlow imported successfully!")
        print(f"   TensorFlow version: {tf.__version__}")
        
        # Test basic functionality
        print("   Testing basic operations...")
        a = tf.constant([1, 2, 3])
        b = tf.constant([4, 5, 6])
        c = tf.add(a, b)
        print(f"   Basic operation test: {c.numpy()}")
        print("✅ TensorFlow is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ TensorFlow import failed: {e}")
        return False

def main():
    """Main function"""
    print("🐱 Cat Identity Recognition - NumPy Compatibility Fix")
    print("="*80)
    
    # Fix NumPy compatibility
    if not fix_numpy_compatibility():
        print("\n❌ Failed to fix NumPy compatibility")
        print("\nManual fix instructions:")
        print("1. pip uninstall numpy -y")
        print("2. pip install 'numpy>=1.21.0,<2.0.0'")
        print("3. pip install -r requirements.txt --force-reinstall")
        return False
    
    # Test TensorFlow
    if not test_tensorflow_import():
        print("\n❌ TensorFlow still not working")
        print("\nTry these additional steps:")
        print("1. pip install --upgrade pip")
        print("2. pip install --upgrade tensorflow")
        print("3. pip install -r requirements.txt --force-reinstall")
        return False
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! NumPy compatibility issue fixed!")
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
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Fix interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
