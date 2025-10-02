#!/usr/bin/env python3
"""
Test script to verify model integration
"""

import sys
import os

def test_imports():
    """Test if all required imports work"""
    print("=== Testing Imports ===")
    
    try:
        import numpy as np
        print("✓ numpy imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import numpy: {e}")
        return False
    
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import TensorFlow: {e}")
        return False
    
    try:
        from PIL import Image
        print("✓ PIL imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PIL: {e}")
        return False
    
    return True

def test_model_integration():
    """Test the model predictor integration"""
    
    print("\n=== Model Integration Test ===")
    
    # Test imports first
    if not test_imports():
        print("Import test failed. Please install requirements: pip install -r requirements.txt")
        return False
    
    # Now try to import our model predictor
    try:
        sys.path.append('/Users/rishi/Documents/plant1/backend')
        from model_predictor import ModelPredictor
        from PIL import Image
        print("✓ ModelPredictor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import ModelPredictor: {e}")
        return False
    
    # Initialize model predictor
    print("\n1. Initializing ModelPredictor...")
    try:
        predictor = ModelPredictor()
        print(f"   ✓ ModelPredictor initialized")
        print(f"   ✓ Model loaded: {predictor.is_loaded()}")
    except Exception as e:
        print(f"   ✗ Failed to initialize ModelPredictor: {e}")
        return False
    
    # Get model info
    print("\n2. Getting model information...")
    try:
        model_info = predictor.get_model_info()
        print(f"   Model Info:")
        for key, value in model_info.items():
            print(f"     {key}: {value}")
    except Exception as e:
        print(f"   ✗ Failed to get model info: {e}")
    
    # Test with a dummy image
    print("\n3. Testing prediction with dummy image...")
    try:
        # Create a dummy RGB image
        dummy_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        
        if predictor.is_loaded():
            result, confidence, all_predictions = predictor.predict_leaf(dummy_image)
            print(f"   ✓ Prediction successful!")
            print(f"     Result: {result}")
            print(f"     Confidence: {confidence:.4f}")
            print(f"     All predictions:")
            for disease, prob in all_predictions.items():
                print(f"       {disease}: {prob:.4f}")
        else:
            print(f"   ⚠ Model not loaded, cannot test prediction")
            
    except Exception as e:
        print(f"   ✗ Prediction failed: {e}")
        return False
    
    print("\n=== Test Complete ===")
    return True

if __name__ == "__main__":
    try:
        test_model_integration()
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()