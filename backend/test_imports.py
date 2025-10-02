#!/usr/bin/env python3

import os
import sys

# Configure TensorFlow settings to prevent segmentation faults
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress all TensorFlow logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

print("Testing basic imports...")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except Exception as e:
    print(f"❌ NumPy import failed: {e}")
    sys.exit(1)

try:
    print("Importing TensorFlow...")
    import tensorflow as tf
    print(f"✅ TensorFlow imported successfully - version: {tf.__version__}")
    
    # Test basic TensorFlow operation
    print("Testing basic TensorFlow operation...")
    with tf.device('/CPU:0'):
        x = tf.constant([1, 2, 3])
        y = tf.constant([4, 5, 6])
        result = tf.add(x, y)
        print(f"✅ TensorFlow operation successful: {result.numpy()}")
        
except Exception as e:
    print(f"❌ TensorFlow import/operation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from PIL import Image
    print("✅ PIL imported successfully")
except Exception as e:
    print(f"❌ PIL import failed: {e}")
    sys.exit(1)

print("All imports successful! Testing model loading...")

try:
    model_path = os.path.join(os.path.dirname(__file__), 'models/my_model.keras')
    if os.path.exists(model_path):
        print(f"Found model at: {model_path}")
        print("Attempting to load model...")
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(model_path)
        print(f"✅ Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
    else:
        print(f"❌ Model file not found: {model_path}")
        
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()

print("Test completed!")