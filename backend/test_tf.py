#!/usr/bin/env python3
"""
Simple test to check TensorFlow import
"""

print("Testing TensorFlow import...")

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    
    import tensorflow as tf
    print(f"✓ TensorFlow {tf.__version__} imported successfully")
    
    # Test basic functionality
    print("Testing basic TensorFlow operations...")
    x = tf.constant([1, 2, 3])
    print(f"✓ Created tensor: {x}")
    
    print("Testing Keras import...")
    from tensorflow import keras
    print("✓ Keras imported successfully")
    
    print("All TensorFlow tests passed!")
    
except Exception as e:
    print(f"✗ TensorFlow test failed: {e}")
    import traceback
    traceback.print_exc()