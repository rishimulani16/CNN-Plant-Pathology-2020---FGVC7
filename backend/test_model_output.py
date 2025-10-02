#!/usr/bin/env python3

import sys
import os
import numpy as np
from PIL import Image
import io

# Add backend to path
sys.path.append('/Users/rishi/Documents/plant1/backend')

def test_model_output():
    print("🔍 Testing your model's actual output format...")
    
    try:
        from model_predictor import ModelPredictor
        
        predictor = ModelPredictor()
        if not predictor.is_loaded():
            print("❌ Model not loaded")
            return
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        print("\n📊 Raw model prediction analysis:")
        
        # Check raw model output
        processed_image = predictor._preprocess_image(test_img)
        print(f"Preprocessed image shape: {processed_image.shape}")
        
        # Get raw predictions from model
        raw_predictions = predictor.model.predict(processed_image, verbose=0)
        print(f"Raw model output shape: {raw_predictions.shape}")
        print(f"Raw model output: {raw_predictions}")
        print(f"Raw model output type: {type(raw_predictions)}")
        
        # Check if it's probabilities or class indices
        if len(raw_predictions.shape) == 2 and raw_predictions.shape[1] > 1:
            print(f"Output appears to be probabilities for {raw_predictions.shape[1]} classes")
            print(f"Sum of probabilities: {np.sum(raw_predictions[0])}")
            print(f"Max probability: {np.max(raw_predictions[0])}")
            print(f"Predicted class index: {np.argmax(raw_predictions[0])}")
        else:
            print("Output might be single class prediction")
        
        # Test current prediction method
        print("\n🧪 Testing current prediction method:")
        try:
            result, confidence, all_predictions = predictor.predict_leaf(test_img)
            print(f"✅ Prediction successful!")
            print(f"Result: {result}")
            print(f"Confidence: {confidence}")
            print(f"All predictions: {all_predictions}")
        except Exception as e:
            print(f"❌ Current prediction method failed: {e}")
            import traceback
            traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_output()