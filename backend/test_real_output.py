#!/usr/bin/env python3

import sys
import os
import numpy as np
from PIL import Image

# Add backend to path
sys.path.append('/Users/rishi/Documents/plant1/backend')

def test_actual_model_output():
    print("🔍 Testing your model's ACTUAL output format...")
    
    try:
        from model_predictor import ModelPredictor
        
        predictor = ModelPredictor()
        if not predictor.is_loaded():
            print("❌ Model not loaded")
            return
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Check raw model output
        processed_image = predictor._preprocess_image(test_img)
        raw_predictions = predictor.model.predict(processed_image, verbose=0)
        
        print(f"Raw model output: {raw_predictions}")
        print(f"Raw model output shape: {raw_predictions.shape}")
        print(f"Raw model output dtype: {raw_predictions.dtype}")
        
        # Check what type of output this is
        if raw_predictions.shape[1] == 1:
            print("Model outputs single class index")
            predicted_class_idx = int(raw_predictions[0][0])
        elif raw_predictions.shape[1] == 4:
            if abs(np.sum(raw_predictions[0]) - 1.0) < 0.01:
                print("Model outputs probability distribution")
                predicted_class_idx = np.argmax(raw_predictions[0])
            else:
                print("Model outputs raw logits or scores")
                predicted_class_idx = np.argmax(raw_predictions[0])
        
        print(f"Predicted class index: {predicted_class_idx}")
        print(f"Predicted class: {predictor.class_names[predicted_class_idx] if predicted_class_idx < len(predictor.class_names) else 'Unknown'}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_actual_model_output()