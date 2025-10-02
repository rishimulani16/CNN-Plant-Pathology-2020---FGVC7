#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/rishi/Documents/plant1/backend')

# Test the ModelPredictor singleton pattern
def test_singleton():
    print("Testing ModelPredictor singleton pattern...")
    
    try:
        from model_predictor import ModelPredictor
        
        # Create first instance
        print("Creating first instance...")
        predictor1 = ModelPredictor()
        print(f"First instance loaded: {predictor1.is_loaded()}")
        
        # Create second instance (should be same)
        print("Creating second instance...")
        predictor2 = ModelPredictor()
        print(f"Second instance loaded: {predictor2.is_loaded()}")
        
        # Check if they're the same object
        print(f"Same object: {predictor1 is predictor2}")
        print(f"Model is same: {predictor1.model is predictor2.model if predictor1.model and predictor2.model else 'N/A'}")
        
        if predictor1.is_loaded():
            print("✅ ModelPredictor singleton pattern working correctly!")
            
            # Test model info
            info = predictor1.get_model_info()
            print(f"Model info: {info}")
            
        else:
            print("❌ Model failed to load")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_singleton()