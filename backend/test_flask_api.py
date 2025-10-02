#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import time

def test_flask_server():
    base_url = "http://localhost:8080"
    
    print("🔍 Testing Flask server connectivity...")
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Model status
    try:
        response = requests.get(f"{base_url}/model/status", timeout=5)
        print(f"✅ Model status: {response.status_code}")
        if response.status_code == 200:
            status = response.json()
            print(f"   Model loaded: {status.get('loaded', 'Unknown')}")
            print(f"   Total params: {status.get('total_params', 'Unknown')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Model status failed: {e}")
        return False
    
    # Test 3: Prediction with test image
    try:
        print("\n🧪 Testing prediction API...")
        
        # Create a simple test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        buffer = io.BytesIO()
        test_img.save(buffer, format='JPEG')
        img_data = base64.b64encode(buffer.getvalue()).decode()
        
        # Test prediction
        start_time = time.time()
        response = requests.post(f"{base_url}/predict", 
                               json={"image": img_data}, 
                               timeout=30)
        end_time = time.time()
        
        print(f"✅ Prediction test: {response.status_code} (took {end_time - start_time:.2f}s)")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   Predicted class: {result.get('predicted_class', 'Unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print("   ✅ Prediction API working correctly!")
            return True
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_flask_server()
    if success:
        print("\n🎉 Flask server is working correctly!")
        print("The 'Analysis failed' error should now be resolved.")
    else:
        print("\n❌ Flask server has issues that need to be fixed.")