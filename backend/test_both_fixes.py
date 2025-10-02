#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image
import io
import numpy as np
import time

def test_both_fixes():
    base_url = "http://localhost:8080"
    
    print("🔍 Testing both fixes...")
    
    # Wait for server to start
    print("⏳ Waiting for server to start...")
    time.sleep(5)
    
    # Test 1: Login functionality
    print("\n1️⃣ Testing LOGIN fix...")
    try:
        login_data = {
            "email": "test@example.com",
            "password": "password123"
        }
        response = requests.post(f"{base_url}/login", json=login_data, timeout=10)
        print(f"   Login status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Login successful!")
            print(f"   User: {result.get('user', {}).get('name', 'Unknown')}")
            print(f"   Token received: {len(result.get('token', ''))} chars")
        else:
            print(f"   ❌ Login failed: {response.text}")
    except Exception as e:
        print(f"   ❌ Login test failed: {e}")
    
    # Test 2: Disease prediction (no confidence scores)
    print("\n2️⃣ Testing PREDICTION fix (no confidence scores)...")
    try:
        # Create a test image
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
        
        print(f"   Prediction status: {response.status_code} (took {end_time - start_time:.2f}s)")
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Prediction successful!")
            print(f"   Disease class: {result.get('predicted_class', 'Unknown')}")
            print(f"   Timestamp: {result.get('timestamp', 'None')}")
            
            # Check that NO confidence scores are returned
            if 'confidence' not in result and 'predictions' not in result:
                print(f"   ✅ Confidence scores correctly removed!")
            else:
                print(f"   ⚠️ Confidence data still present: {result}")
                
        else:
            print(f"   ❌ Prediction failed: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Prediction test failed: {e}")
    
    print(f"\n🎯 Summary:")
    print(f"   • Login: Use test@example.com / password123")
    print(f"   • Disease prediction: Returns only class name (no confidence)")
    print(f"   • Available classes: scab, rust, multiple disease, healthy")

if __name__ == "__main__":
    test_both_fixes()