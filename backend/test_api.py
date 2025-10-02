#!/usr/bin/env python3

import requests
import json
import time

def test_api():
    url = "http://localhost:8080/predict"
    headers = {"Content-Type": "application/json"}
    
    # Test with invalid data first
    print("Testing API with invalid data...")
    start_time = time.time()
    try:
        response = requests.post(url, 
                               headers=headers, 
                               json={"image": "invalid_base64"},
                               timeout=10)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Test with a simple valid base64 image (1x1 pixel PNG)
    print("Testing API with minimal valid image...")
    # Minimal 1x1 PNG image in base64
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    start_time = time.time()
    try:
        response = requests.post(url,
                               headers=headers,
                               json={"image": tiny_png_b64},
                               timeout=30)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f} seconds")
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_api()