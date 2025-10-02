import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
sys.path.append('/Users/rishi/Documents/plant1/backend')
import numpy as np
from PIL import Image
from model_predictor import ModelPredictor

print('=== Testing Different Class Mappings ===')

# Possible class mappings to try
possible_mappings = [
    ['healthy', 'multiple_diseases', 'rust', 'scab'],  # Current
    ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy'],
    ['Black_rot', 'healthy', 'rust', 'scab'],
    ['Cedar_apple_rust', 'Apple_scab', 'Black_rot', 'healthy'],
    ['scab', 'rust', 'multiple_diseases', 'healthy'],
    ['Apple_scab', 'Apple_rust', 'Apple_black_rot', 'Apple_healthy']
]

predictor = ModelPredictor()

# Test with different images
test_colors = [
    ('Dark Brown (diseased)', (101, 67, 33)),
    ('Yellow-Brown (diseased)', (184, 134, 11)),
    ('Bright Green (healthy)', (34, 139, 34)),
    ('Dark spots (diseased)', (139, 90, 43))
]

print('Current mapping:', predictor.class_names)
print()

for name, color in test_colors:
    img = Image.new('RGB', (224, 224), color=color)
    result, conf, all_preds = predictor.predict_leaf(img)
    
    print(f'{name:25}: {result:15} (conf: {conf:.3f})')
    
    # Show what it would be with different mappings
    raw_preds = list(all_preds.values())
    max_idx = np.argmax(raw_preds)
    
    print(f'  Raw predictions: {[f"{p:.3f}" for p in raw_preds]}')
    print(f'  Max index: {max_idx}')
    
    print('  Alternative interpretations:')
    for i, mapping in enumerate(possible_mappings[1:3], 1):  # Show just 2 alternatives
        predicted_class = mapping[max_idx] if max_idx < len(mapping) else 'unknown'
        print(f'    Mapping {i}: {predicted_class}')
    print()