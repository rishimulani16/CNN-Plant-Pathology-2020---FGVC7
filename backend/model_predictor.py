import os
import sys

# Configure TensorFlow settings to prevent segmentation faults
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory issues

import numpy as np
import tensorflow as tf
from PIL import Image

# Configure TensorFlow for better stability
try:
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception:
    pass  # In case these settings are not available

class ModelPredictor:
    _instance = None
    _model_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # Ensure initialization happens only once
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.model = None
        # Apple leaf disease classes - updated to match your actual model
        self.class_names = ['scab', 'rust', 'multiple disease', 'healthy']
        self.input_size = (224, 224)
        self.loaded = False
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load model safely
        try:
            self._load_model()
            self._initialized = True
            ModelPredictor._model_loaded = True
        except Exception as e:
            print(f"Failed to initialize ModelPredictor: {e}")
            import traceback
            traceback.print_exc()
            self.loaded = False
            self._initialized = True  # Mark as initialized even if failed
    
    def _load_model(self):
        """Load the user's trained model with better error handling"""
        # Check if already loaded to prevent double loading
        if ModelPredictor._model_loaded and self.model is not None:
            print("Model already loaded by singleton instance")
            return
            
        try:
            # Only look for the user's specific model file
            keras_model_path = os.path.join(self.script_dir, 'models/my_model.keras')
            
            if not os.path.exists(keras_model_path):
                raise FileNotFoundError(f"Model file not found: {keras_model_path}")
            
            print(f"Loading your trained model from {keras_model_path}...")
            
            # Use safe model loading with CPU to prevent GPU memory issues
            with tf.device('/CPU:0'):
                self.model = tf.keras.models.load_model(keras_model_path, compile=False)
            
            print(f"✅ Successfully loaded your model!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            print(f"Total parameters: {self.model.count_params():,}")
            
            # Update input size from model
            if len(self.model.input_shape) >= 3 and self.model.input_shape[1:3] != (None, None):
                self.input_size = self.model.input_shape[1:3]
                print(f"Updated input size to: {self.input_size}")
            
            # Verify model has correct number of outputs
            num_classes = self.model.output_shape[-1]
            if num_classes != len(self.class_names):
                print(f"Warning: Model has {num_classes} outputs but {len(self.class_names)} class names defined")
                # Optionally adjust class names if needed
                if num_classes < len(self.class_names):
                    self.class_names = self.class_names[:num_classes]
                    print(f"Adjusted class names to match model outputs: {self.class_names}")
            
            self.loaded = True
            
        except Exception as e:
            print(f"❌ Error loading your model: {e}")
            print(f"Exception type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            self.loaded = False
            raise  # Re-raise to handle in __init__

    def _preprocess_image(self, image):
        """Preprocess image for prediction (match training preprocessing)"""
        # Resize image to model input size
        image = image.resize(self.input_size)
        
        # Convert to array 
        image_array = tf.keras.utils.img_to_array(image)
        
        # Try different normalization approaches
        # Option 1: Standard normalization (0-1)
        image_array = image_array / 255.0
        
        # Option 2: ImageNet normalization (uncomment if needed)
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # image_array = (image_array / 255.0 - mean) / std
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.loaded or self.model is None:
            return {"loaded": False, "error": "Model not loaded"}
        
        try:
            return {
                "loaded": True,
                "input_shape": str(self.model.input_shape),
                "output_shape": str(self.model.output_shape),
                "class_names": self.class_names,
                "input_size": self.input_size,
                "total_params": self.model.count_params() if hasattr(self.model, 'count_params') else "Unknown"
            }
        except Exception as e:
            return {"loaded": False, "error": f"Error getting model info: {str(e)}"}

    def predict_leaf(self, image):
        """
        Predict leaf disease from PIL Image
        Returns: only the disease class name (no confidence scores)
        """
        if not self.loaded or self.model is None:
            raise Exception("Model not loaded")

        try:
            # Preprocess the image
            processed_image = self._preprocess_image(image)
            
            # Make prediction with CPU to ensure consistency
            with tf.device('/CPU:0'):
                predictions = self.model.predict(processed_image, verbose=0)
            
            # Get only the predicted class (no confidence)
            predicted_class_idx = np.argmax(predictions[0])
            
            # Map to class name (ensure index is within bounds)
            if predicted_class_idx < len(self.class_names):
                result = self.class_names[predicted_class_idx]
            else:
                result = f"Class_{predicted_class_idx}"
            
            # Return only the class name
            return result
            
        except Exception as e:
            print(f"Prediction error details: {str(e)}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Prediction error: {str(e)}")

    def is_loaded(self):
        """Check if model is loaded"""
        return self.loaded and self.model is not None