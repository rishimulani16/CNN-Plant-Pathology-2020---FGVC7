import numpy as np
from PIL import Image
import os
import json
import tensorflow as tf

class ModelPredictor:
    def __init__(self):
        self.model = None
        # Common apple leaf disease classes
        self.class_names = ['healthy', 'multiple_diseases', 'rust', 'scab']
        self.input_size = (224, 224)
        self.loaded = False
        self._load_model()
    
    def _create_dummy_model(self):
        """Create a dummy model for testing when TensorFlow is not available"""
        print("Creating dummy model for testing purposes...")
        return "dummy_model"
    
    def _load_model(self):
        """Load the trained model or create a dummy one"""
        try:
            # Check if TensorFlow is available
            try:
                import tensorflow as tf
                tensorflow_available = True
                print("TensorFlow is available")
            except ImportError:
                tensorflow_available = False
                print("TensorFlow not available - using dummy model")
            
            if tensorflow_available:
                # Define possible model and weights paths
                model_paths = [
                    'models/apple_leaf_model.h5',
                    'models/model.h5',
                    'models/apple_leaf_model.keras',
                    'models/model.keras'
                ]
                
                weights_paths = [
                    'models/my_weights.weights.h5',
                    'models/weights.h5',
                    'models/model_weights.h5'
                ]
                
                model_loaded = False
                
                # First try to load complete model files
                for model_path in model_paths:
                    if os.path.exists(model_path):
                        try:
                            print(f"Attempting to load complete model from {model_path}...")
                            
                            # Load the model based on file extension
                            if model_path.endswith('.h5'):
                                self.model = tf.keras.models.load_model(
                                    model_path,
                                    custom_objects=None,
                                    compile=True
                                )
                            else:
                                self.model = tf.keras.models.load_model(model_path)
                            
                            print(f"Successfully loaded complete model from {model_path}")
                            print(f"Model input shape: {self.model.input_shape}")
                            print(f"Model output shape: {self.model.output_shape}")
                            
                            # Update input size from model if different
                            if self.model.input_shape[1:3] != (None, None):
                                self.input_size = self.model.input_shape[1:3]
                                print(f"Updated input size to: {self.input_size}")
                            
                            model_loaded = True
                            break
                            
                        except Exception as e:
                            print(f"Failed to load complete model from {model_path}: {e}")
                            continue
                
                # If no complete model found, try to load weights into architecture
                if not model_loaded:
                    print("No complete model found. Checking for weights files...")
                    
                    for weights_path in weights_paths:
                        if os.path.exists(weights_path):
                            try:
                                print(f"Found weights file: {weights_path}")
                                print("Creating model architecture and loading weights...")
                                
                                # Create the model architecture
                                self.model = self._create_model_architecture(tf)
                                
                                # Load the weights
                                self.model.load_weights(weights_path)
                                
                                # Compile the model
                                self.model.compile(
                                    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                                    loss='categorical_crossentropy',
                                    metrics=['accuracy']
                                )
                                
                                print(f"Successfully loaded weights from {weights_path}")
                                print(f"Model input shape: {self.model.input_shape}")
                                print(f"Model output shape: {self.model.output_shape}")
                                
                                model_loaded = True
                                break
                                
                            except Exception as e:
                                print(f"Failed to load weights from {weights_path}: {e}")
                                continue
                
                if not model_loaded:
                    # Create new model with pre-trained weights as fallback
                    print("No trained model or weights found. Creating new model with ImageNet weights...")
                    self.model = self._create_model_architecture(tf)
                    
                    # Compile the model
                    self.model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                        loss='categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    print("Model created successfully (using ImageNet weights)")
            else:
                # TensorFlow not available - create dummy model
                self.model = self._create_dummy_model()
            
            self.loaded = True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
    
    def _create_model_architecture(self, tf):
        """Create the MobileNetV2 model architecture"""
        # Use proper TensorFlow 2.x imports
        MobileNetV2 = tf.keras.applications.MobileNetV2
        Model = tf.keras.models.Model
        GlobalAveragePooling2D = tf.keras.layers.GlobalAveragePooling2D
        Dense = tf.keras.layers.Dense
        Dropout = tf.keras.layers.Dropout
        
        # Load MobileNetV2 base
        base_model = MobileNetV2(
            include_top=False,
            input_shape=(224, 224, 3),
            weights='imagenet'
        )
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        output = Dense(4, activation='softmax')(x)  # 4 classes
        
        model = Model(inputs=base_model.input, outputs=output)
        return model
    
    def _preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image to model input size
        image = image.resize(self.input_size)
        
        # Convert to array and normalize
        image_array = tf.keras.utils.img_to_array(image)
        image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if not self.loaded or self.model is None:
            return {"loaded": False, "error": "Model not loaded"}
        
        try:
            # if self.model == "dummy_model":
            #     return {
            #         "loaded": True,
            #         "model_type": "dummy",
            #         "input_shape": f"(None, {self.input_size[0]}, {self.input_size[1]}, 3)",
            #         "output_shape": f"(None, {len(self.class_names)})",
            #         "class_names": self.class_names,
            #         "input_size": self.input_size,
            #         "note": "This is a dummy model for testing. Install TensorFlow for actual predictions."
            #     }
            # else:
                return {
                    "loaded": True,
                    "model_type": "tensorflow",
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
        Returns: (result_string, confidence_score, all_predictions)
        """
        if not self.loaded or self.model is None:
            raise Exception("Model not loaded")

        try:
            if self.model == "dummy_model":
                # Return dummy predictions for testing
                dummy_predictions = np.random.dirichlet([1, 1, 1, 1])  # Random probabilities that sum to 1
                predicted_class_idx = np.argmax(dummy_predictions)
                confidence = float(dummy_predictions[predicted_class_idx])
                result = self.class_names[predicted_class_idx]
                
                all_predictions = {
                    self.class_names[i]: float(dummy_predictions[i]) 
                    for i in range(len(self.class_names))
                }
                
                return result, confidence, all_predictions
            else:
                # Real TensorFlow prediction
                processed_image = self._preprocess_image(image)
                predictions = self.model.predict(processed_image, verbose=0)
                
                predicted_class_idx = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_class_idx])
                result = self.class_names[predicted_class_idx]
                
                all_predictions = {
                    self.class_names[i]: float(predictions[0][i]) 
                    for i in range(len(self.class_names))
                }
                
                return result, confidence, all_predictions
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def is_loaded(self):
        """Check if model is loaded"""
        return self.loaded and self.model is not None