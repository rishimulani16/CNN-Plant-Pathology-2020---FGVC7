# Model Integration Guide

## How to Integrate Your .h5 Model

### Step 1: Place Your Model File
Place your trained `.h5` model file in the `backend/models/` directory with one of these names:
- `apple_leaf_model.h5` (recommended)
- `model.h5`

### Step 2: Install Dependencies
Navigate to the backend directory and install required packages:
```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Test Model Integration
Run the test script to verify your model loads correctly:
```bash
cd backend
python test_model.py
```

### Step 4: Start the Backend Server
```bash
cd backend
python app.py
```

## Model Requirements

Your `.h5` model should:
- Accept input images of shape `(224, 224, 3)` or `(None, None, 3)`
- Output predictions for 4 classes: `['healthy', 'multiple_diseases', 'rust', 'scab']`
- Be a compiled Keras model with softmax output

## API Endpoints

### Check Model Status
```
GET /model/status
```
Returns detailed information about the loaded model.

### Make Predictions
```
POST /predict
```
Upload an image and get disease predictions with confidence scores.

## Supported Model Formats

The system will automatically detect and load:
- `.h5` files (HDF5 format)
- `.keras` files (Keras SavedModel format)

## Troubleshooting

1. **Model not loading**: Check that your `.h5` file is placed in `backend/models/`
2. **Prediction errors**: Ensure your model expects RGB images of size 224x224
3. **Dependencies missing**: Run `pip install -r requirements.txt`

## Model File Structure Expected

```
backend/
├── models/
│   └── apple_leaf_model.h5  # Your trained model here
├── app.py
├── model_predictor.py
└── requirements.txt
```