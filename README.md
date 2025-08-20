# Stress Detection API

A Flask-based REST API for psychological status prediction using machine learning models trained on physiological data.

## Overview

This API uses a scaler-based classification approach to predict psychological status based on physiological measurements:
- **PSD Theta** - Power Spectral Density in theta frequency band
- **PSD Alpha** - Power Spectral Density in alpha frequency band  
- **PSD Beta** - Power Spectral Density in beta frequency band
- **EDA** - Electrodermal Activity
- **HRV** - Heart Rate Variability

The model can classify individuals into four psychological states:
- `normal` - Normal psychological state
- `anxious` - Anxiety
- `stressed` - Stress
- `ptsd` - Post-Traumatic Stress Disorder

## Features

- Real-time psychological status prediction
- Confidence scores for each prediction
- Health check endpoint
- Model information endpoint
- Example usage endpoint
- CORS enabled for cross-origin requests

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Starting the API

```bash
python app.py
```

The API will start on `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```http
GET /health
```
Returns API status and health information.

#### 2. Model Information
```http
GET /model_info
```
Returns information about the loaded machine learning model.

#### 3. Example Usage
```http
GET /example
```
Returns example request formats for the prediction endpoints.

#### 4. Single Prediction
```http
POST /predict
Content-Type: application/json

{
    "psd_theta": 80.5,
    "psd_alpha": 6.355,
    "psd_beta": 8.08897,
    "eda": 80.054613,
    "hrv": 7.03
}
```

**Response:**
```json
{
    "prediction": "stressed",
    "confidence_scores": {
        "anxious": 0.067,
        "normal": 0.067,
        "ptsd": 0.067,
        "stressed": 0.8
    },
    "input_features": {
        "psd_theta": 80.5,
        "psd_alpha": 6.355,
        "psd_beta": 8.08897,
        "eda": 80.054613,
        "hrv": 7.03
    },
    "scaled_features": [0.5, -0.2, 0.1, 1.8, -0.3],
    "method": "scaler_based_classification"
}
```

## Model Files

The API requires the following model files to be present in the same directory:
- `label_encoder (1).pkl` - Label encoder for class names
- `scaler.pkl` - Standard scaler for feature normalization

**Note:** The API now uses a rule-based classification approach with scaled features instead of a pre-trained Random Forest model.

**Note:** These files are generated when running the Jupyter notebook (`Untitled5.ipynb`). Make sure to run the notebook first to generate these model files.

## Example Python Client

```python
import requests
import json

# API base URL
base_url = "http://localhost:5000"

# Example physiological data
data = {
    "psd_theta": 80.5,
    "psd_alpha": 6.355,
    "psd_beta": 8.08897,
    "eda": 80.054613,
    "hrv": 7.03
}

# Make prediction
response = requests.post(f"{base_url}/predict", json=data)
result = response.json()

print(f"Predicted Status: {result['prediction']}")
print(f"Confidence: {result['confidence_scores'][result['prediction']]:.2%}")
```

## Example cURL Usage

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "psd_theta": 80.5,
    "psd_alpha": 6.355,
    "psd_beta": 8.08897,
    "eda": 80.054613,
    "hrv": 7.03
  }'
```

## Data Format

### Input Features
All features should be numeric values:
- `psd_theta`: Float value for theta frequency power
- `psd_alpha`: Float value for alpha frequency power
- `psd_beta`: Float value for beta frequency power
- `eda`: Float value for electrodermal activity
- `hrv`: Float value for heart rate variability

### Output
- `prediction`: String indicating predicted psychological status
- `confidence_scores`: Dictionary with confidence scores for each class
- `input_features`: Echo of the input features for verification
- `scaled_features`: Array of scaled feature values used for classification
- `method`: Classification method used ("scaler_based_classification")

## Error Handling

The API returns appropriate HTTP status codes:
- `200` - Successful prediction
- `400` - Bad request (missing features, invalid data)
- `500` - Internal server error (model loading issues, prediction errors)

## Development

This API is based on the machine learning analysis in `Untitled5.ipynb`. The notebook includes:
- Data preprocessing and feature engineering
- Model training (Random Forest and SVM)
- Model evaluation and visualization
- Model serialization

The API uses the scaler and label encoder from the notebook to implement a rule-based classification system.

## License

This project is for educational and research purposes.

## Support

For issues or questions, please check the notebook for detailed implementation details or refer to the API documentation above.
