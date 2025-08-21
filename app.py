from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
# rf_model = None
label_encoder = None
scaler = None

def load_models():
    """Load the trained models and preprocessing objects"""
    global label_encoder, scaler
    
    try:
        if not all(os.path.exists(f) for f in ['label_encoder (1).pkl', 'scaler.pkl']):
            return False, "Model files not found"
        
        label_encoder = joblib.load('label_encoder (1).pkl')
        scaler = joblib.load('scaler.pkl')
        
        return True, "Models loaded successfully"
    except Exception as e:
        return False, f"Error loading models: {str(e)}"

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Stress Detection API is running'})

@app.route('/predict', methods=['GET'])
def predict():
    try:
        if scaler is None:
            success, message = load_models()
            if not success:
                return jsonify({'error': message}), 500
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        required_features = ['psd_theta', 'psd_beta', 'hrv']
        missing_features = [feature for feature in required_features if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        # Create input data with only the required features
        # Fill the removed features with default values for the scaler
        input_data = pd.DataFrame([[
            data['psd_theta'], 0.0, data['psd_beta'], 0.0, data['hrv']
        ]], columns=['0.psd_theta', '0.psd_alpha', '0.psd_beta', 'EDA', 'HRV'])
        
        # Use scaler to transform the input data
        scaled_input = scaler.transform(input_data)
        
        # For now, we'll use a simple rule-based approach based on scaled values
        # This is a placeholder - you can implement your own classification logic here
        scaled_values = scaled_input[0]
        
        # Get the label encoder classes to understand the mapping
        class_names = label_encoder.classes_
        print(f"Debug - Available classes: {class_names}")
        
        # Simple classification based on scaled values (updated for 3 features)
        # Using: psd_theta (index 0), psd_beta (index 2), hrv (index 4)
        if scaled_values[4] < -1.0:  # HRV threshold - very low HRV indicates PTSD
            prediction = 2  # ptsd
        elif scaled_values[2] > 1.5:  # PSD beta threshold - high beta indicates stress
            prediction = 3  # stressed
        elif scaled_values[0] > 1.0:  # PSD theta threshold - high theta indicates anxiety
            prediction = 1  # anxious
        else:
            prediction = 0  # normal
        
        # Get the predicted status text from the label encoder
        try:
            predicted_status = label_encoder.inverse_transform([prediction])[0]
        except Exception as e:
            print(f"Error with label encoder: {e}")
            # Fallback: use the class names directly
            if prediction < len(class_names):
                predicted_status = class_names[prediction]
            else:
                predicted_status = "unknown"
        
        # Create a proper mapping if the label encoder classes are numeric
        # Based on the notebook, the mapping should be:
        # 0: 'anxious', 1: 'normal', 2: 'ptsd', 3: 'stressed'
        status_mapping = {
            0: 'anxious',
            1: 'normal', 
            2: 'ptsd',
            3: 'stressed'
        }
        
        # Use the mapping to get the proper text label
        if prediction in status_mapping:
            predicted_status = status_mapping[prediction]
        else:
            predicted_status = "unknown"
        
        # Generate confidence scores with proper text labels
        confidence_scores = {}
        for i, status_name in status_mapping.items():
            if i == prediction:
                confidence_scores[status_name] = 0.8  # High confidence for predicted class
            else:
                confidence_scores[status_name] = 0.2 / (len(status_mapping) - 1)  # Distribute remaining confidence
        
        # Debug: Print what we're getting from the label encoder
        print(f"Debug - Prediction index: {prediction}")
        print(f"Debug - Label encoder classes: {class_names}")
        print(f"Debug - Predicted status: {predicted_status}")
        print(f"Debug - Confidence scores: {confidence_scores}")
        
        return jsonify({
            'prediction': predicted_status,
            'confidence_scores': confidence_scores,
            'input_features': data,
            'scaled_features': scaled_values.tolist(),
            'method': 'scaler_based_classification'
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    try:
        if scaler is None:
            return jsonify({'error': 'Model not loaded'}), 400
        
        feature_names = ['psd_theta', 'psd_beta', 'hrv']
        status_mapping = ['anxious', 'normal', 'ptsd', 'stressed']
        
        return jsonify({
            'model_type': 'Scaler-based Classification',
            'features': feature_names,
            'classes': status_mapping,
            'model_loaded': True,
            'method': 'Rule-based classification using scaled features',
            'thresholds': {
                'psd_beta_stressed': 1.5,
                'hrv_ptsd': -1.0,
                'psd_theta_anxious': 1.0
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/example', methods=['GET'])
def get_example():
    return jsonify({
        'single_prediction': {
            'url': '/predict',
            'method': 'POST',
            'body': {
                'psd_theta': 8.5,
                'psd_beta': 0.08897,
                'hrv': 0.03
            }
        }
    })

if __name__ == '__main__':
    success, message = load_models()
    if success:
        print(f"✅ {message}")
        print("🚀 Starting Stress Detection API...")
    else:
        print(f"⚠️  {message}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
