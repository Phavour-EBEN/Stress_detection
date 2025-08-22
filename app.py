from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import requests
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Global variables for loaded models
# rf_model = None
label_encoder = None
scaler = None

# Firebase configuration from environment variables
FIREBASE_API_KEY = os.getenv('FIREBASE_API_KEY')
FIREBASE_PROJECT_ID = os.getenv('FIREBASE_PROJECT_ID')
FIREBASE_BASE_URL = f"https://{FIREBASE_PROJECT_ID}-default-rtdb.firebaseio.com" if FIREBASE_PROJECT_ID else None

def setup_firebase():
    """Setup Firebase configuration"""
    if not FIREBASE_API_KEY or not FIREBASE_PROJECT_ID:
        print("❌ Firebase configuration not found in environment variables")
        return False
    
    print(f"✅ Firebase configured for project: {FIREBASE_PROJECT_ID}")
    return True

def fetch_all_data(api_key: str, project_id: str) -> dict:
    """Fetch all data from Firebase Realtime Database"""
    base_url = f"https://{project_id}-default-rtdb.firebaseio.com"
    try:
        # Add .json to the end of the URL to get JSON response
        url = f"{base_url}/.json"
        params = {
            'auth': api_key
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {str(e)}")
        raise

def fetch_latest_data():
    """Fetch latest data from Firebase 'latest' table"""
    try:
        all_data = fetch_all_data(FIREBASE_API_KEY, FIREBASE_PROJECT_ID)
        if all_data and 'latest' in all_data:
            latest_data = all_data['latest']
            print(f"✅ Fetched latest data from Firebase: {latest_data}")
            return latest_data
        else:
            print("⚠️  Latest table not found or empty")
            return None
    except Exception as e:
        print(f"❌ Error fetching latest data: {str(e)}")
        return None

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
    return jsonify({
        'status': 'healthy', 
        'message': 'Stress Detection API is running',
        'models_loaded': all([label_encoder, scaler]),
        'firebase_configured': bool(FIREBASE_API_KEY and FIREBASE_PROJECT_ID)
    })


@app.route('/fetch_latest', methods=['GET'])
def get_latest_data():
    """Fetch latest data from Firebase"""
    data = fetch_latest_data()
    if data is None:
        return jsonify({'error': 'Failed to fetch latest data from Firebase'}), 500
    
    return jsonify({
        'data': data,
        'source': 'firebase_latest'
    })

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    """Make stress prediction from Firebase data or request body"""
    try:
        if scaler is None:
            success, message = load_models()
            if not success:
                return jsonify({'error': message}), 500
        
        # Determine data source
        if request.method == 'GET' or not request.is_json:
            # Fetch data from Firebase
            firebase_data = fetch_latest_data()
            if not firebase_data:
                return jsonify({'error': 'Failed to fetch data from Firebase'}), 500
            data = firebase_data
            data_source = 'firebase'
        else:
            # Get data from request body
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No data provided'}), 400
            data_source = 'request_body'
        
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
                confidence_scores[status_name] = 0.4  # High confidence for predicted class
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
            'input_features': {
                'psd_theta': data['psd_theta'],
                'psd_beta': data['psd_beta'],
                'hrv': data['hrv']
            },
            'scaled_features': scaled_values.tolist(),
            'method': 'scaler_based_classification',
            'data_source': data_source
        })
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500




if __name__ == '__main__':
    # Setup Firebase
    if setup_firebase():
        print("🚀 Starting Flask app with Firebase integration...")
    else:
        print("⚠️  Starting Flask app without Firebase integration...")
    
    # Load models
    success, message = load_models()
    if success:
        print(f"✅ {message}")
        print("🚀 Starting Stress Detection API...")
    else:
        print(f"⚠️  {message}")
    
    # Get Flask configuration from environment variables
    flask_host = os.getenv('FLASK_HOST', '0.0.0.0')
    flask_port = int(os.getenv('FLASK_PORT', 5000))
    flask_debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    app.run(debug=flask_debug, host=flask_host, port=flask_port)