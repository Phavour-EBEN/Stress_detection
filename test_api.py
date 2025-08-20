import requests
import json
import time

def test_api():
    """Test the Stress Detection API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🧪 Testing Stress Detection API...")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1. Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:5000")
        return
    
    # Test 2: Model Info
    print("\n2. Testing Model Info...")
    try:
        response = requests.get(f"{base_url}/model_info")
        if response.status_code == 200:
            print("✅ Model info retrieved")
            info = response.json()
            print(f"   Model Type: {info.get('model_type', 'N/A')}")
            print(f"   Features: {info.get('features', [])}")
            print(f"   Classes: {info.get('classes', [])}")
        else:
            print(f"❌ Model info failed: {response.status_code}")
            print(f"   Error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        print(f"❌ Model info error: {e}")
    
    # Test 3: Example Usage
    print("\n3. Testing Example Usage...")
    try:
        response = requests.get(f"{base_url}/example")
        if response.status_code == 200:
            print("✅ Example usage retrieved")
            example = response.json()
            print(f"   Example data: {json.dumps(example['single_prediction']['body'], indent=2)}")
        else:
            print(f"❌ Example usage failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Example usage error: {e}")
    
    # Test 4: Prediction
    print("\n4. Testing Prediction...")
    test_data = {
        "psd_theta": 80.5,
        "psd_alpha": 6.355,
        "psd_beta": 8.08897,
        "eda": 80.054613,
        "hrv": 7.03
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data)
        if response.status_code == 200:
            print("✅ Prediction successful")
            result = response.json()
            print(f"   Predicted Status: {result['prediction']}")
            print(f"   Method: {result.get('method', 'N/A')}")
            print(f"   Confidence Scores:")
            for status, confidence in result['confidence_scores'].items():
                print(f"     {status}: {confidence:.3f}")
            if 'scaled_features' in result:
                print(f"   Scaled Features: {[f'{x:.3f}' for x in result['scaled_features']]}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            error = response.json().get('error', 'Unknown error')
            print(f"   Error: {error}")
            if "Model files not found" in error:
                print("   💡 Make sure to run the Jupyter notebook first to generate model files")
            elif "rf_model.pkl" in error:
                print("   💡 The API now uses scaler-based classification - only label_encoder.pkl and scaler.pkl are needed")
    except Exception as e:
        print(f"❌ Prediction error: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Testing completed!")

if __name__ == "__main__":
    test_api()
