#!/usr/bin/env python3
"""
CVD Risk Prediction Backend Test Script
Tests the API with comprehensive test cases for different risk levels
"""

import json
import requests
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://54.160.253.120:5001"
HEALTH_ENDPOINT = f"{API_BASE_URL}/api/health"
PREDICT_ENDPOINT = f"{API_BASE_URL}/api/predict"

def test_health_endpoint():
    """Test the health endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Model Loaded: {data.get('model_loaded', 'N/A')}")
            print(f"   Model Info: {data.get('model_info', {})}")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_prediction_endpoint(patient_data: Dict[str, Any], expected_stage: str):
    """Test the prediction endpoint with patient data"""
    print(f"\n🔬 Testing {expected_stage} Case...")
    print(f"   Age: {patient_data['Age']}, BMI: {patient_data['BMI']}")
    print(f"   BP: {patient_data['Systolic BP']}/{patient_data['Diastolic BP']}")
    print(f"   CVD Risk Score: {patient_data['CVD Risk Score']}")
    
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            json=patient_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction Successful!")
            print(f"   Risk Level: {result['prediction']['risk_level']}")
            print(f"   Confidence: {result['prediction']['confidence']:.2%}")
            print(f"   Risk Score: {result['prediction']['risk_score']}")
            print(f"   Probabilities: {result['prediction']['probabilities']}")
            
            # Validate prediction makes sense
            predicted_risk = result['prediction']['risk_level']
            if expected_stage.lower() in predicted_risk.lower():
                print(f"✅ Risk Level Validation: Expected {expected_stage}, Got {predicted_risk}")
            else:
                print(f"⚠️  Risk Level Validation: Expected {expected_stage}, Got {predicted_risk}")
            
            return True
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction Error: {e}")
        return False

def run_comprehensive_tests():
    """Run comprehensive tests with all test cases"""
    print("🚀 Starting Comprehensive CVD Risk Prediction Tests")
    print("=" * 60)
    
    # Test health endpoint first
    if not test_health_endpoint():
        print("❌ Health check failed. Please ensure the backend is running.")
        return
    
    # Load test cases
    try:
        with open('test_cases.json', 'r') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("❌ test_cases.json not found!")
        return
    
    # Run tests for each case
    total_tests = len(test_data['test_cases'])
    passed_tests = 0
    
    for i, test_case in enumerate(test_data['test_cases'], 1):
        print(f"\n📊 Test Case {i}/{total_tests}: {test_case['stage']}")
        print("-" * 40)
        
        if test_prediction_endpoint(test_case['data'], test_case['stage']):
            passed_tests += 1
        
        # Small delay between tests
        time.sleep(0.5)
    
    # Summary
    print("\n" + "=" * 60)
    print("📈 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed_tests}/{total_tests}")
    print(f"❌ Failed: {total_tests - passed_tests}/{total_tests}")
    print(f"📊 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 All tests passed! Backend is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the backend configuration.")

def test_specific_case():
    """Test a specific high-risk case"""
    print("\n🎯 Testing Specific High-Risk Case...")
    
    high_risk_data = {
        "Abdominal Circumference (cm)": 100.0,
        "Age": 58,
        "BMI": 31.5,
        "Blood Pressure Category": 3,
        "CVD Risk Score": 36.0,
        "Diabetes Status": 1,
        "Diastolic BP": 90.0,
        "Estimated LDL (mg/dL)": 155.0,
        "Family History of CVD": 1,
        "Fasting Blood Sugar (mg/dL)": 130.0,
        "HDL (mg/dL)": 40.0,
        "Height (m)": 1.68,
        "Physical Activity Level": 0,
        "Sex": 1,
        "Smoking Status": 1,
        "Systolic BP": 150.0,
        "Total Cholesterol (mg/dL)": 240.0,
        "Waist-to-Height Ratio": 0.60,
        "Weight (kg)": 89.0
    }
    
    test_prediction_endpoint(high_risk_data, "High Risk")

if __name__ == "__main__":
    print("🏥 CVD Risk Prediction Backend Test Suite")
    print("By: Mir Mozadded Alfeshani Murad")
    print("Repository: https://github.com/mozaddedalfeshani/cvd-risk-prediction-ai")
    print()
    
    # Check if backend is running
    print("🔍 Checking if backend is running...")
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            print("✅ Backend is running!")
            run_comprehensive_tests()
        else:
            print("❌ Backend is not responding correctly")
    except requests.exceptions.RequestException:
        print("❌ Backend is not running!")
        print("💡 Please start the backend with: cd api && python app/app_simple.py")
        print("   Then run this test script again.") 