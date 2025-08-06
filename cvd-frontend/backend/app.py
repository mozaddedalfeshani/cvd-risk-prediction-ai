#!/usr/bin/env python3
"""
CVD Risk Prediction Backend API
==============================

Flask backend API for serving the CVD risk prediction model
to the NextJS frontend.

Author: AI Assistant
Date: 2024
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for NextJS frontend

# Global model variable
model_package = None
model_info = {}

def load_model():
    """Load the trained CVD model"""
    global model_package, model_info
    
    # Look for the production model file
    model_paths = [
        '../docs/models/cvd_model_production.pkl',
        '../docs/models/MymensingUniversity_ML_Ready_best_model.pkl',
        '../docs/models/MymensingUniversity_ML_Ready_xgboost_model.pkl'
    ]
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                print(f"Loading model from: {model_path}")
                model_package = joblib.load(model_path)
                
                # Extract model information
                metadata = model_package['metadata']
                model_info = {
                    'model_type': metadata['model_type'],
                    'accuracy': metadata['accuracy'],
                    'feature_count': len(metadata['feature_names']),
                    'features': metadata['feature_names'],
                    'target_classes': ['LOW', 'INTERMEDIARY', 'HIGH'],
                    'version': '1.0',
                    'status': 'loaded'
                }
                
                print(f"✅ Model loaded successfully!")
                print(f"   Model type: {model_info['model_type']}")
                print(f"   Accuracy: {model_info['accuracy']:.4f} ({model_info['accuracy']*100:.2f}%)")
                print(f"   Features: {model_info['feature_count']}")
                return True
                
            except Exception as e:
                print(f"❌ Error loading model from {model_path}: {e}")
                continue
    
    print("❌ No valid model found!")
    return False

def predict_risk(patient_data):
    """Predict CVD risk for a patient"""
    try:
        if model_package is None:
            return None, "Model not loaded"
        
        model = model_package['model']
        metadata = model_package['metadata']
        scaler = metadata['scaler']
        feature_names = metadata['feature_names']
        
        # Create DataFrame from patient data
        patient_df = pd.DataFrame([patient_data])
        
        # Select required features
        missing_features = []
        for feature in feature_names:
            if feature not in patient_df.columns:
                missing_features.append(feature)
        
        if missing_features:
            return None, f"Missing required features: {missing_features}"
        
        X_selected = patient_df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(X_selected)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        # Format result
        risk_levels = ['LOW', 'INTERMEDIARY', 'HIGH']
        risk_level = risk_levels[prediction]
        confidence = probabilities[prediction]
        
        result = {
            'risk_level': risk_level,
            'risk_code': int(prediction),
            'confidence': float(confidence),
            'probabilities': {
                'LOW': float(probabilities[0]),
                'INTERMEDIARY': float(probabilities[1]),
                'HIGH': float(probabilities[2])
            },
            'model_accuracy': metadata['accuracy']
        }
        
        return result, None
        
    except Exception as e:
        return None, str(e)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_package is not None,
        'model_info': model_info
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    if model_package is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(model_info)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict CVD risk for a patient"""
    try:
        if model_package is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get patient data from request
        patient_data = request.json
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Make prediction
        result, error = predict_risk(patient_data)
        
        if error:
            return jsonify({'error': error}), 400
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_required_features():
    """Get list of required features for prediction"""
    if model_package is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    metadata = model_package['metadata']
    features = metadata['feature_names']
    
    # Organize features by category for better UX
    feature_categories = {
        'Demographics': [
            'Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI'
        ],
        'Vital Signs': [
            'Systolic BP', 'Diastolic BP', 'Pulse_Pressure', 'Blood Pressure Category'
        ],
        'Lab Values': [
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
            'Fasting Blood Sugar (mg/dL)', 'Cholesterol_HDL_Ratio', 'LDL_HDL_Ratio'
        ],
        'Risk Factors': [
            'Smoking Status', 'Diabetes Status', 'Family History of CVD',
            'Physical Activity Level', 'Multiple_Risk_Factors'
        ],
        'Derived Metrics': [
            'CVD Risk Score', 'Waist-to-Height Ratio', 'Abdominal Circumference (cm)',
            'Age_Group', 'BMI_Category'
        ]
    }
    
    # Filter only required features
    required_by_category = {}
    for category, feature_list in feature_categories.items():
        required_features = [f for f in feature_list if f in features]
        if required_features:
            required_by_category[category] = required_features
    
    return jsonify({
        'required_features': features,
        'feature_count': len(features),
        'categories': required_by_category
    })

@app.route('/api/example', methods=['GET'])
def get_example_patient():
    """Get example patient data for testing"""
    example_patients = {
        'low_risk': {
            'name': 'Low Risk Patient',
            'data': {
                'Sex': 0,  # Female
                'Age': 30,
                'Weight (kg)': 65.0,
                'Height (m)': 1.68,
                'BMI': 23.0,
                'Abdominal Circumference (cm)': 75.0,
                'Total Cholesterol (mg/dL)': 180.0,
                'HDL (mg/dL)': 65.0,
                'Fasting Blood Sugar (mg/dL)': 90.0,
                'Smoking Status': 0,  # No
                'Diabetes Status': 0,  # No
                'Physical Activity Level': 2,  # High
                'Family History of CVD': 0,  # No
                'Waist-to-Height Ratio': 0.45,
                'Systolic BP': 110.0,
                'Diastolic BP': 70.0,
                'Blood Pressure Category': 1,  # Normal
                'Estimated LDL (mg/dL)': 100.0,
                'CVD Risk Score': 12.0,
                'Pulse_Pressure': 40.0,
                'Cholesterol_HDL_Ratio': 2.8,
                'LDL_HDL_Ratio': 1.5,
                'Age_Group': 1,  # 25-34
                'BMI_Category': 2,  # Normal
                'Multiple_Risk_Factors': 0
            }
        },
        'high_risk': {
            'name': 'High Risk Patient',
            'data': {
                'Sex': 1,  # Male
                'Age': 55,
                'Weight (kg)': 95.0,
                'Height (m)': 1.75,
                'BMI': 31.0,
                'Abdominal Circumference (cm)': 105.0,
                'Total Cholesterol (mg/dL)': 280.0,
                'HDL (mg/dL)': 35.0,
                'Fasting Blood Sugar (mg/dL)': 145.0,
                'Smoking Status': 1,  # Yes
                'Diabetes Status': 1,  # Yes
                'Physical Activity Level': 0,  # Low
                'Family History of CVD': 1,  # Yes
                'Waist-to-Height Ratio': 0.60,
                'Systolic BP': 160.0,
                'Diastolic BP': 95.0,
                'Blood Pressure Category': 4,  # Hypertension Stage 2
                'Estimated LDL (mg/dL)': 200.0,
                'CVD Risk Score': 22.0,
                'Pulse_Pressure': 65.0,
                'Cholesterol_HDL_Ratio': 8.0,
                'LDL_HDL_Ratio': 5.7,
                'Age_Group': 4,  # 50-59
                'BMI_Category': 3,  # Overweight
                'Multiple_Risk_Factors': 3
            }
        }
    }
    
    risk_type = request.args.get('type', 'low_risk')
    if risk_type not in example_patients:
        risk_type = 'low_risk'
    
    return jsonify(example_patients[risk_type])

if __name__ == '__main__':
    print("="*60)
    print("CVD RISK PREDICTION BACKEND API")
    print("="*60)
    
    # Load the model
    if load_model():
        print(f"\n🚀 Starting Flask server...")
        print(f"📊 Model ready for predictions!")
        print(f"🌐 API will be available at: http://localhost:5000")
        print(f"🔗 Health check: http://localhost:5000/api/health")
        print(f"📋 Features: http://localhost:5000/api/features")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("❌ Failed to load model. Server not started.")
        print("Please ensure the model file exists in the correct location.")