#!/usr/bin/env python3
"""
Simple CVD Risk Prediction Backend API (Mock Version)
====================================================

Flask backend API for serving mock CVD risk predictions 
to test the NextJS frontend functionality.

Author: AI Assistant
Date: 2025
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for NextJS frontend

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'model_info': {
            'model_type': 'XGBoost',
            'accuracy': 0.95,
            'feature_count': 20,
            'version': '1.0',
            'status': 'loaded'
        }
    })

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """Get model information"""
    return jsonify({
        'model_type': 'XGBoost',
        'accuracy': 0.95,
        'feature_count': 20,
        'features': [
            'Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI',
            'Systolic BP', 'Diastolic BP', 'Blood Pressure Category',
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
            'Fasting Blood Sugar (mg/dL)', 'Smoking Status', 'Diabetes Status',
            'Family History of CVD', 'Physical Activity Level',
            'Abdominal Circumference (cm)', 'Waist-to-Height Ratio', 'CVD Risk Score'
        ],
        'target_classes': ['LOW', 'INTERMEDIARY', 'HIGH'],
        'version': '1.0',
        'status': 'loaded'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict CVD risk for a patient"""
    try:
        # Get patient data from request
        patient_data = request.json
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Mock prediction based on simple risk factors
        age = patient_data.get('Age', 40)
        bp_systolic = patient_data.get('Systolic BP', 120)
        cholesterol = patient_data.get('Total Cholesterol (mg/dL)', 200)
        smoking = patient_data.get('Smoking Status', 0)
        diabetes = patient_data.get('Diabetes Status', 0)
        bmi = patient_data.get('BMI', 25)
        
        # Simple risk scoring
        risk_score = 0
        if age > 45: risk_score += 1
        if age > 60: risk_score += 1
        if bp_systolic > 140: risk_score += 2
        if bp_systolic > 160: risk_score += 1
        if cholesterol > 240: risk_score += 2
        if cholesterol > 200: risk_score += 1
        if smoking: risk_score += 2
        if diabetes: risk_score += 2
        if bmi > 30: risk_score += 1
        if bmi > 35: risk_score += 1
        
        # Determine risk level
        if risk_score <= 2:
            risk_level = 'LOW'
            risk_code = 0
            confidence = 0.85 + random.uniform(0, 0.1)
            probabilities = [confidence, (1-confidence)*0.7, (1-confidence)*0.3]
        elif risk_score <= 5:
            risk_level = 'INTERMEDIARY'
            risk_code = 1
            confidence = 0.75 + random.uniform(0, 0.15)
            probabilities = [(1-confidence)*0.3, confidence, (1-confidence)*0.7]
        else:
            risk_level = 'HIGH'
            risk_code = 2
            confidence = 0.80 + random.uniform(0, 0.15)
            probabilities = [(1-confidence)*0.2, (1-confidence)*0.8, confidence]
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p/total for p in probabilities]
        
        result = {
            'risk_level': risk_level,
            'risk_code': int(risk_code),
            'confidence': float(confidence),
            'probabilities': {
                'LOW': float(probabilities[0]),
                'INTERMEDIARY': float(probabilities[1]),
                'HIGH': float(probabilities[2])
            },
            'model_accuracy': 0.95,
            'risk_score': risk_score
        }
        
        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': '2025-01-08T23:09:00.000Z'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features', methods=['GET'])
def get_required_features():
    """Get list of required features for prediction"""
    features = [
        'Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI',
        'Systolic BP', 'Diastolic BP', 'Blood Pressure Category',
        'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
        'Fasting Blood Sugar (mg/dL)', 'Smoking Status', 'Diabetes Status',
        'Family History of CVD', 'Physical Activity Level',
        'Abdominal Circumference (cm)', 'Waist-to-Height Ratio', 'CVD Risk Score'
    ]
    
    # Organize features by category for better UX
    feature_categories = {
        'Demographics': [
            'Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI'
        ],
        'Vital Signs': [
            'Systolic BP', 'Diastolic BP', 'Blood Pressure Category'
        ],
        'Lab Values': [
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
            'Fasting Blood Sugar (mg/dL)'
        ],
        'Risk Factors': [
            'Smoking Status', 'Diabetes Status', 'Family History of CVD',
            'Physical Activity Level'
        ],
        'Additional Measurements': [
            'CVD Risk Score', 'Waist-to-Height Ratio', 'Abdominal Circumference (cm)'
        ]
    }
    
    return jsonify({
        'required_features': features,
        'feature_count': len(features),
        'categories': feature_categories
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
                'CVD Risk Score': 12.0
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
                'CVD Risk Score': 22.0
            }
        }
    }
    
    risk_type = request.args.get('type', 'low_risk')
    if risk_type not in example_patients:
        risk_type = 'low_risk'
    
    return jsonify(example_patients[risk_type])

if __name__ == '__main__':
    print("="*60)
    print("CVD RISK PREDICTION BACKEND API (MOCK VERSION)")
    print("="*60)
    print(f"\nStarting Flask server...")
    print(f"Mock model ready for predictions!")
    print(f"API available at: http://localhost:5001")
    print(f"Health check: http://localhost:5001/api/health")
    print(f"Features: http://localhost:5001/api/features")
    print(f"Example data: http://localhost:5001/api/example?type=low_risk")
    app.run(host='0.0.0.0', port=5001, debug=True)