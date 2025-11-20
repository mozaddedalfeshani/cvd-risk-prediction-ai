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
        data = request.json
        
        if not data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Handle both direct format and dual API format
        if 'patient_data' in data:
            # Dual API format: {"model_type": "full", "patient_data": {...}}
            model_type = data.get('model_type', 'full')
            patient_data = data.get('patient_data', {})
        else:
            # Direct format: {"Age": 45, "BMI": 25.5, ...}
            model_type = 'full'
            patient_data = data
        
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
        
        # Clinical interpretation
        if risk_level == 'LOW':
            clinical_interpretation = {
                'risk_category': 'Low Risk',
                'confidence_level': 'High',
                'recommendations': {
                    'recommendation': 'Continue current lifestyle with routine monitoring',
                    'follow_up': 'Annual cardiovascular assessment',
                    'lifestyle': 'Maintain healthy diet, regular exercise, and avoid smoking'
                }
            }
        elif risk_level == 'INTERMEDIARY':
            clinical_interpretation = {
                'risk_category': 'Moderate Risk',
                'confidence_level': 'Good',
                'recommendations': {
                    'recommendation': 'Enhanced screening and lifestyle modifications recommended',
                    'follow_up': 'Semi-annual cardiovascular monitoring',
                    'lifestyle': 'Structured exercise program, dietary counseling, stress management'
                }
            }
        else:  # HIGH
            clinical_interpretation = {
                'risk_category': 'High Risk',
                'confidence_level': 'Good',
                'recommendations': {
                    'recommendation': 'Immediate clinical evaluation and intervention required',
                    'follow_up': 'Quarterly monitoring with specialist consultation',
                    'lifestyle': 'Intensive lifestyle intervention, medication review, cardiac rehabilitation'
                }
            }
        
        result = {
            'model_used': {
                'type': model_type,
                'name': 'Full Accuracy Model' if model_type == 'full' else 'Quick Assessment Model',
                'accuracy': 0.95 if model_type == 'full' else 0.87,
                'features_used': 19 if model_type == 'full' else 8
            },
            'prediction': {
                'risk_level': risk_level,
                'risk_code': int(risk_code),
                'confidence': float(confidence),
                'probabilities': {
                    'LOW': float(probabilities[0]),
                    'INTERMEDIARY': float(probabilities[1]),
                    'HIGH': float(probabilities[2])
                }
            },
            'clinical_interpretation': clinical_interpretation
        }
        
        return jsonify({
            'success': True,
            'result': result,
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

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get available model options"""
    models = [
        {
            'id': 'full',
            'name': 'Full Accuracy Model',
            'description': 'Maximum accuracy with comprehensive assessment',
            'accuracy': '95.00%',
            'features': 20,
            'time_required': '5-7 minutes',
            'recommended_for': 'Comprehensive clinical assessment'
        },
        {
            'id': 'quick',
            'name': 'Quick Assessment Model', 
            'description': 'Fast screening with key risk factors',
            'accuracy': '87.00%',
            'features': 8,
            'time_required': '1-2 minutes',
            'recommended_for': 'Initial screening and triage'
        }
    ]
    
    return jsonify({
        'available_models': models,
        'default_model': 'full'
    })

@app.route('/api/features/<model_type>', methods=['GET'])
def get_model_features(model_type):
    """Get required features for specific model"""
    if model_type == 'full':
        features = [
            'Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI',
            'Systolic BP', 'Diastolic BP', 'Blood Pressure Category',
            'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)',
            'Fasting Blood Sugar (mg/dL)', 'Smoking Status', 'Diabetes Status',
            'Family History of CVD', 'Physical Activity Level',
            'Abdominal Circumference (cm)', 'Waist-to-Height Ratio', 'CVD Risk Score'
        ]
        model_info = {
            'name': 'Full Accuracy Model',
            'accuracy': '95.00%',
            'description': 'Comprehensive 19-feature assessment'
        }
        feature_categories = {
            'Demographics': ['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI'],
            'Vital Signs': ['Systolic BP', 'Diastolic BP', 'Blood Pressure Category'],
            'Lab Values': ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Estimated LDL (mg/dL)', 
                          'Fasting Blood Sugar (mg/dL)'],
            'Risk Factors': ['Smoking Status', 'Diabetes Status', 'Family History of CVD',
                           'Physical Activity Level'],
            'Additional Measurements': ['CVD Risk Score', 'Waist-to-Height Ratio', 
                                     'Abdominal Circumference (cm)']
        }
    else:  # quick
        features = [
            'Age', 'Sex', 'BMI', 'Systolic BP', 'Diastolic BP',
            'Total Cholesterol (mg/dL)', 'Smoking Status', 'CVD Risk Score'
        ]
        model_info = {
            'name': 'Quick Assessment Model',
            'accuracy': '87.00%',
            'description': 'Essential 8-feature screening'
        }
        feature_categories = {
            'Demographics': ['Age', 'Sex', 'BMI'],
            'Vital Signs': ['Systolic BP', 'Diastolic BP'],
            'Lab Values': ['Total Cholesterol (mg/dL)'],
            'Risk Assessment': ['Smoking Status', 'CVD Risk Score']
        }
    
    return jsonify({
        'model': model_info,
        'required_features': features,
        'feature_count': len(features),
        'categories': feature_categories
    })

@app.route('/api/example/<model_type>', methods=['GET'])
def get_example_data(model_type):
    """Get example data for specific model"""
    risk_type = request.args.get('risk', 'low')
    
    if model_type == 'full':
        examples = {
            'low': {
                'Sex': 0, 'Age': 30, 'Weight (kg)': 65.0, 'Height (m)': 1.68, 'BMI': 23.0,
                'Systolic BP': 110.0, 'Diastolic BP': 70.0, 'Blood Pressure Category': 1,
                'Total Cholesterol (mg/dL)': 175.0, 'HDL (mg/dL)': 65.0, 'Estimated LDL (mg/dL)': 100.0,
                'Fasting Blood Sugar (mg/dL)': 85.0, 'Smoking Status': 0, 'Diabetes Status': 0,
                'Family History of CVD': 0, 'Physical Activity Level': 2,
                'Abdominal Circumference (cm)': 75.0, 'Waist-to-Height Ratio': 0.45,
                'CVD Risk Score': 10.0
            },
            'high': {
                'Sex': 1, 'Age': 58, 'Weight (kg)': 95.0, 'Height (m)': 1.75, 'BMI': 31.0,
                'Systolic BP': 160.0, 'Diastolic BP': 95.0, 'Blood Pressure Category': 4,
                'Total Cholesterol (mg/dL)': 280.0, 'HDL (mg/dL)': 35.0, 'Estimated LDL (mg/dL)': 200.0,
                'Fasting Blood Sugar (mg/dL)': 145.0, 'Smoking Status': 1, 'Diabetes Status': 1,
                'Family History of CVD': 1, 'Physical Activity Level': 0,
                'Abdominal Circumference (cm)': 105.0, 'Waist-to-Height Ratio': 0.60,
                'CVD Risk Score': 25.0
            }
        }
    else:  # quick
        examples = {
            'low': {
                'Age': 30, 'Sex': 0, 'BMI': 22.0, 'Systolic BP': 110.0, 'Diastolic BP': 70.0,
                'Total Cholesterol (mg/dL)': 175.0, 'Smoking Status': 0, 'CVD Risk Score': 8.0
            },
            'high': {
                'Age': 58, 'Sex': 1, 'BMI': 31.0, 'Systolic BP': 160.0, 'Diastolic BP': 95.0,
                'Total Cholesterol (mg/dL)': 280.0, 'Smoking Status': 1, 'CVD Risk Score': 22.0
            }
        }
    
    return jsonify({
        'model_type': model_type,
        'risk_type': risk_type,
        'example_data': examples.get(risk_type, examples['low'])
    })

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