#!/usr/bin/env python3
"""
Dual-Model CVD Risk Prediction API
==================================

Flask backend API with two model options:
1. Full Accuracy Model: 95.91% accuracy, 23 features
2. Quick Assessment Model: 86.79% accuracy, 8 features

Author: Mir Mozadded Alfeshani Murad
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load both models
FULL_MODEL_PATH = '/Users/murad/Developer/cvd-risk-prediction-ai/ml-models/models/cvd_full_xgb.pkl'
QUICK_MODEL_PATH = '/Users/murad/Developer/cvd-risk-prediction-ai/ml-models/models/cvd_quick_xgb.pkl'

full_model = None
quick_model = None

try:
    print(f"Looking for full model at: {os.path.abspath(FULL_MODEL_PATH)}")
    if os.path.exists(FULL_MODEL_PATH):
        full_model = joblib.load(FULL_MODEL_PATH)
        print("✅ Full Accuracy Model loaded (95.91%, 23 features)")
    else:
        print("⚠️  Full model not found")
        print(f"Current working directory: {os.getcwd()}")
        print("Available files:")
        if os.path.exists('../ml-models/models/'):
            print(os.listdir('../ml-models/models/'))
except Exception as e:
    print(f"⚠️  Error loading full model: {e}")

try:
    if os.path.exists(QUICK_MODEL_PATH):
        quick_model = joblib.load(QUICK_MODEL_PATH)
        print("✅ Quick Assessment Model loaded (86.79%, 8 features)")
    else:
        print("⚠️  Quick model not found")
except Exception as e:
    print(f"⚠️  Error loading quick model: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models': {
            'full_model': {
                'loaded': full_model is not None,
                'accuracy': full_model['accuracy'] if full_model else None,
                'features': len(full_model['feature_names']) if full_model else None
            },
            'quick_model': {
                'loaded': quick_model is not None,
                'accuracy': quick_model['accuracy'] if quick_model else None,
                'features': len(quick_model['feature_names']) if quick_model else None
            }
        },
        'version': '2.0-dual'
    })

@app.route('/api/models', methods=['GET'])
def get_available_models():
    """Get available model options"""
    models = []
    
    if full_model:
        models.append({
            'id': 'full',
            'name': 'Full Accuracy Model',
            'description': 'Maximum accuracy with comprehensive assessment',
            'accuracy': f"{full_model['accuracy']*100:.2f}%",
            'features': len(full_model['feature_names']),
            'time_required': '5-7 minutes',
            'recommended_for': 'Comprehensive clinical assessment'
        })
    
    if quick_model:
        models.append({
            'id': 'quick',
            'name': 'Quick Assessment Model', 
            'description': 'Fast screening with key risk factors',
            'accuracy': f"{quick_model['accuracy']*100:.2f}%",
            'features': len(quick_model['feature_names']),
            'time_required': '1-2 minutes',
            'recommended_for': 'Initial screening and triage'
        })
    
    return jsonify({
        'available_models': models,
        'default_model': 'full' if full_model else 'quick'
    })

@app.route('/api/features/<model_type>', methods=['GET'])
def get_model_features(model_type):
    """Get required features for specific model"""
    if model_type == 'full' and full_model:
        features = full_model['feature_names']
        model_info = {
            'name': 'Full Accuracy Model',
            'accuracy': f"{full_model['accuracy']*100:.2f}%",
            'description': 'Comprehensive 23-feature assessment'
        }
    elif model_type == 'quick' and quick_model:
        features = quick_model['feature_names']
        model_info = {
            'name': 'Quick Assessment Model',
            'accuracy': f"{quick_model['accuracy']*100:.2f}%",
            'description': 'Essential 8-feature screening'
        }
    else:
        return jsonify({'error': 'Model not available'}), 404
    
    # Categorize features
    feature_categories = {}
    
    if model_type == 'full':
        feature_categories = {
            'Demographics': [f for f in features if f in ['Sex', 'Age', 'Weight (kg)', 'Height (m)', 'BMI']],
            'Vital Signs': [f for f in features if 'BP' in f or 'Blood Pressure' in f],
            'Lab Values': [f for f in features if any(x in f for x in ['Cholesterol', 'HDL', 'LDL', 'Blood Sugar'])],
            'Risk Factors': [f for f in features if any(x in f for x in ['Smoking', 'Diabetes', 'Family History', 'Activity'])],
            'Additional': [f for f in features if f not in sum(feature_categories.values(), [])]
        }
    else:  # quick
        feature_categories = {
            'Demographics': ['Age', 'BMI'],
            'Vital Signs': ['Systolic BP', 'Diastolic BP'],
            'Lab Values': ['Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)'],
            'Risk Assessment': ['CVD Risk Score']
        }
    
    return jsonify({
        'model': model_info,
        'required_features': features,
        'feature_count': len(features),
        'categories': feature_categories
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict CVD risk using selected model"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        model_type = data.get('model_type', 'full')
        patient_data = data.get('patient_data', {})
        
        if not patient_data:
            return jsonify({'error': 'No patient data provided'}), 400
        
        # Select model
        if model_type == 'full' and full_model:
            model_artifact = full_model
            model_name = 'Full Accuracy Model'
        elif model_type == 'quick' and quick_model:
            model_artifact = quick_model
            model_name = 'Quick Assessment Model'
        else:
            return jsonify({'error': f'Model type "{model_type}" not available'}), 400
        
        # Extract features
        feature_names = model_artifact['feature_names']
        feature_values = []
        missing_features = []
        
        for feature in feature_names:
            if feature in patient_data:
                feature_values.append(float(patient_data[feature]))
            else:
                missing_features.append(feature)
        
        if missing_features:
            return jsonify({
                'error': 'Missing required features',
                'missing_features': missing_features
            }), 400
        
        # Make prediction
        model = model_artifact['model']
        scaler = model_artifact['scaler']
        
        X = pd.DataFrame([feature_values], columns=feature_names)
        X_scaled = scaler.transform(X)
        
        probabilities = model.predict_proba(X_scaled)[0]
        predicted_class = int(model.predict(X_scaled)[0])
        
        # Map to risk levels
        risk_levels = ['LOW', 'INTERMEDIARY', 'HIGH']
        risk_level = risk_levels[predicted_class]
        
        result = {
            'model_used': {
                'type': model_type,
                'name': model_name,
                'accuracy': model_artifact['accuracy'],
                'features_used': len(feature_names)
            },
            'prediction': {
                'risk_level': risk_level,
                'risk_code': predicted_class,
                'confidence': float(probabilities[predicted_class]),
                'probabilities': {
                    'LOW': float(probabilities[0]),
                    'INTERMEDIARY': float(probabilities[1]),
                    'HIGH': float(probabilities[2])
                }
            },
            'clinical_interpretation': get_clinical_interpretation(risk_level, probabilities[predicted_class])
        }
        
        return jsonify({
            'success': True,
            'result': result,
            'timestamp': pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_clinical_interpretation(risk_level, confidence):
    """Get clinical interpretation of results"""
    interpretations = {
        'LOW': {
            'recommendation': 'Routine monitoring and lifestyle maintenance',
            'follow_up': 'Annual cardiovascular screening',
            'lifestyle': 'Continue healthy lifestyle practices'
        },
        'INTERMEDIARY': {
            'recommendation': 'Enhanced monitoring and targeted interventions',
            'follow_up': 'Semi-annual cardiovascular assessment',
            'lifestyle': 'Lifestyle modifications and risk factor management'
        },
        'HIGH': {
            'recommendation': 'Immediate clinical intervention required',
            'follow_up': 'Frequent monitoring and specialist referral',
            'lifestyle': 'Comprehensive lifestyle changes and medical management'
        }
    }
    
    return {
        'risk_category': risk_level,
        'confidence_level': 'High' if confidence > 0.8 else 'Moderate' if confidence > 0.6 else 'Low',
        'recommendations': interpretations[risk_level]
    }

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
                'CVD Risk Score': 10.0, 'Cholesterol_HDL_Ratio': 2.7, 'LDL_HDL_Ratio': 1.5,
                'Multiple_Risk_Factors': 1, 'Pulse_Pressure': 40.0
            },
            'high': {
                'Sex': 1, 'Age': 58, 'Weight (kg)': 95.0, 'Height (m)': 1.75, 'BMI': 31.0,
                'Systolic BP': 160.0, 'Diastolic BP': 95.0, 'Blood Pressure Category': 4,
                'Total Cholesterol (mg/dL)': 280.0, 'HDL (mg/dL)': 35.0, 'Estimated LDL (mg/dL)': 200.0,
                'Fasting Blood Sugar (mg/dL)': 145.0, 'Smoking Status': 1, 'Diabetes Status': 1,
                'Family History of CVD': 1, 'Physical Activity Level': 0,
                'Abdominal Circumference (cm)': 105.0, 'Waist-to-Height Ratio': 0.60,
                'CVD Risk Score': 25.0, 'Cholesterol_HDL_Ratio': 8.0, 'LDL_HDL_Ratio': 5.7,
                'Multiple_Risk_Factors': 4, 'Pulse_Pressure': 65.0
            }
        }
    else:  # quick
        examples = {
            'low': {
                'CVD Risk Score': 8.0, 'Age': 30, 'Systolic BP': 110.0,
                'Fasting Blood Sugar (mg/dL)': 85.0, 'BMI': 22.0,
                'Total Cholesterol (mg/dL)': 175.0, 'Diastolic BP': 70.0, 'HDL (mg/dL)': 65.0
            },
            'high': {
                'CVD Risk Score': 22.0, 'Age': 58, 'Systolic BP': 160.0,
                'Fasting Blood Sugar (mg/dL)': 145.0, 'BMI': 31.0,
                'Total Cholesterol (mg/dL)': 280.0, 'Diastolic BP': 95.0, 'HDL (mg/dL)': 35.0
            }
        }
    
    return jsonify({
        'model_type': model_type,
        'risk_type': risk_type,
        'example_data': examples.get(risk_type, examples['low'])
    })

if __name__ == '__main__':
    print("="*60)
    print("DUAL-MODEL CVD RISK PREDICTION API")
    print("="*60)
    print(f"Full Model: {'✅ Loaded' if full_model else '❌ Not available'}")
    print(f"Quick Model: {'✅ Loaded' if quick_model else '❌ Not available'}")
    print(f"API available at: http://localhost:5001")
    print(f"Model options: /api/models")
    print(f"Health check: /api/health")
    app.run(host='0.0.0.0', port=5001, debug=True)
