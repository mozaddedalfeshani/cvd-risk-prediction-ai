#!/usr/bin/env python3
"""
CVD Model Usage Example
======================

This script demonstrates how to use the generated CVD risk prediction models
for making predictions on new patient data.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import joblib
import numpy as np
from pathlib import Path

def load_model(model_path):
    """
    Load a saved CVD model
    """
    try:
        model_package = joblib.load(model_path)
        print(f"✅ Model loaded: {model_path}")
        print(f"   Model type: {model_package['metadata']['model_type']}")
        print(f"   Accuracy: {model_package['metadata']['accuracy']:.4f} ({model_package['metadata']['accuracy']*100:.2f}%)")
        print(f"   Created: {model_package['metadata']['created_at']}")
        return model_package
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def predict_patient_risk(model_package, patient_data):
    """
    Predict CVD risk for a patient
    """
    try:
        model = model_package['model']
        metadata = model_package['metadata']
        scaler = metadata['scaler']
        feature_names = metadata['feature_names']
        
        # Prepare patient data
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data
        
        # Select required features
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
            'risk_code': prediction,
            'confidence': confidence,
            'probabilities': {
                'LOW': probabilities[0],
                'INTERMEDIARY': probabilities[1],
                'HIGH': probabilities[2]
            },
            'model_accuracy': metadata['accuracy']
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def main():
    """
    Example usage of the CVD models
    """
    print("="*80)
    print("CVD RISK PREDICTION - MODEL USAGE EXAMPLE")
    print("="*80)
    
    # Load the best performing model (MymensingUniversity dataset)
    model_path = "MymensingUniversity_ML_Ready_best_model.pkl"
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        print("Please run the model generator first!")
        return
    
    model_package = load_model(model_path)
    if not model_package:
        return
    
    print(f"\n📋 Required features for prediction:")
    feature_names = model_package['metadata']['feature_names']
    for i, feature in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feature}")
    
    print(f"\n🧪 Example Predictions:")
    print("="*50)
    
    # Example 1: Low Risk Patient
    print("\n👤 Patient 1 (Low Risk Profile):")
    patient1 = {
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
    
    result1 = predict_patient_risk(model_package, patient1)
    if result1:
        print(f"   🎯 Predicted Risk: {result1['risk_level']}")
        print(f"   📊 Confidence: {result1['confidence']:.3f}")
        print(f"   📈 Probabilities:")
        for risk, prob in result1['probabilities'].items():
            print(f"      {risk}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Example 2: High Risk Patient
    print("\n👤 Patient 2 (High Risk Profile):")
    patient2 = {
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
    
    result2 = predict_patient_risk(model_package, patient2)
    if result2:
        print(f"   🎯 Predicted Risk: {result2['risk_level']}")
        print(f"   📊 Confidence: {result2['confidence']:.3f}")
        print(f"   📈 Probabilities:")
        for risk, prob in result2['probabilities'].items():
            print(f"      {risk}: {prob:.3f} ({prob*100:.1f}%)")
    
    # Example 3: Intermediate Risk Patient
    print("\n👤 Patient 3 (Intermediate Risk Profile):")
    patient3 = {
        'Sex': 0,  # Female
        'Age': 42,
        'Weight (kg)': 78.0,
        'Height (m)': 1.65,
        'BMI': 28.6,
        'Abdominal Circumference (cm)': 88.0,
        'Total Cholesterol (mg/dL)': 220.0,
        'HDL (mg/dL)': 52.0,
        'Fasting Blood Sugar (mg/dL)': 115.0,
        'Smoking Status': 0,  # No
        'Diabetes Status': 0,  # No
        'Physical Activity Level': 1,  # Moderate
        'Family History of CVD': 1,  # Yes
        'Waist-to-Height Ratio': 0.53,
        'Systolic BP': 135.0,
        'Diastolic BP': 85.0,
        'Blood Pressure Category': 3,  # Hypertension Stage 1
        'Estimated LDL (mg/dL)': 140.0,
        'CVD Risk Score': 17.5,
        'Pulse_Pressure': 50.0,
        'Cholesterol_HDL_Ratio': 4.2,
        'LDL_HDL_Ratio': 2.7,
        'Age_Group': 3,  # 40-49
        'BMI_Category': 3,  # Overweight
        'Multiple_Risk_Factors': 1
    }
    
    result3 = predict_patient_risk(model_package, patient3)
    if result3:
        print(f"   🎯 Predicted Risk: {result3['risk_level']}")
        print(f"   📊 Confidence: {result3['confidence']:.3f}")
        print(f"   📈 Probabilities:")
        for risk, prob in result3['probabilities'].items():
            print(f"      {risk}: {prob:.3f} ({prob*100:.1f}%)")
    
    print(f"\n" + "="*80)
    print("USAGE NOTES:")
    print("="*80)
    print("1. Ensure all required features are provided")
    print("2. Use the correct data types and ranges for each feature")
    print("3. Categorical variables should be properly encoded:")
    print("   - Sex: 0=Female, 1=Male")
    print("   - Yes/No fields: 0=No, 1=Yes") 
    print("   - Activity Level: 0=Low, 1=Moderate, 2=High")
    print("   - BP Category: 1=Normal, 2=Elevated, 3=Stage1, 4=Stage2")
    print("4. The model provides probability scores for all risk levels")
    print("5. Higher confidence scores indicate more certain predictions")
    print(f"\n🏥 Model Accuracy: {model_package['metadata']['accuracy']*100:.2f}%")
    print("✅ Model is ready for clinical decision support!")

if __name__ == "__main__":
    main()