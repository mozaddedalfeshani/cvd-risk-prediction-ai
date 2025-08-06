#!/usr/bin/env python3
"""
MymensingUniversity Dataset Cleaning Script
==========================================

This script cleans and prepares the MymensingUniversity dataset for machine learning.
It handles missing values, converts categorical variables, creates engineered features,
and outputs a clean, ML-ready dataset.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def clean_mymensinguniversity_dataset(input_file, output_file):
    """
    Clean and prepare the MymensingUniversity dataset for ML training
    
    Args:
        input_file (str): Path to the raw dataset
        output_file (str): Path for the cleaned dataset
    """
    
    print("="*80)
    print("MYMENSINGUNIVERSITY DATASET CLEANING PIPELINE")
    print("="*80)
    
    # 1. Load the dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. Handle Blood Pressure column parsing
    print("\n2. Parsing Blood Pressure data...")
    
    # Extract systolic and diastolic BP from combined column
    def parse_blood_pressure(bp_str):
        """Parse blood pressure string like '125/79' into systolic and diastolic"""
        if pd.isna(bp_str) or bp_str == '':
            return np.nan, np.nan
        
        bp_str = str(bp_str).strip()
        if '/' in bp_str:
            try:
                systolic, diastolic = bp_str.split('/')
                return float(systolic), float(diastolic)
            except:
                return np.nan, np.nan
        return np.nan, np.nan
    
    # Only parse if Systolic BP and Diastolic BP columns are missing or incomplete
    if 'Blood Pressure (mmHg)' in df.columns:
        missing_systolic = df['Systolic BP'].isna().sum() if 'Systolic BP' in df.columns else len(df)
        missing_diastolic = df['Diastolic BP'].isna().sum() if 'Diastolic BP' in df.columns else len(df)
        
        if missing_systolic > 0 or missing_diastolic > 0:
            print(f"  Parsing {df['Blood Pressure (mmHg)'].notna().sum()} BP values...")
            bp_parsed = df['Blood Pressure (mmHg)'].apply(parse_blood_pressure)
            
            # Fill missing values in Systolic/Diastolic BP columns
            if 'Systolic BP' not in df.columns:
                df['Systolic BP'] = np.nan
            if 'Diastolic BP' not in df.columns:
                df['Diastolic BP'] = np.nan
                
            for i, (sys, dia) in enumerate(bp_parsed):
                if pd.isna(df.iloc[i]['Systolic BP']) and not pd.isna(sys):
                    df.loc[i, 'Systolic BP'] = sys
                if pd.isna(df.iloc[i]['Diastolic BP']) and not pd.isna(dia):
                    df.loc[i, 'Diastolic BP'] = dia
    
    # 3. Handle Height columns (choose most complete one)
    print("\n3. Handling height columns...")
    
    if 'Height (m)' in df.columns and 'Height (cm)' in df.columns:
        # Check which column has more complete data
        height_m_missing = df['Height (m)'].isna().sum()
        height_cm_missing = df['Height (cm)'].isna().sum()
        
        print(f"  Height (m) missing: {height_m_missing}")
        print(f"  Height (cm) missing: {height_cm_missing}")
        
        if height_m_missing <= height_cm_missing:
            # Use Height (m) as primary, fill missing with Height (cm)
            df['Height (m)'] = df['Height (m)'].fillna(df['Height (cm)'] / 100)
            df = df.drop('Height (cm)', axis=1)
            print("  Using Height (m) as primary height column")
        else:
            # Use Height (cm) as primary, convert to meters
            df['Height (m)'] = df['Height (cm)'] / 100
            df['Height (m)'] = df['Height (m)'].fillna(df['Height (m)'])
            df = df.drop('Height (cm)', axis=1)
            print("  Converted Height (cm) to Height (m)")
    
    # 4. Convert categorical variables to numerical
    print("\n4. Converting categorical variables...")
    
    # Sex: F -> 0, M -> 1
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})
        print("  Sex: F=0, M=1")
    
    # Smoking Status: N -> 0, Y -> 1
    if 'Smoking Status' in df.columns:
        df['Smoking Status'] = df['Smoking Status'].map({'N': 0, 'Y': 1})
        print("  Smoking Status: N=0, Y=1")
    
    # Diabetes Status: N -> 0, Y -> 1
    if 'Diabetes Status' in df.columns:
        df['Diabetes Status'] = df['Diabetes Status'].map({'N': 0, 'Y': 1})
        print("  Diabetes Status: N=0, Y=1")
    
    # Family History of CVD: N -> 0, Y -> 1
    if 'Family History of CVD' in df.columns:
        df['Family History of CVD'] = df['Family History of CVD'].map({'N': 0, 'Y': 1})
        print("  Family History of CVD: N=0, Y=1")
    
    # Physical Activity Level: Low -> 0, Moderate -> 1, High -> 2
    if 'Physical Activity Level' in df.columns:
        df['Physical Activity Level'] = df['Physical Activity Level'].map({
            'Low': 0, 'Moderate': 1, 'High': 2
        })
        print("  Physical Activity Level: Low=0, Moderate=1, High=2")
    
    # Blood Pressure Category: Encode based on severity
    if 'Blood Pressure Category' in df.columns:
        bp_category_map = {
            'Normal': 1,
            'Elevated': 2,
            'Hypertension Stage 1': 3,
            'Hypertension Stage 2': 4
        }
        df['Blood Pressure Category'] = df['Blood Pressure Category'].map(bp_category_map)
        print("  Blood Pressure Category: Normal=1, Elevated=2, Stage1=3, Stage2=4")
    
    # Target variable: CVD Risk Level
    if 'CVD Risk Level' in df.columns:
        df['CVD Risk Level'] = df['CVD Risk Level'].map({
            'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2
        })
        print("  CVD Risk Level: LOW=0, INTERMEDIARY=1, HIGH=2")
    
    # 5. Handle missing values
    print("\n5. Handling missing values...")
    
    # Check missing values before imputation
    missing_before = df.isnull().sum().sum()
    print(f"  Total missing values before imputation: {missing_before}")
    
    # Numeric columns: impute with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"    {col}: filled {df[col].isnull().sum()} missing values with median ({median_val:.2f})")
    
    # Categorical columns: impute with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
            df[col] = df[col].fillna(mode_val)
            print(f"    {col}: filled {df[col].isnull().sum()} missing values with mode ({mode_val})")
    
    missing_after = df.isnull().sum().sum()
    print(f"  Total missing values after imputation: {missing_after}")
    
    # 6. Data validation and cleaning
    print("\n6. Data validation and cleaning...")
    
    # Remove rows with invalid age
    if 'Age' in df.columns:
        invalid_age = df[(df['Age'] < 18) | (df['Age'] > 100)]
        if len(invalid_age) > 0:
            df = df.drop(invalid_age.index)
            print(f"  Removed {len(invalid_age)} rows with invalid age")
    
    # Fix impossible BMI values
    if 'BMI' in df.columns:
        invalid_bmi = df[(df['BMI'] < 10) | (df['BMI'] > 60)]
        if len(invalid_bmi) > 0:
            print(f"  Found {len(invalid_bmi)} rows with invalid BMI, clipping to valid range")
            df['BMI'] = df['BMI'].clip(lower=10, upper=60)
    
    # Fix blood pressure values
    if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
        # Fix cases where systolic <= diastolic
        invalid_bp = df[df['Systolic BP'] <= df['Diastolic BP']]
        if len(invalid_bp) > 0:
            print(f"  Found {len(invalid_bp)} rows with invalid BP (systolic <= diastolic), swapping values")
            for idx in invalid_bp.index:
                sys_val = df.loc[idx, 'Systolic BP']
                dia_val = df.loc[idx, 'Diastolic BP']
                df.loc[idx, 'Systolic BP'] = max(sys_val, dia_val)
                df.loc[idx, 'Diastolic BP'] = min(sys_val, dia_val)
    
    # 7. Feature engineering
    print("\n7. Creating engineered features...")
    
    # Pulse Pressure
    if 'Systolic BP' in df.columns and 'Diastolic BP' in df.columns:
        df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']
        print("  Created Pulse_Pressure feature")
    
    # Cholesterol ratios
    if 'Total Cholesterol (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
        df['Cholesterol_HDL_Ratio'] = df['Total Cholesterol (mg/dL)'] / df['HDL (mg/dL)']
        print("  Created Cholesterol_HDL_Ratio feature")
    
    if 'Estimated LDL (mg/dL)' in df.columns and 'HDL (mg/dL)' in df.columns:
        df['LDL_HDL_Ratio'] = df['Estimated LDL (mg/dL)'] / df['HDL (mg/dL)']
        print("  Created LDL_HDL_Ratio feature")
    
    # Age groups
    if 'Age' in df.columns:
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 30, 40, 50, 60, 100], 
                                labels=[1, 2, 3, 4, 5])
        df['Age_Group'] = df['Age_Group'].astype(int)
        print("  Created Age_Group feature")
    
    # BMI categories
    if 'BMI' in df.columns:
        df['BMI_Category'] = pd.cut(df['BMI'], 
                                   bins=[0, 18.5, 24.9, 29.9, 100], 
                                   labels=[1, 2, 3, 4])
        df['BMI_Category'] = df['BMI_Category'].astype(int)
        print("  Created BMI_Category feature")
    
    # Multiple risk factors indicator
    risk_factors = []
    if 'Smoking Status' in df.columns:
        risk_factors.append('Smoking Status')
    if 'Diabetes Status' in df.columns:
        risk_factors.append('Diabetes Status')
    if 'Family History of CVD' in df.columns:
        risk_factors.append('Family History of CVD')
    
    if risk_factors:
        df['Multiple_Risk_Factors'] = df[risk_factors].sum(axis=1)
        print(f"  Created Multiple_Risk_Factors feature from: {', '.join(risk_factors)}")
    
    # 8. Remove redundant columns
    print("\n8. Removing redundant columns...")
    
    columns_to_remove = []
    
    # Remove the original Blood Pressure string column if it exists
    if 'Blood Pressure (mmHg)' in df.columns:
        columns_to_remove.append('Blood Pressure (mmHg)')
    
    # Remove any completely empty columns
    for col in df.columns:
        if df[col].isnull().all():
            columns_to_remove.append(col)
    
    if columns_to_remove:
        df = df.drop(columns_to_remove, axis=1)
        print(f"  Removed columns: {', '.join(columns_to_remove)}")
    
    # 9. Final data type optimization
    print("\n9. Optimizing data types...")
    
    # Convert float columns to appropriate types
    for col in df.columns:
        if df[col].dtype == 'float64':
            # Check if column contains only integers
            if df[col].dropna().apply(lambda x: x.is_integer()).all():
                df[col] = df[col].astype('int32')
            else:
                df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    
    print("  Optimized data types for memory efficiency")
    
    # 10. Final validation and summary
    print("\n10. Final validation and summary...")
    
    print(f"  Final dataset shape: {df.shape}")
    print(f"  Final memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"  Total missing values: {df.isnull().sum().sum()}")
    
    # Check target variable distribution
    if 'CVD Risk Level' in df.columns:
        print(f"  Target variable distribution:")
        target_dist = df['CVD Risk Level'].value_counts().sort_index()
        for risk_level, count in target_dist.items():
            risk_name = {0: 'LOW', 1: 'INTERMEDIARY', 2: 'HIGH'}.get(risk_level, risk_level)
            print(f"    {risk_name}: {count} ({count/len(df)*100:.1f}%)")
    
    # 11. Save cleaned dataset
    print(f"\n11. Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*80)
    print("DATASET CLEANING COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"✅ Input file: {input_file}")
    print(f"✅ Output file: {output_file}")
    print(f"✅ Original shape: {df.shape}")
    print(f"✅ Features: {df.shape[1] - 1}")  # Excluding target column
    print(f"✅ Samples: {df.shape[0]}")
    print(f"✅ Memory saved: {(df.memory_usage(deep=True).sum() / 1024**2):.2f} MB")
    print("\n🚀 Dataset is now ready for machine learning!")
    
    return df


if __name__ == "__main__":
    # Define file paths
    input_file = "../data/MymensingUniversity.csv"
    output_file = "../data/MymensingUniversity_ML_Ready.csv"
    
    # Run the cleaning pipeline
    try:
        cleaned_df = clean_mymensinguniversity_dataset(input_file, output_file)
        
        print(f"\n📊 Quick preview of cleaned dataset:")
        print(cleaned_df.head())
        
        print(f"\n📈 Dataset summary:")
        print(f"Columns: {list(cleaned_df.columns)}")
        
    except FileNotFoundError:
        print(f"❌ Error: Could not find input file '{input_file}'")
        print("Please make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"❌ Error during cleaning: {str(e)}")
        print("Please check the data format and try again.")