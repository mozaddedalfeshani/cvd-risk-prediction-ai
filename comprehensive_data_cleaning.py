import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE DATA CLEANING FOR OPTIMAL ML PERFORMANCE")
print("="*80)

def clean_cvd_dataset(input_file='Raw_Dataset.csv', output_file='CVD_Dataset_ML_Ready.csv'):
    """
    Comprehensive data cleaning pipeline for CVD dataset to achieve maximum ML accuracy
    """
    
    # 1. Load the raw dataset
    print("\n1. Loading raw dataset...")
    df = pd.read_csv(input_file)
    print(f"Original dataset shape: {df.shape}")
    print(f"Original missing values: {df.isnull().sum().sum()}")
    
    # Display data types
    print("\nOriginal data types:")
    print(df.dtypes)
    
    # 2. Handle critical missing values first (as per your colab approach)
    print("\n2. Handling critical missing values...")
    print("Before cleaning critical columns:")
    print(df[['Weight (kg)', 'Height (m)', 'Age']].isnull().sum())
    
    # Drop rows with missing critical demographic/physical data
    df.dropna(subset=['Weight (kg)', 'Height (m)', 'Age'], inplace=True)
    print(f"Shape after dropping critical nulls: {df.shape}")
    
    # 3. Fix BMI calculation (as per your colab approach)
    print("\n3. Recalculating BMI for consistency...")
    df['BMI'] = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    print("BMI recalculated successfully")
    
    # 4. Fix Height (cm) calculation (as per your colab approach)
    print("\n4. Recalculating Height (cm)...")
    df['Height (cm)'] = df['Height (m)'] * 100
    print("Height (cm) recalculated successfully")
    
    # 5. Handle the problematic Blood Pressure (mmHg) column
    print("\n5. Processing Blood Pressure (mmHg) column...")
    # This column has mixed formats (e.g., "125/79"), extract systolic/diastolic if missing
    def extract_bp_values(bp_str, systolic_col, diastolic_col, index):
        if pd.isna(bp_str):
            return systolic_col, diastolic_col
        
        try:
            if '/' in str(bp_str):
                systolic, diastolic = map(int, str(bp_str).split('/'))
                # Fill missing values in separate columns if they exist
                if pd.isna(systolic_col):
                    systolic_col = systolic
                if pd.isna(diastolic_col):
                    diastolic_col = diastolic
        except:
            pass
        
        return systolic_col, diastolic_col
    
    # Apply BP extraction
    for idx in df.index:
        df.loc[idx, 'Systolic BP'], df.loc[idx, 'Diastolic BP'] = extract_bp_values(
            df.loc[idx, 'Blood Pressure (mmHg)'], 
            df.loc[idx, 'Systolic BP'], 
            df.loc[idx, 'Diastolic BP'], 
            idx
        )
    
    # Drop the original mixed-format column
    df = df.drop('Blood Pressure (mmHg)', axis=1)
    print("Blood pressure processing complete")
    
    # 6. Advanced missing value handling for numeric columns
    print("\n6. Advanced imputation for remaining missing values...")
    
    # Separate numeric and categorical columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Categorical columns: {len(categorical_columns)}")
    
    # Handle categorical missing values with mode
    for col in categorical_columns:
        if df[col].isnull().any():
            mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
            df[col].fillna(mode_value, inplace=True)
            print(f"   Filled {col} missing values with mode: {mode_value}")
    
    # Advanced numeric imputation using KNN
    print("   Using KNN imputation for numeric columns...")
    numeric_data = df[numeric_columns]
    
    # Only apply KNN if there are missing values
    if numeric_data.isnull().any().any():
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_columns] = imputer.fit_transform(numeric_data)
        print("   KNN imputation completed")
    else:
        print("   No missing numeric values found")
    
    # 7. Data type optimization and consistency
    print("\n7. Optimizing data types for ML performance...")
    
    # Convert binary categorical variables to numeric
    binary_mappings = {
        'Sex': {'M': 1, 'F': 0},
        'Smoking Status': {'Y': 1, 'N': 0},
        'Diabetes Status': {'Y': 1, 'N': 0},
        'Family History of CVD': {'Y': 1, 'N': 0}
    }
    
    for col, mapping in binary_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"   Converted {col} to numeric (0/1)")
    
    # Convert ordinal categorical variables
    ordinal_mappings = {
        'Physical Activity Level': {'Low': 1, 'Moderate': 2, 'High': 3},
        'Blood Pressure Category': {
            'Normal': 1, 
            'Elevated': 2, 
            'Hypertension Stage 1': 3, 
            'Hypertension Stage 2': 4
        },
        'CVD Risk Level': {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
    }
    
    for col, mapping in ordinal_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
            print(f"   Converted {col} to ordinal numeric")
    
    # 8. Outlier detection and handling
    print("\n8. Handling outliers using IQR method...")
    
    # Define columns that should have outlier treatment
    outlier_columns = ['Weight (kg)', 'BMI', 'Abdominal Circumference (cm)', 
                      'Total Cholesterol (mg/dL)', 'HDL (mg/dL)', 
                      'Fasting Blood Sugar (mg/dL)', 'CVD Risk Score']
    
    for col in outlier_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                # Cap outliers instead of removing them
                df[col] = df[col].clip(lower_bound, upper_bound)
                print(f"   Capped {outliers_count} outliers in {col}")
    
    # 9. Data validation and consistency checks
    print("\n9. Performing data validation...")
    
    # BMI consistency check
    calculated_bmi = df['Weight (kg)'] / (df['Height (m)'] ** 2)
    bmi_diff = abs(df['BMI'] - calculated_bmi)
    inconsistent_bmi = (bmi_diff > 0.1).sum()
    if inconsistent_bmi > 0:
        df['BMI'] = calculated_bmi
        print(f"   Fixed {inconsistent_bmi} inconsistent BMI values")
    
    # Height consistency check
    calculated_height_cm = df['Height (m)'] * 100
    height_diff = abs(df['Height (cm)'] - calculated_height_cm)
    inconsistent_height = (height_diff > 1).sum()
    if inconsistent_height > 0:
        df['Height (cm)'] = calculated_height_cm
        print(f"   Fixed {inconsistent_height} inconsistent height values")
    
    # Blood pressure validation
    invalid_bp = (df['Systolic BP'] <= df['Diastolic BP']).sum()
    if invalid_bp > 0:
        print(f"   Warning: {invalid_bp} rows have invalid BP (systolic <= diastolic)")
        # Fix by adding 20 to systolic if it's invalid
        mask = df['Systolic BP'] <= df['Diastolic BP']
        df.loc[mask, 'Systolic BP'] = df.loc[mask, 'Diastolic BP'] + 20
        print(f"   Fixed invalid BP values")
    
    # 10. Feature engineering for better ML performance
    print("\n10. Adding engineered features for ML optimization...")
    
    # Medical risk indicators
    df['Pulse_Pressure'] = df['Systolic BP'] - df['Diastolic BP']
    df['Cholesterol_HDL_Ratio'] = df['Total Cholesterol (mg/dL)'] / df['HDL (mg/dL)']
    df['LDL_HDL_Ratio'] = df['Estimated LDL (mg/dL)'] / df['HDL (mg/dL)']
    
    # Age groups for better pattern recognition
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=[1, 2, 3, 4])
    df['Age_Group'] = df['Age_Group'].fillna(2).astype(int)  # Fill NaN with middle age group
    
    # BMI categories
    df['BMI_Category'] = pd.cut(df['BMI'], 
                               bins=[0, 18.5, 25, 30, 50], 
                               labels=[1, 2, 3, 4])  # Underweight, Normal, Overweight, Obese
    df['BMI_Category'] = df['BMI_Category'].fillna(2).astype(int)  # Fill NaN with Normal category
    
    # Risk factor combinations
    df['Multiple_Risk_Factors'] = (df['Smoking Status'] + df['Diabetes Status'] + 
                                  df['Family History of CVD']).astype(int)
    
    print(f"   Added 6 engineered features")
    
    # 11. Final cleanup - ensure no missing values remain
    print("\n11. Final validation...")
    
    # Apply your final dropna approach if any missing values remain
    initial_shape = df.shape
    df.dropna(inplace=True)
    final_shape = df.shape
    
    rows_dropped = initial_shape[0] - final_shape[0]
    if rows_dropped > 0:
        print(f"   Dropped {rows_dropped} rows with remaining missing values")
    
    print(f"Final missing values: {df.isnull().sum().sum()}")
    print(f"Final dataset shape: {df.shape}")
    
    # 12. Data type optimization for memory efficiency
    print("\n12. Optimizing data types for memory efficiency...")
    
    # Convert float64 to float32 where appropriate
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        if df[col].max() < 3.4e38 and df[col].min() > -3.4e38:  # float32 range
            df[col] = df[col].astype('float32')
    
    # Convert int64 to int32 where appropriate
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].max() < 2147483647 and df[col].min() > -2147483648:  # int32 range
            df[col] = df[col].astype('int32')
    
    print("   Data types optimized for memory efficiency")
    
    # 13. Save the cleaned dataset
    print(f"\n13. Saving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # 14. Generate summary report
    print("\n" + "="*80)
    print("CLEANING SUMMARY REPORT")
    print("="*80)
    
    print(f"✅ Input file: {input_file}")
    print(f"✅ Output file: {output_file}")
    print(f"✅ Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✅ Missing values: {df.isnull().sum().sum()}")
    print(f"✅ Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nFinal data types:")
    print(df.dtypes.value_counts())
    
    print(f"\nTarget variable distribution:")
    if 'CVD Risk Level' in df.columns:
        print(df['CVD Risk Level'].value_counts().sort_index())
    
    print(f"\nFeature correlation with target (top 10):")
    if 'CVD Risk Level' in df.columns:
        correlations = df.corr()['CVD Risk Level'].abs().sort_values(ascending=False)
        print(correlations.head(10))
    
    print("\n✅ Dataset is now ready for machine learning!")
    print("✅ All data types are consistent and optimized")
    print("✅ No missing values remain")
    print("✅ Outliers have been handled appropriately")
    print("✅ Additional features have been engineered for better performance")
    
    return df

if __name__ == "__main__":
    # Run the comprehensive cleaning pipeline
    cleaned_df = clean_cvd_dataset()
    
    print(f"\n🎯 Cleaned dataset saved successfully!")
    print(f"📊 Ready for high-accuracy machine learning models!")