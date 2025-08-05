import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE DATA CLEANING PIPELINE")
print("="*80)

# 1. Load Raw Dataset
print("\n1. Loading Raw_Dataset.csv...")
df_raw = pd.read_csv('./Raw_Dataset.csv')
print(f"Raw dataset shape: {df_raw.shape}")
print(f"Total missing values: {df_raw.isnull().sum().sum()}")

print("\nMissing values per column:")
missing_counts = df_raw.isnull().sum()
for col, count in missing_counts[missing_counts > 0].items():
    print(f"  {col}: {count} ({count/len(df_raw)*100:.1f}%)")

# 2. Examine data types and unique values
print("\n2. Data overview...")
print(f"\nData types:")
print(df_raw.dtypes)

print(f"\nTarget variable distribution:")
print(df_raw['CVD Risk Level'].value_counts())

print(f"\nCategorical columns unique values:")
categorical_cols = df_raw.select_dtypes(include=['object']).columns
for col in categorical_cols:
    unique_vals = df_raw[col].unique()
    print(f"  {col}: {unique_vals}")

# 3. Start comprehensive cleaning
print("\n3. Starting comprehensive data cleaning...")
df_clean = df_raw.copy()

# Step 3a: Handle categorical missing values first
print("\n   Handling categorical missing values...")
categorical_columns = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                      'Family History of CVD', 'Blood Pressure Category', 'CVD Risk Level']

for col in categorical_columns:
    if col in df_clean.columns and df_clean[col].isnull().any():
        mode_value = df_clean[col].mode()[0]
        missing_count = df_clean[col].isnull().sum()
        df_clean[col].fillna(mode_value, inplace=True)
        print(f"     {col}: filled {missing_count} missing values with '{mode_value}'")

# Step 3b: Handle outliers in numeric columns before imputation
print("\n   Handling outliers in numeric columns...")
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

outlier_summary = []
for col in numeric_cols:
    if col in df_clean.columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers before capping
        outliers_before = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        # Cap extreme outliers (keep some variation but remove extreme values)
        df_clean[col] = df_clean[col].clip(lower=Q1 - 3*IQR, upper=Q3 + 3*IQR)
        
        outliers_after = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
        
        if outliers_before > 0:
            outlier_summary.append(f"     {col}: {outliers_before} outliers handled")

if outlier_summary:
    for summary in outlier_summary:
        print(summary)
else:
    print("     No extreme outliers found")

# Step 3c: Advanced imputation for numeric missing values
print("\n   Imputing numeric missing values with KNN...")
numeric_missing_before = df_clean[numeric_cols].isnull().sum().sum()

if numeric_missing_before > 0:
    # Use KNN imputation for numeric columns
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    df_clean[numeric_cols] = imputer.fit_transform(df_clean[numeric_cols])
    print(f"     Imputed {numeric_missing_before} missing numeric values using KNN")
else:
    print("     No numeric missing values to impute")

# Step 3d: Remove problematic columns
print("\n   Removing problematic columns...")
columns_to_remove = ['Blood Pressure (mmHg)']  # Contains mixed format data like "108/87"

removed_cols = []
for col in columns_to_remove:
    if col in df_clean.columns:
        df_clean = df_clean.drop(col, axis=1)
        removed_cols.append(col)

if removed_cols:
    print(f"     Removed columns: {removed_cols}")
else:
    print("     No problematic columns found")

# Step 3e: Create derived medical features
print("\n   Creating derived medical features...")

# BMI calculation verification and correction
if 'Weight (kg)' in df_clean.columns and 'Height (m)' in df_clean.columns:
    df_clean['BMI_Calculated'] = df_clean['Weight (kg)'] / (df_clean['Height (m)'] ** 2)
    # Use calculated BMI if original BMI is missing or inconsistent
    bmi_diff = abs(df_clean['BMI'] - df_clean['BMI_Calculated'])
    inconsistent_bmi = bmi_diff > 1.0  # Allow 1 unit difference
    if inconsistent_bmi.sum() > 0:
        print(f"     Fixed {inconsistent_bmi.sum()} inconsistent BMI values")
        df_clean.loc[inconsistent_bmi, 'BMI'] = df_clean.loc[inconsistent_bmi, 'BMI_Calculated']
    df_clean = df_clean.drop('BMI_Calculated', axis=1)

# Height consistency check
if 'Height (m)' in df_clean.columns and 'Height (cm)' in df_clean.columns:
    height_cm_calculated = df_clean['Height (m)'] * 100
    height_diff = abs(df_clean['Height (cm)'] - height_cm_calculated)
    inconsistent_height = height_diff > 2.0  # Allow 2 cm difference
    if inconsistent_height.sum() > 0:
        print(f"     Fixed {inconsistent_height.sum()} inconsistent height values")
        df_clean.loc[inconsistent_height, 'Height (cm)'] = height_cm_calculated[inconsistent_height]

# Waist-to-Height ratio verification
if 'Abdominal Circumference (cm)' in df_clean.columns and 'Height (cm)' in df_clean.columns:
    whr_calculated = df_clean['Abdominal Circumference (cm)'] / df_clean['Height (cm)']
    if 'Waist-to-Height Ratio' in df_clean.columns:
        whr_diff = abs(df_clean['Waist-to-Height Ratio'] - whr_calculated)
        inconsistent_whr = whr_diff > 0.05  # Allow 5% difference
        if inconsistent_whr.sum() > 0:
            print(f"     Fixed {inconsistent_whr.sum()} inconsistent Waist-to-Height ratios")
            df_clean.loc[inconsistent_whr, 'Waist-to-Height Ratio'] = whr_calculated[inconsistent_whr]

# Step 3f: Data validation
print("\n   Performing data validation...")

# Age validation
if 'Age' in df_clean.columns:
    invalid_age = (df_clean['Age'] < 18) | (df_clean['Age'] > 100)
    if invalid_age.sum() > 0:
        print(f"     Warning: {invalid_age.sum()} potentially invalid age values")

# Blood pressure validation
if 'Systolic BP' in df_clean.columns and 'Diastolic BP' in df_clean.columns:
    invalid_bp = df_clean['Systolic BP'] <= df_clean['Diastolic BP']
    if invalid_bp.sum() > 0:
        print(f"     Fixed {invalid_bp.sum()} cases where systolic <= diastolic BP")
        # Fix by swapping values
        df_clean.loc[invalid_bp, ['Systolic BP', 'Diastolic BP']] = df_clean.loc[invalid_bp, ['Diastolic BP', 'Systolic BP']].values

# BMI validation
if 'BMI' in df_clean.columns:
    invalid_bmi = (df_clean['BMI'] < 15) | (df_clean['BMI'] > 50)
    if invalid_bmi.sum() > 0:
        print(f"     Warning: {invalid_bmi.sum()} potentially extreme BMI values")

# 4. Final data overview
print("\n4. Final cleaned dataset overview...")
print(f"Final shape: {df_clean.shape}")
print(f"Total missing values: {df_clean.isnull().sum().sum()}")

if df_clean.isnull().sum().sum() > 0:
    print("Remaining missing values:")
    remaining_missing = df_clean.isnull().sum()
    for col, count in remaining_missing[remaining_missing > 0].items():
        print(f"  {col}: {count}")

# 5. Data quality summary
print("\n5. Data quality summary...")
print(f"✅ Dataset successfully cleaned:")
print(f"   - Original records: {len(df_raw)}")
print(f"   - Final records: {len(df_clean)}")
print(f"   - Records retained: {len(df_clean)/len(df_raw)*100:.1f}%")
print(f"   - Original missing values: {df_raw.isnull().sum().sum()}")
print(f"   - Final missing values: {df_clean.isnull().sum().sum()}")
print(f"   - Features: {df_clean.shape[1]}")

# 6. Save cleaned dataset
output_filename = 'CVD_Dataset_Cleaned_Final.csv'
print(f"\n6. Saving cleaned dataset as '{output_filename}'...")

try:
    df_clean.to_csv(output_filename, index=False)
    print(f"✅ Successfully saved cleaned dataset!")
    print(f"   File: {output_filename}")
    print(f"   Size: {df_clean.shape}")
    
    # Verify the saved file
    df_verify = pd.read_csv(output_filename)
    if df_verify.shape == df_clean.shape:
        print(f"✅ File verification successful!")
    else:
        print(f"⚠️  File verification failed - shapes don't match")
        
except Exception as e:
    print(f"❌ Error saving file: {e}")

# 7. Create data dictionary
print(f"\n7. Creating data dictionary...")
data_dict = {
    'Column': df_clean.columns,
    'Data_Type': [str(df_clean[col].dtype) for col in df_clean.columns],
    'Non_Null_Count': [df_clean[col].count() for col in df_clean.columns],
    'Null_Count': [df_clean[col].isnull().sum() for col in df_clean.columns],
    'Unique_Values': [df_clean[col].nunique() for col in df_clean.columns]
}

# Add sample values for categorical columns
sample_values = []
for col in df_clean.columns:
    if df_clean[col].dtype == 'object' or df_clean[col].nunique() < 10:
        unique_vals = df_clean[col].unique()[:5]  # First 5 unique values
        sample_values.append(str(list(unique_vals)))
    else:
        min_val = df_clean[col].min()
        max_val = df_clean[col].max()
        sample_values.append(f"Range: {min_val:.2f} - {max_val:.2f}")

data_dict['Sample_Values'] = sample_values

dict_df = pd.DataFrame(data_dict)
dict_filename = 'CVD_Dataset_Data_Dictionary.csv'
dict_df.to_csv(dict_filename, index=False)
print(f"✅ Data dictionary saved as '{dict_filename}'")

print("\n" + "="*80)
print("DATA CLEANING COMPLETE!")
print("="*80)
print(f"📁 Cleaned dataset: {output_filename}")
print(f"📋 Data dictionary: {dict_filename}")
print(f"🎯 Ready for machine learning modeling!")
print("="*80)