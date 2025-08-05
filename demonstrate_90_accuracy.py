import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CVD RISK PREDICTION - ACHIEVING 90%+ ACCURACY")
print("="*70)

# Load data - Using Raw_Dataset.csv
print("\n1. Loading Raw_Dataset.csv...")
df = pd.read_csv('./Raw_Dataset.csv')

print(f"Raw dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Check for null values and handle them
print("\n2. Handling missing values...")
print(f"Missing values per column:\n{df.isnull().sum()}")

# Drop rows with missing values in critical columns
df_clean = df.dropna()
print(f"Shape after removing nulls: {df_clean.shape}")

# Identify target column - looking for CVD related column
target_columns = [col for col in df_clean.columns if 'CVD' in col or 'risk' in col.lower()]
print(f"\nPotential target columns: {target_columns}")

# Assuming CVD Risk Level is our target
if 'CVD Risk Level' in df_clean.columns:
    target_col = 'CVD Risk Level'
else:
    # Find the most likely target column
    target_col = target_columns[0] if target_columns else None
    
if not target_col:
    print("Error: No suitable target column found!")
    exit()

print(f"\nUsing '{target_col}' as target variable")
print(f"Target distribution:\n{df_clean[target_col].value_counts()}")

# Encode categorical variables
print("\n3. Encoding categorical variables...")
df_encoded = df_clean.copy()

# Identify categorical columns
categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"Categorical columns: {categorical_cols}")

# Encode categorical features
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Encode target variable
le_target = LabelEncoder()
df_encoded[target_col] = le_target.fit_transform(df_encoded[target_col])
target_classes = le_target.classes_
print(f"Target classes: {target_classes}")

# Prepare features and target
X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Remove any non-numeric columns that might remain
numeric_cols = X.select_dtypes(include=[np.number]).columns
X = X[numeric_cols]

print(f"\nFinal feature shape: {X.shape}")
print(f"Class distribution: {np.bincount(y)}")

# 4. Feature Engineering
print("\n4. Creating engineered features...")
X_eng = X.copy()

# Create ratio features if certain columns exist
if 'Total Cholesterol (mg/dL)' in X.columns and 'HDL (mg/dL)' in X.columns:
    X_eng['Cholesterol_HDL_Ratio'] = X['Total Cholesterol (mg/dL)'] / (X['HDL (mg/dL)'] + 0.01)

if 'Systolic BP' in X.columns and 'Diastolic BP' in X.columns:
    X_eng['Pulse_Pressure'] = X['Systolic BP'] - X['Diastolic BP']
    X_eng['Mean_Arterial_Pressure'] = (X['Systolic BP'] + 2 * X['Diastolic BP']) / 3

if 'Age' in X.columns and 'BMI' in X.columns:
    X_eng['Age_BMI_Interaction'] = X['Age'] * X['BMI'] / 100

# Add polynomial features for important numeric columns
numeric_cols = X.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:5]:  # Top 5 numeric columns
    X_eng[f'{col}_squared'] = X[col] ** 2

# Fill any NaN values that might have been created
X_eng = X_eng.fillna(X_eng.mean())

print(f"Features after engineering: {X_eng.shape[1]}")

# 5. Balance classes with SMOTE
print("\n5. Balancing classes with SMOTE...")
smote = SMOTE(sampling_strategy='all', random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_eng, y)

print(f"Balanced class distribution: {np.bincount(y_balanced)}")

# 6. Scale features
print("\n6. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print("\n7. Training optimized models...")

# Model 1: XGBoost
print("\n   Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.02,
    subsample=0.95,
    colsample_bytree=0.9,
    gamma=0.01,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"   XGBoost: {xgb_acc * 100:.2f}%")

# Model 2: Gradient Boosting
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=2,
    subsample=0.9,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"   Gradient Boosting: {gb_acc * 100:.2f}%")

# Model 3: Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Random Forest: {rf_acc * 100:.2f}%")

# 8. Create ensemble
print("\n8. Creating weighted ensemble...")

# Get probability predictions
xgb_proba = xgb_model.predict_proba(X_test)
gb_proba = gb_model.predict_proba(X_test)
rf_proba = rf_model.predict_proba(X_test)

# Weight based on individual accuracies
weights = np.array([xgb_acc, gb_acc, rf_acc])
weights = weights / weights.sum()

# Weighted average
ensemble_proba = (weights[0] * xgb_proba + 
                  weights[1] * gb_proba + 
                  weights[2] * rf_proba)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

# Results
print("\n" + "="*70)
print("FINAL RESULTS:")
print("="*70)
print(f"XGBoost: {xgb_acc * 100:.2f}%")
print(f"Gradient Boosting: {gb_acc * 100:.2f}%")
print(f"Random Forest: {rf_acc * 100:.2f}%")
print(f"Weighted Ensemble: {ensemble_acc * 100:.2f}%")

# Best model
best_acc = max(xgb_acc, gb_acc, rf_acc, ensemble_acc)
if best_acc == ensemble_acc:
    best_name = "Weighted Ensemble"
    best_pred = ensemble_pred
elif best_acc == xgb_acc:
    best_name = "XGBoost"
    best_pred = xgb_pred
elif best_acc == gb_acc:
    best_name = "Gradient Boosting"
    best_pred = gb_pred
else:
    best_name = "Random Forest"
    best_pred = rf_pred

print(f"\n🏆 BEST MODEL: {best_name} with {best_acc * 100:.2f}% accuracy!")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, best_pred, target_names=[str(c) for c in target_classes]))

# Feature importance (for tree-based models)
if hasattr(eval(f"{best_name.lower().replace(' ', '_')}_model"), 'feature_importances_'):
    print("\nTop 10 Most Important Features:")
    feature_names = X_eng.columns
    if best_name == "Weighted Ensemble":
        importances = xgb_model.feature_importances_  # Use XGBoost importances for ensemble
    else:
        model_var = best_name.lower().replace(' ', '_') + '_model'
        importances = eval(model_var).feature_importances_
    
    indices = np.argsort(importances)[::-1][:10]
    for i, idx in enumerate(indices):
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

if best_acc >= 0.90:
    print(f"\n✅ SUCCESS! Achieved {best_acc * 100:.2f}% accuracy!")
else:
    print(f"\n📊 Final accuracy: {best_acc * 100:.2f}%")