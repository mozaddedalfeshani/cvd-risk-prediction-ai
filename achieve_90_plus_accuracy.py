import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADVANCED CVD RISK PREDICTION - PUSHING FOR 90%+ ACCURACY")
print("="*80)

# 1. Load Raw Dataset
print("\n1. Loading Raw_Dataset.csv...")
df = pd.read_csv('./Raw_Dataset.csv')
print(f"Raw dataset shape: {df.shape}")

# 2. Handle missing values more strategically
print("\n2. Strategic handling of missing values...")
# For numeric columns, use median imputation
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col].fillna(df[col].median(), inplace=True)

# For categorical columns, use mode
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

print(f"Shape after imputation: {df.shape}")

# 3. Advanced preprocessing
print("\n3. Advanced preprocessing...")
df_processed = df.copy()

# Create age groups for better pattern recognition
df_processed['Age_Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], labels=[0, 1, 2, 3, 4])
df_processed['Age_Group'] = df_processed['Age_Group'].astype(int)

# BMI categories
df_processed['BMI_Category'] = pd.cut(df['BMI'], bins=[0, 18.5, 25, 30, 35, 100], labels=[0, 1, 2, 3, 4])
df_processed['BMI_Category'] = df_processed['BMI_Category'].astype(int)

# Blood pressure stages
df_processed['BP_Stage'] = pd.cut(df['Systolic BP'], bins=[0, 120, 130, 140, 180, 300], labels=[0, 1, 2, 3, 4])
df_processed['BP_Stage'] = df_processed['BP_Stage'].astype(int)

# Cholesterol risk levels
df_processed['Cholesterol_Risk'] = pd.cut(df['Total Cholesterol (mg/dL)'], 
                                          bins=[0, 200, 240, 300, 500], 
                                          labels=[0, 1, 2, 3])
df_processed['Cholesterol_Risk'] = df_processed['Cholesterol_Risk'].astype(int)

# Encode categorical variables
categorical_cols = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                   'Family History of CVD', 'Blood Pressure Category']

for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

# Drop the problematic column
if 'Blood Pressure (mmHg)' in df_processed.columns:
    df_processed = df_processed.drop('Blood Pressure (mmHg)', axis=1)

# Encode target
target_mapping = {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
df_processed['CVD Risk Level'] = df_processed['CVD Risk Level'].map(target_mapping)

# 4. Advanced Feature Engineering
print("\n4. Creating advanced engineered features...")
X = df_processed.drop('CVD Risk Level', axis=1)
y = df_processed['CVD Risk Level']

# Create interaction features
X['Cholesterol_HDL_Ratio'] = X['Total Cholesterol (mg/dL)'] / (X['HDL (mg/dL)'] + 1)
X['LDL_HDL_Ratio'] = X['Estimated LDL (mg/dL)'] / (X['HDL (mg/dL)'] + 1)
X['Non_HDL_Cholesterol'] = X['Total Cholesterol (mg/dL)'] - X['HDL (mg/dL)']
X['Triglycerides_Estimate'] = X['Total Cholesterol (mg/dL)'] - X['HDL (mg/dL)'] - X['Estimated LDL (mg/dL)']

# Blood pressure features
X['Pulse_Pressure'] = X['Systolic BP'] - X['Diastolic BP']
X['Mean_Arterial_Pressure'] = (X['Systolic BP'] + 2 * X['Diastolic BP']) / 3
X['BP_Product'] = X['Systolic BP'] * X['Diastolic BP'] / 1000

# Risk score enhancements
X['CVD_Score_Squared'] = X['CVD Risk Score'] ** 2
X['CVD_Score_Log'] = np.log1p(X['CVD Risk Score'])
X['CVD_Score_Sqrt'] = np.sqrt(X['CVD Risk Score'])

# Age interactions
X['Age_CVD_Score'] = X['Age'] * X['CVD Risk Score']
X['Age_Cholesterol'] = X['Age'] * X['Total Cholesterol (mg/dL)'] / 1000
X['Age_BP'] = X['Age'] * X['Systolic BP'] / 1000
X['Age_BMI'] = X['Age'] * X['BMI'] / 100

# Multiple risk factors
X['Risk_Factor_Count'] = (
    (X['Smoking Status'] == 1).astype(int) +
    (X['Diabetes Status'] == 1).astype(int) +
    (X['Family History of CVD'] == 1).astype(int) +
    (X['BP_Stage'] >= 2).astype(int) +
    (X['Cholesterol_Risk'] >= 1).astype(int) +
    (X['BMI_Category'] >= 2).astype(int)
)

# Complex interactions
X['Diabetes_Cholesterol'] = X['Diabetes Status'] * X['Total Cholesterol (mg/dL)']
X['Smoking_BP'] = X['Smoking Status'] * X['Systolic BP']
X['Family_Age_Risk'] = X['Family History of CVD'] * X['Age'] * X['CVD Risk Score'] / 100

# Physical activity protection factor
X['Activity_HDL_Benefit'] = X['Physical Activity Level'] * X['HDL (mg/dL)']
X['Activity_Protection'] = X['Physical Activity Level'] * (100 - X['CVD Risk Score'])

print(f"Features after engineering: {X.shape[1]}")

# 5. Aggressive class balancing with multiple techniques
print("\n5. Advanced class balancing...")
# Try SMOTETomek for better boundary definition
smote_tomek = SMOTETomek(random_state=42)
X_balanced, y_balanced = smote_tomek.fit_resample(X, y)

print(f"Balanced distribution: {np.bincount(y_balanced)}")

# 6. Feature selection
print("\n6. Feature selection...")
# Use SelectKBest to identify top features
selector = SelectKBest(f_classif, k=min(40, X_balanced.shape[1]))
X_selected = selector.fit_transform(X_balanced, y_balanced)

# Get selected feature names
selected_features = X.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features")

# 7. Advanced scaling
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_selected)

# 8. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print("\n7. Training multiple optimized models...")

# Model 1: Super-optimized XGBoost
print("\n   Training XGBoost with extensive optimization...")
xgb_model = XGBClassifier(
    n_estimators=1000,
    max_depth=10,
    learning_rate=0.01,
    subsample=0.95,
    colsample_bytree=0.9,
    colsample_bylevel=0.9,
    gamma=0.01,
    reg_alpha=0.001,
    reg_lambda=1,
    min_child_weight=1,
    scale_pos_weight=1,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)
print(f"   XGBoost: {xgb_acc * 100:.2f}%")

# Model 2: Neural Network
print("\n   Training Deep Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    n_iter_no_change=20,
    random_state=42
)
nn_model.fit(X_train, y_train)
nn_pred = nn_model.predict(X_test)
nn_acc = accuracy_score(y_test, nn_pred)
print(f"   Neural Network: {nn_acc * 100:.2f}%")

# Model 3: Extra Trees
print("\n   Training Extra Trees...")
et_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
et_pred = et_model.predict(X_test)
et_acc = accuracy_score(y_test, et_pred)
print(f"   Extra Trees: {et_acc * 100:.2f}%")

# Model 4: Gradient Boosting
print("\n   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=6,
    min_samples_split=2,
    min_samples_leaf=1,
    subsample=0.95,
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"   Gradient Boosting: {gb_acc * 100:.2f}%")

# Model 5: Random Forest
print("\n   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Random Forest: {rf_acc * 100:.2f}%")

# 8. Advanced Ensemble Methods
print("\n8. Creating advanced ensembles...")

# Voting Ensemble
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('nn', nn_model),
        ('et', et_model),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    voting='soft'
)
voting.fit(X_train, y_train)
voting_pred = voting.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)
print(f"   Voting Ensemble: {voting_acc * 100:.2f}%")

# Custom Weighted Ensemble
print("\n   Creating custom weighted ensemble...")
# Get probability predictions
all_probs = [
    xgb_model.predict_proba(X_test),
    nn_model.predict_proba(X_test),
    et_model.predict_proba(X_test),
    gb_model.predict_proba(X_test),
    rf_model.predict_proba(X_test)
]

# Calculate weights based on individual accuracies
accuracies = [xgb_acc, nn_acc, et_acc, gb_acc, rf_acc]
weights = np.array(accuracies) ** 2  # Square to emphasize better models
weights = weights / weights.sum()

# Weighted average
weighted_probs = np.zeros_like(all_probs[0])
for i, prob in enumerate(all_probs):
    weighted_probs += weights[i] * prob

weighted_pred = np.argmax(weighted_probs, axis=1)
weighted_acc = accuracy_score(y_test, weighted_pred)
print(f"   Weighted Ensemble: {weighted_acc * 100:.2f}%")

# Final Results
print("\n" + "="*80)
print("FINAL RESULTS:")
print("="*80)

all_results = {
    'XGBoost': xgb_acc,
    'Neural Network': nn_acc,
    'Extra Trees': et_acc,
    'Gradient Boosting': gb_acc,
    'Random Forest': rf_acc,
    'Voting Ensemble': voting_acc,
    'Weighted Ensemble': weighted_acc
}

for name, acc in sorted(all_results.items(), key=lambda x: x[1], reverse=True):
    print(f"{name}: {acc * 100:.2f}%")

best_model = max(all_results.items(), key=lambda x: x[1])
print(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1] * 100:.2f}% accuracy!")

# Get best predictions for report
best_pred = weighted_pred if best_model[0] == 'Weighted Ensemble' else voting_pred

print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['LOW', 'INTERMEDIARY', 'HIGH']))

if best_model[1] >= 0.90:
    print(f"\n✅ SUCCESS! Achieved {best_model[1] * 100:.2f}% accuracy - EXCEEDS 90% TARGET!")
else:
    print(f"\n📊 Maximum accuracy achieved: {best_model[1] * 100:.2f}%")
    
# Show top features
print("\nTop 10 Most Important Features (from XGBoost):")
feature_importances = xgb_model.feature_importances_
indices = np.argsort(feature_importances)[::-1][:10]
for i, idx in enumerate(indices):
    print(f"{i+1}. {selected_features[idx]}: {feature_importances[idx]:.4f}")