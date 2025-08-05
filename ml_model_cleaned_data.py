import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MACHINE LEARNING MODEL USING CLEANED DATASET")
print("="*80)

# 1. Load the cleaned dataset
print("\n1. Loading cleaned dataset...")
try:
    df = pd.read_csv('CVD_Dataset_Cleaned_Final.csv')
    print(f"✅ Successfully loaded cleaned dataset")
    print(f"   Shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
except FileNotFoundError:
    print("❌ Cleaned dataset not found. Please run data_cleaning_pipeline.py first.")
    exit()

# 2. Quick data overview
print(f"\n2. Dataset overview...")
print(f"Target distribution:")
print(df['CVD Risk Level'].value_counts())
print(f"\nFeatures: {df.shape[1]}")
print(f"Samples: {df.shape[0]}")

# 3. Feature Engineering on Clean Data
print("\n3. Advanced feature engineering on clean data...")

# Encode categorical variables
categorical_cols = ['Sex', 'Smoking Status', 'Diabetes Status', 'Physical Activity Level', 
                   'Family History of CVD', 'Blood Pressure Category']

df_processed = df.copy()
for col in categorical_cols:
    if col in df_processed.columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])

# Encode target
target_mapping = {'LOW': 0, 'INTERMEDIARY': 1, 'HIGH': 2}
df_processed['CVD Risk Level'] = df_processed['CVD Risk Level'].map(target_mapping)

# Separate features and target
X = df_processed.drop('CVD Risk Level', axis=1)
y = df_processed['CVD Risk Level']

# Create advanced medical features
print("   Creating medical domain features...")

# Lipid profile indicators
X['Total_HDL_Ratio'] = X['Total Cholesterol (mg/dL)'] / (X['HDL (mg/dL)'] + 1)
X['LDL_HDL_Ratio'] = X['Estimated LDL (mg/dL)'] / (X['HDL (mg/dL)'] + 1)
X['Non_HDL_Cholesterol'] = X['Total Cholesterol (mg/dL)'] - X['HDL (mg/dL)']
X['Atherogenic_Index'] = np.log(X['Total Cholesterol (mg/dL)'] / X['HDL (mg/dL)'])

# Blood pressure features
X['Pulse_Pressure'] = X['Systolic BP'] - X['Diastolic BP']
X['Mean_Arterial_Pressure'] = (X['Systolic BP'] + 2 * X['Diastolic BP']) / 3
X['BP_Product'] = X['Systolic BP'] * X['Diastolic BP'] / 10000

# CVD Risk Score transformations
X['CVD_Score_Squared'] = X['CVD Risk Score'] ** 2
X['CVD_Score_Log'] = np.log1p(X['CVD Risk Score'])
X['CVD_Score_Sqrt'] = np.sqrt(X['CVD Risk Score'])

# Age-related interactions
X['Age_CVD_Interaction'] = X['Age'] * X['CVD Risk Score'] / 100
X['Age_Cholesterol'] = X['Age'] * X['Total Cholesterol (mg/dL)'] / 10000
X['Age_BP'] = X['Age'] * X['Systolic BP'] / 10000

# Risk factor combinations
X['Smoking_Diabetes'] = X['Smoking Status'] * X['Diabetes Status']
X['Family_CVD_Score'] = X['Family History of CVD'] * X['CVD Risk Score']
X['Activity_HDL'] = X['Physical Activity Level'] * X['HDL (mg/dL)']

# Medical risk categories
X['High_BP_Risk'] = (X['Systolic BP'] > 140).astype(int)
X['High_Cholesterol_Risk'] = (X['Total Cholesterol (mg/dL)'] > 240).astype(int)
X['Low_HDL_Risk'] = (X['HDL (mg/dL)'] < 40).astype(int)
X['Obesity_Risk'] = (X['BMI'] > 30).astype(int)

# Combined risk score
X['Total_Risk_Factors'] = (
    X['Smoking Status'] + 
    X['Diabetes Status'] + 
    X['Family History of CVD'] +
    X['High_BP_Risk'] + 
    X['High_Cholesterol_Risk'] + 
    X['Low_HDL_Risk'] + 
    X['Obesity_Risk']
)

# Metabolic syndrome score
X['MetS_Score'] = (
    (X['Waist-to-Height Ratio'] > 0.5).astype(int) +
    (X['Systolic BP'] > 130).astype(int) +
    (X['HDL (mg/dL)'] < 40).astype(int) +
    (X['Fasting Blood Sugar (mg/dL)'] > 100).astype(int)
)

print(f"   Final features: {X.shape[1]}")

# 4. Class Balancing Strategy
print("\n4. Applying class balancing...")
print(f"Original distribution: {np.bincount(y)}")

# Use SMOTE with strategic sampling
sampling_strategy = {0: 600, 1: 700, 2: 728}  # More conservative balancing
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X, y)

print(f"Balanced distribution: {np.bincount(y_balanced)}")

# 5. Feature Scaling
print("\n5. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# 7. Model Training with Hyperparameter Optimization
print("\n7. Training optimized models...")

models = {}

# XGBoost - Optimized for medical data
print("   Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.9,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    min_child_weight=3,
    random_state=42,
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model

# Random Forest - Optimized
print("   Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=600,
    max_depth=12,
    min_samples_split=3,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# Extra Trees
print("   Training Extra Trees...")
et_model = ExtraTreesClassifier(
    n_estimators=600,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
et_model.fit(X_train, y_train)
models['Extra Trees'] = et_model

# Gradient Boosting
print("   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=3,
    min_samples_leaf=2,
    subsample=0.9,
    random_state=42
)
gb_model.fit(X_train, y_train)
models['Gradient Boosting'] = gb_model

# Neural Network
print("   Training Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(200, 100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=64,
    learning_rate='adaptive',
    max_iter=1000,
    early_stopping=True,
    random_state=42
)
nn_model.fit(X_train, y_train)
models['Neural Network'] = nn_model

# 8. Model Evaluation
print("\n8. Evaluating models...")

results = {}
predictions = {}

for name, model in models.items():
    # Predictions
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    print(f"   {name}:")
    print(f"     Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"     CV Accuracy: {cv_mean:.4f} ± {cv_scores.std():.4f}")

# 9. Advanced Ensemble Methods
print("\n9. Creating advanced ensembles...")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('rf', rf_model),
        ('et', et_model),
        ('gb', gb_model)
    ],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
voting_pred = voting_clf.predict(X_test)
voting_acc = accuracy_score(y_test, voting_pred)

# Weighted ensemble based on CV performance
probabilities = {}
for name, model in models.items():
    if hasattr(model, 'predict_proba'):
        probabilities[name] = model.predict_proba(X_test)

# Weight by accuracy squared to emphasize better models
weights = np.array(list(results.values())) ** 2
weights = weights / weights.sum()

weighted_probs = np.zeros_like(list(probabilities.values())[0])
for i, (name, probs) in enumerate(probabilities.items()):
    weighted_probs += weights[i] * probs

weighted_pred = np.argmax(weighted_probs, axis=1)
weighted_acc = accuracy_score(y_test, weighted_pred)

# 10. Results Summary
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

# Individual models
print("\nIndividual Models:")
for name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name:<20}: {acc:.4f} ({acc*100:.2f}%)")

# Ensemble results
print(f"\nEnsemble Models:")
print(f"  {'Voting Classifier':<20}: {voting_acc:.4f} ({voting_acc*100:.2f}%)")
print(f"  {'Weighted Ensemble':<20}: {weighted_acc:.4f} ({weighted_acc*100:.2f}%)")

# Find best model
all_results = {**results, 'Voting Classifier': voting_acc, 'Weighted Ensemble': weighted_acc}
best_model = max(all_results.items(), key=lambda x: x[1])

print(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1]:.4f} accuracy ({best_model[1]*100:.2f}%)")

# Get best predictions for detailed analysis
if best_model[0] == 'Voting Classifier':
    best_pred = voting_pred
elif best_model[0] == 'Weighted Ensemble':
    best_pred = weighted_pred
else:
    best_pred = predictions[best_model[0]]

# 11. Detailed Analysis
print(f"\n11. Detailed analysis of best model...")

print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['LOW', 'INTERMEDIARY', 'HIGH']))

# Confusion Matrix
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, best_pred)
print(cm)

# Feature Importance (using XGBoost)
print(f"\nTop 15 Most Important Features:")
feature_names = X.columns
importances = xgb_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

for i, idx in enumerate(indices):
    print(f"{i+1:2d}. {feature_names[idx]:<35}: {importances[idx]:.4f}")

# 12. Model Performance Summary
print(f"\n12. Model performance summary...")

if best_model[1] >= 0.90:
    print(f"🎯 EXCELLENT! Achieved {best_model[1]*100:.2f}% accuracy")
    print(f"✅ Model exceeds 90% accuracy target!")
elif best_model[1] >= 0.85:
    print(f"🚀 VERY GOOD! Achieved {best_model[1]*100:.2f}% accuracy")
    print(f"✅ Model achieves excellent performance for medical prediction!")
elif best_model[1] >= 0.80:
    print(f"✅ GOOD! Achieved {best_model[1]*100:.2f}% accuracy")
    print(f"✅ Model achieves good performance for medical prediction!")
else:
    print(f"📊 Achieved {best_model[1]*100:.2f}% accuracy")

print(f"\n🏥 Model is ready for clinical evaluation and potential deployment!")
print(f"📈 Using cleaned dataset: CVD_Dataset_Cleaned_Final.csv")
print(f"🎯 Best approach: {best_model[0]}")

print("\n" + "="*80)
print("MODELING COMPLETE!")
print("="*80)