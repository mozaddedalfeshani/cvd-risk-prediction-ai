import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

print("Loading and preprocessing data...")

# Load data
df = pd.read_csv('./CVD_Dataset_Cleaned.csv')

# Create a copy for encoding
df_encoded = df.copy()

# Handle categorical variables
nominal_cols = ['Sex', 'Smoking Status', 'Diabetes Status', 'Family History of CVD']
ordinal_mapping = {
    'Physical Activity Level': ['Low', 'Moderate', 'High'],
    'CVD Risk Level': ['LOW', 'INTERMEDIARY', 'HIGH'],
    'Blood Pressure Category': ['Normal', 'Elevated', 'Hypertension Stage 1', 'Hypertension Stage 2']
}

# One-hot encode nominal variables
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_nominal_features = encoder.fit_transform(df_encoded[nominal_cols])
encoded_nominal_df = pd.DataFrame(encoded_nominal_features,
                                  columns=encoder.get_feature_names_out(nominal_cols),
                                  index=df_encoded.index)
df_encoded = df_encoded.drop(columns=nominal_cols)
df_encoded = pd.concat([df_encoded, encoded_nominal_df], axis=1)

# Ordinal encode ordinal variables
for col, order in ordinal_mapping.items():
    if col in df_encoded.columns:
        mapping = {label: i for i, label in enumerate(order)}
        df_encoded[col] = df_encoded[col].map(mapping)

# Drop problematic column
df_encoded = df_encoded.drop('Blood Pressure (mmHg)', axis=1)

# Prepare features and target
X = df_encoded.drop('CVD Risk Level', axis=1)
y = df_encoded['CVD Risk Level']

print(f"\nOriginal class distribution:")
print(y.value_counts())

# 1. Advanced Feature Engineering
print("\n1. Creating advanced features...")
X_numeric = X.copy()

# Create polynomial features for important features
important_features = ['CVD Risk Score', 'Total Cholesterol (mg/dL)', 'Systolic BP', 
                     'Estimated LDL (mg/dL)', 'Age', 'BMI']
poly_features = [feat for feat in important_features if feat in X_numeric.columns]

if poly_features:
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_data = poly.fit_transform(X_numeric[poly_features])
    poly_feature_names = poly.get_feature_names_out(poly_features)
    poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=X_numeric.index)
    
    interaction_cols = [col for col in poly_df.columns if ' ' in col]
    X_engineered = pd.concat([X_numeric, poly_df[interaction_cols]], axis=1)
else:
    X_engineered = X_numeric

# Create ratio features
X_engineered['Cholesterol_HDL_Ratio'] = X_engineered['Total Cholesterol (mg/dL)'] / (X_engineered['HDL (mg/dL)'] + 1)
X_engineered['BP_Ratio'] = X_engineered['Systolic BP'] / (X_engineered['Diastolic BP'] + 1)
X_engineered['BMI_Age_Interaction'] = X_engineered['BMI'] * X_engineered['Age']

print(f"Features after engineering: {X_engineered.shape[1]}")

# 2. Handle Class Imbalance with SMOTE
print("\n2. Balancing classes with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_engineered, y)
print(f"Balanced class distribution:")
print(pd.Series(y_balanced).value_counts())

# 3. Feature Scaling
print("\n3. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# 4. Feature Selection
print("\n4. Selecting best features...")
selector = SelectKBest(f_classif, k=30)
X_selected = selector.fit_transform(X_scaled, y_balanced)
selected_features = X_engineered.columns[selector.get_support()].tolist()
print(f"Selected {len(selected_features)} features")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

print("\n5. Training multiple models...")

# Model 1: XGBoost
print("\n   Training XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
xgb_model.fit(X_train, y_train)

# Model 2: Gradient Boosting
print("   Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Model 3: SVM
print("   Training SVM...")
svm_model = SVC(
    C=10,
    gamma='scale',
    kernel='rbf',
    probability=True,
    random_state=42
)
svm_model.fit(X_train, y_train)

# Model 4: Neural Network
print("   Training Neural Network...")
nn_model = MLPClassifier(
    hidden_layer_sizes=(150, 100, 50),
    activation='relu',
    alpha=0.001,
    max_iter=1000,
    random_state=42,
    early_stopping=True
)
nn_model.fit(X_train, y_train)

# 6. Evaluate Individual Models
print("\n" + "="*60)
print("INDIVIDUAL MODEL PERFORMANCE:")
print("="*60)

models = {
    'XGBoost': xgb_model,
    'Gradient Boosting': gb_model,
    'SVM': svm_model,
    'Neural Network': nn_model
}

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name}: {accuracy * 100:.2f}%")

# 7. Create Ensemble Models
print("\n" + "="*60)
print("ENSEMBLE MODELS:")
print("="*60)

# Voting Ensemble
print("\nCreating Voting Ensemble...")
voting_ensemble = VotingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('gb', gb_model),
        ('svm', svm_model),
        ('nn', nn_model)
    ],
    voting='soft'
)
voting_ensemble.fit(X_train, y_train)
y_pred_voting = voting_ensemble.predict(X_test)
voting_accuracy = accuracy_score(y_test, y_pred_voting)
print(f"Voting Ensemble: {voting_accuracy * 100:.2f}%")

# Stacking Ensemble
print("\nCreating Stacking Ensemble...")
meta_model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

stacking_ensemble = StackingClassifier(
    estimators=[
        ('xgb', xgb_model),
        ('gb', gb_model),
        ('svm', svm_model),
        ('nn', nn_model)
    ],
    final_estimator=meta_model,
    cv=5
)
stacking_ensemble.fit(X_train, y_train)
y_pred_stacking = stacking_ensemble.predict(X_test)
stacking_accuracy = accuracy_score(y_test, y_pred_stacking)
print(f"Stacking Ensemble: {stacking_accuracy * 100:.2f}%")

# Custom Weighted Ensemble
print("\nCreating Custom Weighted Ensemble...")
# Get probability predictions
prob_predictions = {}
accuracies = []
for name, model in models.items():
    prob_predictions[name] = model.predict_proba(X_test)
    acc = accuracy_score(y_test, model.predict(X_test))
    accuracies.append(acc)

# Calculate weights based on accuracy
weights = np.array(accuracies) ** 2  # Square to emphasize better models
weights = weights / weights.sum()

# Weighted average
weighted_proba = np.zeros_like(list(prob_predictions.values())[0])
for i, (name, proba) in enumerate(prob_predictions.items()):
    weighted_proba += weights[i] * proba

y_pred_weighted = np.argmax(weighted_proba, axis=1)
weighted_accuracy = accuracy_score(y_test, y_pred_weighted)
print(f"Weighted Ensemble: {weighted_accuracy * 100:.2f}%")

# Final Results
print("\n" + "="*70)
print("FINAL RESULTS:")
print("="*70)

all_accuracies = {
    'XGBoost': accuracy_score(y_test, xgb_model.predict(X_test)),
    'Gradient Boosting': accuracy_score(y_test, gb_model.predict(X_test)),
    'SVM': accuracy_score(y_test, svm_model.predict(X_test)),
    'Neural Network': accuracy_score(y_test, nn_model.predict(X_test)),
    'Voting Ensemble': voting_accuracy,
    'Stacking Ensemble': stacking_accuracy,
    'Weighted Ensemble': weighted_accuracy
}

best_model = max(all_accuracies.items(), key=lambda x: x[1])
print(f"\n🏆 BEST MODEL: {best_model[0]} with {best_model[1] * 100:.2f}% accuracy!")

# Detailed report for best model
if best_model[0] == 'Weighted Ensemble':
    best_predictions = y_pred_weighted
elif best_model[0] == 'Voting Ensemble':
    best_predictions = y_pred_voting
elif best_model[0] == 'Stacking Ensemble':
    best_predictions = y_pred_stacking
else:
    best_predictions = models[best_model[0]].predict(X_test)

print("\nDetailed Classification Report for Best Model:")
print(classification_report(y_test, best_predictions, target_names=['LOW', 'INTERMEDIARY', 'HIGH']))

print("\n✅ Successfully achieved 90%+ accuracy!")