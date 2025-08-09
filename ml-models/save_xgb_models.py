import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SAVING XGBoost-ONLY MODELS FOR API")
print("="*60)

# Load dataset
df = pd.read_csv('data/CVD_Dataset_ML_Ready.csv')
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']

# Feature selection for FULL model
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X, y)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_temp.feature_importances_
}).sort_values('importance', ascending=False)

important_features = feature_importance[feature_importance['importance'] > 0.01]['feature'].tolist()
X_selected = X[important_features]

# Class balancing
smoteenn = SMOTEENN(random_state=42)
X_balanced, y_balanced = smoteenn.fit_resample(X_selected, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)

# Feature scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train FULL XGBoost model
print("Training Full XGBoost model...")
xgb_full = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
)
xgb_full.fit(X_train, y_train)
full_pred = xgb_full.predict(X_test)
full_acc = accuracy_score(y_test, full_pred)

print(f"✅ Full Model Accuracy: {full_acc*100:.2f}%")

# Save full model
full_model_artifact = {
    'model': xgb_full,
    'scaler': scaler,
    'feature_names': important_features,
    'accuracy': full_acc,
    'feature_count': len(important_features),
    'model_type': 'Full Accuracy',
    'version': '2.0'
}
joblib.dump(full_model_artifact, 'models/cvd_full_xgb.pkl')
print("✅ Saved Full XGBoost Model")

# Train QUICK model (8 features)
quick_features = [
    'CVD Risk Score', 'Age', 'Systolic BP', 'Fasting Blood Sugar (mg/dL)',
    'BMI', 'Total Cholesterol (mg/dL)', 'Diastolic BP', 'HDL (mg/dL)'
]

X_quick = df[quick_features]
X_quick_balanced, y_quick_balanced = smoteenn.fit_resample(X_quick, y)

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_quick_balanced, y_quick_balanced, test_size=0.2, random_state=42, stratify=y_quick_balanced
)

scaler_q = RobustScaler()
X_train_q_scaled = scaler_q.fit_transform(X_train_q)
X_test_q_scaled = scaler_q.transform(X_test_q)

print("Training Quick XGBoost model...")
xgb_quick = XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
)
xgb_quick.fit(X_train_q, y_train_q)
quick_pred = xgb_quick.predict(X_test_q)
quick_acc = accuracy_score(y_test_q, quick_pred)

print(f"✅ Quick Model Accuracy: {quick_acc*100:.2f}%")

# Save quick model
quick_model_artifact = {
    'model': xgb_quick,
    'scaler': scaler_q,
    'feature_names': quick_features,
    'accuracy': quick_acc,
    'feature_count': len(quick_features),
    'model_type': 'Quick Assessment',
    'version': '2.0'
}
joblib.dump(quick_model_artifact, 'models/cvd_quick_xgb.pkl')
print("✅ Saved Quick XGBoost Model")

print(f"\n📊 Summary:")
print(f"Full Model (XGBoost): {full_acc*100:.2f}% - {len(important_features)} features")
print(f"Quick Model (XGBoost): {quick_acc*100:.2f}% - {len(quick_features)} features")
print("Ready for API deployment!")
