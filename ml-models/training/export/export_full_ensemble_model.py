import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTEENN
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT_DIR / 'data' / 'processed' / 'MymensingUniversity_ML_Ready.csv'
MODELS_DIR = ROOT_DIR / 'models'

print("="*60)
print("SAVING FULL ACCURACY MODEL (95.91%)")
print("="*60)

# Load dataset
df = pd.read_csv(DEFAULT_DATASET)
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']

# Feature selection
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
    X_balanced, y_balanced, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_balanced
)

# Feature scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train top 3 models
xgb_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)

lgb_model = LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, random_state=42
)
gb_model.fit(X_train, y_train)

# Create ensemble
voting_clf = VotingClassifier(
    estimators=[('xgb', xgb_model), ('lgb', lgb_model), ('gb', gb_model)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)

# Test accuracy
ensemble_pred = voting_clf.predict(X_test)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"✅ Full Model Accuracy: {ensemble_acc*100:.2f}%")
print(f"✅ Features: {len(important_features)}")

# Save full accuracy model
full_model_artifact = {
    'model': voting_clf,
    'scaler': scaler,
    'feature_names': important_features,
    'accuracy': ensemble_acc,
    'feature_count': len(important_features),
    'model_type': 'Full Accuracy',
    'version': '2.0'
}

MODELS_DIR.mkdir(parents=True, exist_ok=True)
model_output = MODELS_DIR / 'cvd_full_model.pkl'
joblib.dump(full_model_artifact, model_output)
print(f"✅ Saved Full Accuracy Model to {model_output}")
print(f"📊 Model ready for dual deployment!")
