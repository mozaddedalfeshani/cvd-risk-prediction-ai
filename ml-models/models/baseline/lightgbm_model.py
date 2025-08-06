import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("SIMPLE LIGHTGBM MODEL")
print("="*60)

# Load data
print("Loading dataset...")
df = pd.read_csv('../data/CVD_Dataset_ML_Ready.csv')
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']

print(f"Dataset shape: {df.shape}")
print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Split data
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# Train LightGBM model
print("\nTraining LightGBM model...")
model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
target_names = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk']
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
print("\nTop 10 Most Important Features:")
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance_df.head(10).iterrows():
    print(f"{row.name+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")

print(f"\n✅ Model trained successfully!")
print(f"🎯 Accuracy: {accuracy*100:.2f}%")
print(f"🏆 Model: LightGBM") 