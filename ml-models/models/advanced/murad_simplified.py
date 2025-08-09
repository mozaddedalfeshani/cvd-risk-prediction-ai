import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("QUICK ASSESSMENT MODEL - 8 KEY FEATURES ONLY")
print("="*80)

def train_quick_model(dataset_file='data/CVD_Dataset_ML_Ready.csv'):
    """
    Train a quick assessment model using only 8 most important features
    Target: 90%+ accuracy with minimal input burden
    """
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_file)
    print(f"Dataset shape: {df.shape}")
    
    # 2. Use only top 8 most important features for quick assessment
    quick_features = [
        'CVD Risk Score',
        'Age', 
        'Systolic BP',
        'Fasting Blood Sugar (mg/dL)',
        'BMI',
        'Total Cholesterol (mg/dL)',
        'Diastolic BP',
        'HDL (mg/dL)'
    ]
    
    print(f"\n2. Quick Assessment Features ({len(quick_features)} features):")
    for i, feat in enumerate(quick_features, 1):
        print(f"   {i}. {feat}")
    
    # 3. Prepare data
    X = df[quick_features]
    y = df['CVD Risk Level']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().sort_index().tolist()}")
    
    # 4. Class balancing
    print("\n3. Applying class balancing...")
    smoteenn = SMOTEENN(random_state=42)
    X_balanced, y_balanced = smoteenn.fit_resample(X, y)
    print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().sort_index().tolist()}")
    
    # 5. Train-test split
    print("\n4. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_balanced
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 6. Feature scaling
    print("\n5. Feature scaling...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Train optimized models for 90%+ target
    print("\n6. Training models for 90%+ accuracy...")
    
    # XGBoost - tuned for higher accuracy
    print("   Training XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # LightGBM - tuned for higher accuracy
    print("   Training LightGBM...")
    lgb_model = LGBMClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_test)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    
    # Gradient Boosting - tuned for higher accuracy
    print("   Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_acc = accuracy_score(y_test, gb_pred)
    
    # 8. Create ensemble
    print("\n7. Creating optimized ensemble...")
    voting_clf = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('lgb', lgb_model),
            ('gb', gb_model)
        ],
        voting='soft'
    )
    
    voting_clf.fit(X_train, y_train)
    ensemble_pred = voting_clf.predict(X_test)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    # 9. Results
    print("\n" + "="*80)
    print("QUICK ASSESSMENT MODEL RESULTS")
    print("="*80)
    
    print(f"\nIndividual Models:")
    print(f"   XGBoost        : {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    print(f"   LightGBM       : {lgb_acc:.4f} ({lgb_acc*100:.2f}%)")
    print(f"   Gradient Boost : {gb_acc:.4f} ({gb_acc*100:.2f}%)")
    print(f"   Quick Ensemble : {ensemble_acc:.4f} ({ensemble_acc*100:.2f}%)")
    
    # 10. Detailed analysis
    print(f"\nClassification Report (Quick Ensemble):")
    target_names = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk']
    print(classification_report(y_test, ensemble_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, ensemble_pred)
    print(cm)
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"{target_names[i]} accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # 11. Feature importance
    print(f"\n8. Feature importance analysis...")
    importance_df = pd.DataFrame({
        'feature': quick_features,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Feature importance (Quick Assessment Model):")
    for i, row in importance_df.iterrows():
        print(f"   {row['feature']:<30}: {row['importance']:.4f}")
    
    # 12. Save the quick model
    print(f"\n9. Saving Quick Assessment Model...")
    
    model_artifact = {
        'model': voting_clf,
        'scaler': scaler,
        'feature_names': quick_features,
        'accuracy': ensemble_acc,
        'feature_count': len(quick_features),
        'model_type': 'Quick Assessment',
        'version': '2.0'
    }
    
    joblib.dump(model_artifact, 'models/cvd_quick_model.pkl')
    print("✅ Saved model to models/cvd_quick_model.pkl")
    
    # 13. Summary
    print("\n" + "="*80)
    print("QUICK ASSESSMENT MODEL SUMMARY")
    print("="*80)
    print(f"✅ Model: Quick Assessment Ensemble (XGB + LGB + GB)")
    print(f"✅ Features: {len(quick_features)} key features only")
    print(f"✅ Accuracy: {ensemble_acc*100:.2f}%")
    print(f"✅ Training samples: {X_train.shape[0]:,}")
    print(f"✅ Test samples: {X_test.shape[0]:,}")
    
    if ensemble_acc >= 0.90:
        print(f"🎉 EXCELLENT! Target 90%+ accuracy achieved!")
    else:
        print(f"📊 Good performance for quick assessment!")
    
    print(f"✅ Status: Ready for dual-model deployment!")
    
    return {
        'model': voting_clf,
        'scaler': scaler,
        'feature_names': quick_features,
        'accuracy': ensemble_acc,
        'individual_accuracies': {
            'XGBoost': xgb_acc,
            'LightGBM': lgb_acc,
            'GradientBoosting': gb_acc,
            'QuickEnsemble': ensemble_acc
        }
    }

if __name__ == "__main__":
    print("🚀 Training Quick Assessment CVD risk prediction model...")
    results = train_quick_model('data/CVD_Dataset_ML_Ready.csv')
    print(f"\n🎉 Quick model training complete!")
    print(f"🎯 Quick Assessment accuracy: {results['accuracy']*100:.2f}%")
    print(f"📊 Using only {len(results['feature_names'])} key features")
    print(f"💾 Model saved and ready for dual-model API!")
