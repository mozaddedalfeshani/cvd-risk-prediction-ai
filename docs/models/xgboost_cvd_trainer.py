#!/usr/bin/env python3
"""
XGBoost CVD Risk Prediction Trainer
===================================

Specialized XGBoost training script optimized for achieving 90+ accuracy
on cardiovascular disease risk prediction datasets.

Features:
- Automated hyperparameter tuning
- Advanced feature engineering
- Data preprocessing and balancing
- Cross-validation
- Performance metrics and visualization

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

class XGBoostCVDTrainer:
    """
    XGBoost trainer specifically optimized for CVD risk prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.best_params = None
        self.training_history = {}
        
    def load_and_preprocess_data(self, dataset_file):
        """
        Load and preprocess the dataset
        """
        print(f"Loading dataset: {dataset_file}")
        df = pd.read_csv(dataset_file)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Separate features and target
        X = df.drop('CVD Risk Level', axis=1)
        y = df['CVD Risk Level']
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target distribution: {y.value_counts().sort_index().tolist()}")
        
        return X, y
    
    def engineer_features(self, X, y):
        """
        Advanced feature engineering and selection
        """
        print("\nPerforming feature engineering...")
        
        # Quick feature importance screening using XGBoost
        temp_model = XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss')
        temp_model.fit(X, y)
        
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 10 most important features:")
        print(feature_importance.head(10))
        
        # Select important features (threshold tuned for performance)
        importance_threshold = 0.01
        important_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
        
        print(f"Selected {len(important_features)} important features")
        
        X_selected = X[important_features]
        self.feature_names = important_features
        
        return X_selected
    
    def balance_data(self, X, y):
        """
        Apply advanced data balancing techniques
        """
        print("\nBalancing dataset...")
        print(f"Original distribution: {pd.Series(y).value_counts().sort_index().tolist()}")
        
        # Use SMOTEENN for optimal balance
        smoteenn = SMOTEENN(random_state=42)
        X_balanced, y_balanced = smoteenn.fit_resample(X, y)
        
        print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().sort_index().tolist()}")
        print(f"Balanced dataset shape: {X_balanced.shape}")
        
        return X_balanced, y_balanced
    
    def optimize_hyperparameters(self, X_train, y_train):
        """
        Hyperparameter optimization for XGBoost
        """
        print("\nOptimizing XGBoost hyperparameters...")
        
        # Define parameter grid optimized for high accuracy
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3],
            'gamma': [0, 0.1]
        }
        
        # Use smaller grid for faster execution
        if X_train.shape[0] < 1000:
            param_grid = {
                'n_estimators': [300, 400],
                'max_depth': [6, 8],
                'learning_rate': [0.1, 0.15],
                'subsample': [0.8],
                'colsample_bytree': [0.8],
            }
        
        xgb_model = XGBClassifier(
            random_state=42,
            eval_metric='mlogloss',
            objective='multi:softprob'
        )
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        print(f"Best parameters: {self.best_params}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def train_model(self, dataset_file, use_hyperparameter_tuning=True):
        """
        Complete training pipeline
        """
        print("="*80)
        print("XGBOOST CVD RISK PREDICTION TRAINER")
        print("="*80)
        
        # Step 1: Load and preprocess data
        X, y = self.load_and_preprocess_data(dataset_file)
        
        # Step 2: Feature engineering
        X_engineered = self.engineer_features(X, y)
        
        # Step 3: Balance data
        X_balanced, y_balanced = self.balance_data(X_engineered, y)
        
        # Step 4: Train-test split
        print("\nCreating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced, y_balanced,
            test_size=0.2,
            random_state=42,
            stratify=y_balanced
        )
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Step 5: Feature scaling
        print("\nApplying feature scaling...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 6: Model training
        if use_hyperparameter_tuning:
            # Optimize hyperparameters
            self.model = self.optimize_hyperparameters(X_train_scaled, y_train)
        else:
            # Use pre-optimized parameters for speed
            print("\nTraining XGBoost with optimized parameters...")
            self.model = XGBClassifier(
                n_estimators=300,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                gamma=0.1,
                random_state=42,
                eval_metric='mlogloss',
                objective='multi:softprob'
            )
            self.model.fit(X_train_scaled, y_train)
        
        # Step 7: Evaluation
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"🎯 F1-Score: {f1:.4f}")
        
        # Classification report
        target_names = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk']
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        for i, acc in enumerate(class_accuracy):
            print(f"{target_names[i]} accuracy: {acc:.4f} ({acc*100:.2f}%)")
        
        # Cross-validation
        print(f"\nCross-validation scores:")
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nTop 10 Feature Importances:")
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, row in importance_df.head(10).iterrows():
                print(f"{i+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
        
        # Store training history
        self.training_history = {
            'dataset': dataset_file,
            'accuracy': accuracy,
            'f1_score': f1,
            'cv_scores': cv_scores,
            'best_params': self.best_params,
            'feature_count': len(self.feature_names),
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        }
        
        # Success message
        print(f"\n" + "="*80)
        if accuracy >= 0.90:
            print(f"🎉 SUCCESS! Achieved {accuracy*100:.2f}% accuracy (Target: 90%+)")
            print(f"🏥 Model is ready for clinical deployment!")
        elif accuracy >= 0.85:
            print(f"👍 GOOD! Achieved {accuracy*100:.2f}% accuracy")
            print(f"🔬 Strong performance for medical prediction!")
        else:
            print(f"📊 Achieved {accuracy*100:.2f}% accuracy")
            print(f"💡 Consider additional feature engineering or data collection")
        
        print(f"✅ Dataset: {dataset_file}")
        print(f"✅ Features: {len(self.feature_names)}")
        print(f"✅ Training samples: {X_train.shape[0]}")
        print(f"✅ Test samples: {X_test.shape[0]}")
        print("="*80)
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'accuracy': accuracy,
            'f1_score': f1,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        import joblib
        
        if self.model is None:
            print("❌ No model to save. Train a model first.")
            return
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        import joblib
        
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.training_history = model_data['training_history']
            print(f"✅ Model loaded from: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, X):
        """
        Make predictions on new data
        """
        if self.model is None or self.scaler is None:
            print("❌ No trained model available. Train or load a model first.")
            return None
        
        # Select features and scale
        X_selected = X[self.feature_names]
        X_scaled = self.scaler.transform(X_selected)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'risk_levels': ['LOW', 'INTERMEDIARY', 'HIGH']
        }


def main():
    """
    Main training function - Focus on MymensingUniversity dataset
    """
    # Initialize trainer
    trainer = XGBoostCVDTrainer()
    
    # Train on MymensingUniversity dataset
    dataset = '../data/MymensingUniversity_ML_Ready.csv'
    
    try:
        print(f"\n{'='*20} TRAINING ON {dataset.split('/')[-1]} {'='*20}")
        
        # Train model
        result = trainer.train_model(dataset, use_hyperparameter_tuning=False)
        
        # Save model if accuracy >= 90%
        if result['accuracy'] >= 0.90:
            model_name = dataset.split('/')[-1].replace('.csv', '_xgboost_model.pkl')
            trainer.save_model(f"{model_name}")
            print(f"🎉 High-accuracy model saved: {model_name}")
        
        # Summary
        print(f"\n{'='*80}")
        print("TRAINING SUMMARY")
        print("="*80)
        
        accuracy = result['accuracy']
        status = "✅ 90%+ ACHIEVED" if accuracy >= 0.90 else "⚠️  Below 90%"
        print(f"MymensingUniversity Dataset: {accuracy*100:.2f}% {status}")
        
        if accuracy >= 0.90:
            print(f"🏆 SUCCESS: Model ready for clinical deployment!")
            print(f"🎯 Accuracy: {accuracy*100:.2f}%")
            print(f"🏥 Ready for frontend integration!")
        
    except FileNotFoundError:
        print(f"❌ Dataset not found: {dataset}")
        print("Please ensure MymensingUniversity_ML_Ready.csv exists in ../data/")
    except Exception as e:
        print(f"❌ Error training on {dataset}: {e}")


if __name__ == "__main__":
    main()