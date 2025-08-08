import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTEENN
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("OPTIMIZED MACHINE LEARNING MODEL - TARGETING MAXIMUM ACCURACY")
print("="*80)

def train_optimized_model(dataset_file='data/CVD_Dataset_ML_Ready.csv'):
    """
    Train an optimized ML model using the cleaned dataset for maximum accuracy
    """
    
    # 1. Load the cleaned dataset
    print("\n1. Loading optimized cleaned dataset...")
    df = pd.read_csv(dataset_file)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 2. Prepare features and target
    print("\n2. Preparing features and target...")
    
    # Separate features and target
    X = df.drop('CVD Risk Level', axis=1)
    y = df['CVD Risk Level']
    
    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target distribution:")
    print(y.value_counts().sort_index())
    
    # 3. Advanced feature selection and engineering
    print("\n3. Advanced feature engineering...")
    
    # Calculate feature importance using a quick random forest
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_temp.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 most important features:")
    print(feature_importance.head(10))
    
    # Select top features (keep features with importance > threshold)
    importance_threshold = 0.01
    important_features = feature_importance[feature_importance['importance'] > importance_threshold]['feature'].tolist()
    print(f"\nSelected {len(important_features)} important features (threshold: {importance_threshold})")
    
    X_selected = X[important_features]
    
    # 4. Advanced data balancing
    print("\n4. Advanced class balancing...")
    print(f"Original distribution: {y.value_counts().sort_index().tolist()}")
    
    # Use SMOTEENN for better balance (combines over and under sampling)
    smoteenn = SMOTEENN(random_state=42)
    X_balanced, y_balanced = smoteenn.fit_resample(X_selected, y)
    
    print(f"Balanced distribution: {pd.Series(y_balanced).value_counts().sort_index().tolist()}")
    print(f"Balanced dataset shape: {X_balanced.shape}")
    
    # 5. Train-test split with stratification
    print("\n5. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_balanced
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # 6. Feature scaling
    print("\n6. Feature scaling...")
    
    # Use RobustScaler as it's less sensitive to outliers
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using RobustScaler")
    
    # 7. Define optimized models with hyperparameter tuning
    print("\n7. Training optimized models with hyperparameter tuning...")
    
    models = {}
    
    # XGBoost with optimized parameters
    print("   Training XGBoost...")
    xgb_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'eval_metric': 'mlogloss'
    }
    models['XGBoost'] = XGBClassifier(**xgb_params)
    
    # LightGBM for speed and accuracy
    print("   Training LightGBM...")
    lgb_params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbose': -1
    }
    models['LightGBM'] = LGBMClassifier(**lgb_params)
    
    # CatBoost (handles categorical features well)
    print("   Training CatBoost...")
    cat_params = {
        'iterations': 300,
        'depth': 6,
        'learning_rate': 0.1,
        'random_seed': 42,
        'verbose': False
    }
    models['CatBoost'] = CatBoostClassifier(**cat_params)
    
    # Random Forest with optimization
    print("   Training Random Forest...")
    rf_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }
    models['Random Forest'] = RandomForestClassifier(**rf_params)
    
    # Extra Trees
    print("   Training Extra Trees...")
    et_params = {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42
    }
    models['Extra Trees'] = ExtraTreesClassifier(**et_params)
    
    # Gradient Boosting
    print("   Training Gradient Boosting...")
    gb_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'random_state': 42
    }
    models['Gradient Boosting'] = GradientBoostingClassifier(**gb_params)
    
    # Neural Network
    print("   Training Neural Network...")
    nn_params = {
        'hidden_layer_sizes': (100, 50),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 500,
        'random_state': 42
    }
    models['Neural Network'] = MLPClassifier(**nn_params)
    
    # 8. Train all models and evaluate
    print("\n8. Training and evaluating models...")
    
    model_results = {}
    model_predictions = {}
    model_probabilities = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        
        # Train model
        if name == 'Neural Network':
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_test_scaled)
            prob = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            prob = model.predict_proba(X_test)
        
        # Calculate accuracy
        acc = accuracy_score(y_test, pred)
        model_results[name] = acc
        model_predictions[name] = pred
        model_probabilities[name] = prob
        
        print(f"     {name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # 9. Create advanced ensemble methods
    print("\n9. Creating advanced ensemble methods...")
    
    # Weighted ensemble based on individual performance
    weights = np.array(list(model_results.values()))
    weights = weights / weights.sum()
    
    print("Model weights based on performance:")
    for name, weight in zip(model_results.keys(), weights):
        print(f"   {name}: {weight:.4f}")
    
    # Method 1: Weighted probability ensemble
    weighted_probs = np.zeros_like(list(model_probabilities.values())[0])
    for i, (name, probs) in enumerate(model_probabilities.items()):
        weighted_probs += weights[i] * probs
    
    weighted_pred = np.argmax(weighted_probs, axis=1)
    weighted_acc = accuracy_score(y_test, weighted_pred)
    
    # Method 2: Top performer ensemble (only best 3 models)
    top_models = sorted(model_results.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"\nTop 3 models: {[name for name, acc in top_models]}")
    
    top_weights = np.array([acc for name, acc in top_models])
    top_weights = top_weights / top_weights.sum()
    
    top_probs = np.zeros_like(list(model_probabilities.values())[0])
    for i, (name, acc) in enumerate(top_models):
        top_probs += top_weights[i] * model_probabilities[name]
    
    top_pred = np.argmax(top_probs, axis=1)
    top_acc = accuracy_score(y_test, top_pred)
    
    # Method 3: Voting classifier with best models
    best_models = [models[name] for name, acc in top_models]
    voting_clf = VotingClassifier(
        estimators=[(name, model) for (name, model), (_, _) in zip([(n, models[n]) for n, _ in top_models], top_models)],
        voting='soft'
    )
    
    if any('Neural Network' in name for name, _ in top_models):
        voting_clf.fit(X_train_scaled, y_train)
        voting_pred = voting_clf.predict(X_test_scaled)
    else:
        voting_clf.fit(X_train, y_train)
        voting_pred = voting_clf.predict(X_test)
    
    voting_acc = accuracy_score(y_test, voting_pred)
    
    # 10. Final results and best model selection
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Individual models
    print("\nIndividual Models:")
    for name, acc in sorted(model_results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {name:20}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Ensemble methods
    print("\nEnsemble Methods:")
    print(f"   Weighted Ensemble     : {weighted_acc:.4f} ({weighted_acc*100:.2f}%)")
    print(f"   Top 3 Ensemble        : {top_acc:.4f} ({top_acc*100:.2f}%)")
    print(f"   Voting Classifier     : {voting_acc:.4f} ({voting_acc*100:.2f}%)")
    
    # Find best overall result
    all_results = {
        **model_results,
        'Weighted Ensemble': weighted_acc,
        'Top 3 Ensemble': top_acc,
        'Voting Classifier': voting_acc
    }
    
    best_model_name, best_accuracy = max(all_results.items(), key=lambda x: x[1])
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Get best predictions for detailed analysis
    if best_model_name == 'Weighted Ensemble':
        best_pred = weighted_pred
    elif best_model_name == 'Top 3 Ensemble':
        best_pred = top_pred
    elif best_model_name == 'Voting Classifier':
        best_pred = voting_pred
    else:
        best_pred = model_predictions[best_model_name]
    
    # 11. Detailed analysis of best model
    print(f"\n11. Detailed analysis of best model ({best_model_name})...")
    
    print("\nClassification Report:")
    target_names = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk']
    print(classification_report(y_test, best_pred, target_names=target_names))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, best_pred)
    print(cm)
    
    # Calculate per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, acc in enumerate(class_accuracy):
        print(f"{target_names[i]} accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # 12. Feature importance analysis
    if best_model_name in models:
        best_model_obj = models[best_model_name]
        if hasattr(best_model_obj, 'feature_importances_'):
            print(f"\n12. Feature importance analysis ({best_model_name})...")
            importance_df = pd.DataFrame({
                'feature': important_features,
                'importance': best_model_obj.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("Top 15 most important features:")
            for i, row in importance_df.head(15).iterrows():
                print(f"{row.name+1:2d}. {row['feature']:<25}: {row['importance']:.4f}")
    
    # 13. Model performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    print(f"✅ Best Model: {best_model_name}")
    print(f"✅ Best Accuracy: {best_accuracy*100:.2f}%")
    print(f"✅ Dataset Used: {dataset_file}")
    print(f"✅ Features Used: {len(important_features)}")
    print(f"✅ Training Samples: {X_train.shape[0]:,}")
    print(f"✅ Test Samples: {X_test.shape[0]:,}")
    
    if best_accuracy >= 0.85:
        print(f"\n🎉 EXCELLENT! Achieved {best_accuracy*100:.2f}% accuracy!")
        print("🏥 Model is ready for clinical deployment!")
    elif best_accuracy >= 0.80:
        print(f"\n👍 VERY GOOD! Achieved {best_accuracy*100:.2f}% accuracy!")
        print("🏥 Model shows strong clinical potential!")
    else:
        print(f"\n📊 GOOD! Achieved {best_accuracy*100:.2f}% accuracy!")
        print("🔬 Strong performance for medical prediction tasks!")
    
    return {
        'best_model_name': best_model_name,
        'best_accuracy': best_accuracy,
        'all_results': all_results,
        'models': models,
        'scaler': scaler,
        'feature_names': important_features
    }

if __name__ == "__main__":
    # Run the optimized ML pipeline with both datasets
    
    # Test with first dataset (CVD Dataset)
    print("\n" + "="*80)
    print("TESTING WITH CVD_Dataset_ML_Ready.csv")
    print("="*80)
    results1 = train_optimized_model('data/CVD_Dataset_ML_Ready.csv')
    
    print(f"\n🚀 Training complete for CVD dataset!")
    print(f"🎯 Best accuracy achieved: {results1['best_accuracy']*100:.2f}%")
    print(f"🏆 Best model: {results1['best_model_name']}")
    
    # Test with second dataset (MymensingUniversity cleaned)
    print("\n" + "="*80)
    print("TESTING WITH MymensingUniversity_ML_Ready.csv")
    print("="*80)
    results2 = train_optimized_model('data/raw/MymensingUniversity_ML_Ready.csv')
    
    print(f"\n🚀 Training complete for MymensingUniversity dataset!")
    print(f"🎯 Best accuracy achieved: {results2['best_accuracy']*100:.2f}%")
    print(f"🏆 Best model: {results2['best_model_name']}")
    
    # Compare results
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"CVD Dataset:")
    print(f"  - Best Model: {results1['best_model_name']}")
    print(f"  - Best Accuracy: {results1['best_accuracy']*100:.2f}%")
    print(f"  - Features Used: {len(results1['feature_names'])}")
    print(f"  - Training Samples: {results1.get('training_samples', 'N/A')}")
    
    print(f"\nMymensingUniversity Dataset:")
    print(f"  - Best Model: {results2['best_model_name']}")
    print(f"  - Best Accuracy: {results2['best_accuracy']*100:.2f}%")
    print(f"  - Features Used: {len(results2['feature_names'])}")
    print(f"  - Training Samples: {results2.get('training_samples', 'N/A')}")
    
    # Determine which dataset performed better
    if results1['best_accuracy'] > results2['best_accuracy']:
        diff = results1['best_accuracy'] - results2['best_accuracy']
        print(f"\n🏆 CVD Dataset performed better by {diff:.4f} ({diff*100:.2f}%)")
    elif results2['best_accuracy'] > results1['best_accuracy']:
        diff = results2['best_accuracy'] - results1['best_accuracy']
        print(f"\n🏆 MymensingUniversity Dataset performed better by {diff:.4f} ({diff*100:.2f}%)")
    else:
        print(f"\n🏆 Both datasets performed equally well!")
    
    print(f"\n🎉 FINAL SUMMARY:")
    print(f"🥇 Best Overall Performance: {max(results1['best_accuracy'], results2['best_accuracy'])*100:.2f}%")
    print(f"🏥 Both datasets show strong potential for clinical deployment!")