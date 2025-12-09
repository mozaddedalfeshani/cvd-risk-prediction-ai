import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             f1_score, precision_score, recall_score, log_loss)
from xgboost import XGBClassifier
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("COMPREHENSIVE CVD RISK PREDICTION MODEL WITH DETAILED ANALYSIS")
print("="*80)

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', filename='confusion_matrix.png'):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_training_history(history, filename='training_history.png'):
    """Plot training and validation metrics over folds"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy plot
    axes[0, 0].plot(history['fold'], history['train_accuracy'], 
                    marker='o', label='Training', linewidth=2)
    axes[0, 0].plot(history['fold'], history['val_accuracy'], 
                    marker='s', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Fold', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_title('Accuracy: Training vs Validation', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[0, 1].plot(history['fold'], history['train_f1'], 
                    marker='o', label='Training', linewidth=2)
    axes[0, 1].plot(history['fold'], history['val_f1'], 
                    marker='s', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Fold', fontsize=12)
    axes[0, 1].set_ylabel('F1 Score', fontsize=12)
    axes[0, 1].set_title('F1 Score: Training vs Validation', fontsize=14, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1, 0].plot(history['fold'], history['train_loss'], 
                    marker='o', label='Training', linewidth=2)
    axes[1, 0].plot(history['fold'], history['val_loss'], 
                    marker='s', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Fold', fontsize=12)
    axes[1, 0].set_ylabel('Log Loss', fontsize=12)
    axes[1, 0].set_title('Log Loss: Training vs Validation', fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Precision & Recall plot
    axes[1, 1].plot(history['fold'], history['train_precision'], 
                    marker='o', label='Training Precision', linewidth=2)
    axes[1, 1].plot(history['fold'], history['val_precision'], 
                    marker='s', label='Validation Precision', linewidth=2)
    axes[1, 1].plot(history['fold'], history['train_recall'], 
                    marker='^', label='Training Recall', linewidth=2, linestyle='--')
    axes[1, 1].plot(history['fold'], history['val_recall'], 
                    marker='v', label='Validation Recall', linewidth=2, linestyle='--')
    axes[1, 1].set_xlabel('Fold', fontsize=12)
    axes[1, 1].set_ylabel('Score', fontsize=12)
    axes[1, 1].set_title('Precision & Recall: Training vs Validation', fontsize=14, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_feature_importance(model, feature_names, filename='feature_importance.png'):
    """Plot feature importance"""
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(12, 10))
    plt.barh(importance_df['feature'], importance_df['importance'], color='skyblue')
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title('Feature Importance Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_class_distribution(y_train, y_test, class_names, filename='class_distribution.png'):
    """Plot class distribution"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training distribution
    train_counts = pd.Series(y_train).value_counts().sort_index()
    axes[0].bar(range(len(train_counts)), train_counts.values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Training Set Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Test distribution
    test_counts = pd.Series(y_test).value_counts().sort_index()
    axes[1].bar(range(len(test_counts)), test_counts.values, color=['#3498db', '#e74c3c', '#2ecc71'])
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Test Set Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def plot_per_class_metrics(cm, class_names, filename='per_class_metrics.png'):
    """Plot per-class precision, recall, and F1 scores"""
    # Calculate per-class metrics
    precision = cm.diagonal() / cm.sum(axis=0)
    recall = cm.diagonal() / cm.sum(axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    # Create plot
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precision, width, label='Precision', color='#3498db')
    ax.bar(x, recall, width, label='Recall', color='#e74c3c')
    ax.bar(x + width, f1, width, label='F1 Score', color='#2ecc71')
    
    ax.set_xlabel('Risk Level', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        ax.text(i - width, p + 0.02, f'{p:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i, r + 0.02, f'{r:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width, f + 0.02, f'{f:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filename}")
    plt.close()

def train_comprehensive_model(dataset_file='data/CVD_Dataset_ML_Ready.csv', n_folds=5):
    """
    Train comprehensive XGBoost model with detailed analysis and visualization
    """
    
    # 1. Load dataset
    print("\n1. Loading dataset...")
    df = pd.read_csv(dataset_file)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Missing values: {df.isnull().sum().sum()}")
    
    # 2. Prepare features and target
    print("\n2. Preparing features and target...")
    X = df.drop('CVD Risk Level', axis=1)
    y = df['CVD Risk Level']
    
    feature_names = X.columns.tolist()
    class_names = ['LOW Risk', 'INTERMEDIARY Risk', 'HIGH Risk']
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Target distribution: {y.value_counts().sort_index().tolist()}")
    
    # 3. Apply SMOTEENN for class balancing
    print("\n3. Applying class balancing (SMOTEENN)...")
    smoteenn = SMOTEENN(random_state=42)
    X_balanced, y_balanced = smoteenn.fit_resample(X, y)
    print(f"   Balanced distribution: {pd.Series(y_balanced).value_counts().sort_index().tolist()}")
    print(f"   Balanced dataset shape: {X_balanced.shape}")
    
    # 4. Train-test split
    print("\n4. Creating train-test split (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2,
        random_state=42,
        stratify=y_balanced
    )
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    
    # 5. Feature scaling
    print("\n5. Feature scaling (RobustScaler)...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot class distribution
    plot_class_distribution(y_train, y_test, class_names, 'evaluation/class_distribution.png')
    
    # 6. Cross-validation with detailed tracking
    print("\n6. Performing 5-fold cross-validation...")
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    history = {
        'fold': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'train_f1': [],
        'val_f1': [],
        'train_loss': [],
        'val_loss': [],
        'train_precision': [],
        'val_precision': [],
        'train_recall': [],
        'val_recall': []
    }
    
    fold_num = 0
    for train_idx, val_idx in skf.split(X_train_scaled, y_train):
        fold_num += 1
        print(f"\n   Fold {fold_num}/{n_folds}:")
        
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Train model
        model_fold = XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='mlogloss',
            verbosity=0
        )
        
        model_fold.fit(X_fold_train, y_fold_train)
        
        # Predictions
        y_train_pred = model_fold.predict(X_fold_train)
        y_val_pred = model_fold.predict(X_fold_val)
        
        y_train_proba = model_fold.predict_proba(X_fold_train)
        y_val_proba = model_fold.predict_proba(X_fold_val)
        
        # Calculate metrics
        train_acc = accuracy_score(y_fold_train, y_train_pred)
        val_acc = accuracy_score(y_fold_val, y_val_pred)
        
        train_f1 = f1_score(y_fold_train, y_train_pred, average='weighted')
        val_f1 = f1_score(y_fold_val, y_val_pred, average='weighted')
        
        train_loss = log_loss(y_fold_train, y_train_proba)
        val_loss = log_loss(y_fold_val, y_val_proba)
        
        train_precision = precision_score(y_fold_train, y_train_pred, average='weighted')
        val_precision = precision_score(y_fold_val, y_val_pred, average='weighted')
        
        train_recall = recall_score(y_fold_train, y_train_pred, average='weighted')
        val_recall = recall_score(y_fold_val, y_val_pred, average='weighted')
        
        # Store metrics
        history['fold'].append(fold_num)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_precision'].append(train_precision)
        history['val_precision'].append(val_precision)
        history['train_recall'].append(train_recall)
        history['val_recall'].append(val_recall)
        
        print(f"      Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"      Train F1:  {train_f1:.4f} | Val F1:  {val_f1:.4f}")
        print(f"      Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    # 7. Train final model on full training set
    print("\n7. Training final model on full training set...")
    final_model = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='mlogloss',
        verbosity=0
    )
    
    final_model.fit(X_train_scaled, y_train)
    
    # 8. Make predictions on test set
    print("\n8. Evaluating on test set...")
    y_test_pred = final_model.predict(X_test_scaled)
    y_test_proba = final_model.predict_proba(X_test_scaled)
    
    # Calculate final metrics
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    test_loss = log_loss(y_test, y_test_proba)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    # 9. Generate visualizations
    print("\n9. Generating visualizations...")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plot_confusion_matrix(cm, class_names, 
                         title='Confusion Matrix - XGBoost Model',
                         filename='evaluation/confusion_matrix.png')
    
    # Training history
    plot_training_history(history, filename='evaluation/training_history.png')
    
    # Feature importance
    plot_feature_importance(final_model, feature_names, 
                           filename='evaluation/feature_importance.png')
    
    # Per-class metrics
    plot_per_class_metrics(cm, class_names, filename='evaluation/per_class_metrics.png')
    
    # 10. Print detailed results
    print("\n" + "="*80)
    print("COMPREHENSIVE MODEL RESULTS")
    print("="*80)
    
    print(f"\n📊 Cross-Validation Results (Average over {n_folds} folds):")
    print(f"   Training Accuracy:   {np.mean(history['train_accuracy']):.4f} ± {np.std(history['train_accuracy']):.4f}")
    print(f"   Validation Accuracy: {np.mean(history['val_accuracy']):.4f} ± {np.std(history['val_accuracy']):.4f}")
    print(f"   Training F1 Score:   {np.mean(history['train_f1']):.4f} ± {np.std(history['train_f1']):.4f}")
    print(f"   Validation F1 Score: {np.mean(history['val_f1']):.4f} ± {np.std(history['val_f1']):.4f}")
    print(f"   Training Loss:       {np.mean(history['train_loss']):.4f} ± {np.std(history['train_loss']):.4f}")
    print(f"   Validation Loss:     {np.mean(history['val_loss']):.4f} ± {np.std(history['val_loss']):.4f}")
    
    print(f"\n🎯 Test Set Performance:")
    print(f"   Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"   F1 Score:  {test_f1:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   Log Loss:  {test_loss:.4f}")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    print("\n🔢 Confusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    print("\n📊 Per-Class Accuracy:")
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, (name, acc) in enumerate(zip(class_names, class_accuracy)):
        print(f"   {name:25}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Feature importance
    print("\n🔝 Top 10 Most Important Features:")
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance_df.head(10).iterrows():
        print(f"   {row.name+1:2d}. {row['feature']:<30}: {row['importance']:.4f}")
    
    # 11. Save model
    print("\n10. Saving model and artifacts...")
    model_artifact = {
        'model': final_model,
        'scaler': scaler,
        'feature_names': feature_names,
        'class_names': class_names,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_loss': test_loss,
        'cv_history': history,
        'confusion_matrix': cm,
        'version': '3.0'
    }
    
    joblib.dump(model_artifact, 'models/cvd_comprehensive_model.pkl')
    print("   ✅ Saved model to: models/cvd_comprehensive_model.pkl")
    
    # 12. Summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"✅ Model: XGBoost with optimized hyperparameters")
    print(f"✅ Features: {len(feature_names)} features")
    print(f"✅ Training samples: {X_train.shape[0]:,}")
    print(f"✅ Test samples: {X_test.shape[0]:,}")
    print(f"✅ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"✅ Test F1 Score: {test_f1:.4f}")
    print(f"✅ Test Log Loss: {test_loss:.4f}")
    print(f"✅ Cross-validation: {n_folds}-fold stratified")
    print(f"✅ Visualizations saved to: evaluation/")
    
    if test_accuracy >= 0.90:
        print("\n🎉 EXCELLENT! Achieved 90%+ accuracy!")
        print("🏥 Model is ready for clinical deployment!")
    elif test_accuracy >= 0.85:
        print("\n👍 VERY GOOD! Achieved 85%+ accuracy!")
        print("🏥 Model shows strong clinical potential!")
    else:
        print("\n📊 GOOD! Strong performance for medical prediction!")
    
    return model_artifact

if __name__ == "__main__":
    print("\n🚀 Starting comprehensive CVD risk prediction model training...")
    print("📊 This will generate detailed metrics and visualizations\n")
    
    # Train the model
    results = train_comprehensive_model('data/CVD_Dataset_ML_Ready.csv', n_folds=5)
    
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE!")
    print("="*80)
    print(f"✅ Model accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"✅ Model F1 score: {results['test_f1']:.4f}")
    print(f"✅ All visualizations saved to: evaluation/")
    print("   - confusion_matrix.png")
    print("   - training_history.png")
    print("   - feature_importance.png")
    print("   - per_class_metrics.png")
    print("   - class_distribution.png")
    print(f"✅ Model saved to: models/cvd_comprehensive_model.pkl")
    print("\n💾 Ready for deployment!")
