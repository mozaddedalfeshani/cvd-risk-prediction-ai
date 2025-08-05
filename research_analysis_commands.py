# =============================================================================
# MURAD RESEARCH PAPER ANALYSIS COMMANDS
# =============================================================================
# This file contains all Python commands for research paper analysis
# Use with CVD_Dataset_ML_Ready.csv for best results
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("RESEARCH PAPER ANALYSIS COMMANDS")
print("="*80)

# =============================================================================
# 1. LOAD AND EXPLORE DATA
# =============================================================================

print("\n1. Loading and exploring the ML-ready dataset...")

# Load the cleaned dataset
df = pd.read_csv('CVD_Dataset_ML_Ready.csv')
print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Display basic info
print("\nDataset Info:")
print(df.info())

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# =============================================================================
# 2. TARGET VARIABLE ANALYSIS
# =============================================================================

print("\n2. Target variable analysis...")

# Target distribution
target_counts = df['CVD Risk Level'].value_counts()
print(f"Target distribution:\n{target_counts}")

# Create target distribution plot
plt.figure(figsize=(10, 6))
target_counts.plot(kind='bar', color=['#2E8B57', '#FF6B6B', '#4ECDC4'])
plt.title('CVD Risk Level Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Risk Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 3. FEATURE CORRELATION ANALYSIS
# =============================================================================

print("\n3. Feature correlation analysis...")

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create correlation heatmap
plt.figure(figsize=(16, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# Top correlations with target
target_correlations = correlation_matrix['CVD Risk Level'].abs().sort_values(ascending=False)
print(f"\nTop 10 features correlated with CVD Risk Level:")
print(target_correlations.head(10))

# =============================================================================
# 4. FEATURE IMPORTANCE ANALYSIS
# =============================================================================

print("\n4. Feature importance analysis...")

# Prepare data for feature importance
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']

# Train Random Forest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 15 most important features:")
print(feature_importance.head(15))

# Create feature importance plot
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 15 Feature Importance (Random Forest)', fontsize=16, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 5. DEMOGRAPHIC ANALYSIS
# =============================================================================

print("\n5. Demographic analysis...")

# Age analysis
plt.figure(figsize=(15, 5))

# Age distribution by risk level
plt.subplot(1, 3, 1)
for risk_level in [0, 1, 2]:
    risk_data = df[df['CVD Risk Level'] == risk_level]['Age']
    plt.hist(risk_data, alpha=0.7, label=f'Risk Level {risk_level}', bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution by Risk Level')
plt.legend()

# BMI analysis
plt.subplot(1, 3, 2)
for risk_level in [0, 1, 2]:
    risk_data = df[df['CVD Risk Level'] == risk_level]['BMI']
    plt.hist(risk_data, alpha=0.7, label=f'Risk Level {risk_level}', bins=20)
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('BMI Distribution by Risk Level')
plt.legend()

# Gender analysis
plt.subplot(1, 3, 3)
gender_risk = pd.crosstab(df['Sex'], df['CVD Risk Level'])
gender_risk.plot(kind='bar', stacked=True)
plt.xlabel('Sex (0=Female, 1=Male)')
plt.ylabel('Count')
plt.title('Gender Distribution by Risk Level')
plt.legend(['Low Risk', 'Intermediate Risk', 'High Risk'])

plt.tight_layout()
plt.savefig('demographic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 6. MEDICAL PARAMETERS ANALYSIS
# =============================================================================

print("\n6. Medical parameters analysis...")

# Create medical parameters plots
medical_features = ['Systolic BP', 'Diastolic BP', 'Total Cholesterol (mg/dL)', 
                   'HDL (mg/dL)', 'Fasting Blood Sugar (mg/dL)', 'CVD Risk Score']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(medical_features):
    for risk_level in [0, 1, 2]:
        risk_data = df[df['CVD Risk Level'] == risk_level][feature]
        axes[i].hist(risk_data, alpha=0.7, label=f'Risk Level {risk_level}', bins=15)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'{feature} Distribution')
    axes[i].legend()

plt.tight_layout()
plt.savefig('medical_parameters.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 7. MODEL PERFORMANCE ANALYSIS
# =============================================================================

print("\n7. Model performance analysis...")

# Prepare data
X = df.drop('CVD Risk Level', axis=1)
y = df['CVD Risk Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Create model comparison plot
plt.figure(figsize=(10, 6))
models_list = list(results.keys())
accuracies = list(results.values())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = plt.bar(models_list, accuracies, color=colors)
plt.xlabel('Model', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.ylim(0, 1)

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 8. CONFUSION MATRIX ANALYSIS
# =============================================================================

print("\n8. Confusion matrix analysis...")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
y_pred = best_model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
class_names = ['Low Risk', 'Intermediate Risk', 'High Risk']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Print classification report
print(f"\nClassification Report - {best_model_name}:")
print(classification_report(y_test, y_pred, target_names=class_names))

# =============================================================================
# 9. CROSS-VALIDATION ANALYSIS
# =============================================================================

print("\n9. Cross-validation analysis...")

# Perform cross-validation
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    cv_scores[name] = scores
    print(f"{name} CV Scores: {scores}")
    print(f"{name} CV Mean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

# Create cross-validation comparison plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
for name, scores in cv_scores.items():
    plt.plot(range(1, 6), scores, marker='o', label=name, linewidth=2)
plt.xlabel('Fold')
plt.ylabel('Accuracy')
plt.title('Cross-Validation Scores')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cv_means = [scores.mean() for scores in cv_scores.values()]
cv_stds = [scores.std() for scores in cv_scores.values()]
plt.bar(cv_scores.keys(), cv_means, yerr=cv_stds, capsize=5)
plt.xlabel('Model')
plt.ylabel('Mean CV Accuracy')
plt.title('Cross-Validation Performance')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('cross_validation.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 10. FEATURE ENGINEERING ANALYSIS
# =============================================================================

print("\n10. Feature engineering analysis...")

# Analyze engineered features
engineered_features = ['Pulse_Pressure', 'Cholesterol_HDL_Ratio', 'LDL_HDL_Ratio', 
                      'Age_Group', 'BMI_Category', 'Multiple_Risk_Factors']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(engineered_features):
    plt.subplot(2, 3, i+1)
    for risk_level in [0, 1, 2]:
        risk_data = df[df['CVD Risk Level'] == risk_level][feature]
        plt.hist(risk_data, alpha=0.7, label=f'Risk Level {risk_level}', bins=15)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.title(f'{feature} Distribution')
    plt.legend()

plt.tight_layout()
plt.savefig('engineered_features.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 11. STATISTICAL SUMMARY
# =============================================================================

print("\n11. Statistical summary...")

# Create statistical summary
summary_stats = df.describe()
print("\nStatistical Summary:")
print(summary_stats)

# Save summary to CSV
summary_stats.to_csv('statistical_summary.csv')
print("\nStatistical summary saved to 'statistical_summary.csv'")

# =============================================================================
# 12. RESEARCH PAPER FIGURES
# =============================================================================

print("\n12. Creating research paper figures...")

# Figure 1: Dataset Overview
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: Target distribution
target_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[0,0])
axes[0,0].set_title('CVD Risk Level Distribution', fontweight='bold')

# Subplot 2: Age vs Risk Level
for risk_level in [0, 1, 2]:
    risk_data = df[df['CVD Risk Level'] == risk_level]['Age']
    axes[0,1].hist(risk_data, alpha=0.7, label=f'Risk Level {risk_level}', bins=20)
axes[0,1].set_xlabel('Age')
axes[0,1].set_ylabel('Frequency')
axes[0,1].set_title('Age Distribution by Risk Level')
axes[0,1].legend()

# Subplot 3: Model performance
bars = axes[1,0].bar(models_list, accuracies, color=colors)
axes[1,0].set_xlabel('Model')
axes[1,0].set_ylabel('Accuracy')
axes[1,0].set_title('Model Performance Comparison')
axes[1,0].set_ylim(0, 1)

# Subplot 4: Feature importance
top_10_features = feature_importance.head(10)
axes[1,1].barh(range(len(top_10_features)), top_10_features['importance'])
axes[1,1].set_yticks(range(len(top_10_features)))
axes[1,1].set_yticklabels(top_10_features['feature'])
axes[1,1].set_xlabel('Feature Importance')
axes[1,1].set_title('Top 10 Feature Importance')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.savefig('research_paper_figure1.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# 13. EXPORT RESULTS
# =============================================================================

print("\n13. Exporting results...")

# Export model results
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': list(results.values())
})
results_df.to_csv('model_results.csv', index=False)
print("Model results saved to 'model_results.csv'")

# Export feature importance
feature_importance.to_csv('feature_importance.csv', index=False)
print("Feature importance saved to 'feature_importance.csv'")

# Export correlation matrix
correlation_matrix.to_csv('correlation_matrix.csv')
print("Correlation matrix saved to 'correlation_matrix.csv'")

# =============================================================================
# 14. SUMMARY STATISTICS FOR RESEARCH PAPER
# =============================================================================

print("\n14. Summary statistics for research paper...")

print("\n" + "="*80)
print("RESEARCH PAPER SUMMARY STATISTICS")
print("="*80)

print(f"Dataset Size: {df.shape[0]} samples, {df.shape[1]} features")
print(f"Missing Values: {df.isnull().sum().sum()}")
print(f"Target Distribution:")
for level, count in target_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  Risk Level {level}: {count} ({percentage:.1f}%)")

print(f"\nBest Model: {best_model_name}")
print(f"Best Accuracy: {results[best_model_name]:.4f} ({results[best_model_name]*100:.2f}%)")

print(f"\nTop 5 Most Important Features:")
for i, row in feature_importance.head(5).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\nGenerated Files:")
print("  - target_distribution.png")
print("  - correlation_heatmap.png")
print("  - feature_importance.png")
print("  - demographic_analysis.png")
print("  - medical_parameters.png")
print("  - model_comparison.png")
print("  - confusion_matrix.png")
print("  - cross_validation.png")
print("  - engineered_features.png")
print("  - research_paper_figure1.png")
print("  - statistical_summary.csv")
print("  - model_results.csv")
print("  - feature_importance.csv")
print("  - correlation_matrix.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE - READY FOR RESEARCH PAPER!")
print("="*80)

# =============================================================================
# 15. ADDITIONAL RESEARCH COMMANDS
# =============================================================================

print("\n15. Additional research commands...")

# Command 1: Create box plots for key features
print("\nCreating box plots for key features...")
key_features = ['Age', 'BMI', 'Systolic BP', 'HDL (mg/dL)', 'CVD Risk Score']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, feature in enumerate(key_features):
    if i < 5:  # Only use first 5 subplots
        df.boxplot(column=feature, by='CVD Risk Level', ax=axes[i])
        axes[i].set_title(f'{feature} by Risk Level')
        axes[i].set_xlabel('CVD Risk Level')

plt.tight_layout()
plt.savefig('boxplots_key_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Command 2: Create scatter plots
print("\nCreating scatter plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Age vs BMI
axes[0,0].scatter(df['Age'], df['BMI'], c=df['CVD Risk Level'], cmap='viridis', alpha=0.6)
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('BMI')
axes[0,0].set_title('Age vs BMI (colored by Risk Level)')

# Systolic vs Diastolic BP
axes[0,1].scatter(df['Systolic BP'], df['Diastolic BP'], c=df['CVD Risk Level'], cmap='viridis', alpha=0.6)
axes[0,1].set_xlabel('Systolic BP')
axes[0,1].set_ylabel('Diastolic BP')
axes[0,1].set_title('Systolic vs Diastolic BP (colored by Risk Level)')

# Cholesterol vs HDL
axes[1,0].scatter(df['Total Cholesterol (mg/dL)'], df['HDL (mg/dL)'], c=df['CVD Risk Level'], cmap='viridis', alpha=0.6)
axes[1,0].set_xlabel('Total Cholesterol')
axes[1,0].set_ylabel('HDL')
axes[1,0].set_title('Cholesterol vs HDL (colored by Risk Level)')

# Age vs CVD Risk Score
axes[1,1].scatter(df['Age'], df['CVD Risk Score'], c=df['CVD Risk Level'], cmap='viridis', alpha=0.6)
axes[1,1].set_xlabel('Age')
axes[1,1].set_ylabel('CVD Risk Score')
axes[1,1].set_title('Age vs CVD Risk Score (colored by Risk Level)')

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll research analysis commands completed!")
print("Check the generated PNG files and CSV files for your research paper.") 