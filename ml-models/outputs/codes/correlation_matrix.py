import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────
# LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/raw/MymensingUniversity.csv")

# Remove leakage-prone columns
df.drop(columns=[
    "Blood Pressure (mmHg)", "Height (m)",
    "CVD Risk Score", "Blood Pressure Category"
], inplace=True, errors="ignore")

# Define the 17 usable features
BINARY_FEATURES = [
    "Sex",
    "Age",
    "Weight (kg)",
    "BMI",
    "Abdominal Circumference (cm)",
    "Total Cholesterol (mg/dL)",
    "HDL (mg/dL)",
    "Fasting Blood Sugar (mg/dL)",
    "Smoking Status",
    "Diabetes Status",
    "Physical Activity Level",
    "Family History of CVD",
    "Height (cm)",
    "Waist-to-Height Ratio",
    "Systolic BP",
    "Diastolic BP",
    "Estimated LDL (mg/dL)",
]

# Encode categorical variables
X = df[BINARY_FEATURES].copy()

for column in X.columns:
    if X[column].dtype == 'object':
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column].astype(str))

# Convert to numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median(numeric_only=True))

# ─────────────────────────────────────────────────────────────
# CORRELATION MATRIX - LARGE COMPREHENSIVE
# ─────────────────────────────────────────────────────────────

corr_matrix = X.corr()

fig, ax = plt.subplots(figsize=(16, 14))

# Create heatmap with better formatting
sns.heatmap(corr_matrix, 
            annot=True,  # Show correlation values
            fmt='.2f',   # 2 decimal places
            cmap='coolwarm',  # Color scheme
            center=0,    # Center color at 0
            square=True,  # Square cells
            linewidths=0.5,  # Grid lines
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            annot_kws={'fontsize': 8, 'fontweight': 'bold'},
            vmin=-1, vmax=1,
            ax=ax)

ax.set_title('Feature Correlation Matrix - 17 Binary Classification Features', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')

# Rotate labels for readability
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/Correlation_Matrix_Heatmap.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Correlation_Matrix_Heatmap.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# CORRELATION MATRIX - COMPACT VERSION
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 12))

# Shorter feature names for compact view
short_names = [
    "Sex", "Age", "Weight", "BMI", "Abd Circ",
    "Total Chol", "HDL", "Fasting BS", "Smoking", "Diabetes",
    "Phys Act", "Fam Hist CVD", "Height", "Waist/Height",
    "Systolic BP", "Diastolic BP", "LDL"
]

sns.heatmap(corr_matrix, 
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',  # Red-Blue reversed
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation', 'shrink': 0.9},
            annot_kws={'fontsize': 7.5, 'weight': 'bold'},
            xticklabels=short_names,
            yticklabels=short_names,
            vmin=-1, vmax=1,
            ax=ax)

ax.set_title('Feature Correlation Matrix (Compact) - 17 Features', 
             fontsize=14, fontweight='bold', pad=15)

plt.xticks(rotation=45, ha='right', fontsize=8.5)
plt.yticks(rotation=0, fontsize=8.5)

plt.tight_layout()
plt.savefig("outputs/Correlation_Matrix_Compact.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Correlation_Matrix_Compact.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# CORRELATION MATRIX - TOP CORRELATIONS ANALYSIS
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(14, 10))

# Extract upper triangle to avoid duplicates
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
top_corr = corr_matrix.mask(mask)

sns.heatmap(top_corr,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',  # Yellow-Orange-Red
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Correlation Coefficient'},
            annot_kws={'fontsize': 8, 'weight': 'bold'},
            xticklabels=short_names,
            yticklabels=short_names,
            vmin=0, vmax=1,
            ax=ax)

ax.set_title('Feature Correlation Matrix (Lower Triangle Only) - Unique Relationships', 
             fontsize=14, fontweight='bold', pad=15)

plt.xticks(rotation=45, ha='right', fontsize=8.5)
plt.yticks(rotation=0, fontsize=8.5)

plt.tight_layout()
plt.savefig("outputs/Correlation_Matrix_Lower_Triangle.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Correlation_Matrix_Lower_Triangle.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("CORRELATION MATRIX ANALYSIS - 17 FEATURES")
print("="*90)

# Find strongest correlations
print("\n🔗 TOP 10 STRONGEST CORRELATIONS (excluding self-correlation):")
print("-"*90)

# Get all correlation pairs
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append({
            'Feature1': corr_matrix.columns[i],
            'Feature2': corr_matrix.columns[j],
            'Correlation': corr_matrix.iloc[i, j]
        })

corr_pairs_df = pd.DataFrame(corr_pairs)
corr_pairs_df['Abs_Corr'] = corr_pairs_df['Correlation'].abs()
top_10 = corr_pairs_df.nlargest(10, 'Abs_Corr')

for idx, row in top_10.iterrows():
    symbol = "📈" if row['Correlation'] > 0 else "📉"
    print(f"{symbol} {row['Feature1']:<25} ↔ {row['Feature2']:<25} = {row['Correlation']:>7.4f}")

# Feature with highest average correlation (multicollinearity risk)
print("\n⚠️  AVERAGE CORRELATION BY FEATURE (Multicollinearity Risk):")
print("-"*90)
avg_corr = corr_matrix.abs().mean().sort_values(ascending=False)
for feat, corr_val in avg_corr.items():
    print(f"  {feat:<35} Avg Correlation: {corr_val:.4f}")

# Correlation distribution
print("\n📊 CORRELATION DISTRIBUTION STATISTICS:")
print("-"*90)
all_corr_values = corr_pairs_df['Correlation'].values
print(f"  Mean Correlation:        {np.mean(all_corr_values):>8.4f}")
print(f"  Median Correlation:      {np.median(all_corr_values):>8.4f}")
print(f"  Std Dev:                 {np.std(all_corr_values):>8.4f}")
print(f"  Min Correlation:         {np.min(all_corr_values):>8.4f}")
print(f"  Max Correlation:         {np.max(all_corr_values):>8.4f}")
print(f"  Total Unique Pairs:      {len(corr_pairs_df):>8}")

print("\n" + "="*90)
