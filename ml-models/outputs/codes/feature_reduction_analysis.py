import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create visualization explaining feature reduction from 22 to 17

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# ==================== LEFT: FEATURE ELIMINATION ====================
categories = ['Original\nColumns', 'Removed\nColumns', 'Final\nFeatures']
counts = [22, 5, 17]
colors_bars = ['#FF6B6B', '#FFA500', '#4ECDC4']

bars = ax1.bar(categories, counts, color=colors_bars, edgecolor='black', linewidth=2, alpha=0.8, width=0.6)
ax1.set_ylabel('Number of Features/Columns', fontsize=12, fontweight='bold')
ax1.set_title('Feature Reduction: From 22 to 17', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylim(0, 25)

# Add value labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{int(count)}', ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add reduction text
ax1.text(0.5, 11, '↓ Remove 5 ↓', ha='center', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

ax1.grid(axis='y', alpha=0.3, linestyle='--')

# ==================== RIGHT: REMOVED COLUMNS & REASONS ====================
ax2.axis('off')

removed_info = [
    ("Removed Columns (5)", None, '#FF6B6B', 14, 'bold'),
    ("", None, 'white', 11, 'normal'),
    ("1. Blood Pressure (mmHg)", "→ Duplicate: Use Systolic/Diastolic BP instead", '#FFE0E0', 11, 'normal'),
    ("2. Height (m)", "→ Duplicate: Use Height (cm) instead", '#FFE0E0', 11, 'normal'),
    ("3. CVD Risk Score", "→ Data Leakage: Directly derived from target label", '#FFD0D0', 11, 'normal'),
    ("4. Blood Pressure Category", "→ Data Leakage: Derived from target variable", '#FFD0D0', 11, 'normal'),
    ("5. CVD Risk Level", "→ This is the TARGET variable (not a feature)", '#FFC0C0', 11, 'normal'),
    ("", None, 'white', 11, 'normal'),
    ("✓ Final Features (17)", None, '#4ECDC4', 14, 'bold'),
    ("", None, 'white', 11, 'normal'),
    ("Sex, Age, Weight (kg), BMI,", None, 'white', 10, 'normal'),
    ("Abdominal Circumference, Total Cholesterol,", None, 'white', 10, 'normal'),
    ("HDL, Fasting Blood Sugar, Smoking Status,", None, 'white', 10, 'normal'),
    ("Diabetes Status, Physical Activity Level,", None, 'white', 10, 'normal'),
    ("Family History of CVD, Height (cm),", None, 'white', 10, 'normal'),
    ("Waist-to-Height Ratio, Systolic BP,", None, 'white', 10, 'normal'),
    ("Diastolic BP, Estimated LDL (mg/dL)", None, 'white', 10, 'normal'),
]

y_position = 0.95
for i, item in enumerate(removed_info):
    text, detail, bg_color, font_size, font_weight = item
    
    if detail:
        # Main text with detail
        ax2.text(0.05, y_position, text, fontsize=font_size, fontweight=font_weight,
                transform=ax2.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7, pad=0.5, edgecolor='black', linewidth=0.5))
        ax2.text(0.05, y_position - 0.035, detail, fontsize=9, style='italic',
                transform=ax2.transAxes, va='top', color='#333333')
        y_position -= 0.07
    else:
        # Just text or spacing
        if text:
            ax2.text(0.05, y_position, text, fontsize=font_size, fontweight=font_weight,
                    transform=ax2.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.7, pad=0.5, edgecolor='black', linewidth=1))
        y_position -= 0.035

plt.suptitle('Feature Selection: Why 22 → 17 (Data Quality & Leakage Prevention)', 
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig("outputs/Feature_Reduction_Explanation_22to17.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Feature_Reduction_Explanation_22to17.png")
plt.close()

# ==================== SUMMARY REPORT ====================
print("\n" + "="*90)
print("FEATURE REDUCTION ANALYSIS: 22 COLUMNS → 17 USABLE FEATURES")
print("="*90)

print("\n📊 ORIGINAL DATASET: 22 Columns")
print("-" * 90)
original_cols = [
    "Sex", "Age", "Weight (kg)", "BMI", "Abdominal Circumference (cm)",
    "Total Cholesterol (mg/dL)", "HDL (mg/dL)", "Fasting Blood Sugar (mg/dL)",
    "Smoking Status", "Diabetes Status", "Physical Activity Level",
    "Family History of CVD", "Height (cm)", "Waist-to-Height Ratio",
    "Blood Pressure (mmHg)", "Height (m)", "Systolic BP", "Diastolic BP",
    "Estimated LDL (mg/dL)", "CVD Risk Score", "Blood Pressure Category",
    "CVD Risk Level"
]
print(f"Total: {len(original_cols)} columns")
for i, col in enumerate(original_cols, 1):
    print(f"  {i:2d}. {col}")

print("\n🗑️  REMOVED COLUMNS: 5 Reasons for Exclusion")
print("-" * 90)

removed_details = {
    "Duplicates (2 columns)": [
        ("Blood Pressure (mmHg)", "Redundant with Systolic BP & Diastolic BP"),
        ("Height (m)", "Redundant with Height (cm) - same information, different unit"),
    ],
    "Data Leakage (2 columns)": [
        ("CVD Risk Score", "Directly derived from CVD Risk Level → LEAKAGE"),
        ("Blood Pressure Category", "Derived from blood pressure values & target → LEAKAGE"),
    ],
    "Target Variable (1 column)": [
        ("CVD Risk Level", "This is the Y-label, not a feature (X)"),
    ]
}

for category, cols in removed_details.items():
    print(f"\n  {category}:")
    for col_name, reason in cols:
        print(f"    • {col_name:<30} → {reason}")

print("\n✅ FINAL FEATURES: 17 Usable Predictors")
print("-" * 90)
final_features = [
    "Sex", "Age", "Weight (kg)", "BMI", "Abdominal Circumference (cm)",
    "Total Cholesterol (mg/dL)", "HDL (mg/dL)", "Fasting Blood Sugar (mg/dL)",
    "Smoking Status", "Diabetes Status", "Physical Activity Level",
    "Family History of CVD", "Height (cm)", "Waist-to-Height Ratio",
    "Systolic BP", "Diastolic BP", "Estimated LDL (mg/dL)"
]
print(f"Total: {len(final_features)} features")
for i, col in enumerate(final_features, 1):
    print(f"  {i:2d}. {col}")

print("\n📌 KEY REASONS FOR REDUCTION:")
print("-" * 90)
print("""
1️⃣  DATA LEAKAGE PREVENTION (Most Critical)
   • CVD Risk Score and Blood Pressure Category are DERIVED from the target variable
   • Including them would give models unfair advantage → unrealistic performance metrics
   • In production, these values won't be available at prediction time

2️⃣  FEATURE REDUNDANCY (Avoiding Multicollinearity)
   • Blood Pressure (mmHg): Duplicate of Systolic BP & Diastolic BP
   • Height (m): Duplicate of Height (cm) in different units
   • Keeping duplicates increases model complexity without new information

3️⃣  TARGET VARIABLE SEPARATION
   • CVD Risk Level is the prediction TARGET, not a feature
   • Models learn to predict HIGH vs NON-HIGH based on independent variables

✓ RESULT: 17 independent, non-leakage, non-redundant features for robust model training
""")

print("="*90)
