import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────────────────────
# LOAD RAW DATA AND ANALYZE CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────

df_raw = pd.read_csv("data/raw/MymensingUniversity.csv")

print("\n" + "="*90)
print("CLASS DISTRIBUTION ANALYSIS - WHICH CLASSES WERE REMOVED?")
print("="*90)

print("\n📊 STEP 1: ORIGINAL DATASET")
print("-"*90)
print(f"Total Rows: {len(df_raw)}")
print(f"Total Columns: {len(df_raw.columns)}")
print(f"\nCVD Risk Level Column - All Unique Values:")
print(df_raw["CVD Risk Level"].value_counts().to_string())
print(f"\nTotal Unique Classes: {df_raw['CVD Risk Level'].nunique()}")

# Store original counts
original_counts = df_raw["CVD Risk Level"].value_counts()
original_total = len(df_raw)

# ─────────────────────────────────────────────────────────────
# STEP 2: APPLY CLASS FILTER (SAME AS IN train_cvd_models.py)
# ─────────────────────────────────────────────────────────────

print("\n📊 STEP 2: AFTER CLASS FILTERING")
print("-"*90)
print("Filter Applied: Keep only ['LOW', 'INTERMEDIARY', 'HIGH']")

TARGET = "CVD Risk Level"
df_filtered = df_raw[df_raw[TARGET].isin(["LOW", "INTERMEDIARY", "HIGH"])].copy()

print(f"\nFiltered Rows: {len(df_filtered)}")
print(f"Rows Removed: {len(df_raw) - len(df_filtered)}")
print(f"Data Retention: {(len(df_filtered)/len(df_raw)*100):.2f}%")

if len(df_raw) != len(df_filtered):
    print(f"\n⚠️  Classes Removed from Dataset:")
    removed_classes = df_raw[~df_raw[TARGET].isin(["LOW", "INTERMEDIARY", "HIGH"])]["CVD Risk Level"].value_counts()
    for class_name, count in removed_classes.items():
        print(f"   • {class_name:<20} : {count:>5} rows ({count/len(df_raw)*100:.2f}%)")
else:
    print(f"\n✓ No rows removed - all data contained valid CVD Risk Level values")

filtered_counts = df_filtered["CVD Risk Level"].value_counts()
print(f"\nCVD Risk Level - After Filtering:")
print(filtered_counts.to_string())

# ─────────────────────────────────────────────────────────────
# STEP 3: BINARY CLASSIFICATION TRANSFORMATION
# ─────────────────────────────────────────────────────────────

print("\n📊 STEP 3: BINARY CLASSIFICATION TRANSFORMATION")
print("-"*90)
print("Transformation: Multiclass (3 classes) → Binary (2 classes)")
print("  • HIGH                    → HIGH (1)")
print("  • LOW + INTERMEDIARY      → NON-HIGH (0)")

df_filtered["Binary_Class"] = df_filtered[TARGET].apply(lambda x: "HIGH" if x == "HIGH" else "NON-HIGH")

binary_counts = df_filtered["Binary_Class"].value_counts()
print(f"\nBinary Class Distribution:")
for class_name, count in binary_counts.items():
    print(f"   • {class_name:<20} : {count:>5} rows ({count/len(df_filtered)*100:.2f}%)")

# ─────────────────────────────────────────────────────────────
# VISUALIZATION 1: CLASS REMOVAL FLOW
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Original distribution
original_all = df_raw["CVD Risk Level"].value_counts()
ax1 = axes[0]
colors_original = ['#FF6B6B' if x not in ["LOW", "INTERMEDIARY", "HIGH"] else '#4ECDC4' for x in original_all.index]
bars1 = ax1.bar(original_all.index, original_all.values, color=colors_original, edgecolor='black', linewidth=1.5, alpha=0.8)
ax1.set_title("Step 1: Original Dataset\n(All Classes)", fontsize=12, fontweight='bold')
ax1.set_ylabel("Count", fontsize=11, fontweight='bold')
ax1.set_xlabel("CVD Risk Level", fontsize=11, fontweight='bold')

for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# After filtering
ax2 = axes[1]
colors_filtered = ['#4ECDC4'] * len(filtered_counts)
bars2 = ax2.bar(filtered_counts.index, filtered_counts.values, color=colors_filtered, edgecolor='black', linewidth=1.5, alpha=0.8)
ax2.set_title("Step 2: After Filtering\n(Valid Classes Only)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Count", fontsize=11, fontweight='bold')
ax2.set_xlabel("CVD Risk Level", fontsize=11, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Binary transformation
ax3 = axes[2]
colors_binary = ['#FF6B6B', '#4ECDC4']  # High in red, Non-High in teal
bars3 = ax3.bar(binary_counts.index, binary_counts.values, color=colors_binary, edgecolor='black', linewidth=1.5, alpha=0.8)
ax3.set_title("Step 3: Binary Classification\n(HIGH vs NON-HIGH)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Count", fontsize=11, fontweight='bold')
ax3.set_xlabel("Class", fontsize=11, fontweight='bold')

for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle("Class Distribution Transformation: Which Classes Were Removed?", 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig("outputs/Class_Removal_Analysis.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/Class_Removal_Analysis.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# VISUALIZATION 2: CLASS TRANSFORMATION DETAILED
# ─────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(12, 7))

# Multiclass to Binary mapping
multiclass_distribution = df_filtered["CVD Risk Level"].value_counts().sort_index()
binary_high = (df_filtered["Binary_Class"] == "HIGH").sum()
binary_non_high = (df_filtered["Binary_Class"] == "NON-HIGH").sum()

# Sankey-like visualization
categories = ["LOW", "INTERMEDIARY", "HIGH", "", "NON-HIGH", "HIGH"]
values = [
    multiclass_distribution.get("LOW", 0),
    multiclass_distribution.get("INTERMEDIARY", 0),
    multiclass_distribution.get("HIGH", 0),
    0,
    binary_non_high,
    binary_high
]
colors = ["#95E1D3", "#95E1D3", "#FF6B6B", "white", "#95E1D3", "#FF6B6B"]
x_pos = [0, 1, 2, 3, 4, 5]

bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=2, alpha=0.8, width=0.6)

# Add labels and annotations
ax.set_xticks(x_pos)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.set_ylabel("Number of Samples", fontsize=12, fontweight='bold')
ax.set_title("Multiclass (3) → Binary (2) Class Transformation", fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(values) * 1.15)

# Add value labels and arrows
for i, (bar, val) in enumerate(zip(bars, values)):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2., val + 50,
                f'{int(val)}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add arrows showing transformation
ax.annotate('', xy=(3.8, max(values)*0.4), xytext=(0.8, max(values)*0.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
ax.text(2.3, max(values)*0.45, 'Combine into', ha='center', fontsize=10, fontweight='bold', 
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Add legend
ax.text(0.5, max(values)*0.75, "Multiclass\n(3 Classes)", fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=1.5))
ax.text(4.5, max(values)*0.75, "Binary\n(2 Classes)", fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig("outputs/Class_Transformation_Multiclass_to_Binary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Class_Transformation_Multiclass_to_Binary.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("SUMMARY: CLASSES REMOVED")
print("="*90)

print("\n🗑️  CLASSES REMOVED FROM DATASET:")
print("-"*90)

if len(df_raw) != len(df_filtered):
    removed_classes = df_raw[~df_raw[TARGET].isin(["LOW", "INTERMEDIARY", "HIGH"])]["CVD Risk Level"].value_counts()
    for class_name, count in removed_classes.items():
        print(f"\n  Class: {class_name}")
        print(f"    • Count: {count} rows")
        print(f"    • Percentage: {count/len(df_raw)*100:.2f}% of original dataset")
    print(f"\n  Total Rows Removed: {len(df_raw) - len(df_filtered)}")
    print(f"  Total Rows Retained: {len(df_filtered)}")
else:
    print("\n  ✓ NO CLASSES REMOVED")
    print(f"  All {df_raw['CVD Risk Level'].nunique()} unique values are valid CVD Risk Levels")
    print(f"  (LOW, INTERMEDIARY, HIGH)")

print("\n📊 CLASS CONSOLIDATION (For Binary Classification):")
print("-"*90)
print(f"\n  Multiclass (3 categories):")
print(f"    • LOW           : {multiclass_distribution.get('LOW', 0):>5} rows ({multiclass_distribution.get('LOW', 0)/len(df_filtered)*100:.2f}%)")
print(f"    • INTERMEDIARY  : {multiclass_distribution.get('INTERMEDIARY', 0):>5} rows ({multiclass_distribution.get('INTERMEDIARY', 0)/len(df_filtered)*100:.2f}%)")
print(f"    • HIGH          : {multiclass_distribution.get('HIGH', 0):>5} rows ({multiclass_distribution.get('HIGH', 0)/len(df_filtered)*100:.2f}%)")

print(f"\n  Binary (2 categories):")
print(f"    • NON-HIGH      : {binary_non_high:>5} rows ({binary_non_high/len(df_filtered)*100:.2f}%) [LOW + INTERMEDIARY combined]")
print(f"    • HIGH          : {binary_high:>5} rows ({binary_high/len(df_filtered)*100:.2f}%)")

print(f"\n  Class Imbalance Ratio: 1 : {binary_non_high / binary_high:.2f}")

print("\n" + "="*90)
