import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10

# Load binary accuracy data
binary_df = pd.read_csv("outputs/accuracy/binary_results.csv")
smote_df = pd.read_csv("outputs/accuracy/smote_binary_results.csv")

# Sort by accuracy (descending)
binary_df = binary_df.sort_values("Accuracy (%)", ascending=False)
smote_df = smote_df.set_index("Model").loc[binary_df["Model"]].reset_index()

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Color palettes
colors_binary = sns.color_palette("viridis", len(binary_df))
colors_smote = sns.color_palette("plasma", len(smote_df))

# Plot 1: Regular Binary Results
ax1 = axes[0]
bars1 = ax1.barh(binary_df["Model"], binary_df["Accuracy (%)"], color=colors_binary)
ax1.set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax1.set_title("Binary Classification - Accuracy Comparison\n(Without SMOTE)", fontsize=12, fontweight='bold')
ax1.set_xlim(60, 82)
ax1.invert_yaxis()

# Add value labels
for i, (model, acc) in enumerate(zip(binary_df["Model"], binary_df["Accuracy (%)"])):
    ax1.text(acc + 0.3, i, f"{acc:.2f}%", va='center', fontweight='bold', fontsize=10)

# Plot 2: SMOTE Binary Results
ax2 = axes[1]
bars2 = ax2.barh(smote_df["Model"], smote_df["Accuracy (%)"], color=colors_smote)
ax2.set_xlabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax2.set_title("Binary Classification - Accuracy Comparison\n(With SMOTE)", fontsize=12, fontweight='bold')
ax2.set_xlim(60, 82)
ax2.invert_yaxis()

# Add value labels
for i, (model, acc) in enumerate(zip(smote_df["Model"], smote_df["Accuracy (%)"])):
    ax2.text(acc + 0.3, i, f"{acc:.2f}%", va='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig("outputs/Binary_Accuracy_Comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Binary_Accuracy_Comparison.png")
plt.close()

# Create a combined comparison chart
fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(binary_df))
width = 0.35

bars1 = ax.bar(x - width/2, binary_df["Accuracy (%)"], width, label="Without SMOTE", color="steelblue", alpha=0.8)
bars2 = ax.bar(x + width/2, smote_df["Accuracy (%)"], width, label="With SMOTE", color="coral", alpha=0.8)

ax.set_xlabel("Models", fontsize=12, fontweight='bold')
ax.set_ylabel("Accuracy (%)", fontsize=12, fontweight='bold')
ax.set_title("Binary Classification - Accuracy Comparison (All 9 Models)", fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(binary_df["Model"], rotation=45, ha='right')
ax.legend(fontsize=11)
ax.set_ylim(60, 82)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/Binary_Accuracy_Combined_Comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Binary_Accuracy_Combined_Comparison.png")
plt.close()

# Print summary statistics
print("\n" + "="*60)
print("BINARY CLASSIFICATION ACCURACY SUMMARY")
print("="*60)
print("\nWithout SMOTE:")
print(binary_df[["Model", "Accuracy (%)"]].to_string(index=False))
print(f"\nTop 3 Models: {', '.join(binary_df.head(3)['Model'].tolist())}")
print(f"Average Accuracy: {binary_df['Accuracy (%)'].mean():.2f}%")

print("\n" + "-"*60)
print("With SMOTE:")
print(smote_df[["Model", "Accuracy (%)"]].to_string(index=False))
print(f"\nTop 3 Models: {', '.join(smote_df.head(3)['Model'].tolist())}")
print(f"Average Accuracy: {smote_df['Accuracy (%)'].mean():.2f}%")
print("="*60)
