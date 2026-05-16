import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9

# Load binary accuracy data
df = pd.read_csv("outputs/accuracy/binary_results.csv")

# Sort by accuracy for consistency
df = df.sort_values("Accuracy (%)", ascending=False)

# Extract metrics
models = df["Model"]
accuracy = df["Accuracy (%)"]
precision = df["Precision (%)"]
recall = df["Recall (%)"]
f1_score = df["F1-Score (%)"]

# ==================== VISUALIZATION 1: SUBPLOTS ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Binary Classification - All 9 Models: Comprehensive Metrics Comparison", 
             fontsize=14, fontweight='bold', y=1.00)

metrics = [
    ("Accuracy (%)", accuracy, "steelblue"),
    ("Precision (%)", precision, "coral"),
    ("Recall (%)", recall, "mediumseagreen"),
    ("F1-Score (%)", f1_score, "mediumpurple")
]

for idx, (metric_name, metric_values, color) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    bars = ax.bar(range(len(models)), metric_values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
    ax.set_title(metric_name, fontsize=12, fontweight='bold', pad=10)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_ylim(60, 82)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.5, f'{value:.2f}%',
                ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/Binary_Metrics_Subplots.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Binary_Metrics_Subplots.png")
plt.close()

# ==================== VISUALIZATION 2: GROUPED BAR CHART ====================
fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(models))
width = 0.2

bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='steelblue', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='coral', alpha=0.85, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='mediumseagreen', alpha=0.85, edgecolor='black', linewidth=0.5)
bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='mediumpurple', alpha=0.85, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Models', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Binary Classification - All 9 Models: Accuracy vs Precision vs Recall vs F1-Score', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
ax.legend(fontsize=11, loc='lower left', framealpha=0.95)
ax.set_ylim(55, 85)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/Binary_Metrics_Grouped_Comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Binary_Metrics_Grouped_Comparison.png")
plt.close()

# ==================== VISUALIZATION 3: HEATMAP ====================
fig, ax = plt.subplots(figsize=(10, 8))

# Create data matrix for heatmap
metrics_data = df[["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]].values
metrics_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

# Create heatmap
sns.heatmap(metrics_data, annot=True, fmt='.2f', cmap='RdYlGn', 
            xticklabels=metrics_labels, yticklabels=models,
            cbar_kws={'label': 'Score (%)'}, ax=ax, linewidths=0.5,
            vmin=60, vmax=82, annot_kws={'fontsize': 10, 'fontweight': 'bold'})

ax.set_title('Binary Classification - Metrics Heatmap (All 9 Models)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_ylabel('Models', fontsize=11, fontweight='bold')
ax.set_xlabel('Metrics', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/Binary_Metrics_Heatmap.png", dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/Binary_Metrics_Heatmap.png")
plt.close()

# ==================== DETAILED SUMMARY ====================
print("\n" + "="*80)
print("BINARY CLASSIFICATION - ALL 9 MODELS: COMPREHENSIVE METRICS SUMMARY")
print("="*80)

summary_df = df[["Model", "Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]].copy()
summary_df = summary_df.round(2)

print("\n" + summary_df.to_string(index=False))

print("\n" + "-"*80)
print("METRICS STATISTICS (All 9 Models):")
print("-"*80)

stats_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Mean': [
        accuracy.mean(),
        precision.mean(),
        recall.mean(),
        f1_score.mean()
    ],
    'Std Dev': [
        accuracy.std(),
        precision.std(),
        recall.std(),
        f1_score.std()
    ],
    'Min': [
        accuracy.min(),
        precision.min(),
        recall.min(),
        f1_score.min()
    ],
    'Max': [
        accuracy.max(),
        precision.max(),
        recall.max(),
        f1_score.max()
    ],
    'Range': [
        accuracy.max() - accuracy.min(),
        precision.max() - precision.min(),
        recall.max() - recall.min(),
        f1_score.max() - f1_score.min()
    ]
}

stats_df = pd.DataFrame(stats_data)
stats_df = stats_df.round(2)
print("\n" + stats_df.to_string(index=False))

print("\n" + "-"*80)
print("TOP PERFORMERS BY METRIC:")
print("-"*80)
print(f"🏆 Best Accuracy:  {df.loc[df['Accuracy (%)'].idxmax(), 'Model']} ({accuracy.max():.2f}%)")
print(f"🏆 Best Precision: {df.loc[df['Precision (%)'].idxmax(), 'Model']} ({precision.max():.2f}%)")
print(f"🏆 Best Recall:    {df.loc[df['Recall (%)'].idxmax(), 'Model']} ({recall.max():.2f}%)")
print(f"🏆 Best F1-Score:  {df.loc[df['F1-Score (%)'].idxmax(), 'Model']} ({f1_score.max():.2f}%)")

print("\n" + "="*80)
