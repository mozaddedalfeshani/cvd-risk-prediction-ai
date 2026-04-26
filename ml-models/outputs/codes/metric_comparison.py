import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

print("Generating Figure: Metric Comparison...")
df = pd.read_csv("outputs/accuracy/binary_results.csv").head(5) # Top 5

df_melted = df.melt(id_vars="Model", value_vars=["Precision (%)", "Recall (%)", "F1-Score (%)"], 
                    var_name="Metric", value_name="Score")

plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="muted")
plt.title("Precision, Recall, and F1-Score Comparison (Top 5 Models)")
plt.ylim(65, 85)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("outputs/Fig4.4_Metric_Comparison.png")
plt.close()
print("Done! Saved to outputs/Fig4.4_Metric_Comparison.png")
