import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

print("Generating Figure: Binary vs Multiclass Comparison...")
bin_df = pd.read_csv("outputs/accuracy/binary_results.csv")
multi_df = pd.read_csv("outputs/accuracy/multiclass_results.csv")

# Merge and compare
combined = []
for model in bin_df["Model"].unique():
    b_acc = bin_df[bin_df["Model"] == model]["Accuracy (%)"].values[0]
    m_acc = multi_df[multi_df["Model"] == model]["Accuracy (%)"].values[0]
    combined.append({"Model": model, "Mode": "Binary", "Accuracy": b_acc})
    combined.append({"Model": model, "Mode": "Multiclass", "Accuracy": m_acc})

comp_df = pd.DataFrame(combined)

plt.figure(figsize=(12, 6))
sns.barplot(data=comp_df, x="Model", y="Accuracy", hue="Mode", palette="coolwarm")
plt.title("Accuracy Comparison: Binary vs Multiclass Classification")
plt.ylim(50, 85)
plt.xticks(rotation=15)

plt.tight_layout()
plt.savefig("outputs/Fig4.5_Binary_vs_Multiclass.png")
plt.close()
print("Done! Saved to outputs/Fig4.5_Binary_vs_Multiclass.png")
