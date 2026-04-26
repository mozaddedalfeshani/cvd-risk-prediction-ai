import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data

print("Generating Figure: Model Comparison...")
df = pd.read_csv("outputs/accuracy/binary_results.csv")
df = df.sort_values("Accuracy (%)", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Accuracy (%)", y="Model", palette="viridis")
plt.title("Model Accuracy Comparison (Binary)")
plt.xlim(60, 85)
for i, v in enumerate(df["Accuracy (%)"]):
    plt.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')
    
plt.tight_layout()
plt.savefig("outputs/Fig4.2_Model_Comparison.png")
plt.close()
print("Done! Saved to outputs/Fig4.2_Model_Comparison.png")
