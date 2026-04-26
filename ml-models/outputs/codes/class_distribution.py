import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from utils import load_data

print("Generating Figure: Class Distribution...")
X, y = load_data(binary=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

before = y_train.value_counts().sort_index()
after = y_res.value_counts().sort_index()

labels = ["Non-High", "High"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=labels, y=before.values, ax=axes[0], palette="Blues_d")
axes[0].set_title("Before SMOTE (Training Set)")
axes[0].set_ylabel("Count")
for i, v in enumerate(before.values):
    axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
sns.barplot(x=labels, y=after.values, ax=axes[1], palette="Greens_d")
axes[1].set_title("After SMOTE (Balanced Training Set)")
axes[1].set_ylabel("Count")
for i, v in enumerate(after.values):
    axes[1].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
plt.tight_layout()
plt.savefig("outputs/Fig4.1_Class_Distribution.png")
plt.close()
print("Done! Saved to outputs/Fig4.1_Class_Distribution.png")
