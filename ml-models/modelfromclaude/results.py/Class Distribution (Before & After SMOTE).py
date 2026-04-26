import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("ml-models/data/raw/MymensingUniversity.csv")

# Encode target
# Convert to binary
y = df["CVD Risk Level"].apply(lambda x: 1 if x == "HIGH" else 0)
X = df.drop(columns=["CVD Risk Level"])

# Train-test split
# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Before SMOTE
before = Counter(y_train)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_sm, y_sm = smote.fit_resample(X_train, y_train)

# After SMOTE
after = Counter(y_sm)

# Plot
labels = ["NON-HIGH", "HIGH"]
before_vals = [before[0], before[1]]
after_vals = [after[0], after[1]]

x = range(len(labels))

plt.figure()
# Plot before
plt.figure()
plt.bar(labels, before_vals)
plt.title("Binary Class Distribution (Before SMOTE)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot after
plt.figure()
plt.bar(labels, after_vals)
plt.title("Binary Class Distribution (After SMOTE)")
plt.tight_layout()
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()