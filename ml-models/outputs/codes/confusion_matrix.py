import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from utils import load_data

print("Generating Figure: Confusion Matrix...")
X, y = load_data(binary=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

model = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
model.fit(X_train_sm, y_train_sm)
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-High", "High"], yticklabels=["Non-High", "High"])
plt.title("Confusion Matrix: XGBoost (Binary + SMOTE)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("outputs/Fig4.3_Confusion_Matrix.png")
plt.close()
print("Done! Saved to outputs/Fig4.3_Confusion_Matrix.png")
