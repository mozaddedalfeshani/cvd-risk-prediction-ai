import pandas as pd
import sys
sys.path.insert(0, '.')

# Load both datasets
normal = pd.read_csv("outputs/accuracy/binary_results.csv")
smote = pd.read_csv("outputs/accuracy/smote_binary_results.csv")

# Rename columns to distinguish
normal = normal.rename(columns={
    "Accuracy (%)": "Accuracy",
    "Precision (%)": "Precision",
    "Recall (%)": "Recall",
    "F1-Score (%)": "F1"
})
smote = smote.rename(columns={
    "Accuracy (%)": "SMOTE Accuracy",
    "Precision (%)": "SMOTE Precision",
    "Recall (%)": "SMOTE Recall",
    "F1-Score (%)": "SMOTE F1"
})

# Merge on Model
merged = pd.merge(normal[["Model", "Accuracy", "Precision", "Recall", "F1"]],
                   smote[["Model", "SMOTE Accuracy", "SMOTE Precision", "SMOTE Recall", "SMOTE F1"]],
                   on="Model", how="outer")

# Reorder columns
merged = merged[["Model", "Accuracy", "Precision", "Recall", "F1",
                 "SMOTE Accuracy", "SMOTE Precision", "SMOTE Recall", "SMOTE F1"]]

# Sort by normal Accuracy descending
merged = merged.sort_values("Accuracy", ascending=False)

# Format to 2 decimals
for col in merged.columns:
    if col != "Model":
        merged[col] = merged[col].apply(lambda x: f"{x:.2f}%")

print("\n=== CVD Risk Prediction Model Comparison ===")
print(merged.to_string(index=False))
print("\n✓ Table includes: Accuracy, Precision, Recall, F1 for both normal and SMOTE versions")

# Save to CSV
merged.to_csv("outputs/accuracy/full_model_comparison.csv", index=False)
print("✓ Saved to outputs/accuracy/full_model_comparison.csv")
