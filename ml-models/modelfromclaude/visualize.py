import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# Ensure output directory exists
os.makedirs("outputs/figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING (Logic from model.py)
# ─────────────────────────────────────────────────────────────
def load_data(binary=True):
    df = pd.read_csv("data/raw/MymensingUniversity.csv")
    df.drop(columns=[
        "Blood Pressure (mmHg)", "Height (m)",
        "CVD Risk Score", "Blood Pressure Category"
    ], inplace=True, errors="ignore")

    cat_cols = [c for c in df.select_dtypes("object").columns if c != "CVD Risk Level"]
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    if binary:
        df["CVD Risk Level"] = df["CVD Risk Level"].apply(lambda x: 1 if x == "HIGH" else 0)
    else:
        df["CVD Risk Level"] = df["CVD Risk Level"].map({"LOW": 0, "INTERMEDIARY": 1, "HIGH": 2})

    df.fillna(df.median(numeric_only=True), inplace=True)
    return df.drop(columns=["CVD Risk Level"]), df["CVD Risk Level"]

# ─────────────────────────────────────────────────────────────
# FIGURE 4.1: Class Distribution (Binary)
# ─────────────────────────────────────────────────────────────
def plot_f1():
    print("Generating Figure 4.1...")
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
    plt.savefig("outputs/figures/Fig4.1_Class_Distribution.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 4.2: Model Accuracy Comparison (Binary)
# ─────────────────────────────────────────────────────────────
def plot_f2():
    print("Generating Figure 4.2...")
    df = pd.read_csv("outputs/binary_results.csv")
    df = df.sort_values("Accuracy (%)", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Accuracy (%)", y="Model", palette="viridis")
    plt.title("Model Accuracy Comparison (Binary)")
    plt.xlim(60, 85)
    for i, v in enumerate(df["Accuracy (%)"]):
        plt.text(v + 0.5, i, f"{v:.2f}%", va='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("outputs/figures/Fig4.2_Model_Comparison.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 4.3: Confusion Matrix (XGBoost + SMOTE)
# ─────────────────────────────────────────────────────────────
def plot_f3():
    print("Generating Figure 4.3...")
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
    plt.savefig("outputs/figures/Fig4.3_Confusion_Matrix.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 4.4: Precision / Recall / F1 Comparison
# ─────────────────────────────────────────────────────────────
def plot_f4():
    print("Generating Figure 4.4...")
    df = pd.read_csv("outputs/binary_results.csv").head(5) # Top 5
    
    df_melted = df.melt(id_vars="Model", value_vars=["Precision (%)", "Recall (%)", "F1-Score (%)"], 
                        var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x="Model", y="Score", hue="Metric", palette="muted")
    plt.title("Precision, Recall, and F1-Score Comparison (Top 5 Models)")
    plt.ylim(65, 85)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig("outputs/figures/Fig4.4_Metric_Comparison.png")
    plt.close()

# ─────────────────────────────────────────────────────────────
# FIGURE 4.5: Binary vs Multiclass Accuracy
# ─────────────────────────────────────────────────────────────
def plot_f5():
    print("Generating Figure 4.5...")
    bin_df = pd.read_csv("outputs/binary_results.csv")
    multi_df = pd.read_csv("outputs/multiclass_results.csv")
    
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
    plt.savefig("outputs/figures/Fig4.5_Binary_vs_Multiclass.png")
    plt.close()

# Run all
plot_f1()
plot_f2()
plot_f3()
plot_f4()
plot_f5()

print("\nDone! All figures saved in ml-models/outputs/figures/")
