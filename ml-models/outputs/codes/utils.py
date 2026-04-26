import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# Set aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

def load_data(binary=True):
    # Path relative to ml-models directory
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
