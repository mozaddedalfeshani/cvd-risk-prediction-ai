#!/usr/bin/env python3
"""Train the research-aligned binary CVD screening model.

The paper frames the deployable problem as HIGH vs NON-HIGH. It also removes
leakage-prone columns before training, especially CVD Risk Score and Blood
Pressure Category.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "ml-models" / "data" / "raw" / "MymensingUniversity.csv"
MODEL_DIR = ROOT / "ml-models" / "models"
OUTPUT_DIR = ROOT / "ml-models" / "outputs"
TARGET = "CVD Risk Level"
CLASS_NAMES = ["NON_HIGH", "HIGH"]

# 17 usable features after removing leakage-prone and duplicate columns:
# Blood Pressure (mmHg), Height (m), CVD Risk Score, Blood Pressure Category.
BINARY_FEATURES = [
    "Sex",
    "Age",
    "Weight (kg)",
    "BMI",
    "Abdominal Circumference (cm)",
    "Total Cholesterol (mg/dL)",
    "HDL (mg/dL)",
    "Fasting Blood Sugar (mg/dL)",
    "Smoking Status",
    "Diabetes Status",
    "Physical Activity Level",
    "Family History of CVD",
    "Height (cm)",
    "Waist-to-Height Ratio",
    "Systolic BP",
    "Diastolic BP",
    "Estimated LDL (mg/dL)",
]


def encode_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    X = df[BINARY_FEATURES].copy()
    encoders = {}

    for column in X.columns:
        if not is_numeric_dtype(X[column]):
            encoder = LabelEncoder()
            X[column] = encoder.fit_transform(X[column].astype(str))
            encoders[column] = {
                "classes": encoder.classes_.tolist(),
                "mapping": {
                    label: int(index)
                    for index, label in enumerate(encoder.classes_)
                },
            }

    X = X.apply(pd.to_numeric, errors="coerce")
    defaults = X.median(numeric_only=True).fillna(0).to_dict()
    X = X.fillna(defaults)
    return X, encoders, defaults


def train_binary_artifact() -> dict:
    df = pd.read_csv(DATA_PATH)
    df = df[df[TARGET].isin(["LOW", "INTERMEDIARY", "HIGH"])].copy()

    X, encoders, defaults = encode_frame(df)
    y = df[TARGET].apply(lambda value: 1 if value == "HIGH" else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    model = XGBClassifier(
        n_estimators=220,
        max_depth=4,
        learning_rate=0.055,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train_balanced, y_train_balanced)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cv_scores = cross_val_score(
        model,
        X_train_balanced,
        y_train_balanced,
        cv=5,
        scoring="accuracy",
    )
    report = classification_report(
        y_test,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    artifact = {
        "model": model,
        "scaler": None,
        "feature_names": BINARY_FEATURES,
        "encoders": encoders,
        "defaults": defaults,
        "class_names": CLASS_NAMES,
        "positive_class": "HIGH",
        "negative_class": "NON_HIGH",
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "classification_report": report,
        "dataset": str(DATA_PATH.relative_to(ROOT)),
        "training_framing": "Binary classification: HIGH=1, NON_HIGH=0",
        "leakage_removed": [
            "Blood Pressure (mmHg)",
            "Height (m)",
            "CVD Risk Score",
            "Blood Pressure Category",
        ],
        "balancing": "SMOTE applied to training partition only",
        "version": "2026.05-binary-high-vs-non-high",
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_DIR / "cvd_binary_xgb.pkl")
    return artifact


def main() -> None:
    artifact = train_binary_artifact()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        [
            {
                "model": "binary",
                "artifact": "cvd_binary_xgb.pkl",
                "target": "HIGH vs NON_HIGH",
                "accuracy": artifact["accuracy"],
                "f1_score": artifact["f1_score"],
                "cv_mean": artifact["cv_mean"],
                "cv_std": artifact["cv_std"],
                "feature_count": len(artifact["feature_names"]),
            }
        ]
    )
    summary.to_csv(OUTPUT_DIR / "trained_backend_models.csv", index=False)

    print("Binary training complete")
    print(summary.to_string(index=False))
    print("Features:", ", ".join(artifact["feature_names"]))


if __name__ == "__main__":
    main()
