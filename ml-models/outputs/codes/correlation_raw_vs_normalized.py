import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

df = pd.read_csv("data/raw/MymensingUniversity.csv")

TARGET = "CVD Risk Level"

# 4 removed for quality reasons + target = 5 total excluded from training
REMOVED_FEATURES = [
    "Blood Pressure (mmHg)",    # redundant: split into Systolic + Diastolic BP
    "Height (m)",               # redundant: same as Height (cm), different unit only
    "CVD Risk Score",           # data leakage: derived from target variable
    "Blood Pressure Category",  # data leakage: derived from BP thresholds tied to risk
]

BINARY_FEATURES = [
    "Sex", "Age", "Weight (kg)", "BMI", "Abdominal Circumference (cm)",
    "Total Cholesterol (mg/dL)", "HDL (mg/dL)", "Fasting Blood Sugar (mg/dL)",
    "Smoking Status", "Diabetes Status", "Physical Activity Level",
    "Family History of CVD", "Height (cm)", "Waist-to-Height Ratio",
    "Systolic BP", "Diastolic BP", "Estimated LDL (mg/dL)",
]

ALL_22_COLS = REMOVED_FEATURES + BINARY_FEATURES + [TARGET]


def parse_bp_string(series: pd.Series) -> pd.Series:
    """Convert 'systolic/diastolic' strings like '125/79' to mean BP float."""
    def _parse(val):
        try:
            parts = str(val).split("/")
            return (float(parts[0]) + float(parts[1])) / 2
        except Exception:
            return np.nan
    return series.apply(_parse)


def label_encode_df(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    # Handle "systolic/diastolic" format before encoding
    if "Blood Pressure (mmHg)" in out.columns:
        out["Blood Pressure (mmHg)"] = parse_bp_string(out["Blood Pressure (mmHg)"])
    for col in out.columns:
        # pandas 3.0 uses StringDtype (dtype="str") — must cast to float explicitly
        if not pd.api.types.is_numeric_dtype(out[col]):
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].astype(str)).astype(float)
    # Ensure all columns are numeric float
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    medians = out.median(numeric_only=True)
    out = out.fillna(medians)
    return out


# ─────────────────────────────────────────────────────────────
# MATRIX 1 — ALL 22 ORIGINAL FEATURES (RAW)
# ─────────────────────────────────────────────────────────────

raw_df = label_encode_df(df[ALL_22_COLS])
corr_raw = raw_df.corr()

fig, ax = plt.subplots(figsize=(18, 16))

# Mark removed/target columns in tick labels with symbols
def annotate_label(col):
    if col in REMOVED_FEATURES:
        return f"[X] {col}"
    if col == TARGET:
        return f"[T] {col}"
    return col

tick_labels_22 = [annotate_label(c) for c in ALL_22_COLS]

sns.heatmap(
    corr_raw,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.4,
    cbar_kws={"label": "Pearson Correlation", "shrink": 0.75},
    annot_kws={"fontsize": 6.5, "fontweight": "bold"},
    xticklabels=tick_labels_22,
    yticklabels=tick_labels_22,
    vmin=-1, vmax=1,
    ax=ax,
)

ax.set_title(
    "Matrix 1 — All 22 Original Features  |  Raw Training Data\n"
    "[X] = removed before training   [T] = target variable",
    fontsize=13, fontweight="bold", pad=16,
)

plt.xticks(rotation=50, ha="right", fontsize=7)
plt.yticks(rotation=0, fontsize=7)

# Color removed-feature tick labels red
for label in ax.get_xticklabels():
    if label.get_text().startswith("[X]") or label.get_text().startswith("[T]"):
        label.set_color("firebrick")
for label in ax.get_yticklabels():
    if label.get_text().startswith("[X]") or label.get_text().startswith("[T]"):
        label.set_color("firebrick")

removed_patch = mpatches.Patch(color="firebrick", label="[X] Removed feature  [T] Target")
kept_patch = mpatches.Patch(color="steelblue", label="Kept training feature")
ax.legend(handles=[removed_patch, kept_patch], loc="upper left",
          bbox_to_anchor=(0, -0.18), fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("outputs/Correlation_Matrix_1_Raw_22_Features.png", dpi=300, bbox_inches="tight")
print("Saved: outputs/Correlation_Matrix_1_Raw_22_Features.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# MATRIX 2 — 17 TRAINING FEATURES (MinMax Normalized)
# ─────────────────────────────────────────────────────────────

encoded_17 = label_encode_df(df[BINARY_FEATURES])
scaler = MinMaxScaler()
normalized_17 = pd.DataFrame(
    scaler.fit_transform(encoded_17),
    columns=BINARY_FEATURES,
)

corr_norm = normalized_17.corr()

short_names = [
    "Sex", "Age", "Weight", "BMI", "Abd Circ",
    "Total Chol", "HDL", "Fasting BS", "Smoking", "Diabetes",
    "Phys Act", "Fam Hist", "Height cm", "Waist/H",
    "Systolic", "Diastolic", "LDL",
]

fig, ax = plt.subplots(figsize=(15, 13))

sns.heatmap(
    corr_norm,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.4,
    cbar_kws={"label": "Pearson Correlation", "shrink": 0.8},
    annot_kws={"fontsize": 8, "fontweight": "bold"},
    xticklabels=short_names,
    yticklabels=short_names,
    vmin=-1, vmax=1,
    ax=ax,
)

ax.set_title(
    "Matrix 2 — 17 Training Features  |  MinMax Normalized [0, 1]\n"
    "5 removed features excluded  •  Data used for model training",
    fontsize=13, fontweight="bold", pad=16,
)

plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)

plt.tight_layout()
plt.savefig("outputs/Correlation_Matrix_2_Normalized_17_Features.png", dpi=300, bbox_inches="tight")
print("Saved: outputs/Correlation_Matrix_2_Normalized_17_Features.png")
plt.close()

# ─────────────────────────────────────────────────────────────
# CONSOLE REPORT
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("WHY 5 FEATURES WERE REMOVED FROM MODEL TRAINING")
print("=" * 80)

removal_table = [
    ("Blood Pressure (mmHg)", "Redundancy",
     "Identical info to Systolic BP + Diastolic BP (just combined). Keeping both causes multicollinearity."),
    ("Height (m)",            "Redundancy",
     "Same measurement as Height (cm). Unit conversion only — zero added information."),
    ("CVD Risk Score",        "Data Leakage",
     "Numerically derived from target variable (CVD Risk Level). Model would learn to memorize it, not predict."),
    ("Blood Pressure Category","Data Leakage",
     "Categorical label derived from BP values + risk thresholds. Leaks target information into features."),
    ("CVD Risk Level",        "Target Variable",
     "This IS what we predict. Cannot be used as an input feature."),
]

for i, (feat, reason, detail) in enumerate(removal_table, 1):
    print(f"\n  {i}. {feat}")
    print(f"     Reason   : {reason}")
    print(f"     Detail   : {detail}")

print("\n" + "=" * 80)
print("KEY CORRELATIONS FROM MATRIX 1 JUSTIFYING REMOVAL")
print("=" * 80)

pairs_to_check = [
    ("Height (m)",              "Height (cm)"),
    ("Blood Pressure (mmHg)",   "Systolic BP"),
    ("Blood Pressure (mmHg)",   "Diastolic BP"),
    ("CVD Risk Score",          "CVD Risk Level"),
    ("Blood Pressure Category", "Systolic BP"),
]

for f1, f2 in pairs_to_check:
    if f1 in corr_raw.columns and f2 in corr_raw.columns:
        val = corr_raw.loc[f1, f2]
        flag = "HIGH REDUNDANCY" if abs(val) >= 0.8 else ("MODERATE" if abs(val) >= 0.5 else "")
        print(f"  {f1:<30} <-> {f2:<25}  r = {val:+.4f}  {flag}")

print("\n" + "=" * 80)
print("TOP 10 CORRELATIONS IN NORMALIZED 17-FEATURE MATRIX")
print("=" * 80)

pairs = []
for i in range(len(corr_norm.columns)):
    for j in range(i + 1, len(corr_norm.columns)):
        pairs.append((corr_norm.columns[i], corr_norm.columns[j], corr_norm.iloc[i, j]))

pairs.sort(key=lambda x: abs(x[2]), reverse=True)
for f1, f2, val in pairs[:10]:
    print(f"  {f1:<35} <-> {f2:<35}  r = {val:+.4f}")

print()
