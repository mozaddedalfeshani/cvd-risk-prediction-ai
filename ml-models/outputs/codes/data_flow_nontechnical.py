import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with high DPI for clarity
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Color scheme for non-technical appeal
color_data = '#E8F4F8'      # Light blue for data
color_process = '#FFF4E6'   # Light orange for processing
color_model = '#F0E8F4'     # Light purple for model
color_result = '#E8F8E8'    # Light green for results

# ==================== TITLE ====================
ax.text(5, 13.2, '📊 CVD Risk Prediction - How It Works', 
        fontsize=24, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', edgecolor='navy', linewidth=2))

# ==================== STAGE 1: DATA COLLECTION ====================
y_pos = 12

# Box 1
rect1 = FancyBboxPatch((0.3, y_pos-1), 2.5, 1.2, boxstyle="round,pad=0.1", 
                        edgecolor='navy', facecolor=color_data, linewidth=2)
ax.add_patch(rect1)
ax.text(1.55, y_pos-0.4, '🏥 Patient Data\nCollection', fontsize=11, fontweight='bold', 
        ha='center', va='center')
ax.text(1.55, y_pos-0.8, '1,529 Patients', fontsize=9, ha='center', style='italic')

# Arrow down
arrow1 = FancyArrowPatch((1.55, y_pos-1.1), (1.55, y_pos-1.8),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkblue')
ax.add_patch(arrow1)

# ==================== STAGE 2: DATA CLEANING ====================
y_pos = 10.2

# Title
ax.text(1.55, y_pos+0.4, '🧹 Data Cleaning & Preparation', fontsize=12, fontweight='bold', ha='center')

# Removed items
rect2a = FancyBboxPatch((0.1, y_pos-0.8), 1.9, 0.9, boxstyle="round,pad=0.05", 
                         edgecolor='#CC0000', facecolor='#FFE0E0', linewidth=2, linestyle='--')
ax.add_patch(rect2a)
ax.text(1.05, y_pos-0.35, '❌ Remove Leakage:\n• CVD Risk Score\n• Blood Pressure Cat.', 
        fontsize=8, ha='center', va='center')

# Keep items
rect2b = FancyBboxPatch((2.2, y_pos-0.8), 1.9, 0.9, boxstyle="round,pad=0.05", 
                         edgecolor='green', facecolor='#E0FFE0', linewidth=2)
ax.add_patch(rect2b)
ax.text(3.15, y_pos-0.35, '✅ Keep Valid Data:\n• All 1,529 rows\n• Remove duplicates', 
        fontsize=8, ha='center', va='center')

# Arrow down
arrow2 = FancyArrowPatch((1.55, y_pos-0.9), (1.55, y_pos-1.6),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkblue')
ax.add_patch(arrow2)

# ==================== STAGE 3: FEATURE SELECTION ====================
y_pos = 8.1

# Main box
rect3 = FancyBboxPatch((0.3, y_pos-1.8), 2.5, 1.9, boxstyle="round,pad=0.1", 
                        edgecolor='darkred', facecolor=color_process, linewidth=2)
ax.add_patch(rect3)

ax.text(1.55, y_pos-0.2, '🔍 Feature Selection', fontsize=11, fontweight='bold', ha='center')
ax.text(1.55, y_pos-0.7, '22 Columns → 17 Features', fontsize=9, ha='center', fontweight='bold')
ax.text(1.55, y_pos-1.4, 'Medical Predictors:\nAge, BMI, Cholesterol,\nBlood Pressure, Diabetes,\nSmoking, Family History...', 
        fontsize=8, ha='center', va='center')

# Arrow down
arrow3 = FancyArrowPatch((1.55, y_pos-1.9), (1.55, y_pos-2.6),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkblue')
ax.add_patch(arrow3)

# ==================== STAGE 4: CLASS TRANSFORMATION ====================
y_pos = 5.3

# Title
ax.text(1.55, y_pos+0.4, '🎯 Target Definition', fontsize=12, fontweight='bold', ha='center')

# From multiclass
rect4a = FancyBboxPatch((0.1, y_pos-0.9), 1.8, 1, boxstyle="round,pad=0.05", 
                         edgecolor='purple', facecolor='#F0E8F4', linewidth=2)
ax.add_patch(rect4a)
ax.text(1.0, y_pos-0.4, 'Multiclass:\n• Low: 220\n• Intermediary: 581\n• High: 728', 
        fontsize=8, ha='center', va='center', fontweight='bold')

# Arrow
arrow4 = FancyArrowPatch((1.9, y_pos-0.4), (2.3, y_pos-0.4),
                        arrowstyle='->', mutation_scale=25, linewidth=2.5, color='purple')
ax.add_patch(arrow4)
ax.text(2.1, y_pos+0.1, '→', fontsize=14, ha='center', fontweight='bold')

# To binary
rect4b = FancyBboxPatch((2.4, y_pos-0.9), 1.8, 1, boxstyle="round,pad=0.05", 
                         edgecolor='green', facecolor='#E8F8E8', linewidth=2)
ax.add_patch(rect4b)
ax.text(3.3, y_pos-0.4, 'Binary:\n• Non-High: 801\n  (Low+Intermediary)\n• High: 728', 
        fontsize=8, ha='center', va='center', fontweight='bold')

# Arrow down from middle
arrow4b = FancyArrowPatch((1.55, y_pos-1), (1.55, y_pos-1.7),
                         arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkblue')
ax.add_patch(arrow4b)

# ==================== STAGE 5: DATA SPLIT ====================
y_pos = 3.4

# Title
ax.text(1.55, y_pos+0.3, '✂️ Split for Training', fontsize=11, fontweight='bold', ha='center')

# Training set
rect5a = FancyBboxPatch((0.1, y_pos-0.7), 1.8, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor='#0066CC', facecolor='#CCE5FF', linewidth=2)
ax.add_patch(rect5a)
ax.text(1.0, y_pos-0.35, '📚 Training Set (80%)\n1,223 patients\nTeach the model', 
        fontsize=8, ha='center', va='center', fontweight='bold')

# Test set
rect5b = FancyBboxPatch((2.2, y_pos-0.7), 1.8, 0.8, boxstyle="round,pad=0.05", 
                         edgecolor='#CC6600', facecolor='#FFCCAA', linewidth=2)
ax.add_patch(rect5b)
ax.text(3.1, y_pos-0.35, '🧪 Test Set (20%)\n306 patients\nEvaluate accuracy', 
        fontsize=8, ha='center', va='center', fontweight='bold')

# Arrow to training
arrow5a = FancyArrowPatch((1.0, y_pos-0.75), (1.0, y_pos-1.3),
                         arrowstyle='->', mutation_scale=25, linewidth=2, color='#0066CC')
ax.add_patch(arrow5a)

# Arrow to model
arrow5b = FancyArrowPatch((1.55, y_pos-1.1), (3.5, y_pos-1.3),
                         arrowstyle='->', mutation_scale=25, linewidth=2.5, color='darkblue')
ax.add_patch(arrow5b)

# ==================== STAGE 6: MODEL TRAINING ====================
y_pos = 1.5

# Title
ax.text(3.5, y_pos+0.5, '🤖 Model Training', fontsize=12, fontweight='bold', ha='center')

# Models box
rect6 = FancyBboxPatch((2.3, y_pos-1), 2.4, 1.2, boxstyle="round,pad=0.1", 
                        edgecolor='darkblue', facecolor=color_model, linewidth=2)
ax.add_patch(rect6)

ax.text(3.5, y_pos-0.2, '9 Different AI Models:', fontsize=10, fontweight='bold', ha='center')
ax.text(3.5, y_pos-0.65, 'SVM • XGBoost • Random Forest\nGradient Boosting • Logistic Regression\nand more...', 
        fontsize=8, ha='center', va='center')

# Arrow to evaluation
arrow6 = FancyArrowPatch((3.5, y_pos-1.15), (3.5, y_pos-1.8),
                        arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkblue')
ax.add_patch(arrow6)

# ==================== STAGE 7: EVALUATION ====================
y_pos = -0.4

# Evaluation box
rect7 = FancyBboxPatch((2.3, y_pos-1.1), 2.4, 1.2, boxstyle="round,pad=0.1", 
                        edgecolor='green', facecolor=color_result, linewidth=2)
ax.add_patch(rect7)

ax.text(3.5, y_pos-0.1, '📊 Evaluate Performance', fontsize=10, fontweight='bold', ha='center')
ax.text(3.5, y_pos-0.65, 'Accuracy • Precision • Recall\nF1-Score • Cross-Validation', 
        fontsize=8, ha='center', va='center')

# ==================== RIGHT SIDE: PREDICTION PIPELINE ====================
y_pos = 12

# Input box
rect_input = FancyBboxPatch((5.5, y_pos-1), 3.5, 1.2, boxstyle="round,pad=0.1", 
                            edgecolor='navy', facecolor=color_data, linewidth=2)
ax.add_patch(rect_input)
ax.text(7.25, y_pos-0.4, '👨‍⚕️ New Patient Data', fontsize=11, fontweight='bold', ha='center')
ax.text(7.25, y_pos-0.8, '17 Medical Measurements', fontsize=9, ha='center', style='italic')

# Arrow down
arrow_p1 = FancyArrowPatch((7.25, y_pos-1.1), (7.25, y_pos-1.8),
                          arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkgreen')
ax.add_patch(arrow_p1)

# Preprocessing box
y_pos = 10.2
rect_prep = FancyBboxPatch((5.5, y_pos-1), 3.5, 1.2, boxstyle="round,pad=0.1", 
                           edgecolor='darkred', facecolor=color_process, linewidth=2)
ax.add_patch(rect_prep)
ax.text(7.25, y_pos-0.4, '🔧 Prepare Data', fontsize=11, fontweight='bold', ha='center')
ax.text(7.25, y_pos-0.8, 'Clean & Encode\n(Same as training)', fontsize=9, ha='center')

# Arrow down
arrow_p2 = FancyArrowPatch((7.25, y_pos-1.1), (7.25, y_pos-1.8),
                          arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkgreen')
ax.add_patch(arrow_p2)

# Best Model box
y_pos = 8.2
rect_best = FancyBboxPatch((5.5, y_pos-1), 3.5, 1.2, boxstyle="round,pad=0.1", 
                           edgecolor='darkblue', facecolor=color_model, linewidth=2)
ax.add_patch(rect_best)
ax.text(7.25, y_pos-0.4, '🏆 Best Model\n(SVM: 79% accuracy)', fontsize=11, fontweight='bold', ha='center')

# Arrow down
arrow_p3 = FancyArrowPatch((7.25, y_pos-1.1), (7.25, y_pos-1.8),
                          arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkgreen')
ax.add_patch(arrow_p3)

# Prediction box
y_pos = 6.2
rect_pred = FancyBboxPatch((5.5, y_pos-1.2), 3.5, 1.4, boxstyle="round,pad=0.1", 
                           edgecolor='green', facecolor=color_result, linewidth=3)
ax.add_patch(rect_pred)
ax.text(7.25, y_pos-0.3, '✅ CVD Risk Prediction', fontsize=12, fontweight='bold', ha='center')
ax.text(7.25, y_pos-0.8, 'HIGH RISK 🔴\nor\nNON-HIGH RISK 🟢', 
        fontsize=10, ha='center', va='center', fontweight='bold')

# Arrow down
arrow_p4 = FancyArrowPatch((7.25, y_pos-1.3), (7.25, y_pos-2),
                          arrowstyle='->', mutation_scale=30, linewidth=2.5, color='darkgreen')
ax.add_patch(arrow_p4)

# Doctor recommendation
y_pos = 3.9
rect_doc = FancyBboxPatch((5.5, y_pos-1), 3.5, 1.2, boxstyle="round,pad=0.1", 
                          edgecolor='darkgreen', facecolor='#FFFFCC', linewidth=2)
ax.add_patch(rect_doc)
ax.text(7.25, y_pos-0.4, '📋 Doctor Review', fontsize=11, fontweight='bold', ha='center')
ax.text(7.25, y_pos-0.8, 'Take appropriate action\nif needed', fontsize=9, ha='center')

# ==================== BOTTOM STATS ====================
y_pos = 1.8

# Training stats
rect_stats1 = FancyBboxPatch((0.3, y_pos-1.5), 2.5, 1.6, boxstyle="round,pad=0.1", 
                             edgecolor='#0066CC', facecolor='#E0ECFF', linewidth=1.5)
ax.add_patch(rect_stats1)
ax.text(1.55, y_pos-0.2, '📈 Model Results', fontsize=10, fontweight='bold', ha='center')
ax.text(1.55, y_pos-1.1, 'Best: SVM\nAccuracy: 79.08%\nPrecision: 78.87%\nRecall: 76.71%', 
        fontsize=8, ha='center', va='center', family='monospace')

# Data summary
rect_stats2 = FancyBboxPatch((5.5, y_pos-1.5), 3.5, 1.6, boxstyle="round,pad=0.1", 
                             edgecolor='#006600', facecolor='#E0FFE0', linewidth=1.5)
ax.add_patch(rect_stats2)
ax.text(7.25, y_pos-0.2, '💾 Dataset Summary', fontsize=10, fontweight='bold', ha='center')
ax.text(7.25, y_pos-1.1, 'Total Patients: 1,529\nFeatures: 17 medical factors\nClass Balance: 52% vs 48%\nNon-Leakage: ✅', 
        fontsize=8, ha='center', va='center', family='monospace')

# Legend/Key at bottom
y_pos = -0.8
ax.text(5, y_pos, '🔵 Blue boxes = Data stages  |  🟠 Orange boxes = Processing steps  |  🟣 Purple boxes = Model training  |  🟢 Green boxes = Results', 
        fontsize=9, ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', linewidth=1))

plt.tight_layout()
plt.savefig("outputs/Data_Flow_Diagram_NonTechnical.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: outputs/Data_Flow_Diagram_NonTechnical.png")
plt.close()

# ==================== CREATE SIMPLIFIED VERSION ====================
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.3, '🏥 How We Predict Heart Disease Risk - Simple Overview', 
        fontsize=20, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightcoral', edgecolor='darkred', linewidth=2))

# Left column - Process
ax.text(2.5, 10.2, 'HOW WE BUILD THE SYSTEM', fontsize=12, fontweight='bold', ha='center', 
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

steps_build = [
    ('1️⃣', 'Collect 1,529 Patient Records', 9.2),
    ('2️⃣', 'Remove Bad/Duplicate Data', 8.2),
    ('3️⃣', 'Keep 17 Important Medical Factors', 7.2),
    ('4️⃣', 'Define Target: HIGH Risk vs NON-HIGH', 6.2),
    ('5️⃣', 'Train 9 Different AI Models', 5.2),
    ('6️⃣', 'Pick Best Model (SVM: 79% accurate)', 4.2),
]

for emoji, text, y in steps_build:
    rect = FancyBboxPatch((0.5, y-0.4), 4, 0.7, boxstyle="round,pad=0.05", 
                          edgecolor='blue', facecolor='#CCE5FF', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(0.7, y, emoji, fontsize=14, ha='left', va='center')
    ax.text(2.5, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    if y > 4.2:
        arrow = FancyArrowPatch((2.5, y-0.45), (2.5, y-0.95),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='blue')
        ax.add_patch(arrow)

# Right column - Usage
ax.text(7.5, 10.2, 'HOW DOCTORS USE IT', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

steps_use = [
    ('1️⃣', 'Enter Patient Information', 9.2),
    ('2️⃣', 'System Processes Data', 8.2),
    ('3️⃣', 'AI Model Makes Prediction', 7.2),
]

for emoji, text, y in steps_use:
    rect = FancyBboxPatch((5.5, y-0.4), 4, 0.7, boxstyle="round,pad=0.05", 
                          edgecolor='green', facecolor='#CCF0CC', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(5.7, y, emoji, fontsize=14, ha='left', va='center')
    ax.text(7.5, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
    
    if y > 7.2:
        arrow = FancyArrowPatch((7.5, y-0.45), (7.5, y-0.95),
                              arrowstyle='->', mutation_scale=20, linewidth=2, color='green')
        ax.add_patch(arrow)

# Result box
rect_result = FancyBboxPatch((5.5, 6.2-0.5), 4, 1, boxstyle="round,pad=0.1", 
                             edgecolor='darkgreen', facecolor='#FFFF99', linewidth=3)
ax.add_patch(rect_result)
ax.text(7.5, 6.6, '🎯 RESULT', fontsize=12, fontweight='bold', ha='center')
ax.text(7.5, 6.1, 'HIGH RISK 🔴  or  NON-HIGH RISK 🟢', fontsize=11, ha='center', fontweight='bold', color='darkgreen')

# Key benefits section
y_benefit = 4.5
ax.text(5, y_benefit, '✨ WHY THIS MATTERS ✨', fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

benefits = [
    '✅ Catches High-Risk Patients Early',
    '✅ 79% Accurate Predictions',
    '✅ Fast Results (seconds)',
    '✅ Uses Real Medical Data',
    '✅ No False Information (Data Leakage Prevention)',
]

for i, benefit in enumerate(benefits):
    ax.text(5, y_benefit-0.6-(i*0.5), benefit, fontsize=10, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.6, edgecolor='orange'))

plt.tight_layout()
plt.savefig("outputs/Data_Flow_Simplified_NonTechnical.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: outputs/Data_Flow_Simplified_NonTechnical.png")
plt.close()

print("\n" + "="*80)
print("DATA FLOW DIAGRAMS CREATED FOR NON-TECHNICAL USERS")
print("="*80)
print("\n✓ Two comprehensive diagrams generated:")
print("  1. Data_Flow_Diagram_NonTechnical.png - Detailed step-by-step flow")
print("  2. Data_Flow_Simplified_NonTechnical.png - Simple side-by-side comparison")
print("\n📌 Key Points Covered:")
print("  • Data collection and cleaning process")
print("  • Feature selection (22 → 17)")
print("  • Class transformation (3 classes → 2 for binary prediction)")
print("  • Model training pipeline")
print("  • Real-world prediction workflow")
print("  • Easy-to-understand results")
print("\n✨ Perfect for: Doctors, Hospital Admin, Patients, Stakeholders")
print("="*80)
