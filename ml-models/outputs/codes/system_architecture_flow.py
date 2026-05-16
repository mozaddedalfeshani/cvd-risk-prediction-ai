import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# ==================== DETAILED ARCHITECTURE DIAGRAM ====================
fig = plt.figure(figsize=(18, 12))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors
color_frontend = '#E3F2FD'     # Light blue
color_backend = '#FFF3E0'      # Light orange
color_model = '#F3E5F5'        # Light purple
color_database = '#E8F5E9'     # Light green
color_arrow = '#1976D2'        # Dark blue

# ==================== TITLE ====================
title_box = FancyBboxPatch((1, 12.8), 16, 1, boxstyle="round,pad=0.15",
                           edgecolor='navy', facecolor='#B3E5FC', linewidth=3)
ax.add_patch(title_box)
ax.text(9, 13.3, '🏥 CVD Risk Prediction System - Complete Data Flow', 
        fontsize=22, fontweight='bold', ha='center', color='navy')

# ==================== SECTION 1: FRONTEND (LEFT) ====================
y_start = 11.5

# Frontend container
frontend_box = FancyBboxPatch((0.3, 2), 4.5, 9.3, boxstyle="round,pad=0.1",
                              edgecolor='#0052CC', facecolor=color_frontend, linewidth=2.5)
ax.add_patch(frontend_box)

ax.text(2.55, 11, '📱 FRONTEND', fontsize=14, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#64B5F6', edgecolor='navy', linewidth=2))
ax.text(2.55, 10.4, '(Web Application - React/Next.js)', fontsize=9, ha='center', style='italic')

# Frontend components
frontend_items = [
    ('🎨 User Interface\n(Input Form)', 9.8),
    ('📊 Display Results\n(Dashboard)', 8.8),
    ('🔄 Real-time Status\n(Progress)', 7.8),
    ('💾 Patient History\n(Local Storage)', 6.8),
    ('⚙️ Settings\n(User Preferences)', 5.8),
    ('📱 Responsive Design\n(Mobile/Desktop)', 4.8),
]

for item, y in frontend_items:
    rect = FancyBboxPatch((0.6, y-0.45), 3.9, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='#0052CC', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(2.55, y, item, fontsize=9, ha='center', va='center', fontweight='bold')

# Frontend functions
ax.text(2.55, 3.8, '✨ Frontend Functions:', fontsize=10, fontweight='bold', ha='center')
functions_fe = [
    '• Collects 17 patient inputs',
    '• Validates data format',
    '• Shows prediction results',
    '• Communicates with API'
]
for i, func in enumerate(functions_fe):
    ax.text(2.55, 3.2-i*0.4, func, fontsize=8, ha='center', style='italic')

# ==================== SECTION 2: BACKEND (CENTER) ====================
y_start = 11.5

# Backend container
backend_box = FancyBboxPatch((5.5, 2), 7, 9.3, boxstyle="round,pad=0.1",
                             edgecolor='#E65100', facecolor=color_backend, linewidth=2.5)
ax.add_patch(backend_box)

ax.text(9, 11, '⚙️ BACKEND API SERVER', fontsize=14, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFB74D', edgecolor='#E65100', linewidth=2))
ax.text(9, 10.4, '(Flask/FastAPI - Python)', fontsize=9, ha='center', style='italic')

# Backend components - Top section
ax.text(9, 9.8, '📥 Input Processing', fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFECB3', alpha=0.8))

input_items = [
    'Receive data from Frontend',
    'Validate input format',
    'Check for missing values'
]
for i, item in enumerate(input_items):
    rect = FancyBboxPatch((5.8, 9.1-i*0.5), 6.4, 0.4, boxstyle="round,pad=0.03",
                          edgecolor='#FF6F00', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(9, 8.95-i*0.5, item, fontsize=8, ha='center', va='center')

# Processing section
ax.text(9, 7.2, '🔧 Data Processing', fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFECB3', alpha=0.8))

process_items = [
    'Normalize/Scale features',
    'Encode categorical data',
    'Apply transformations'
]
for i, item in enumerate(process_items):
    rect = FancyBboxPatch((5.8, 6.5-i*0.5), 6.4, 0.4, boxstyle="round,pad=0.03",
                          edgecolor='#FF6F00', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(9, 6.35-i*0.5, item, fontsize=8, ha='center', va='center')

# Model calling section
ax.text(9, 4.7, '🤖 Model Integration', fontsize=11, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFECB3', alpha=0.8))

model_items = [
    'Load trained model (SVM)',
    'Pass processed data',
    'Get prediction result'
]
for i, item in enumerate(model_items):
    rect = FancyBboxPatch((5.8, 4-i*0.5), 6.4, 0.4, boxstyle="round,pad=0.03",
                          edgecolor='#FF6F00', facecolor='white', linewidth=1)
    ax.add_patch(rect)
    ax.text(9, 3.85-i*0.5, item, fontsize=8, ha='center', va='center')

# Response section
ax.text(9, 2.2, '📤 Response Generation', fontsize=9, fontweight='bold', ha='center')
ax.text(9, 1.8, 'Format result as JSON\nReturn to Frontend\nLog transaction', 
        fontsize=8, ha='center', style='italic')

# ==================== SECTION 3: ML MODEL (RIGHT) ====================
y_start = 11.5

# Model container
model_box = FancyBboxPatch((13, 2), 4.7, 9.3, boxstyle="round,pad=0.1",
                           edgecolor='#6A1B9A', facecolor=color_model, linewidth=2.5)
ax.add_patch(model_box)

ax.text(15.35, 11, '🧠 ML MODEL PIPELINE', fontsize=14, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#CE93D8', edgecolor='#6A1B9A', linewidth=2))
ax.text(15.35, 10.4, '(Trained SVM Model)', fontsize=9, ha='center', style='italic')

# Model components
model_items = [
    ('📦 Model Artifact\n(SVM.pkl)', 9.8),
    ('🔢 17 Features\nProcessed', 8.8),
    ('⚡ Prediction\nEngine', 7.8),
    ('📊 Output\n0 or 1', 6.8),
    ('📈 Confidence\nScore', 5.8),
    ('💾 Saved\nModel File', 4.8),
]

for item, y in model_items:
    rect = FancyBboxPatch((13.2, y-0.45), 4.3, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='#6A1B9A', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(15.35, y, item, fontsize=9, ha='center', va='center', fontweight='bold')

# Model stats
ax.text(15.35, 3.8, '📊 Model Stats:', fontsize=10, fontweight='bold', ha='center')
stats = [
    'Accuracy: 79.08%',
    'Precision: 78.87%',
    'Recall: 76.71%',
]
for i, stat in enumerate(stats):
    ax.text(15.35, 3.2-i*0.35, stat, fontsize=8, ha='center', family='monospace', fontweight='bold')

# ==================== DATA FLOW ARROWS ====================

# 1. Frontend to Backend (User Input)
arrow1 = FancyArrowPatch((4.8, 7.5), (5.5, 7.5),
                        arrowstyle='<->', mutation_scale=35, linewidth=3, color='#1976D2')
ax.add_patch(arrow1)
ax.text(5.15, 7.9, '📨 HTTP/JSON\nRequest/Response', fontsize=8, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

# 2. Backend to Model (Process & Predict)
arrow2 = FancyArrowPatch((12.5, 7.5), (13, 7.5),
                        arrowstyle='<->', mutation_scale=35, linewidth=3, color='#6A1B9A')
ax.add_patch(arrow2)
ax.text(12.75, 7.9, '🔄 Data\nPassing', fontsize=8, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.2', facecolor='#F3E5F5', alpha=0.8))

# ==================== DATA FLOW DETAILS ====================

# Step-by-step flow
y_flow = 0.5
ax.text(9, y_flow, '📊 COMPLETE DATA JOURNEY:', fontsize=11, fontweight='bold', ha='center')

flow_steps = [
    '1️⃣ Patient fills 17 medical inputs (Age, Weight, BMI, etc.) → Frontend',
    '2️⃣ Clicks "Predict Risk" button → Request sent to Backend API',
    '3️⃣ Backend validates & processes the data → Scales/Encodes values',
    '4️⃣ Backend loads trained SVM model → Passes processed data',
    '5️⃣ Model generates prediction: HIGH RISK (1) or NON-HIGH (0)',
    '6️⃣ Backend formats result + confidence → Sends back to Frontend',
    '7️⃣ Frontend displays result to doctor/patient → Ready for action'
]

for i, step in enumerate(flow_steps):
    ax.text(9, y_flow-0.35-(i*0.3), step, fontsize=8, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF9C4', alpha=0.7, edgecolor='gray'))

plt.tight_layout()
plt.savefig("outputs/System_Architecture_Complete_Flow.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: outputs/System_Architecture_Complete_Flow.png")
plt.close()

# ==================== SIMPLIFIED ARCHITECTURE ====================
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
title_box = FancyBboxPatch((1, 10.8), 14, 1.2, boxstyle="round,pad=0.15",
                           edgecolor='darkblue', facecolor='#B3E5FC', linewidth=3)
ax.add_patch(title_box)
ax.text(8, 11.4, '🏥 Three-Tier System Architecture - Easy View', 
        fontsize=20, fontweight='bold', ha='center')

# Three main boxes
# Frontend
frontend = FancyBboxPatch((0.5, 5), 4.5, 4.5, boxstyle="round,pad=0.15",
                          edgecolor='#0052CC', facecolor='#E3F2FD', linewidth=2.5)
ax.add_patch(frontend)
ax.text(2.75, 9, '📱 FRONTEND', fontsize=14, fontweight='bold', ha='center', color='#0052CC')
ax.text(2.75, 8.3, '(What Users See)', fontsize=10, ha='center', style='italic', fontweight='bold')

frontend_text = """
✓ Input Form
  Enter 17 medical values
  
✓ Display Results
  HIGH RISK or NON-HIGH
  
✓ User-Friendly
  Click buttons, see answers
"""
ax.text(2.75, 6.5, frontend_text, fontsize=9, ha='center', va='center',
       family='monospace', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

# Backend
backend = FancyBboxPatch((5.75, 5), 4.5, 4.5, boxstyle="round,pad=0.15",
                         edgecolor='#E65100', facecolor='#FFF3E0', linewidth=2.5)
ax.add_patch(backend)
ax.text(8, 9, '⚙️ BACKEND SERVER', fontsize=14, fontweight='bold', ha='center', color='#E65100')
ax.text(8, 8.3, '(Brain of System)', fontsize=10, ha='center', style='italic', fontweight='bold')

backend_text = """
✓ Receives Data
  From frontend (HTTP request)
  
✓ Processes Data
  Validates, cleans, prepares
  
✓ Calls Model
  Sends to ML engine
  
✓ Sends Response
  Back to frontend (JSON)
"""
ax.text(8, 6.2, backend_text, fontsize=9, ha='center', va='center',
       family='monospace', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

# Model
model = FancyBboxPatch((11, 5), 4.5, 4.5, boxstyle="round,pad=0.15",
                       edgecolor='#6A1B9A', facecolor='#F3E5F5', linewidth=2.5)
ax.add_patch(model)
ax.text(13.25, 9, '🧠 ML MODEL', fontsize=14, fontweight='bold', ha='center', color='#6A1B9A')
ax.text(13.25, 8.3, '(AI Decision Maker)', fontsize=10, ha='center', style='italic', fontweight='bold')

model_text = """
✓ Pre-trained SVM
  Learned from 1,529 patients
  
✓ Takes 17 Features
  Medical measurements
  
✓ Makes Prediction
  HIGH (1) or NON-HIGH (0)
  
✓ Returns Result
  With confidence score
"""
ax.text(13.25, 6.2, model_text, fontsize=9, ha='center', va='center',
       family='monospace', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

# Communication arrows
arrow_right1 = FancyArrowPatch((5.2, 7.5), (5.75, 7.5),
                             arrowstyle='->', mutation_scale=30, linewidth=3, color='#1976D2')
ax.add_patch(arrow_right1)
ax.text(5.45, 8, 'Request', fontsize=9, ha='center', fontweight='bold', color='blue')

arrow_left1 = FancyArrowPatch((5.75, 7), (5.2, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='#D32F2F')
ax.add_patch(arrow_left1)
ax.text(5.45, 6.5, 'Response', fontsize=9, ha='center', fontweight='bold', color='red')

arrow_right2 = FancyArrowPatch((9.8, 7.5), (11, 7.5),
                             arrowstyle='->', mutation_scale=30, linewidth=3, color='#6A1B9A')
ax.add_patch(arrow_right2)
ax.text(10.4, 8, 'Data', fontsize=9, ha='center', fontweight='bold', color='purple')

arrow_left2 = FancyArrowPatch((11, 7), (9.8, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3, color='#00897B')
ax.add_patch(arrow_left2)
ax.text(10.4, 6.5, 'Prediction', fontsize=9, ha='center', fontweight='bold', color='teal')

# Bottom explanation
explanation_box = FancyBboxPatch((0.5, 0.3), 15, 4, boxstyle="round,pad=0.15",
                               edgecolor='darkgreen', facecolor='#FFFDE7', linewidth=2)
ax.add_patch(explanation_box)

ax.text(8, 4, '🔄 HOW THEY WORK TOGETHER', fontsize=13, fontweight='bold', ha='center', color='darkgreen')

steps_text = """
1. DOCTOR ENTERS DATA 👨‍⚕️
   Doctor fills patient info in the Frontend form (Age, Weight, BMI, Blood Pressure, etc.)
   
2. DATA TRAVELS TO BACKEND 📤
   Frontend sends encrypted data to Backend server via secure HTTPS connection
   
3. BACKEND PROCESSES & VALIDATES ⚙️
   Backend checks if data is correct, fills missing values, scales numbers (0-1 range)
   
4. BACKEND CALLS MODEL 🤖
   Backend loads the trained SVM model from storage file (model.pkl)
   
5. MODEL MAKES PREDICTION 🎯
   AI analyzes 17 features using patterns learned from 1,529 past patients
   
6. RESULT RETURNS TO FRONTEND 📥
   Model prediction (HIGH RISK or NON-HIGH) sent back to Frontend
   
7. DOCTOR SEES RESULT 👀
   Result displayed on screen with confidence score and recommendation
   
8. DOCTOR TAKES ACTION 💊
   Doctor reviews result and decides next steps for patient care
"""

ax.text(8, 2, steps_text, fontsize=8.5, ha='center', va='center',
       family='monospace', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

plt.tight_layout()
plt.savefig("outputs/System_Architecture_Simplified.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: outputs/System_Architecture_Simplified.png")
plt.close()

# ==================== DATA FLOW SEQUENCE DIAGRAM ====================
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 14)
ax.axis('off')

# Title
ax.text(8, 13.2, '📊 Complete Data Flow - Step by Step', 
        fontsize=20, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE082', edgecolor='#F57C00', linewidth=2))

# Three vertical lines for entities
y_top = 12
y_bottom = 1

# Column positions
col_x = [2, 8, 14]
col_names = ['👨‍⚕️ DOCTOR', '⚙️ BACKEND', '🧠 MODEL']
col_colors = ['#E3F2FD', '#FFF3E0', '#F3E5F5']

# Draw vertical lines
for i, (x, name, color) in enumerate(zip(col_x, col_names, col_colors)):
    # Header box
    header = FancyBboxPatch((x-1.2, y_top-0.4), 2.4, 0.6, boxstyle="round,pad=0.05",
                            edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(header)
    ax.text(x, y_top-0.1, name, fontsize=11, fontweight='bold', ha='center')
    
    # Vertical dotted line
    ax.plot([x, x], [y_top-0.5, y_bottom], 'k--', linewidth=1.5, alpha=0.5)

# Sequence steps
steps_sequence = [
    (2, 11, '1. Input Data\n(17 values)', 'Enter Age, Weight,\nBMI, etc.', '#E3F2FD'),
    (2, 10.2, '1a. Click "Predict"', 'Submit form', '#E3F2FD'),
    (8, 9.5, '2. Receive Request', 'HTTP POST\nwith data', '#FFF3E0'),
    (8, 8.8, '3. Validate Data', 'Check format,\nfill missing', '#FFF3E0'),
    (8, 8, '4. Process Data', 'Scale features,\nencode categories', '#FFF3E0'),
    (14, 7.3, '5. Load Model', 'Read SVM.pkl\nfrom disk', '#F3E5F5'),
    (14, 6.5, '6. Process Input', '17 features\nprepared', '#F3E5F5'),
    (14, 5.7, '7. Make Prediction', 'SVM outputs\n0 or 1', '#F3E5F5'),
    (8, 5, '8. Receive Result', 'Model sends\nprediction', '#FFF3E0'),
    (8, 4.2, '9. Format Response', 'Create JSON:\nresult + confidence', '#FFF3E0'),
    (2, 3.5, '10. Display Result', 'HIGH RISK 🔴\nor NON-HIGH 🟢', '#E3F2FD'),
    (2, 2.7, '11. Doctor Action', 'Review & decide\nnext steps', '#E3F2FD'),
]

for x, y, title, desc, color in steps_sequence:
    rect = FancyBboxPatch((x-1, y-0.35), 2, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y-0.05, title, fontsize=8, ha='center', va='center', fontweight='bold')
    ax.text(x, y-0.8, desc, fontsize=7, ha='center', va='top', style='italic')

# Horizontal arrows showing communication
communications = [
    (2.8, 10, 8, 9.7, '📨 Send\nRequest'),
    (7.2, 5.2, 14, 5.8, '📤 Prediction'),
    (13.2, 3.8, 2, 3.2, '📬 Response'),
]

for x1, y1, x2, y2, label in communications:
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=25, linewidth=2.5, color='#D32F2F')
    ax.add_patch(arrow)
    mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
    ax.text(mid_x, mid_y+0.2, label, fontsize=8, ha='center', fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', alpha=0.8))

# Bottom summary
summary_box = FancyBboxPatch((0.5, 0.1), 15, 1, boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(summary_box)
ax.text(8, 0.7, '✅ Result: Patient gets prediction in seconds  |  🔒 All data encrypted  |  📊 Logged for audit trail', 
        fontsize=9, ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/System_Data_Flow_Sequence.png", dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Saved: outputs/System_Data_Flow_Sequence.png")
plt.close()

print("\n" + "="*100)
print("COMPLETE SYSTEM ARCHITECTURE DIAGRAMS CREATED")
print("="*100)
print("\n✅ Three comprehensive diagrams generated:")
print("   1. System_Architecture_Complete_Flow.png - Detailed backend/frontend/model diagram")
print("   2. System_Architecture_Simplified.png - Easy-to-understand three-tier architecture")
print("   3. System_Data_Flow_Sequence.png - Step-by-step sequence diagram")
print("\n📌 Coverage:")
print("   ✓ Frontend: User interface, input forms, result display")
print("   ✓ Backend: API server, data validation, model calling")
print("   ✓ Model: SVM prediction engine, confidence scoring")
print("   ✓ Data Flow: Request/Response cycle")
print("   ✓ Communication: HTTP/JSON, data passing, results")
print("\n👥 Perfect for: Technical teams, stakeholders, presentations, documentation")
print("="*100)
