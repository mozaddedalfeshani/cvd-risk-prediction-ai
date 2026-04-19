import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / 'models'
FIGURES_DIR = ROOT_DIR / 'evaluation' / 'reports' / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("Loading trained model...")
model_data = joblib.load(MODELS_DIR / 'xgboost_model.pkl')
history = model_data['cv_history']

print("Generating accuracy and loss graphs...")

# Create figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ============================================
# Accuracy Graph
# ============================================
axes[0].plot(history['fold'], history['train_accuracy'], 
             marker='o', linewidth=2.5, markersize=8, 
             label='Training Accuracy', color='#2ecc71')
axes[0].plot(history['fold'], history['val_accuracy'], 
             marker='s', linewidth=2.5, markersize=8,
             label='Validation Accuracy', color='#e74c3c')

# Add mean lines
train_mean = np.mean(history['train_accuracy'])
val_mean = np.mean(history['val_accuracy'])
axes[0].axhline(train_mean, color='#2ecc71', linestyle='--', 
                linewidth=1.5, alpha=0.5, label=f'Train Mean: {train_mean:.4f}')
axes[0].axhline(val_mean, color='#e74c3c', linestyle='--', 
                linewidth=1.5, alpha=0.5, label=f'Val Mean: {val_mean:.4f}')

axes[0].set_xlabel('Fold', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Accuracy', fontsize=14, fontweight='bold')
axes[0].set_title('Model Accuracy Across 5-Fold Cross-Validation', 
                  fontsize=16, fontweight='bold', pad=20)
axes[0].legend(fontsize=11, loc='lower right')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_ylim([0.8, 1.02])
axes[0].set_xticks(history['fold'])

# Add value annotations
for i, (f, train_acc, val_acc) in enumerate(zip(history['fold'], 
                                                 history['train_accuracy'], 
                                                 history['val_accuracy'])):
    axes[0].annotate(f'{train_acc:.3f}', 
                     (f, train_acc), 
                     textcoords="offset points", 
                     xytext=(0, 8), 
                     ha='center', 
                     fontsize=9, 
                     color='#27ae60')
    axes[0].annotate(f'{val_acc:.3f}', 
                     (f, val_acc), 
                     textcoords="offset points", 
                     xytext=(0, -15), 
                     ha='center', 
                     fontsize=9, 
                     color='#c0392b')

# ============================================
# Loss Graph
# ============================================
axes[1].plot(history['fold'], history['train_loss'], 
             marker='o', linewidth=2.5, markersize=8,
             label='Training Loss', color='#3498db')
axes[1].plot(history['fold'], history['val_loss'], 
             marker='s', linewidth=2.5, markersize=8,
             label='Validation Loss', color='#e67e22')

# Add mean lines
train_loss_mean = np.mean(history['train_loss'])
val_loss_mean = np.mean(history['val_loss'])
axes[1].axhline(train_loss_mean, color='#3498db', linestyle='--', 
                linewidth=1.5, alpha=0.5, label=f'Train Mean: {train_loss_mean:.4f}')
axes[1].axhline(val_loss_mean, color='#e67e22', linestyle='--', 
                linewidth=1.5, alpha=0.5, label=f'Val Mean: {val_loss_mean:.4f}')

axes[1].set_xlabel('Fold', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Log Loss', fontsize=14, fontweight='bold')
axes[1].set_title('Model Loss Across 5-Fold Cross-Validation', 
                  fontsize=16, fontweight='bold', pad=20)
axes[1].legend(fontsize=11, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xticks(history['fold'])

# Add value annotations
for i, (f, train_loss, val_loss) in enumerate(zip(history['fold'], 
                                                   history['train_loss'], 
                                                   history['val_loss'])):
    axes[1].annotate(f'{train_loss:.3f}', 
                     (f, train_loss), 
                     textcoords="offset points", 
                     xytext=(0, -15), 
                     ha='center', 
                     fontsize=9, 
                     color='#2980b9')
    axes[1].annotate(f'{val_loss:.3f}', 
                     (f, val_loss), 
                     textcoords="offset points", 
                     xytext=(0, 8), 
                     ha='center', 
                     fontsize=9, 
                     color='#d35400')

# Add overall statistics text box
stats_text = f"Final Test Performance:\nAccuracy: {model_data['test_accuracy']*100:.2f}%\nF1 Score: {model_data['test_f1']:.4f}\nLog Loss: {model_data['test_loss']:.4f}"
fig.text(0.5, 0.02, stats_text, ha='center', fontsize=12, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout(rect=[0, 0.06, 1, 1])
combined_output = FIGURES_DIR / 'accuracy_loss_graph.png'
plt.savefig(combined_output, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {combined_output}")
plt.close()

# ============================================
# Create separate high-quality accuracy plot
# ============================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(history['fold'], history['train_accuracy'], 
        marker='o', linewidth=3, markersize=10, 
        label='Training Accuracy', color='#2ecc71', alpha=0.8)
ax.plot(history['fold'], history['val_accuracy'], 
        marker='s', linewidth=3, markersize=10,
        label='Validation Accuracy', color='#e74c3c', alpha=0.8)

# Fill between
ax.fill_between(history['fold'], history['train_accuracy'], 
                history['val_accuracy'], alpha=0.2, color='gray')

ax.axhline(val_mean, color='#e74c3c', linestyle='--', 
           linewidth=2, alpha=0.6, label=f'Average Val Accuracy: {val_mean:.4f}')

ax.set_xlabel('Cross-Validation Fold', fontsize=15, fontweight='bold')
ax.set_ylabel('Accuracy Score', fontsize=15, fontweight='bold')
ax.set_title('XGBoost Model Accuracy - Training vs Validation', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_ylim([0.85, 1.02])
ax.set_xticks(history['fold'])

# Add test accuracy annotation
test_acc = model_data['test_accuracy']
ax.text(0.02, 0.98, f'Final Test Accuracy: {test_acc*100:.2f}%', 
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

plt.tight_layout()
accuracy_output = FIGURES_DIR / 'model_accuracy.png'
plt.savefig(accuracy_output, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {accuracy_output}")
plt.close()

# ============================================
# Create separate high-quality loss plot
# ============================================
fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(history['fold'], history['train_loss'], 
        marker='o', linewidth=3, markersize=10,
        label='Training Loss', color='#3498db', alpha=0.8)
ax.plot(history['fold'], history['val_loss'], 
        marker='s', linewidth=3, markersize=10,
        label='Validation Loss', color='#e67e22', alpha=0.8)

# Fill between
ax.fill_between(history['fold'], history['train_loss'], 
                history['val_loss'], alpha=0.2, color='gray')

ax.axhline(val_loss_mean, color='#e67e22', linestyle='--', 
           linewidth=2, alpha=0.6, label=f'Average Val Loss: {val_loss_mean:.4f}')

ax.set_xlabel('Cross-Validation Fold', fontsize=15, fontweight='bold')
ax.set_ylabel('Log Loss', fontsize=15, fontweight='bold')
ax.set_title('XGBoost Model Loss - Training vs Validation', 
             fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.4, linestyle='--')
ax.set_xticks(history['fold'])

# Add test loss annotation
test_loss = model_data['test_loss']
ax.text(0.02, 0.98, f'Final Test Loss: {test_loss:.4f}', 
        transform=ax.transAxes, fontsize=14, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
loss_output = FIGURES_DIR / 'model_loss.png'
plt.savefig(loss_output, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {loss_output}")
plt.close()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Training Accuracy: {train_mean:.4f} ± {np.std(history['train_accuracy']):.4f}")
print(f"Validation Accuracy: {val_mean:.4f} ± {np.std(history['val_accuracy']):.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"\nTraining Loss: {train_loss_mean:.4f} ± {np.std(history['train_loss']):.4f}")
print(f"Validation Loss: {val_loss_mean:.4f} ± {np.std(history['val_loss']):.4f}")
print(f"Test Loss: {test_loss:.4f}")
print("\n✅ All graphs generated successfully!")
print(f"   - {combined_output} (combined)")
print(f"   - {accuracy_output} (detailed)")
print(f"   - {loss_output} (detailed)")
