import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import json
import pandas as pd
import os

# Paths
PROJECT_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project"
MODELS_DIR = PROJECT_DIR + "/models"
TEST_DIR = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
RESULTS_TESTING_DIR = PROJECT_DIR + "/results_testing"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
os.makedirs(RESULTS_TESTING_DIR, exist_ok=True)

print("=" * 70)
print(" COMPREHENSIVE EVALUATION: SOLO MODELS + FUSION")
print("=" * 70)

# Load fusion config
with open(MODELS_DIR + "/fusion_config.json") as f:
    config = json.load(f)

w_b0 = config["models"]["efficientnetb0"]["weight"]
w_b2 = config["models"]["efficientnetb2"]["weight"]
w_resnet = config["models"]["resnet50"]["weight"]

# Load models
print("\nLoading models...")
model_b0 = tf.keras.models.load_model(MODELS_DIR + "/best_acne_model.keras")
model_b2 = tf.keras.models.load_model(MODELS_DIR + "/efficientnetb2_acne.keras")
model_resnet = tf.keras.models.load_model(MODELS_DIR + "/resnet50_acne.keras")

# Load test datasets (handle different sizes)
AUTOTUNE = tf.data.AUTOTUNE
test_ds_224 = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(224,224),
    batch_size=32, shuffle=False
).prefetch(AUTOTUNE)

test_ds_260 = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(260,260),
    batch_size=32, shuffle=False
).prefetch(AUTOTUNE)

def get_probs(model, ds):
    probs, labels = [], []
    for imgs, lbls in ds:
        p = model.predict(imgs, verbose=0)
        probs.append(p)
        labels.append(lbls.numpy())
    return np.vstack(probs), np.vstack(labels)

# Predictions
print("\nRunning predictions...")
probs_b0, labels = get_probs(model_b0, test_ds_224)
probs_resnet, _ = get_probs(model_resnet, test_ds_224)
probs_b2, _ = get_probs(model_b2, test_ds_260)

# Fusion
probs_fusion = (w_b0 * probs_b0) + (w_b2 * probs_b2) + (w_resnet * probs_resnet)
y_true = np.argmax(labels, axis=1)

probs_dict = {
    "B0": probs_b0, "B2": probs_b2, "ResNet": probs_resnet, "Fusion": probs_fusion
}

# Metrics function
def compute_metrics(probs, y_true):
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    _, _, f1_micro, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    
    # Specificity macro (avg recall for negative classes)
    cm = confusion_matrix(y_true, y_pred)
  specificity = []
    for i in range(len(CLASS_NAMES)):
        tn = cm[i, i]
        fp = cm[:, i].sum() - tn
        specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 1.0)
    spec_macro = np.mean(specificity)
    
    return {
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "specificity_macro": spec_macro
    }

# Compute metrics
metrics = {}
for name, probs in probs_dict.items():
    metrics[name] = compute_metrics(probs, y_true)
    print(f"{name} Accuracy: {metrics[name]['accuracy']:.4f}")

# 1. Per-model confusion matrices & heatmaps
fig_cm_all, axes_all = plt.subplots(2, 2, figsize=(16, 14))
fig_cm_all.suptitle("All Confusion Matrices", fontsize=16, fontweight="bold")

for idx, (name, probs) in enumerate(probs_dict.items()):
    row, col = idx // 2, idx % 2
    ax = axes_all[row, col]
    y_pred = np.argmax(probs, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"{name} (Acc: {metrics[name]['accuracy']:.3f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    
    # Individual full plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=axes[0])
    axes[0].set_title(f"{name} Confusion Matrix (Counts)")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1], vmin=0, vmax=1)
    axes[1].set_title(f"{name} Normalized")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_TESTING_DIR}/{name.lower()}_cm.png", dpi=150, bbox_inches="tight")
    plt.close()

plt.tight_layout()
plt.savefig(f"{RESULTS_TESTING_DIR}/all_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# 2. Test samples images
plt.figure(figsize=(15, 10))
for imgs, lbls in test_ds_224.take(1):
    for i in range(12):
        plt.subplot(3, 4, i+1)
        plt.imshow(imgs[i].numpy().astype("uint8"))
        pred_class = CLASS_NAMES[np.argmax(lbls[i])]
        plt.title(pred_class, fontsize=10)
        plt.axis("off")
plt.suptitle("Test Set Sample Images", fontsize=14)
plt.tight_layout()
plt.savefig(f"{RESULTS_TESTING_DIR}/test_samples.png", dpi=150, bbox_inches="tight")
plt.close()

# 3. Metrics table
df_metrics = pd.DataFrame(metrics).T.round(4)
plt.figure(figsize=(12, 6))
plt.axis("tight")
plt.axis("off")
table = plt.table(cellText=df_metrics.values, colLabels=df_metrics.columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
plt.title("Full Metrics Comparison", fontsize=14, fontweight="bold")
plt.savefig(f"{RESULTS_TESTING_DIR}/metrics_table.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close()

# 4. Grouped bar plots for metrics
metrics_6 = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro", "specificity_macro"]
solo_names = ["B0", "B2", "ResNet"]
solo_avg = np.mean([metrics[n][m] for n in solo_names for m in metrics_6], axis=0)  # Avg solo per metric
fusion_vals = [metrics["Fusion"][m] for m in metrics_6]
solo_vals = [np.mean([metrics[n][m] for n in solo_names]) for m in metrics_6]

x = np.arange(len(metrics_6))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
bars1 = ax.bar(x - width/2, solo_vals, width, label="Solo Avg", color="#3498DB")
bars2 = ax.bar(x + width/2, fusion_vals, width, label="Fusion", color="#9B59B6")
ax.set_xlabel("Metrics")
ax.set_ylabel("Score")
ax.set_title("Solo vs Fusion Metrics Comparison")
ax.set_xticks(x)
ax.set_xticklabels(metrics_6)
ax.legend()
ax.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{RESULTS_TESTING_DIR}/metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. Improvement deltas
deltas = [fusion_vals[i] - solo_vals[i] for i in range(len(metrics_6))]
colors_delta = ["green" if d > 0 else "red" for d in deltas]
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(metrics_6, deltas, color=colors_delta, alpha=0.7)
ax.axhline(0, color="black", linestyle="--")
ax.set_ylabel("Delta (Fusion - Solo Avg)")
ax.set_title("Fusion Improvements (6 Parameters)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{RESULTS_TESTING_DIR}/improvement_deltas.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. Save JSON summary
summary = {
    "metrics": metrics,
    "deltas_6_params": dict(zip(metrics_6, deltas)),
    "six_params": {
        "accuracy": metrics["Fusion"]["accuracy"],
        "precision_macro": metrics["Fusion"]["precision_macro"],
        "recall_macro": metrics["Fusion"]["recall_macro"],
        "f1_macro": metrics["Fusion"]["f1_macro"],
        "f1_micro": metrics["Fusion"]["f1_micro"],
        "specificity_macro": metrics["Fusion"]["specificity_macro"]
    }
}
with open(f"{RESULTS_TESTING_DIR}/metrics_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll images saved to results_testing/:")
print("- *_cm.png (per model)")
print("- test_samples.png")
print("- metrics_table.png")
print("- metrics_comparison.png")
print("- improvement_deltas.png")
print("- all_confusion_matrices.png")
print("- metrics_summary.json")
print("\nCOMPLETE!")

