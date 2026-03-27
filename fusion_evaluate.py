import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json

TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
AUTOTUNE    = tf.data.AUTOTUNE
MODELS_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/models"
RESULTS_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/results"

print("="*55)
print("  FUSION MODEL — FULL EVALUATION")
print("="*55)

# Load fusion config
with open(MODELS_DIR + "/fusion_config.json") as f:
    config = json.load(f)

w_b0     = config["models"]["efficientnetb0"]["weight"]
w_b2     = config["models"]["efficientnetb2"]["weight"]
w_resnet = config["models"]["resnet50"]["weight"]

# Load models
print("\nLoading all 3 models...")
model_b0     = tf.keras.models.load_model(MODELS_DIR + "/best_acne_model.keras")
model_b2     = tf.keras.models.load_model(MODELS_DIR + "/efficientnetb2_acne.keras")
model_resnet = tf.keras.models.load_model(MODELS_DIR + "/resnet50_acne.keras")

# Load test data
test_224 = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(224,224),
    batch_size=32, shuffle=False
).prefetch(AUTOTUNE)

test_260 = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=(260,260),
    batch_size=32, shuffle=False
).prefetch(AUTOTUNE)

# Collect predictions
def get_probs(model, ds):
    probs, labels = [], []
    for imgs, lbls in ds:
        probs.append(model.predict(imgs, verbose=0))
        labels.append(lbls.numpy())
    return np.vstack(probs), np.vstack(labels)

print("Running predictions...")
probs_b0,     labels = get_probs(model_b0,     test_224)
probs_resnet, _      = get_probs(model_resnet, test_224)
probs_b2,     _      = get_probs(model_b2,     test_260)

# Weighted fusion
probs_fusion = (w_b0 * probs_b0) + (w_b2 * probs_b2) + (w_resnet * probs_resnet)

y_true = np.argmax(labels, axis=1)
y_pred = np.argmax(probs_fusion, axis=1)

fusion_acc = np.mean(y_pred == y_true)
print(f"\nFusion Model Test Accuracy: {fusion_acc:.4f} ({fusion_acc*100:.2f}%)")

print("\nCLASSIFICATION REPORT — FUSION MODEL:")
print("-"*60)
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# Confusion matrix
cm      = confusion_matrix(y_true, y_pred)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[0], linewidths=0.5, annot_kws={"size":12})
axes[0].set_title(
    f"Fusion Model Confusion Matrix\\nTest Accuracy: {fusion_acc*100:.2f}%",
    fontsize=13, fontweight="bold"
)
axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
axes[0].tick_params(axis="x", rotation=45)

sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            ax=axes[1], vmin=0, vmax=1, linewidths=0.5,
            annot_kws={"size":12})
axes[1].set_title("Fusion Model Confusion Matrix (Normalized)",
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle(
    f"Multi-Fusion Model Evaluation  |  Accuracy: {fusion_acc*100:.2f}%",
    fontsize=14, fontweight="bold"
)
plt.tight_layout()
plt.savefig(RESULTS_DIR + "/07_fusion_confusion_matrix.png",
            dpi=150, bbox_inches="tight")
plt.show()
print("Saved: results/07_fusion_confusion_matrix.png")
print("\nSTEP 5 COMPLETE — Fusion evaluation done!")

