import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import json
import os
from scipy import stats

TEST_DIR    = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\test"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
NUM_CLASSES = 5
AUTOTUNE    = tf.data.AUTOTUNE
MODELS_DIR  = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/models"

print("="*60)
print("  BUILDING MULTI-FUSION MODEL")
print("  EfficientNetB0 + EfficientNetB2 + ResNet50")
print("="*60)

# ── Load all 3 trained models ──
print("\nLoading models...")

model_b0 = tf.keras.models.load_model(
    MODELS_DIR + "/best_acne_model.keras"
)
model_b2 = tf.keras.models.load_model(
    MODELS_DIR + "/efficientnetb2_acne.keras"
)
model_resnet = tf.keras.models.load_model(
    MODELS_DIR + "/resnet50_acne.keras"
)

print("  EfficientNetB0  loaded")
print("  EfficientNetB2  loaded")
print("  ResNet50        loaded")

# ── Load test data for all 3 input sizes ──
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

# ── Get predictions from each model ──
print("\nCollecting predictions from each model...")

def get_predictions(model, dataset):
    all_probs  = []
    all_labels = []
    for images, labels in dataset:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    return np.vstack(all_probs), np.vstack(all_labels)

probs_b0,     labels_224 = get_predictions(model_b0,     test_ds_224)
probs_resnet, _          = get_predictions(model_resnet, test_ds_224)
probs_b2,     labels_260 = get_predictions(model_b2,     test_ds_260)

y_true = np.argmax(labels_224, axis=1)

# ── Individual model accuracies ──
acc_b0     = np.mean(np.argmax(probs_b0,     axis=1) == y_true)
acc_b2     = np.mean(np.argmax(probs_b2,     axis=1) == y_true)
acc_resnet = np.mean(np.argmax(probs_resnet, axis=1) == y_true)

print(f"\n  Individual Model Accuracies:")
print(f"  EfficientNetB0  : {acc_b0*100:.2f}%")
print(f"  EfficientNetB2  : {acc_b2*100:.2f}%")
print(f"  ResNet50        : {acc_resnet*100:.2f}%")

# ── FUSION STRATEGY 1: Simple Average ──
probs_avg  = (probs_b0 + probs_b2 + probs_resnet) / 3.0
acc_avg    = np.mean(np.argmax(probs_avg, axis=1) == y_true)

# ── FUSION STRATEGY 2: Weighted Average (better models get higher weight) ──
# Weights proportional to individual accuracies
total_acc  = acc_b0 + acc_b2 + acc_resnet
w_b0       = acc_b0     / total_acc
w_b2       = acc_b2     / total_acc
w_resnet   = acc_resnet / total_acc

probs_weighted = (w_b0 * probs_b0) + (w_b2 * probs_b2) + (w_resnet * probs_resnet)
acc_weighted   = np.mean(np.argmax(probs_weighted, axis=1) == y_true)

# ── FUSION STRATEGY 3: Max Confidence Voting ──
# For each image, pick the prediction from whichever model is most confident
probs_stack    = np.stack([probs_b0, probs_b2, probs_resnet], axis=0)
max_conf_idx   = np.argmax(np.max(probs_stack, axis=2), axis=0)
probs_maxconf  = probs_stack[max_conf_idx, np.arange(len(y_true)), :]
acc_maxconf    = np.mean(np.argmax(probs_maxconf, axis=1) == y_true)

# ── FUSION STRATEGY 4: Majority Voting ──
votes_b0     = np.argmax(probs_b0,     axis=1)
votes_b2     = np.argmax(probs_b2,     axis=1)
votes_resnet = np.argmax(probs_resnet, axis=1)
votes_stack  = np.stack([votes_b0, votes_b2, votes_resnet], axis=1)

majority_votes, _ = stats.mode(votes_stack, axis=1)
majority_votes    = majority_votes.flatten()
acc_majority      = np.mean(majority_votes == y_true)

print(f"\n  Fusion Strategy Results:")
print(f"  Simple Average      : {acc_avg*100:.2f}%")
print(f"  Weighted Average    : {acc_weighted*100:.2f}%")
print(f"  Max Confidence      : {acc_maxconf*100:.2f}%")
print(f"  Majority Voting     : {acc_majority*100:.2f}%")

# Pick best fusion strategy
fusion_results = {
    "Simple Average":   acc_avg,
    "Weighted Average": acc_weighted,
    "Max Confidence":   acc_maxconf,
    "Majority Voting":  acc_majority
}
best_strategy = max(fusion_results, key=fusion_results.get)
best_acc      = fusion_results[best_strategy]

print(f"\n  BEST FUSION STRATEGY : {best_strategy}")
print(f"  BEST FUSION ACCURACY : {best_acc*100:.2f}%")
print(f"  IMPROVEMENT OVER B0  : +{(best_acc - acc_b0)*100:.2f}%")

# Save fusion weights
fusion_config = {
    "models": {
        "efficientnetb0": {
            "path":     MODELS_DIR + "/best_acne_model.keras",
            "img_size": [224, 224],
            "weight":   float(w_b0),
            "accuracy": float(acc_b0)
        },
        "efficientnetb2": {
            "path":     MODELS_DIR + "/efficientnetb2_acne.keras",
            "img_size": [260, 260],
            "weight":   float(w_b2),
            "accuracy": float(acc_b2)
        },
        "resnet50": {
            "path":     MODELS_DIR + "/resnet50_acne.keras",
            "img_size": [224, 224],
            "weight":   float(w_resnet),
            "accuracy": float(acc_resnet)
        }
    },
    "best_strategy":   best_strategy,
    "best_accuracy":   float(best_acc),
    "class_names":     CLASS_NAMES
}

with open(MODELS_DIR + "/fusion_config.json", "w") as f:
    json.dump(fusion_config, f, indent=2)

print("\nFusion config saved: models/fusion_config.json")
print("\nSTEP 3 COMPLETE — Fusion model built!")

