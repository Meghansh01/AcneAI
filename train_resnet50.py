import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import json

TRAIN_DIR   = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\train"
VALID_DIR   = r"C:\Users\Kartik\Downloads\acne dataset img\AcneDataset\valid"
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = 5
AUTOTUNE    = tf.data.AUTOTUNE
MODELS_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/models"

print("="*55)
print("  TRAINING MODEL 3 — ResNet50")
print("="*55)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=True, seed=42
).cache().shuffle(1000).prefetch(AUTOTUNE)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR, labels="inferred", label_mode="categorical",
    class_names=CLASS_NAMES, image_size=IMG_SIZE,
    batch_size=BATCH_SIZE, shuffle=False
).cache().prefetch(AUTOTUNE)

# Augmentation
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
], name="augmentation_resnet")

# ResNet50 backbone
base = ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3)
)
base.trainable = False

inputs  = layers.Input(shape=(224, 224, 3), name="input_resnet")
x       = augment(inputs)
# ResNet50 needs specific preprocessing
x       = tf.keras.applications.resnet50.preprocess_input(x)
x       = base(x, training=False)
x       = layers.GlobalAveragePooling2D(name="gap_resnet")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(512, activation="relu")(x)
x       = layers.Dropout(0.5)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(128, activation="relu")(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model_resnet = models.Model(inputs, outputs, name="ResNet50_Acne")

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

os.makedirs(MODELS_DIR, exist_ok=True)

callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6,
                  restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        MODELS_DIR + "/resnet50_acne.keras",
        monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3,
                      patience=3, min_lr=1e-7, verbose=1),
]

# Phase 1
print("\nPhase 1 — Frozen backbone...")
history1 = model_resnet.fit(
    train_ds, validation_data=valid_ds,
    epochs=25, callbacks=callbacks, verbose=1
)
best_p1 = max(history1.history["val_accuracy"])
print(f"Phase 1 best: {best_p1:.4f} ({best_p1*100:.1f}%)")

# Phase 2 - Fine-tune
print("\nPhase 2 — Fine-tuning top 30 layers...")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model_resnet.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

history2 = model_resnet.fit(
    train_ds, validation_data=valid_ds,
    epochs=15, callbacks=callbacks, verbose=1
)
best_p2 = max(history2.history["val_accuracy"])
print(f"Phase 2 best: {best_p2:.4f} ({best_p2*100:.1f}%)")

model_resnet.save(MODELS_DIR + "/resnet50_acne.keras")

history_combined = {
    "accuracy":     history1.history["accuracy"]     + history2.history["accuracy"],
    "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
    "loss":         history1.history["loss"]         + history2.history["loss"],
    "val_loss":     history1.history["val_loss"]     + history2.history["val_loss"],
    "phase1_end":   len(history1.history["accuracy"]),
    "best_val_acc": best_p2
}
with open(MODELS_DIR + "/resnet50_history.json", "w") as f:
    json.dump(history_combined, f)

print(f"\nResNet50 saved!")
print(f"Best val accuracy: {best_p2:.4f} ({best_p2*100:.1f}%)")
print("\nSTEP 2 COMPLETE — ResNet50 trained!")

