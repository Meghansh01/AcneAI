# 🧴 AcneAI - Acne Type Classification Project

A deep learning-based acne classification system that identifies 5 types of acne lesions from skin images using Transfer Learning with EfficientNetB0 .

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Evaluation](#evaluation)
6. [Explainability (Grad-CAM)](#explainability-grad-cam)
7. [Web Application](#web-application)
8. [Project Structure](#project-structure)
9. [Requirements](#requirements)
10. [How to Run](#how-to-run)
11. [Results](#results)

---

## 🧠 Project Overview

This project builds a complete end-to-end acne classification system:

- **Input**: Skin images (224×224 RGB)
- **Output**: Classification into 5 acne types with confidence scores
- **Additional Features**: Severity assessment, treatment recommendations, model explainability

### Acne Types Classification

| Type | Severity | Description |
|------|----------|-------------|
| 🟢 Blackheads | Mild | Open comedones. Dark spots caused by oxidized melanin. |
| 🟢 Whiteheads | Mild | Closed comedones. Small white bumps under skin. |
| 🟡 Papules | Moderate | Inflamed red bumps without pus. Do not squeeze. |
| 🟡 Pustules | Moderate | Pus-filled red bumps. Classic pimples. |
| 🔴 Cyst | Severe | Deep pus-filled lesion. Most severe type. Can cause scarring. |

---

## 📊 Dataset

The model is trained on a custom acne dataset with the following structure:

```
AcneDataset/
├── train/           # Training set
│   ├── Blackheads/
│   ├── Whiteheads/
│   ├── Papules/
│   ├── Pustules/
│   └── Cyst/
├── valid/           # Validation set
│   ├── Blackheads/
│   ├── Whiteheads/
│   ├── Papules/
│   ├── Pustules/
│   └── Cyst/
└── test/            # Test set
    ├── Blackheads/
    ├── Whiteheads/
    ├── Papules/
    ├── Pustules/
    └── Cyst/
```

### Dataset Configuration

```python
CLASS_NAMES = ["Blackheads", "Whiteheads", "Papules", "Pustules", "Cyst"]
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
```

---

## 🏗️ Model Architecture

The model uses **Transfer Learning** with **EfficientNetB0** as the backbone.

### Data Augmentation

Applied during training to improve generalization:

```python
augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
    layers.RandomBrightness(0.15),
])
```

### Network Architecture

```
Input (224×224×3)
    │
    ▼
[Data Augmentation]
    │
    ▼
EfficientNetB0 (pretrained on ImageNet)
    │ (Frozen in Phase 1)
    ▼
GlobalAveragePooling2D
    │
    ▼
BatchNormalization
    │
    ▼
Dense(256) → ReLU → Dropout(0.4)
    │
    ▼
Dense(128) → ReLU → Dropout(0.3)
    │
    ▼
Dense(5) → Softmax
```

### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | ~4.8M |
| Trainable (Phase 1) | ~5.8K |
| Frozen Parameters | ~4.2M |
| Output Classes | 5 |

---

## 🔄 Training Process

Two-phase training approach for optimal performance:

### Phase 1: Training with Frozen Backbone

- **Duration**: 25 epochs
- **Learning Rate**: 1e-3
- **Strategy**: Train only custom classification head
- **Best Validation Accuracy**: **82.74%**
- **Loss Function**: Categorical Crossentropy with label smoothing (0.1)

### Phase 2: Fine-Tuning

- **Duration**: 15 epochs
- **Learning Rate**: 1e-5 (10× lower)
- **Strategy**: Unfreeze top 20 layers of EfficientNetB0
- **Best Validation Accuracy**: **80.02%**

### Callbacks Used

1. **EarlyStopping** - Stop if no improvement for 6 epochs
2. **ModelCheckpoint** - Save best model based on val_accuracy
3. **ReduceLROnPlateau** - Reduce learning rate by 0.3× every 3 epochs

---

## 📈 Evaluation

The model is evaluated using multiple metrics:

### Metrics Calculated

- **Test Accuracy**
- **Precision**, **Recall**, **F1-Score** (per class)
- **Confusion Matrix** (raw counts and normalized)

### Classification Report Example

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Blackheads | X.XX | X.XX | X.XX | XXX |
| Whiteheads | X.XX | X.XX | X.XX | XXX |
| Papules | X.XX | X.XX | X.XX | XXX |
| Pustules | X.XX | X.XX | X.XX | XXX |
| Cyst | X.XX | X.XX | X.XX | XXX |

---

## 🔍 Explainability (Grad-CAM)

Gradient-weighted Class Activation Mapping (Grad-CAM) is implemented to visualize where the model focuses its attention when making predictions.

### How It Works

1. Extract gradients from the target conv layer (top_conv)
2. Pool gradients to create importance weights
3. Generate heatmap showing attention regions
4. Overlay heatmap on original image

### Output

The Grad-CAM visualization shows:
- Original image
- Attention heatmap (red = high focus)
- Overlay image with highlighted regions

---

## 🌐 Web Application

A **Streamlit** web application provides an intuitive interface for acne classification.

### Features

1. **Image Upload** - Drag & drop or select skin images
2. **Real-time Analysis** - Instant classification results
3. **Severity Assessment** - Color-coded severity levels
4. **Treatment Recommendations** - Personalized advice for each acne type
5. **Probability Charts** - Interactive bar charts (Plotly)
6. **Top 3 Predictions** - Show most likely classes
7. **JSON Export** - Download results

### App Preview

```
┌─────────────────────────────────────────────────────┐
│  🧴 AcneAI Skin Analysis                            │
├─────────────────────────────────────────────────────┤
│  📤 Upload Image                                    │
│                                                     │
│  ┌─────────────┐  ┌──────────────────────────────┐  │
│  │             │  │  🟡 Papules                   │  │
│  │  [Image]    │  │  Confidence: 78.5%           │  │
│  │             │  │  Severity: Moderate          │  │
│  └─────────────┘  │  ████████████░░░░░░          │  │
│                   │                              │  │
│                   │  💡 Recommended Action:      │  │
│                   │  Apply benzoyl peroxide.     │  │
│                   │  Do NOT squeeze or pop.      │  │
│                   └──────────────────────────────┘  │
│                                                     │
│  📊 All Class Probabilities                         │
│  [████████████████████░░░░░░░░░░░] 82.3%            │
│  [████████████░░░░░░░░░░░░░░░░░░░░] 78.5%            │
│  ...                                                │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
acne_project/
├── step1_verify.py         # Dataset verification & visualization
├── step2_model.py         # Model architecture definition
├── step3_train.py         # Training pipeline (2 phases)
├── step5_evaluate.py      # Model evaluation & metrics
├── step6_gradcam.py       # Grad-CAM explainability
├── step7_predict.py       # Single image prediction demo
│
├── app/
│   └── app.py             # Streamlit web application
│
├── models/
│   ├── best_acne_model.keras      # Trained model
│   └── training_history.json      # Training history
│
├── results/
│   ├── 01_sample_images.png       # Dataset sample visualization
│   ├── 02_training_history.png    # Training curves
│   ├── 03_confusion_matrix.png    # Confusion matrix
│   ├── 04_gradcam.png            # Grad-CAM visualizations
│   └── 05_prediction_demo.png    # Prediction demo
│
└── README.md               # This file
```

---

## 🛠️ Requirements

### Python Dependencies

```
tensorflow>=2.10.0
numpy
matplotlib
seaborn
scikit-learn
streamlit
plotly
pillow
```

### Install Dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn streamlit plotly pillow
```

---

## 🚀 How to Run

### Step 1: Dataset Verification

```bash
python step1_verify.py
```

This will:
- Load and verify the dataset
- Display sample images
- Save visualization to `results/01_sample_images.png`

### Step 2: Model Definition

```bash
python step2_model.py
```

This will:
- Display model architecture
- Show parameter statistics

### Step 3: Training

```bash
python step3_train.py
```

This will:
- Train the model in 2 phases
- Save best model to `models/best_acne_model.keras`
- Save training history to `models/training_history.json`

### Step 4: Evaluation

```bash
python step5_evaluate.py
```

This will:
- Evaluate model on test set
- Generate classification report
- Create confusion matrix plots

### Step 5: Grad-CAM Visualization

```bash
python step6_gradcam.py
```

This will:
- Generate Grad-CAM visualizations
- Save to `results/04_gradcam.png`

### Step 6: Prediction Demo

```bash
python step7_predict.py
```

This will:
- Run single image prediction
- Display results with severity and advice

### Running the Web App

```bash
cd app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📊 Results Summary

### Training Performance

| Phase | Epochs | Best Val Accuracy | Learning Rate |
|-------|--------|-------------------|---------------|
| Phase 1 (Frozen) | 25 | 82.74% | 1e-3 |
| Phase 2 (Fine-tune) | 15 | 80.02% | 1e-5 |

### Model Capabilities

- ✅ 5-class acne classification
- ✅ Severity assessment (Mild/Moderate/Severe)
- ✅ Treatment recommendations
- ✅ Model explainability with Grad-CAM
- ✅ Interactive web interface

---

## ⚠️ Disclaimer

> **This project is for educational/research purposes only.**
> 
> The model is not a medical device and should NOT be used for actual medical diagnosis. Always consult a qualified dermatologist for skin conditions.

---

## 📝 License

This project is for educational purposes.

---

## 👤 Author

Created by: Kartik

---

## 🔗 Additional Notes

- The model uses EfficientNetB0 preprocessing (normalization built into the model)
- All image paths use absolute Windows paths - modify as needed for your system
- The web app runs locally and does not require internet after model loading

