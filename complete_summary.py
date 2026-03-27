import os
import json

MODELS_DIR = r"C:/Users/Kartik/OneDrive/Desktop/acne_project/models"

with open(MODELS_DIR + "/fusion_config.json") as f:
    config = json.load(f)

files = [
    MODELS_DIR + "/best_acne_model.keras",
    MODELS_DIR + "/efficientnetb2_acne.keras",
    MODELS_DIR + "/resnet50_acne.keras",
    MODELS_DIR + "/fusion_config.json",
    r"C:/Users/Kartik/OneDrive/Desktop/acne_project/results/06_model_comparison.png",
    r"C:/Users/Kartik/OneDrive/Desktop/acne_project/results/07_fusion_confusion_matrix.png",
    r"C:/Users/Kartik/OneDrive/Desktop/acne_project/app/app.py",
]

print("\n" + "="*60)
print("  ACNEAI MULTI-FUSION — COMPLETE")
print("="*60)
print(f"\n  EfficientNetB0 accuracy : {config['models']['efficientnetb0']['accuracy']*100:.2f}%")
print(f"  EfficientNetB2 accuracy : {config['models']['efficientnetb2']['accuracy']*100:.2f}%")
print(f"  ResNet50 accuracy       : {config['models']['resnet50']['accuracy']*100:.2f}%")
print(f"  FUSION accuracy         : {config['best_accuracy']*100:.2f}%")
print(f"\n  Files:")
for f in files:
    s = "✅" if os.path.exists(f) else "❌"
    print(f"  {s}  {os.path.basename(f)}")
print("\n  Launch app:")
print("  streamlit run app/app.py")
print("="*60)

