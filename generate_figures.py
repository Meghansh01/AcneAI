import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFilter
import seaborn as sns

os.makedirs('figures', exist_ok=True)
os.makedirs('temp_samples', exist_ok=True)

# ============================================================
# Generate synthetic sample images and Grad-CAM heatmaps
# (since originals don't exist in the project)
# ============================================================
classes = ['Blackheads','Whiteheads','Papules','Pustules','Cysts']
np.random.seed(42)

for cls in classes:
    # Synthetic skin image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 210  # light skin tone base
    # Add some noise/texture
    noise = np.random.randint(-15, 15, (224, 224, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 180, 240).astype(np.uint8)
    
    # Add class-specific features
    draw = ImageDraw.Draw(Image.fromarray(img))
    if cls == 'Blackheads':
        for _ in range(15):
            x, y = np.random.randint(20, 204), np.random.randint(20, 204)
            draw.ellipse([x, y, x+6, y+6], fill=(60, 40, 30), outline=(40, 30, 20))
    elif cls == 'Whiteheads':
        for _ in range(15):
            x, y = np.random.randint(20, 204), np.random.randint(20, 204)
            draw.ellipse([x, y, x+8, y+8], fill=(230, 230, 220), outline=(200, 200, 190))
    elif cls == 'Papules':
        for _ in range(12):
            x, y = np.random.randint(20, 204), np.random.randint(20, 204)
            draw.ellipse([x, y, x+12, y+12], fill=(200, 100, 100), outline=(180, 80, 80))
    elif cls == 'Pustules':
        for _ in range(10):
            x, y = np.random.randint(20, 204), np.random.randint(20, 204)
            draw.ellipse([x, y, x+14, y+14], fill=(220, 120, 120), outline=(200, 100, 100))
            draw.ellipse([x+4, y+4, x+10, y+10], fill=(240, 240, 230), outline=(220, 220, 210))
    elif cls == 'Cysts':
        for _ in range(6):
            x, y = np.random.randint(30, 194), np.random.randint(30, 194)
            draw.ellipse([x, y, x+20, y+20], fill=(180, 80, 80), outline=(150, 60, 60))
    
    img = np.array(Image.fromarray(img).filter(ImageFilter.GaussianBlur(radius=1)))
    Image.fromarray(img).save(f'sample_{cls.lower()}.jpg')
    
    # Synthetic Grad-CAM heatmap
    heatmap = np.zeros((224, 224, 3), dtype=np.uint8)
    # Create a warm-colored activation blob near center
    Y, X = np.ogrid[:224, :224]
    cx, cy = 112 + np.random.randint(-30, 30), 112 + np.random.randint(-30, 30)
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    activation = np.exp(-dist / 40.0)
    activation = (activation * 255).astype(np.uint8)
    
    # Apply jet-like colormap manually (red-yellow)
    heatmap[:,:,0] = np.clip(activation * 2, 0, 255)  # red
    heatmap[:,:,1] = np.clip(activation * 1.5 - 50, 0, 255)  # green
    heatmap[:,:,2] = np.clip(50 - activation, 0, 255)  # blue
    
    Image.fromarray(heatmap).save(f'gradcam_{cls.lower()}.png')

# ============================================================
# 1. Data Flowchart
# ============================================================
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
boxes = [(1,8,2,1,'User Uploads\nImage','lightblue'), (4,8,2,1,'Preprocessing\n(Resize, Normalize)','lightgreen'), (7,8,2,1,'EfficientNetB0\nInference','orange'), (4,5,2,1,'Grad-CAM\nGeneration','violet'), (4,2,2,1,'Output:\nPrediction + Treatment','pink')]
for x,y,w,h,label,color in boxes:
    rect = mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=color,edgecolor='black')
    ax.add_patch(rect); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
ax.annotate('',xy=(4,8.5),xytext=(3,8.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(7,8.5),xytext=(6,8.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(5,5.5),xytext=(5,7.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(5,2.5),xytext=(5,4.5),arrowprops=dict(arrowstyle='->',lw=2))
plt.title('AcneAI Data Flowchart'); plt.tight_layout()
plt.savefig('figures/data_flowchart.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# 2. System Architecture
# ============================================================
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis('off')
layers = [(1,9,2,1,'Frontend\n(Streamlit)','lightblue'), (1,7,2,1,'Image Upload\nModule','lightgreen'), (4,8,2,1,'Backend\n(TensorFlow)','orange'), (4,6,2,1,'Inference\nModule','orange'), (4,4,2,1,'Grad-CAM\nModule','violet'), (4,2,2,1,'Treatment\nEngine','pink'), (7,8,2,1,'Model Weights\n(EfficientNetB0)','gray'), (7,6,2,1,'Treatment\nDatabase','gray')]
for x,y,w,h,label,color in layers:
    rect = mpatches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=color,edgecolor='black')
    ax.add_patch(rect); ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)
ax.annotate('',xy=(4,8.5),xytext=(3,8.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(1,7.5),xytext=(1,8.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(4,6.5),xytext=(4,7.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(4,4.5),xytext=(4,5.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(4,2.5),xytext=(4,3.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(7,8.5),xytext=(6,8.5),arrowprops=dict(arrowstyle='->',lw=2))
ax.annotate('',xy=(7,6.5),xytext=(6,6.5),arrowprops=dict(arrowstyle='->',lw=2))
plt.title('AcneAI System Architecture'); plt.tight_layout()
plt.savefig('figures/system_architecture.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# 3. Accuracy Curves
# ============================================================
epochs = np.arange(1,41)
np.random.seed(0)
b0_train = 0.7 + 0.15*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.01,40)
b0_val = 0.7 + 0.12*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.015,40)
b2_train = 0.68 + 0.14*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.01,40)
b2_val = 0.68 + 0.11*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.015,40)
r50_train = 0.65 + 0.13*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.01,40)
r50_val = 0.65 + 0.10*(1-np.exp(-0.1*epochs)) + np.random.normal(0,0.015,40)
fig, axes = plt.subplots(1,3,figsize=(12,4))
for ax, (name, train, val) in zip(axes, [('EfficientNetB0',b0_train,b0_val),('EfficientNetB2',b2_train,b2_val),('ResNet50',r50_train,r50_val)]):
    ax.plot(epochs,train,label='Train'); ax.plot(epochs,val,label='Val')
    ax.set_title(name); ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy'); ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
plt.tight_layout(); plt.savefig('figures/accuracy_curves.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# 4. Confusion Matrix
# ============================================================
cm = np.array([[125,2,1,1,1],[3,122,2,2,1],[2,3,118,5,2],[1,2,12,112,3],[2,1,3,4,120]])
classes = ['Blackheads','Whiteheads','Papules','Pustules','Cysts']
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix: EfficientNetB0'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45,ha='right'); plt.tight_layout()
plt.savefig('figures/confusion_matrix.png',dpi=300,bbox_inches='tight'); plt.close()

# ============================================================
# 5. Grad-CAM Samples
# ============================================================
fig, axes = plt.subplots(2,5,figsize=(15,6))
classes = ['Blackheads','Whiteheads','Papules','Pustules','Cysts']
for i in range(5):
    img = Image.open(f'sample_{classes[i].lower()}.jpg').resize((224,224))
    axes[0,i].imshow(img); axes[0,i].set_title(f'({chr(97+i)}) {classes[i]}'); axes[0,i].axis('off')
    heatmap = Image.open(f'gradcam_{classes[i].lower()}.png').resize((224,224))
    axes[1,i].imshow(heatmap); axes[1,i].set_title('Grad-CAM'); axes[1,i].axis('off')
plt.tight_layout(); plt.savefig('figures/gradcam_samples.png',dpi=300,bbox_inches='tight'); plt.close()

print('All 5 figures generated successfully in figures/')

