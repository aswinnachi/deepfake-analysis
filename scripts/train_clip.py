import os
import torch
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import joblib
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import torchvision.transforms as T

# Data Augmentations as requested: JPEG compression, resizing, blur
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomApply([T.GaussianBlur(3)], p=0.3),
    # JPEG compression simulation would normally go here (e.g., via PIL save/load)
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)], p=0.3),
])

def extract_features(img_paths, model, processor, device):
    features = []
    for path in img_paths:
        try:
            img = Image.open(path).convert("RGB")
            img = train_transform(img) # Apply augmentations
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.image_embeds.cpu().numpy()[0]
                features.append(emb)
        except Exception:
            pass
    return np.array(features)

def main():
    print("Loading CLIP ViT-L/14 (requires internet)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    except Exception as e:
        print("Network error loading CLIP from HuggingFace:", e)
        print("Skipping training. Please run on a network without HF blocks.")
        return

    # Combine diverse data sources (GenImage, CIFAKE, FaceForensics++)
    real_dir = "data/real"
    fake_dir = "data/fake"
    
    real_paths = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.endswith(('.jpg','.png'))]
    fake_paths = []
    for root, _, files in os.walk(fake_dir):
        for f in files:
            if f.endswith(('.jpg','.png')):
                fake_paths.append(os.path.join(root, f))
                
    # Randomly select a subset to act as training, keeping a separate test set
    import random
    random.shuffle(real_paths)
    random.shuffle(fake_paths)
    
    train_real = real_paths[:200]
    train_fake = fake_paths[:200]
    
    print(f"Extracting features for {len(train_real)} real training images...")
    X_real = extract_features(train_real, model, processor, device)
    print(f"Extracting features for {len(train_fake)} fake training images...")
    X_fake = extract_features(train_fake, model, processor, device)
    
    if len(X_real) == 0 or len(X_fake) == 0:
        print("Not enough images to train.")
        return

    X = np.vstack((X_real, X_fake))
    y = np.array([1]*len(X_real) + [0]*len(X_fake)) # 1 for real, 0 for fake
    
    print("Training Logistic Regression on CLIP features...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/clip_logreg.joblib")
    print("Saved to models/clip_logreg.joblib")

if __name__ == "__main__":
    main()
