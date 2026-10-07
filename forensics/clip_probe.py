"""Fast & Robust CLIP Vision Probe for AI-full and AI-edit Detection."""
import os
from typing import List, Optional
import numpy as np
import torch
import cv2
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib

from .utils import Signal
from .face import detect_face

CLIP_MODEL_ID = "openai/clip-vit-base-patch16"
PROBE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "clip_probe.joblib")
FEATURES_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "clip_features.pkl")

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_DEVICE = None

def _get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE
        
    from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
    _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
    _CLIP_PROCESSOR = CLIPImageProcessor.from_pretrained(CLIP_MODEL_ID)
    _CLIP_MODEL = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_ID).to(_DEVICE)
    _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PROCESSOR, _DEVICE

def extract_features(pil_img):
    model, processor, device = _get_clip()
    inputs = processor(images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model(**inputs).image_embeds.cpu().float().numpy()[0]
    emb = np.nan_to_num(emb, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb

def train_clip_probe(manifest_path="data/manifest.csv"):
    import pandas as pd
    df = pd.read_csv(manifest_path)
    df["is_fake"] = (~df["category"].str.startswith("real")).astype(int)
    
    # Train primarily on train split (sample for speed: up to 300 per class)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    # Stratified fast sample for speedy 15-min completion
    train_sample = train_df.groupby("category").apply(lambda x: x.sample(min(len(x), 80), random_state=42)).reset_index(drop=True)
    val_sample = val_df.groupby("category").apply(lambda x: x.sample(min(len(x), 30), random_state=42)).reset_index(drop=True)
    
    print(f"[clip_probe] Extracting features for {len(train_sample)} train images and {len(val_sample)} val images...")
    
    def get_feats(subset):
        X, y = [], []
        for _, row in subset.iterrows():
            try:
                pil_img = Image.open(row["path"]).convert("RGB")
                X.append(extract_features(pil_img))
                y.append(row["is_fake"])
            except Exception:
                continue
        return np.array(X), np.array(y)
        
    X_train, y_train = get_feats(train_sample)
    X_val, y_val = get_feats(val_sample)
    
    clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    
    val_preds = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_preds)
    print(f"[clip_probe] Trained model Val AUC: {val_auc:.4f}")
    
    os.makedirs(os.path.dirname(PROBE_MODEL_PATH), exist_ok=True)
    joblib.dump(clf, PROBE_MODEL_PATH)
    return clf

_LOADED_PROBE = None
def analyze(rgb: np.ndarray) -> Optional[Signal]:
    global _LOADED_PROBE
    if not os.path.exists(PROBE_MODEL_PATH):
        return None
    try:
        if _LOADED_PROBE is None:
            _LOADED_PROBE = joblib.load(PROBE_MODEL_PATH)
            
        pil_img = Image.fromarray(rgb)
        feat = extract_features(pil_img)
        fake_prob = float(_LOADED_PROBE.predict_proba([feat])[0, 1])
        fake_prob = float(np.clip(fake_prob, 0.0, 1.0))
        
        detail = f"CLIP probe score: {fake_prob:.2f}."
        return Signal("clip_probe", fake_prob, 1.5, detail)
    except Exception as e:
        return None
