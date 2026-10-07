"""Optional learned detector (ResNet-18 fine-tuned on native-resolution patches).
Runs only if models/cnn.pt exists (train with train/train_cnn.py). Scoring many
patches gives a *localisation* map: which regions look generated."""
import os
import numpy as np
import cv2
from .utils import Signal

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "cnn.pt")
_MODEL = None
PATCH = 224


def build_model():
    import torch.nn as nn
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m


def _load():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import torch
        m = build_model()
        m.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        m.eval()
        _MODEL = m
    except Exception:
        return None
    return _MODEL


def analyze(rgb, max_tiles=16):
    model = _load()
    if model is None:
        return None
    import torch
    h, w = rgb.shape[:2]
    if min(h, w) < PATCH:
        rgb = cv2.resize(rgb, (PATCH, PATCH)); h = w = PATCH
    ys = list(range(0, h - PATCH + 1, PATCH)) or [0]
    xs = list(range(0, w - PATCH + 1, PATCH)) or [0]
    coords = [(y, x) for y in ys for x in xs][:max_tiles]
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = np.stack([((rgb[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255 - mean) / std) for y, x in coords])
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(batch).permute(0, 3, 1, 2))).squeeze(1).numpy()
    heat = np.zeros((h, w), np.float32)
    for (y, x), v in zip(coords, p):
        heat[y:y + PATCH, x:x + PATCH] = v
    score = float(np.mean(p))
    detail = (f"Learned CNN scored {len(coords)} native-resolution patches: mean fake probability "
              f"{score:.2f} (max {p.max():.2f}) -> {'generator-like texture in several regions' if score > 0.5 else 'patches look camera-like'}.")
    # Sensible weight: contributes to logit voting instead of being display-only
    return Signal("cnn", score, 1.5, detail, heatmap=heat)
