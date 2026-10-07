import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from forensics.cnn import build_model, MODEL_PATH, PATCH

_GOD_MODEL = None
_OPTIM = None

def init_god_mode():
    global _GOD_MODEL, _OPTIM
    if _GOD_MODEL is not None:
        return
    _GOD_MODEL = build_model()
    if os.path.exists(MODEL_PATH):
        try:
            _GOD_MODEL.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        except:
            pass
    _GOD_MODEL.train()
    _OPTIM = torch.optim.Adam(_GOD_MODEL.parameters(), lr=1e-4)

def train_image(image_path, label):
    """
    label: 1.0 for FAKE, 0.0 for REAL
    """
    init_god_mode()
    
    img = cv2.imread(image_path)
    if img is None:
        return "Failed to load image."
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = rgb.shape[:2]
    if min(h, w) < PATCH:
        rgb = cv2.resize(rgb, (PATCH, PATCH))
        h = w = PATCH
        
    ys = list(range(0, h - PATCH + 1, max(1, PATCH // 2)))[:8]
    xs = list(range(0, w - PATCH + 1, max(1, PATCH // 2)))[:8]
    coords = [(y, x) for y in ys for x in xs][:32]
    if not coords:
        coords = [(0,0)]
        
    mean = np.array([0.485, 0.456, 0.406], np.float32)
    std = np.array([0.229, 0.224, 0.225], np.float32)
    batch = np.stack([((rgb[y:y + PATCH, x:x + PATCH].astype(np.float32) / 255 - mean) / std) for y, x in coords])
    
    X = torch.from_numpy(batch).permute(0, 3, 1, 2)
    y = torch.full((len(coords), 1), label, dtype=torch.float32)
    
    _OPTIM.zero_grad()
    logits = _GOD_MODEL(X)
    loss = nn.BCEWithLogitsLoss()(logits, y)
    loss.backward()
    _OPTIM.step()
    
    # Save the updated model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(_GOD_MODEL.state_dict(), MODEL_PATH)
    
    # Force reload of CNN model in prediction mode
    import forensics.cnn as cnn
    cnn._MODEL = None
    
    return f"Successfully trained God Mode on {len(coords)} patches as {'FAKE' if label==1.0 else 'REAL'}. Loss: {loss.item():.4f}"
