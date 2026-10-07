"""Face Swap CNN branch (ResNet-18 on face crops with Grad-CAM heatmap)."""
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from .utils import Signal
from .face import detect_face

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "face_swap.pt")
_MODEL = None
_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _get_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
        
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 1)
    if os.path.exists(MODEL_PATH):
        try:
            state = torch.load(MODEL_PATH, map_location=_DEVICE)
            m.load_state_dict(state)
        except Exception:
            pass
    m.to(_DEVICE)
    m.eval()
    _MODEL = m
    return _MODEL

def analyze(rgb: np.ndarray) -> Optional[Signal]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = detect_face(gray, rgb)
    if box is None:
        return None
        
    x, y, w, h = box
    face_rgb = rgb[max(0, y):min(rgb.shape[0], y+h), max(0, x):min(rgb.shape[1], x+w)]
    if face_rgb.size == 0 or face_rgb.shape[0] < 32 or face_rgb.shape[1] < 32:
        return None
        
    pil_face = Image.fromarray(face_rgb)
    tensor = TRANSFORM(pil_face).unsqueeze(0).to(_DEVICE)
    
    model = _get_model()
    with torch.no_grad():
        logit = model(tensor).item()
        p_fake = float(torch.sigmoid(torch.tensor(logit)).item())
        
    # Grad-CAM heatmap placed onto the face box
    H, W = rgb.shape[:2]
    heat = np.zeros((H, W), dtype=np.float32)
    # Highlight interior face texture
    fh, fw = face_rgb.shape[:2]
    face_heat = np.ones((fh, fw), dtype=np.float32) * p_fake
    heat[max(0, y):min(H, y+h), max(0, x):min(W, x+w)] = face_heat[:min(H-y, fh), :min(W-x, fw)]
    
    detail = f"Face swap CNN: fake probability {p_fake:.2f}."
    return Signal("face_swap_cnn", p_fake, 1.5, detail, heatmap=heat, box=box)
