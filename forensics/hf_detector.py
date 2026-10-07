"""Hugging Face pretrained classifiers for AI-generated images and deepfakes.

Configured via configs/detectors.json.
Runs each detector on both:
 1. Full image (<key>_full)
 2. Face crop (<key>_face)
"""
import os
import json
from typing import List, Optional
import numpy as np
import cv2
from PIL import Image
from .utils import Signal
from .face import detect_face

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "detectors.json")

def load_detector_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    # Default fallback
    return [
        {"key": "hf_ai_image", "model_id": "umm-maybe/AI-image-detector"},
        {"key": "hf_deepfake", "model_id": "dima806/deepfake_vs_real_image_detection"}
    ]

_PIPELINES = {}
_PRINTED_LABELS = set()
FAKE_KEYWORDS = ("fake", "artificial", "ai", "deepfake", "generated", "synthetic", "label_1")

def _get_pipeline(key: str, model_id: str):
    if key in _PIPELINES:
        return _PIPELINES[key]
    from transformers import pipeline
    try:
        # Try MPS first, fallback to CPU
        import torch
        device = "mps" if torch.backends.mps.is_available() else -1
        pipe = pipeline("image-classification", model=model_id, device=device)
    except Exception:
        pipe = pipeline("image-classification", model=model_id, device=-1)
    _PIPELINES[key] = pipe
    return pipe

def _is_fake_label(label: str) -> bool:
    l = label.lower()
    return any(k in l for k in FAKE_KEYWORDS) and "real" not in l and "human" not in l and "label_0" not in l

def analyze_crop(pipe, pil_img, name: str, model_id: str) -> float:
    results = pipe(pil_img, top_k=None)
    if name not in _PRINTED_LABELS:
        labels = [r["label"] for r in results]
        print(f"[hf_detector] Model {name} ({model_id}) raw labels: {labels}")
        _PRINTED_LABELS.add(name)

    fake_prob = sum(float(r["score"]) for r in results if _is_fake_label(r["label"]))
    return float(np.clip(fake_prob, 0.0, 1.0))

def analyze(rgb: np.ndarray) -> List[Signal]:
    signals = []
    pil_full = Image.fromarray(rgb)
    
    # Extract face crop
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = detect_face(gray, rgb)
    if box is not None:
        x, y, w, h = box
        face_rgb = rgb[max(0, y):min(rgb.shape[0], y+h), max(0, x):min(rgb.shape[1], x+w)]
        pil_face = Image.fromarray(face_rgb) if face_rgb.size > 0 else pil_full
    else:
        # Center crop fallback
        h, w = rgb.shape[:2]
        ch, cw = int(h * 0.6), int(w * 0.6)
        sy, sx = (h - ch) // 2, (w - cw) // 2
        pil_face = Image.fromarray(rgb[sy:sy+ch, sx:sx+cw])

    configs = load_detector_config()

    for item in configs:
        key = item["key"]
        model_id = item["model_id"]
        try:
            pipe = _get_pipeline(key, model_id)
            if pipe is None:
                continue

            # Full image evaluation
            fake_prob_full = analyze_crop(pipe, pil_full, f"{key}_full", model_id)
            detail_full = f"HF detector ({key}_full): fake prob {fake_prob_full:.2f}."
            signals.append(Signal(f"{key}_full", fake_prob_full, 1.0, detail_full))

            # Face crop evaluation
            fake_prob_face = analyze_crop(pipe, pil_face, f"{key}_face", model_id)
            detail_face = f"HF detector ({key}_face): fake prob {fake_prob_face:.2f}."
            signals.append(Signal(f"{key}_face", fake_prob_face, 1.0, detail_face))

            # Legacy fallback signal name for backward compatibility
            if key in ["hf_ai_image", "hf_deepfake"]:
                signals.append(Signal(key, fake_prob_full, 1.0, detail_full))

        except Exception as e:
            print(f"[warn] HF detector {key} failed: {e}")

    return signals
