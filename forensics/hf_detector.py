"""Hugging Face pretrained classifiers for AI-generated images and deepfakes.

Integrates:
 - umm-maybe/AI-image-detector: detects diffusion / GAN / synthetic AI images
 - dima806/deepfake_vs_real_image_detection: detects deepfake / face-swap manipulations
"""
from typing import List, Optional
import numpy as np
from PIL import Image
from .utils import Signal

MODELS = {
    "hf_ai_image": "umm-maybe/AI-image-detector",
    "hf_deepfake": "dima806/deepfake_vs_real_image_detection",
}

# Forensic signal weight for the pretrained HF models (none should be display-only)
HF_SIGNAL_WEIGHT = 2.0

FAKE_KEYWORDS = ("fake", "artificial", "ai", "deepfake", "generated", "synthetic")

_PIPELINES = {}
_PRINTED_LABELS = set()


def _get_pipeline(name: str):
    """Lazily load and cache the Hugging Face image classification pipeline."""
    if name in _PIPELINES:
        return _PIPELINES[name]
    model_id = MODELS.get(name)
    if not model_id:
        return None
    from transformers import pipeline
    pipe = pipeline("image-classification", model=model_id)
    _PIPELINES[name] = pipe
    return pipe


def _is_fake_label(label: str) -> bool:
    """Check if label indicates synthetic / fake content and not real."""
    l = label.lower()
    return any(k in l for k in FAKE_KEYWORDS) and "real" not in l


def analyze(rgb: np.ndarray) -> List[Signal]:
    """Analyze an RGB numpy array using Hugging Face classifiers.
    
    Returns a list of Signal objects (weight 2.0).
    Wraps each model in try/except so a failed download or model never crashes the app.
    """
    signals = []
    pil_img = Image.fromarray(rgb)

    for name, model_id in MODELS.items():
        try:
            pipe = _get_pipeline(name)
            if pipe is None:
                continue

            results = pipe(pil_img, top_k=None)
            
            # Print raw labels once for debugging
            if name not in _PRINTED_LABELS:
                labels = [r["label"] for r in results]
                print(f"[hf_detector] Model {name} ({model_id}) raw labels: {labels}")
                _PRINTED_LABELS.add(name)

            # Fake probability is sum of scores for fake/ai/synthetic labels (excluding 'real')
            fake_prob = sum(
                float(r["score"]) for r in results if _is_fake_label(r["label"])
            )
            fake_prob = float(np.clip(fake_prob, 0.0, 1.0))

            detail = (
                f"HF detector ({name}): fake prob {fake_prob:.2f} "
                f"({'AI/manipulated pattern' if fake_prob > 0.5 else 'natural capture pattern'})."
            )
            signals.append(Signal(name, fake_prob, HF_SIGNAL_WEIGHT, detail))
        except Exception as e:
            # Model download failure or inference error should never crash the app
            print(f"[warn] HF detector {name} failed: {e}")

    return signals
