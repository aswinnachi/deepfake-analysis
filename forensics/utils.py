from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

@dataclass
class Signal:
    """One forensic cue. score: 0 = looks real, 1 = looks fake."""
    name: str
    score: float
    weight: float
    detail: str
    heatmap: Optional[np.ndarray] = None   # float 0..1, any size (resized later)
    box: Optional[Tuple[int, int, int, int]] = None

    def to_dict(self):
        return {"name": self.name, "fake_score": round(float(self.score), 3),
                "weight": self.weight, "detail": self.detail}

def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))

def logit(p: float) -> float:
    p = min(max(float(p), 0.02), 0.98)
    return float(np.log(p / (1 - p)))

def to_unit(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo + 1e-12), 0.0, 1.0))

def load_rgb(path, max_side: int = 1600) -> np.ndarray:
    """Load image as RGB. If image has alpha channel, composite onto gray (128)."""
    pil_img = Image.open(str(path))
    if pil_img.mode in ("RGBA", "LA") or (pil_img.mode == "P" and "transparency" in pil_img.info):
        rgba = pil_img.convert("RGBA")
        # Composite onto gray (128)
        gray_bg = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        composited = Image.alpha_composite(gray_bg, rgba).convert("RGB")
        rgb = np.array(composited)
    else:
        rgb = np.array(pil_img.convert("RGB"))

    h, w = rgb.shape[:2]
    s = max(h, w)
    if s > max_side:
        k = max_side / s
        rgb = cv2.resize(rgb, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    return rgb
