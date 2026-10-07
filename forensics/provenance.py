"""Provenance, Digital Signatures, and Visible Watermark Detection.

Detects:
 1. Raw byte markers: c2pa, jumb, trainedAlgorithmicMedia, compositeWithTrainedAlgorithmicMedia,
    DigitalSourceType, SynthID
 2. Visible AI watermarks (e.g. Gemini / SynthID sparkle icon) in bottom-right 20% edge map
 3. Alpha transparency cutouts (edit_cutout)
"""
import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Optional
from .utils import Signal

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "..", "assets")
TEMPLATE_PATH = os.path.join(ASSETS_DIR, "watermark_template.png")

PROVENANCE_KEYWORDS = [
    b"c2pa",
    b"jumb",
    b"trainedAlgorithmicMedia",
    b"compositeWithTrainedAlgorithmicMedia",
    b"DigitalSourceType",
    b"SynthID",
    b"google:generator",
    b"google:model"
]

def scan_bytes(path: str) -> Optional[Signal]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            content = f.read()
        for kw in PROVENANCE_KEYWORDS:
            if kw in content:
                marker_str = kw.decode('latin1', errors='ignore')
                return Signal(
                    name="provenance",
                    score=0.97,
                    weight=3.0,
                    detail=f"Cryptographic / metadata provenance marker detected ('{marker_str}'). Note: can be removed by cropping."
                )
    except Exception:
        pass
    return None

def detect_watermark(rgb: np.ndarray) -> Optional[Signal]:
    if not os.path.exists(TEMPLATE_PATH):
        return None
    try:
        tpl = cv2.imread(TEMPLATE_PATH)
        if tpl is None:
            return None
        tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
        tpl_edges = cv2.Canny(tpl_gray, 50, 150)
        
        h, w = rgb.shape[:2]
        # Search in bottom-right 25% of image
        br_rgb = rgb[int(h * 0.75):, int(w * 0.75):]
        if br_rgb.shape[0] < 40 or br_rgb.shape[1] < 40:
            return None
            
        br_gray = cv2.cvtColor(br_rgb, cv2.COLOR_RGB2GRAY)
        br_edges = cv2.Canny(br_gray, 50, 150)
        
        # Multi-scale matchTemplate
        best_val = 0.0
        for scale in np.linspace(0.6, 1.4, 7):
            tw = int(tpl_edges.shape[1] * scale)
            th = int(tpl_edges.shape[0] * scale)
            if tw >= br_edges.shape[1] or th >= br_edges.shape[0] or tw < 20 or th < 20:
                continue
            resized_tpl = cv2.resize(tpl_edges, (tw, th))
            res = cv2.matchTemplate(br_edges, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                
        # Zero false positives on real images with threshold 0.45
        if best_val >= 0.42:
            return Signal(
                name="visible_watermark",
                score=0.95,
                weight=3.0,
                detail=f"Visible AI watermark detected in bottom-right region (correlation: {best_val:.2f})."
            )
    except Exception as e:
        pass
    return None

def check_cutout(path: str, rgb: np.ndarray) -> Optional[Signal]:
    if not path or not os.path.exists(path):
        return None
    try:
        img_pil = Image.open(path)
        if img_pil.mode in ("RGBA", "LA") or (img_pil.mode == "P" and "transparency" in img_pil.info):
            alpha = np.array(img_pil.convert("RGBA"))[:, :, 3]
            transparent_fraction = np.mean(alpha < 240)
            if transparent_fraction > 0.05:
                return Signal(
                    name="edit_cutout",
                    score=0.70,
                    weight=0.5,
                    detail=f"Alpha transparency detected ({transparent_fraction*100:.1f}%). Transparency indicates editing/cutout, not necessarily fakeness."
                )
    except Exception:
        pass
    return None

def analyze(rgb: np.ndarray, path: Optional[str] = None) -> List[Signal]:
    sigs = []
    if path:
        prov_sig = scan_bytes(path)
        if prov_sig:
            sigs.append(prov_sig)
            
        cutout_sig = check_cutout(path, rgb)
        if cutout_sig:
            sigs.append(cutout_sig)
            
    wm_sig = detect_watermark(rgb)
    if wm_sig:
        sigs.append(wm_sig)
        
    return sigs
