"""Error Level Analysis: re-save as JPEG and look at where the error differs.
Spliced / inpainted regions often have a different compression history."""
import io
import numpy as np
import cv2
from PIL import Image
from .utils import Signal, to_unit


def ela_map(rgb, quality=90):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re = np.array(Image.open(buf).convert("RGB")).astype(np.float32)
    return np.abs(rgb.astype(np.float32) - re).mean(axis=2)


def analyze(rgb):
    d = cv2.GaussianBlur(ela_map(rgb), (0, 0), 3)
    med = float(np.median(d)) + 1e-3
    p99 = float(np.percentile(d, 99))
    ratio = p99 / med
    hot_frac = float((d > 3 * med + 1.0).mean())
    score = 0.6 * to_unit(ratio, 3.0, 12.0) + 0.4 * to_unit(hot_frac, 0.01, 0.15)
    heat = d / (d.max() + 1e-6)
    verdict = ("regions with a different compression history (possible edit/splice)"
               if score > 0.5 else "uniform compression history")
    detail = (f"Error-level analysis: peak/median ratio {ratio:.1f}, {hot_frac*100:.1f}% of pixels "
              f"stand out -> {verdict}.")
    # Sensible forensic weight: contributes to logit voting instead of being display-only
    return Signal("ela", score, 1.0, detail, heatmap=heat)
