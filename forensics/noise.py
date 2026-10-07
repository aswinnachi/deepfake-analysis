"""Noise-residual consistency. Real camera noise is fairly uniform across the
frame; pasted / generated regions carry different noise statistics.

Thresholds are deliberately conservative: heavy social-media compression
(WhatsApp, Instagram) creates block-boundary noise variation that should NOT
trigger a false positive."""
import numpy as np
import cv2
from .utils import Signal, to_unit


def analyze(rgb, block=32):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    res = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    h, w = res.shape
    h2, w2 = h // block * block, w // block * block
    if h2 < block * 2 or w2 < block * 2:
        return None
    blocks = res[:h2, :w2].reshape(h2 // block, block, w2 // block, block)
    lv = blocks.std(axis=(1, 3)) + 1e-3
    mean_noise = float(lv.mean())
    cv_ = float(lv.std() / lv.mean())
    dev = np.abs(np.log(lv / np.median(lv)))

    # --- Scoring ---
    # cv_ (coefficient of variation) tells us how *uneven* the noise is.
    #   Real photos compressed once: cv_ ~ 0.15-0.50
    #   WhatsApp / heavy recompression: cv_ ~ 0.50-1.10  (creates block artifacts, NOT fake)
    #   Actual splices / composites: cv_ > 1.20
    # mean_noise tells us how much noise exists overall.
    #   AI images are unnaturally clean: mean_noise < 0.8
    #   Real photos: mean_noise ~ 2.0-8.0
    #   Heavily compressed: mean_noise ~ 1.0-2.5

    # Widen the "real" thresholds so compression artifacts don't trigger false positives
    variation_score = to_unit(cv_, 0.70, 1.50)   # was (0.35, 1.2) — too sensitive
    cleanliness_score = 1 - to_unit(mean_noise, 0.5, 2.5)  # was (0.8, 3.0)

    score = 0.5 * variation_score + 0.5 * cleanliness_score

    heat = cv2.resize((dev / (dev.max() + 1e-6)).astype(np.float32), (w2, h2),
                      interpolation=cv2.INTER_NEAREST)
    verdict = ("inconsistent or unnaturally clean noise (synthetic / composited)"
               if score > 0.5 else "consistent sensor-like noise")
    detail = (f"Noise residual: mean level {mean_noise:.2f}, block-to-block variation "
              f"{cv_:.2f} -> {verdict}.")
    return Signal("noise", score, 1.5, detail, heatmap=heat)
