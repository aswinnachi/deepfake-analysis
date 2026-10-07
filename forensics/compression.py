"""Estimate how degraded (compressed / small) the media is, so the fusion layer
lowers its confidence instead of over-claiming on heavily re-encoded input."""
import numpy as np
import cv2
from .utils import to_unit


def blockiness(gray):
    d = np.abs(np.diff(gray.astype(np.float32), axis=1)).mean(axis=0)
    if len(d) < 16:
        return 1.0
    on = d[7::8].mean()
    off = np.mean([d[i::8].mean() for i in range(8) if i != 7])
    return float(on / (off + 1e-6))


def degradation(rgb):
    """returns (0..1 degradation, human readable note)"""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    b = blockiness(gray)
    small = 1 - to_unit(min(gray.shape), 128, 512)
    deg = float(max(small, to_unit(b, 1.02, 1.40)))
    return deg, (f"Compression/resolution check: 8x8 blockiness {b:.2f}, "
                 f"min side {min(gray.shape)}px -> degradation {deg:.2f}")
