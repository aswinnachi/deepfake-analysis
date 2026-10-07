"""Frequency-domain fingerprints. GAN / diffusion up-sampling leaves periodic
spikes and unusual high-frequency decay in the Fourier spectrum."""
import numpy as np
import cv2
from scipy.ndimage import uniform_filter1d
from .utils import Signal, to_unit


def radial_profile(gray):
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(gray * win))
    p = np.log1p(np.abs(f))
    y, x = np.indices(p.shape)
    r = np.hypot(y - h // 2, x - w // 2).astype(int)
    prof = np.bincount(r.ravel(), p.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return prof[: min(h, w) // 2]


def analyze(rgb, size=256):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    if min(h, w) < size:
        gray = cv2.resize(gray, (size, size))
        h = w = size
    # crops keep native pixel statistics (resizing would erase the artefacts)
    crops = [gray[(h - size) // 2:(h + size) // 2, (w - size) // 2:(w + size) // 2],
             gray[:size, :size], gray[-size:, -size:]]
    spikes, hfs = [], []
    for c in crops:
        prof = radial_profile(c)
        n = len(prof)
        smooth = uniform_filter1d(prof, 7)
        band = slice(n // 2, n - 2)
        spikes.append(float(np.max(prof[band] - smooth[band])))
        hfs.append(float(prof[int(.75 * n):].mean() / (prof[:int(.25 * n)].mean() + 1e-6)))
    spike, hf = float(np.mean(spikes)), float(np.mean(hfs))
    score = 0.6 * to_unit(spike, 0.05, 0.30) + 0.4 * to_unit(0.45 - hf, -0.10, 0.25)
    verdict = ("up-sampling / generator-like spectral pattern" if score > 0.5
               else "natural-looking spectrum")
    detail = (f"Fourier spectrum: periodic high-freq spike strength {spike:.3f}, "
              f"HF/LF energy ratio {hf:.2f} -> {verdict}.")
    # Sensible forensic weight: contributes to logit voting instead of being display-only
    return Signal("frequency", score, 1.0, detail)
