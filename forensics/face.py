"""Face-region consistency: face swaps blend a face whose sharpness / noise
differs from its surroundings and from its own border.

This detects spliced faces in real photos and says nothing about fully generated images.

Uses OpenCV DNN face detector (Caffe model) for much better detection than
Haar cascades, with a Haar fallback if the DNN model files aren't present.
"""
import numpy as np
import cv2
import os
from .utils import Signal, to_unit

_DETECTOR = None
_DETECTOR_TYPE = None  # "dnn" or "haar"

# Paths for the DNN face detector model files
_DNN_PROTO = os.path.join(os.path.dirname(__file__), "..", "models", "deploy.prototxt")
_DNN_MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "res10_300x300_ssd_iter_140000.caffemodel")


def _init_detector():
    """Initialize the best available face detector."""
    global _DETECTOR, _DETECTOR_TYPE

    # Try DNN first (much better than Haar)
    if os.path.exists(_DNN_PROTO) and os.path.exists(_DNN_MODEL):
        try:
            _DETECTOR = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
            _DETECTOR_TYPE = "dnn"
            return
        except Exception:
            pass

    # Fallback to Haar cascade
    _DETECTOR = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _DETECTOR_TYPE = "haar"


def detect_face(gray, rgb=None):
    """Detect the largest face. Returns (x, y, w, h) or None.
    If rgb is provided and DNN is available, uses it for better accuracy.
    """
    global _DETECTOR, _DETECTOR_TYPE
    if _DETECTOR is None:
        _init_detector()

    if _DETECTOR_TYPE == "dnn" and rgb is not None:
        h, w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        _DETECTOR.setInput(blob)
        detections = _DETECTOR.forward()
        best_conf, best_box = 0, None
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                bw, bh = x2 - x1, y2 - y1
                if bw >= 48 and bh >= 48 and confidence > best_conf:
                    best_conf = confidence
                    best_box = (int(x1), int(y1), int(bw), int(bh))
        return best_box

    # Haar fallback
    faces = _DETECTOR.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
    if len(faces) == 0:
        return None
    return tuple(int(v) for v in max(faces, key=lambda b: b[2] * b[3]))


def detect_all_faces(gray, rgb=None):
    """Detect all faces. Returns list of (x, y, w, h) tuples."""
    global _DETECTOR, _DETECTOR_TYPE
    if _DETECTOR is None:
        _init_detector()

    faces = []
    if _DETECTOR_TYPE == "dnn" and rgb is not None:
        h, w = rgb.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(rgb, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        _DETECTOR.setInput(blob)
        detections = _DETECTOR.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                bw, bh = x2 - x1, y2 - y1
                if bw >= 48 and bh >= 48:
                    faces.append((int(x1), int(y1), int(bw), int(bh)))
    else:
        detected = _DETECTOR.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(detected) > 0:
            faces = [tuple(int(v) for v in f) for f in detected]
    return faces


def _stats(lap, res, rgb, mask):
    if mask.sum() < 200:
        return None
    color = rgb[mask].mean(axis=0) # [R, G, B]
    return float(lap[mask].var()) + 1.0, float(res[mask].std()) + 1e-3, color


def analyze(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = detect_face(gray, rgb)
    if box is None:
        return None
    x, y, w, h = box
    H, W = gray.shape
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    res = g - cv2.GaussianBlur(g, (0, 0), 1.2)

    full = np.zeros((H, W), bool)
    full[y:y + h, x:x + w] = True
    inner = np.zeros((H, W), bool)
    inner[y + int(.2 * h):y + int(.8 * h), x + int(.2 * w):x + int(.8 * w)] = True
    ring = full.copy()
    ring[y + int(.1 * h):y + int(.9 * h), x + int(.1 * w):x + int(.9 * w)] = False
    bg = np.ones((H, W), bool)
    pad = int(.3 * w)
    bg[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad] = False

    si = _stats(lap, res, rgb, inner)
    sr = _stats(lap, res, rgb, ring)
    sb = _stats(lap, res, rgb, bg)
    
    diffs, notes = [], []
    if si and sr:
        b_s = abs(np.log(si[0] / sr[0])) / 2.0
        b_n = abs(np.log(si[1] / sr[1]))
        b_c = np.linalg.norm(si[2] - sr[2]) / 255.0
        notes.append(f"vs face border: sharpness x{np.sqrt(si[0]/sr[0]):.2f}, noise x{si[1]/sr[1]:.2f}, color_gap {b_c:.2f}")
    else:
        b_s = b_n = b_c = 0.0

    if si and sb:
        bg_s = abs(np.log(si[0] / sb[0])) / 2.0
        bg_n = abs(np.log(si[1] / sb[1]))
        bg_c = np.linalg.norm(si[2] - sb[2]) / 255.0
        notes.append(f"vs background: sharpness x{np.sqrt(si[0]/sb[0]):.2f}, noise x{si[1]/sb[1]:.2f}, color_gap {bg_c:.2f}")
    else:
        bg_s = bg_n = bg_c = 0.0

    if not si:
        return None

    # Realistic calibration:
    # 1. Border discontinuity (face swap paste seam):
    border_discontinuity = max(b_s, b_n)
    s_border = to_unit(border_discontinuity, 0.70, 1.15)

    # 2. Contextual background mismatch (face pasted onto foreign background):
    bg_discontinuity = max(bg_s, bg_n, bg_c * 2.0)
    s_bg = to_unit(bg_discontinuity, 1.10, 1.80)

    # Face swap score combines border seam and background mismatch
    score_swap = 0.5 * s_border + 0.5 * s_bg

    # Synthetic portrait color gap (e.g., stylized AI lighting boundary)
    score_color = to_unit(b_c, 0.25, 0.60)

    score = float(np.clip(max(score_swap, score_color), 0.0, 1.0))

    heat = np.zeros((H, W), np.float32)
    heat[y:y + h, x:x + w] = score
    verdict = ("strong physical mismatch with surroundings (Face Swap / Deepfake)"
               if score > 0.5 else "face statistics blend naturally with frame")
    detail = f"Face consistency: {'; '.join(notes)} -> {verdict}."

    return Signal("face_consistency", score, 0.5, detail, heatmap=heat, box=box)

