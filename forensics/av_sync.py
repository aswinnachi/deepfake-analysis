"""Audio-visual consistency: does mouth movement follow the speech energy?
Lip-sync fakes (Wav2Lip etc.) and dubbed/cloned audio break this correlation."""
import numpy as np
import cv2
from .face import detect_face
from .utils import Signal, to_unit


def analyze(video_path, y, sr, max_seconds=12):
    if y is None or len(y) == 0:
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    motion, prev, n = [], None, 0
    while n < fps * max_seconds:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        box = detect_face(gray, rgb_small)
        if box is None:
            motion.append(np.nan); prev = None
            continue
        x, yy, w, h = box
        mouth = cv2.resize(gray[yy + int(.62 * h):yy + h, x + int(.2 * w):x + int(.8 * w)], (32, 16)).astype(np.float32)
        motion.append(np.nan if prev is None else float(np.abs(mouth - prev).mean()))
        prev = mouth
    cap.release()
    motion = np.array(motion, float)
    if np.isfinite(motion).sum() < 20:
        return None
    hop = int(sr / fps)
    env = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) for i in range(len(motion)) if (i + 1) * hop <= len(y)])
    m = motion[: len(env)]
    ok = np.isfinite(m)
    if ok.sum() < 20:
        return None
    mz = (m[ok] - m[ok].mean()) / (m[ok].std() + 1e-6)
    ez = (env[ok] - env[ok].mean()) / (env[ok].std() + 1e-6)
    best, best_lag = -1.0, 0
    for lag in range(-6, 7):
        a, b = (mz[lag:], ez[:len(ez) - lag]) if lag >= 0 else (mz[:lag], ez[-lag:])
        k = min(len(a), len(b))
        if k > 10:
            c = float(np.corrcoef(a[:k], b[:k])[0, 1])
            if c > best:
                best, best_lag = c, lag
    score = 0.75 * (1 - to_unit(best, 0.05, 0.45)) + 0.25 * to_unit(abs(best_lag), 2, 5)
    detail = (f"Lip-sync check: mouth-motion vs speech-energy correlation {best:.2f} at offset "
              f"{best_lag/fps*1000:.0f} ms -> "
              f"{'mouth movement does not follow the audio (lip-sync / dubbed-audio suspicion)' if score > 0.5 else 'mouth movement follows the speech'}.")
    return Signal("av_sync", float(score), 1.5, detail)
