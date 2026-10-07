"""Video analysis: per-frame image forensics + temporal consistency."""
import numpy as np
import cv2
from . import ela, frequency, noise, face, cnn
from .face import detect_face
from .utils import Signal, to_unit


def sample_frames(path, n=12):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 1, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            if max(f.shape[:2]) > 960:
                k = 960 / max(f.shape[:2]); f = cv2.resize(f, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            frames.append(f)
    cap.release()
    return frames


def temporal_face(path, max_frames=48):
    """Face flicker: fakes often show frame-to-frame jitter in face texture/box."""
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(total // 2 - max_frames // 2, 0))
    sharp, centers, sizes = [], [], []
    for _ in range(max_frames):
        ok, f = cap.read()
        if not ok:
            break
        small = cv2.resize(f, None, fx=0.5, fy=0.5)
        g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        b = detect_face(g, rgb_small)
        if b is None:
            continue
        x, y, w, h = b
        sharp.append(np.log(cv2.Laplacian(g[y:y + h, x:x + w], cv2.CV_32F).var() + 1.0))
        centers.append((x + w / 2, y + h / 2)); sizes.append(w)
    cap.release()
    if len(sharp) < 12:
        return None
    flick = float(np.std(np.diff(sharp)))
    c = np.array(centers)
    jit = float(np.mean(np.linalg.norm(np.diff(c, axis=0), axis=1)) / np.mean(sizes))
    s = 0.6 * to_unit(flick, 0.08, 0.5) + 0.4 * to_unit(jit, 0.02, 0.15)
    return Signal("temporal_face", s, 1.0,
                  f"Temporal: face-texture flicker {flick:.2f}, box jitter {jit:.3f} of face width over {len(sharp)} frames -> "
                  f"{'unstable face rendering between frames (blend/swap flicker)' if s > 0.5 else 'temporally stable face'}.")


def analyze_frames(frames):
    """Return aggregated signals and the most suspicious frame (for localisation)."""
    per = {}
    worst, worst_score, worst_sigs = None, -1, []
    for f in frames:
        sigs = [m.analyze(f) for m in (ela, frequency, noise, face, cnn)]
        sigs = [s for s in sigs if s is not None]
        tot = np.mean([s.score for s in sigs]) if sigs else 0
        if tot > worst_score:
            worst, worst_score, worst_sigs = f, tot, sigs
        for s in sigs:
            per.setdefault(s.name, []).append(s)
    agg = []
    for name, lst in per.items():
        mean = float(np.mean([s.score for s in lst]))
        mx = float(np.max([s.score for s in lst]))
        proto = lst[0]
        agg.append(Signal(name, 0.7 * mean + 0.3 * mx, proto.weight,
                          f"[{len(lst)} frames, mean {mean:.2f}, max {mx:.2f}] {proto.detail}",
                          heatmap=next((s.heatmap for s in worst_sigs if s.name == name), None)))
    return agg, worst
