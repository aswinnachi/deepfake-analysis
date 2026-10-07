"""Robustness + generalisation report. Per fake generator and per perturbation
(JPEG q30, half-resolution, 80% centre crop, blur) -> AUC and accuracy.
    python -m scripts.evaluate --max-per-class 200
"""
import argparse, io, json, os, random
import numpy as np, cv2
from PIL import Image
from sklearn.metrics import roc_auc_score
from forensics.pipeline import analyze_array
from forensics.utils import load_rgb
from train.common import load_dataset


def jpeg(a, q=30):
    b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG", quality=q); b.seek(0); return np.array(Image.open(b).convert("RGB"))

def half(a): return cv2.resize(a, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
def crop(a):
    h, w = a.shape[:2]; return a[int(.1*h):int(.9*h), int(.1*w):int(.9*w)]
def blur(a): return cv2.GaussianBlur(a, (0, 0), 1.2)

PERTS = {"original": lambda a: a, "jpeg_q30": jpeg, "half_res": half, "crop_80": crop, "blur": blur}


def score(path, fn):
    try:
        return analyze_array(fn(load_rgb(path)), None)["p_fake"]
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max-per-class", type=int, default=200)
    a = ap.parse_args(); random.seed(1)
    real, fakes = load_dataset("data")
    real = random.sample(real, min(len(real), a.max_per_class))
    rows = {}
    for pname, fn in PERTS.items():
        rs = [s for s in (score(p, fn) for p in real) if s is not None]
        for g, v in fakes.items():
            fs = [s for s in (score(p, fn) for p in random.sample(v, min(len(v), a.max_per_class))) if s is not None]
            y = [0] * len(rs) + [1] * len(fs); sc = rs + fs
            auc = roc_auc_score(y, sc); acc = float(np.mean((np.array(sc) > .5) == np.array(y)))
            fpr = float(np.mean(np.array(rs) > .5))
            rows.setdefault(g, {})[pname] = {"auc": round(auc, 3), "acc": round(acc, 3), "false_positive_rate": round(fpr, 3)}
            print(f"{g:14s} {pname:10s} AUC {auc:.3f}  ACC {acc:.3f}  FPR(real flagged) {fpr:.3f}")
    os.makedirs("results", exist_ok=True); json.dump(rows, open("results/robustness.json", "w"), indent=2)


if __name__ == "__main__":
    main()
