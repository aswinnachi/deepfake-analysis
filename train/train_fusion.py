"""Train the small fusion model on handcrafted forensic signals (fast, CPU only,
and generalises better than a big CNN on tiny data). Reports LEAVE-ONE-GENERATOR-OUT
AUC: each fake method is held out in turn -> an honest 'unseen technique' score.

    python -m train.train_fusion --max-per-class 400
"""
import argparse, random, json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from forensics.pipeline import analyze_array
from forensics.fusion import feature_vector, FEATURE_ORDER
from forensics.utils import load_rgb
from train.common import load_dataset


def feats(path):
    try:
        res = analyze_array(load_rgb(path), path, use_learned=False)
        return feature_vector(res["_signals"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--max-per-class", type=int, default=400)
    a = ap.parse_args()
    random.seed(0)
    real, fakes = load_dataset(a.data)
    assert real and fakes, "Put images in data/real and data/fake/<generator>/"
    random.shuffle(real)
    X_real = [f for f in map(feats, real[: a.max_per_class]) if f]
    X_fake = {g: [f for f in map(feats, random.sample(v, min(len(v), a.max_per_class))) if f] for g, v in fakes.items()}
    split = int(0.8 * len(X_real))
    r_tr, r_te = X_real[:split], X_real[split:]
    report = {}
    for g in X_fake:
        tr_f = [f for k, v in X_fake.items() if k != g for f in v]
        if not tr_f:
            continue
        X = np.array(r_tr + tr_f); y = np.array([0] * len(r_tr) + [1] * len(tr_f))
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, y)
        Xt = np.array(r_te + X_fake[g]); yt = np.array([0] * len(r_te) + [1] * len(X_fake[g]))
        report[g] = round(float(roc_auc_score(yt, clf.predict_proba(Xt)[:, 1])), 3)
        print(f"held-out generator {g:15s} AUC = {report[g]}")
    if report:
        print("mean unseen-generator AUC:", round(float(np.mean(list(report.values()))), 3))
    X = np.array(X_real + [f for v in X_fake.values() for f in v])
    y = np.array([0] * len(X_real) + [1] * sum(len(v) for v in X_fake.values()))
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(X, y)
    joblib.dump(clf, "models/fusion.joblib")
    print("feature weights:", dict(zip(FEATURE_ORDER, np.round(clf.coef_[0], 2))))
    json.dump(report, open("models/leave_one_generator_out.json", "w"), indent=2)
    print("saved models/fusion.joblib")


if __name__ == "__main__":
    main()
