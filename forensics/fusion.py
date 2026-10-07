"""Fuse independent forensic signals into one authenticity score + explanation.

Architecture:
 - Signals with weight > 0 vote in logit space (weighted sum).
 - No signals are display-only; all forensic signals contribute to the score.
 - An "UNCERTAIN" zone (40-60%) avoids embarrassing forced verdicts.
 - Low confidence (< 0.30) results in "INCONCLUSIVE" verdict to prevent false alarms
   on compressed, re-shared, or ambiguous images. Never show LIKELY FAKE with low confidence.
 - Heavy degradation (WhatsApp, screenshots) shrinks confidence toward 50%.
"""
import os
import numpy as np
from .utils import logit, sigmoid, Signal

FEATURE_ORDER = [
    "metadata",
    "ela",
    "frequency",
    "noise",
    "face_consistency",
    "cnn",
    "hf_ai_image",
    "hf_deepfake",
]

FUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fusion.joblib")

def feature_vector(signals):
    """Extract a fixed-order feature vector for learned fusion models."""
    d = {s.name: s.score for s in signals if s is not None}
    return [float(d.get(k, 0.5)) for k in FEATURE_ORDER]

def fuse(signals, degradation=0.0, use_learned=True):
    sigs = [s for s in signals if s is not None]
    if not sigs:
        return {"p_fake": 0.5, "authenticity_real_pct": 50.0, "verdict": "NO EVIDENCE",
                "confidence": "none", "explanation": ["No analysable signals found."], "signals": []}

    tier_a_names = {"hf_ai_image", "hf_deepfake"}
    tier_a = []
    tier_b = []

    for s in sigs:
        if s.name in tier_a_names:
            s.weight = 2.0
            tier_a.append(s)
        else:
            if s.name == "ela": s.weight = 1.0
            elif s.name == "frequency": s.weight = 1.0
            elif s.name == "noise": s.weight = 0.8
            elif s.name == "face_consistency": s.weight = 0.5
            elif s.name == "metadata": s.weight = 0.5
            elif s.name == "cnn": s.weight = 0.5
            else: s.weight = 0.5
            tier_b.append(s)

    z_A = 0.0
    if tier_a:
        w_A = np.array([s.weight for s in tier_a], float)
        z_A = float(np.sum(w_A * np.array([logit(s.score) for s in tier_a])) / w_A.sum())
    
    z_B = 0.0
    if tier_b:
        w_B = np.array([s.weight for s in tier_b], float)
        if w_B.sum() > 0:
            z_B = float(np.sum(w_B * np.array([logit(s.score) for s in tier_b])) / w_B.sum())
            
    if tier_a:
        z = z_A + 0.5 * np.clip(z_B, -0.5, 2.0)
    else:
        z = z_B
        
    p = sigmoid(1.6 * z)
    mode = "heuristic fusion"
    
    if use_learned and os.path.exists(FUSION_MODEL_PATH):
        try:
            import joblib
            clf = joblib.load(FUSION_MODEL_PATH)
            fv = feature_vector(sigs)
            if hasattr(clf, "n_features_in_") and clf.n_features_in_ != len(fv):
                pass
            else:
                p = float(clf.predict_proba([fv])[0, 1])
                mode = "learned fusion (logistic regression)"
        except Exception:
            pass

    p = 0.5 + (p - 0.5) * (1 - 0.5 * degradation)

    all_scores = np.array([s.score for s in sigs if s.weight > 0])
    agreement = float(np.clip(1 - 2 * all_scores.std(), 0, 1)) if len(all_scores) > 0 else 0.0
    coverage = min(1.0, len(sigs) / 3)
    conf = agreement * coverage * (1 - 0.5 * degradation)
    
    if len(tier_a) >= 2:
        diff = abs(tier_a[0].score - tier_a[1].score)
        if diff > 0.5:
            conf *= 0.5
            
    conf_label = "high" if conf > 0.6 else "medium" if conf > 0.35 else "low"

    real = 1 - p
    real_pct = round(real * 100, 1)
    
    max_tier_a = max([s.score for s in tier_a]) if tier_a else 0.0
    
    if real_pct >= 65 and max_tier_a < 0.60:
        verdict = "LIKELY REAL"
    elif (real_pct <= 35 and conf_label != "low") or (max_tier_a >= 0.90 and real_pct <= 45):
        verdict = "LIKELY FAKE / MANIPULATED"
    else:
        verdict = "INCONCLUSIVE"
        
    ranked = sorted(sigs, key=lambda s: -abs(s.score - 0.5) * max(s.weight, 0.01))
    lines = []
    
    for s in ranked:
        arrow = "FAKE-leaning" if s.score > 0.6 else "REAL-leaning" if s.score < 0.4 else "neutral"
        lines.append(f"[{arrow} {s.score:.2f} | w={s.weight:.1f}] {s.detail}")
        
    lines.append("Note: Real-leaning heuristic scores are weak evidence for fully synthetic images.")
    if max_tier_a >= 0.60:
        lines.append("Pretrained detector flags this image; heuristic checks cannot rule out fully synthetic content.")
        
    return {"p_fake": p, "authenticity_real_pct": real_pct, "verdict": verdict,
            "confidence": f"{conf_label} ({conf:.2f})", "mode": mode, "degradation": round(degradation, 2),
            "explanation": lines, "signals": [s.to_dict() for s in sigs]}
