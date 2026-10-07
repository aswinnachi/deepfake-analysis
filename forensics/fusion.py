"""Data-Driven Multimodal Fusion & Calibrated Verdict Engine."""
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
    "hf_ai_image_full",
    "hf_ai_image_face",
    "hf_deepfake_full",
    "hf_deepfake_face",
    "hf_community_forensics_full",
    "hf_community_forensics_face",
    "clip_probe",
    "face_boundary",
    "region_disagreement",
    "face_swap_cnn",
    "provenance",
    "visible_watermark",
    "edit_cutout"
]

FUSION_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "fusion.joblib")
CALIBRATOR_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "calibrator.joblib")

_FUSION_CLF = None
_CALIBRATOR = None

def _load_models():
    global _FUSION_CLF, _CALIBRATOR
    if _FUSION_CLF is None and os.path.exists(FUSION_MODEL_PATH):
        try:
            import joblib
            _FUSION_CLF = joblib.load(FUSION_MODEL_PATH)
        except Exception:
            pass
    if _CALIBRATOR is None and os.path.exists(CALIBRATOR_PATH):
        try:
            import joblib
            _CALIBRATOR = joblib.load(CALIBRATOR_PATH)
        except Exception:
            pass

def feature_vector(signals):
    """Extract features + missing value indicators in fixed FEATURE_ORDER."""
    d = {s.name: s.score for s in signals if s is not None}
    feats = []
    missing = []
    for f in FEATURE_ORDER:
        if f in d:
            feats.append(float(d[f]))
            missing.append(0.0)
        else:
            feats.append(0.5)
            missing.append(1.0)
    return feats + missing

def fuse(signals, degradation=0.0, use_learned=True):
    _load_models()
    sigs = [s for s in signals if s is not None]
    if not sigs:
        return {
            "p_fake": 0.5, "authenticity_real_pct": 50.0, "verdict": "NO EVIDENCE",
            "confidence": "none", "explanation": ["No analysable signals found."], "signals": []
        }

    # 1. Check Provenance & Watermark first (Step 5)
    provenance_firing = [s for s in sigs if s.name in ("provenance", "visible_watermark") and s.score >= 0.90]
    
    # 2. Base weighted logit fusion (Model A fallback)
    scores = np.array([logit(s.score) for s in sigs if s.weight > 0], dtype=float)
    weights = np.array([s.weight for s in sigs if s.weight > 0], dtype=float)
    if weights.sum() > 0:
        z = float(np.sum(weights * scores) / weights.sum())
    else:
        z = 0.0
    p = sigmoid(1.2 * z)
    mode = "symmetric weighted logit fusion"

    # 3. Learned Fusion Model (Logistic Regression / GBDT)
    if use_learned and _FUSION_CLF is not None:
        try:
            fv = feature_vector(sigs)
            if hasattr(_FUSION_CLF, "n_features_in_") and len(fv) == _FUSION_CLF.n_features_in_:
                p = float(_FUSION_CLF.predict_proba([fv])[0, 1])
                mode = f"learned fusion ({type(_FUSION_CLF).__name__})"
        except Exception as e:
            pass

    # 4. Calibration (Step 8)
    if _CALIBRATOR is not None:
        try:
            p_cal = float(_CALIBRATOR.predict([p])[0])
            p = float(np.clip(p_cal, 0.01, 0.99))
        except Exception:
            pass

    # Adjust for image degradation
    p = 0.5 + (p - 0.5) * (1 - 0.4 * degradation)
    p = float(np.clip(p, 0.01, 0.99))

    # Calculate confidence
    all_scores = np.array([s.score for s in sigs if s.weight > 0])
    std = float(all_scores.std()) if len(all_scores) > 0 else 0.5
    conf = float(np.clip(1.0 - 1.8 * std, 0.1, 1.0) * (1.0 - 0.4 * degradation))
    conf_label = "high" if conf > 0.65 else "medium" if conf > 0.35 else "low"

    real_pct = round((1.0 - p) * 100, 1)

    # 5. Safety Rules & Decision Logic (Step 5, 6, 8)
    high_fake_alerts = [s for s in sigs if s.name != "metadata" and s.score >= 0.75]

    if provenance_firing:
        # Step 5: Provenance override
        p = 0.98
        real_pct = 2.0
        verdict = "AI-GENERATED (provenance marker)"
    elif p >= 0.65:
        verdict = "LIKELY FAKE / MANIPULATED"
    elif p <= 0.30:
        if high_fake_alerts:
            # Step 6 Safety Rule: Never output LIKELY REAL if any signal >= 0.75
            firing_names = ", ".join([f"{s.name} ({s.score:.2f})" for s in high_fake_alerts])
            verdict = f"INCONCLUSIVE (alert: {firing_names})"
        else:
            verdict = "LIKELY REAL"
    else:
        verdict = "INCONCLUSIVE"

    # Sorted explanation by influence (distance from 0.5 * weight)
    ranked = sorted(sigs, key=lambda s: -abs(s.score - 0.5) * max(s.weight, 0.1))
    lines = []
    
    if provenance_firing:
        lines.append(f"Provenance match: {provenance_firing[0].detail}")
        
    for s in ranked:
        arrow = "FAKE-leaning" if s.score > 0.55 else "REAL-leaning" if s.score < 0.45 else "neutral"
        lines.append(f"[{arrow} {s.score:.2f} | w={s.weight:.1f}] {s.detail}")

    return {
        "p_fake": p,
        "authenticity_real_pct": real_pct,
        "verdict": verdict,
        "confidence": f"{conf_label} ({conf:.2f})",
        "mode": mode,
        "degradation": round(degradation, 2),
        "explanation": lines,
        "signals": [s.to_dict() for s in sigs]
    }
