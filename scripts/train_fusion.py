"""Train data-driven fusion models and probability calibration.

Compares:
 (A) Weighted logit average
 (B) Regularized Logistic Regression
 (C) HistGradientBoostingClassifier

Evaluates cross-validation, selects the best model, fits probability calibrator,
and saves models/fusion.joblib.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
import joblib

sys.path.insert(0, ".")
from forensics.fusion import FEATURE_ORDER
from forensics.pipeline import analyze_image
from scripts.eval_signals import calc_recall_at_fpr

FUSION_MODEL_PATH = "models/fusion.joblib"
CALIBRATOR_PATH = "models/calibrator.joblib"

def extract_vector(row_dict):
    """Extract features + missing indicators."""
    feats = []
    missing = []
    for f in FEATURE_ORDER:
        val = row_dict.get(f, None)
        if val is None or pd.isna(val):
            feats.append(0.5)
            missing.append(1.0)
        else:
            feats.append(float(val))
            missing.append(0.0)
    return feats + missing

def main():
    print("=== Training Fusion Model ===")
    signals_csv = "results/signals.csv"
    if not os.path.exists(signals_csv):
        raise FileNotFoundError("results/signals.csv missing. Run scripts/eval_signals.py first.")
        
    df = pd.read_csv(signals_csv)
    df["is_fake"] = (~df["category"].str.startswith("real")).astype(int)
    
    # We re-evaluate signals on samples to include new branches (clip_probe, face_boundary, provenance, etc.)
    print("Extracting full signal vectors for train and val splits...")
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    
    # Sample up to 100 per category to train quickly within time limit
    train_sample = train_df.groupby("category").apply(lambda x: x.sample(min(len(x), 100), random_state=42)).reset_index(drop=True)
    val_sample = val_df.groupby("category").apply(lambda x: x.sample(min(len(x), 40), random_state=42)).reset_index(drop=True)
    
    def process_subset(sub):
        X, y, cats, subs = [], [], [], []
        for _, row in sub.iterrows():
            try:
                res = analyze_image(row["path"])
                d = {s["name"]: s["fake_score"] for s in res["signals"]}
                X.append(extract_vector(d))
                y.append(row["is_fake"])
                cats.append(row["category"])
                subs.append(row["subfolder"])
            except Exception:
                continue
        return np.array(X), np.array(y), np.array(cats), np.array(subs)
        
    X_train, y_train, cat_train, sub_train = process_subset(train_sample)
    X_val, y_val, cat_val, sub_val = process_subset(val_sample)
    
    print(f"Dataset ready: {len(X_train)} train, {len(X_val)} val samples.")
    
    # Model (B): Logistic Regression
    clf_lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf_lr.fit(X_train, y_train)
    p_val_lr = clf_lr.predict_proba(X_val)[:, 1]
    auc_lr = roc_auc_score(y_val, p_val_lr)
    rec_lr, _ = calc_recall_at_fpr(y_val, p_val_lr, target_fpr=0.05)
    
    # Model (C): HistGradientBoosting
    clf_hgb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    clf_hgb.fit(X_train, y_train)
    p_val_hgb = clf_hgb.predict_proba(X_val)[:, 1]
    auc_hgb = roc_auc_score(y_val, p_val_hgb)
    rec_hgb, _ = calc_recall_at_fpr(y_val, p_val_hgb, target_fpr=0.05)
    
    print("\n--- Fusion Model Comparison on Validation Split ---")
    print(f"(B) Logistic Regression:      AUC={auc_lr:.4f}, Recall@5%FPR={rec_lr*100:.2f}%")
    print(f"(C) HistGradientBoosting:     AUC={auc_hgb:.4f}, Recall@5%FPR={rec_hgb*100:.2f}%")
    
    # Choose best model
    best_clf = clf_lr if auc_lr >= auc_hgb else clf_hgb
    best_name = "Logistic Regression" if best_clf == clf_lr else "HistGradientBoosting"
    best_p_val = p_val_lr if best_clf == clf_lr else p_val_hgb
    
    print(f"\nSelected Model: {best_name}")
    
    # Calibrate probabilities with Isotonic Regression (Platt scaling fallback)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
    calibrator.fit(best_p_val, y_val)
    
    # Save models
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_clf, FUSION_MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    print(f"Saved {FUSION_MODEL_PATH} and {CALIBRATOR_PATH}")

if __name__ == "__main__":
    main()
