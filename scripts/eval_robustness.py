import os
import sys
import cv2
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
import forensics.pipeline as pipeline
from scripts.eval_signals import calc_recall_at_fpr

RESULTS_DIR = "results"
ROBUSTNESS_CSV = os.path.join(RESULTS_DIR, "robustness.csv")

def apply_transform(img_np, transform_type):
    h, w = img_np.shape[:2]
    if transform_type == "jpeg_q30":
        _, enc = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
        return cv2.imdecode(enc, 1)
    elif transform_type == "resize_0.5":
        small = cv2.resize(img_np, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif transform_type == "crop_80":
        ch, cw = int(h * 0.8), int(w * 0.8)
        sy, sx = (h - ch) // 2, (w - cw) // 2
        cropped = img_np[sy:sy+ch, sx:sx+cw]
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    elif transform_type == "blur":
        return cv2.GaussianBlur(img_np, (7, 7), 0)
    else:
        return img_np

def run_robustness_eval():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    manifest_path = "data/manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("data/manifest.csv missing.")
        
    df = pd.read_csv(manifest_path)
    df_val = df[df["split"] == "val"].copy()
    
    transforms = ["clean", "jpeg_q30", "resize_0.5", "crop_80", "blur"]
    results = []
    
    print(f"Running Robustness Evaluation on {len(df_val)} validation images...")
    
    for t_name in transforms:
        scores = []
        labels = []
        cats = []
        
        for idx, row in df_val.iterrows():
            path = row["path"]
            is_fake = 0 if row["category"].startswith("real") else 1
            try:
                img_np = cv2.imread(path)
                if img_np is None:
                    continue
                aug_img = apply_transform(img_np, t_name)
                rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                res = pipeline.analyze_array(rgb, path=path)
                scores.append(res["p_fake"])
                labels.append(is_fake)
                cats.append(row["category"])
            except Exception as e:
                continue
                
        df_t = pd.DataFrame({"category": cats, "label": labels, "score": scores})
        
        rec_overall, _ = calc_recall_at_fpr(df_t["label"], df_t["score"], target_fpr=0.05)
        try:
            auc_overall = roc_auc_score(df_t["label"], df_t["score"])
        except Exception:
            auc_overall = 0.0
            
        rec_row = {
            "Transform": t_name,
            "Overall_AUC": f"{auc_overall:.4f}",
            "Overall_Recall@5%FPR": f"{rec_overall*100:.2f}%"
        }
        
        for fcat in ["ai_full", "ai_edit", "faceswap"]:
            sub_t = df_t[(df_t["category"].str.startswith("real")) | (df_t["category"] == fcat)]
            rec, _ = calc_recall_at_fpr(sub_t["label"], sub_t["score"], target_fpr=0.05)
            rec_row[f"{fcat}_Recall@5%FPR"] = f"{rec*100:.2f}%"
            
        results.append(rec_row)
        print(f"Transform: {t_name:10s} -> Overall Recall@5%FPR: {rec_overall*100:.2f}%")
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(ROBUSTNESS_CSV, index=False)
    print("\n=== Robustness Summary Table ===")
    print(res_df.to_markdown(index=False))
    return res_df

if __name__ == "__main__":
    run_robustness_eval()
