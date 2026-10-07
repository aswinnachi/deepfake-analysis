import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0, ".")
from scripts.eval_signals import calc_recall_at_fpr

RESULTS_DIR = "results"
UNSEEN_CSV = os.path.join(RESULTS_DIR, "unseen.csv")

def run_unseen_eval():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    signals_csv = os.path.join(RESULTS_DIR, "signals.csv")
    if not os.path.exists(signals_csv):
        raise FileNotFoundError("results/signals.csv missing. Run scripts/eval_signals.py first.")
        
    df = pd.read_csv(signals_csv)
    df["is_fake"] = (~df["category"].str.startswith("real")).astype(int)
    
    fake_subfolders = df[df["is_fake"] == 1]["subfolder"].unique().tolist()
    
    unseen_results = []
    print(f"Evaluating Leave-One-Subfolder-Out across {len(fake_subfolders)} subfolders...")
    
    real_df = df[df["is_fake"] == 0]
    
    for held_out in sorted(fake_subfolders):
        held_out_fake_df = df[(df["is_fake"] == 1) & (df["subfolder"] == held_out)]
        combined_df = pd.concat([real_df, held_out_fake_df])
        
        category = held_out_fake_df["category"].iloc[0] if len(held_out_fake_df) > 0 else "unknown"
        scores = combined_df["fused_p_fake"]
        labels = combined_df["is_fake"]
        
        try:
            auc = roc_auc_score(labels, scores)
        except Exception:
            auc = 0.0
            
        rec, thresh = calc_recall_at_fpr(labels, scores, target_fpr=0.05)
        
        unseen_results.append({
            "Held_Out_Subfolder": held_out,
            "Category": category,
            "N_Fake_Samples": len(held_out_fake_df),
            "AUC": f"{auc:.4f}",
            "Recall@5%FPR": f"{rec*100:.2f}%",
            "Status": "Honest Evaluation (Held-Out)"
        })
        print(f"Held-out Subfolder '{held_out}': AUC={auc:.4f}, Recall@5%FPR={rec*100:.2f}%")
        
    res_df = pd.DataFrame(unseen_results)
    res_df.to_csv(UNSEEN_CSV, index=False)
    print("\n=== Leave-One-Subfolder-Out (Unseen Generators) Summary ===")
    print(res_df.to_markdown(index=False))
    return res_df

if __name__ == "__main__":
    run_unseen_eval()
