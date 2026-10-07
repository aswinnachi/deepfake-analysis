import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score, confusion_matrix
sys.path.insert(0, ".")
import forensics.pipeline as pipeline

RESULTS_DIR = "results"
SIGNALS_CSV = os.path.join(RESULTS_DIR, "signals.csv")
EVAL_REPORT_MD = os.path.join(RESULTS_DIR, "eval_report.md")

def calc_recall_at_fpr(y_true, y_scores, target_fpr=0.05):
    """Compute Recall at <= target_fpr (False Positive Rate)."""
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    reals = y_scores[y_true == 0]
    fakes = y_scores[y_true == 1]
    
    if len(reals) == 0 or len(fakes) == 0:
        return 0.0, 0.0
        
    sorted_reals = np.sort(reals)[::-1]
    k = int(np.floor(target_fpr * len(reals)))
    thresh = sorted_reals[k] if k < len(reals) else sorted_reals[-1]
    
    actual_fpr = np.mean(reals >= thresh)
    recall = np.mean(fakes >= thresh)
    return float(recall), float(thresh)

def run_evaluation(output_report_path=EVAL_REPORT_MD):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    manifest_path = "data/manifest.csv"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError("data/manifest.csv missing. Run scripts/build_manifest.py first.")
        
    df_manifest = pd.read_csv(manifest_path)
    
    if os.path.exists(SIGNALS_CSV):
        df_cached = pd.read_csv(SIGNALS_CSV)
        cached_paths = set(df_cached["path"].tolist())
    else:
        df_cached = pd.DataFrame()
        cached_paths = set()
        
    records = df_cached.to_dict("records") if not df_cached.empty else []
    
    print(f"Evaluating signals across {len(df_manifest)} images ({len(cached_paths)} cached)...")
    
    new_rows = 0
    for idx, row in df_manifest.iterrows():
        path = row["path"]
        if path in cached_paths:
            continue
            
        try:
            res = pipeline.analyze_image(path)
            record = {
                "path": path,
                "category": row["category"],
                "subfolder": row["subfolder"],
                "source_group_id": row["source_group_id"],
                "split": row["split"],
                "fused_p_fake": res["p_fake"],
                "verdict": res["verdict"]
            }
            for sig in res.get("_signals", []):
                if sig is not None:
                    record[sig.name] = sig.score
            records.append(record)
            cached_paths.add(path)
            new_rows += 1
            if new_rows % 50 == 0:
                print(f" Analyzed {new_rows} new images...")
        except Exception as e:
            print(f"Error analyzing {path}: {e}")
            continue
            
    df_results = pd.DataFrame(records)
    df_results.to_csv(SIGNALS_CSV, index=False)
    print(f"Updated {SIGNALS_CSV} (total {len(df_results)} records).")
    
    report_lines = ["# Deepfake Forensics Signal & Pipeline Evaluation Report", ""]
    
    # 1. Dataset Counts
    report_lines.append("## 1. Dataset Counts")
    summary_table = df_results.groupby(["split", "category", "subfolder"]).size().unstack(fill_value=0)
    report_lines.append(summary_table.to_markdown())
    report_lines.append("")
    
    df_results["is_fake"] = (~df_results["category"].str.startswith("real")).astype(int)
    
    # 2. Per-Signal AUC
    report_lines.append("## 2. Per-Signal AUC")
    signal_cols = [c for c in df_results.columns if c not in ["path", "category", "subfolder", "source_group_id", "split", "fused_p_fake", "verdict", "is_fake"]]
    
    auc_data = []
    fake_cats = ["ai_full", "ai_edit", "faceswap"]
    
    for sig in signal_cols + ["fused_p_fake"]:
        valid_df = df_results.dropna(subset=[sig])
        row_auc = {"Signal": sig}
        useful_any = False
        
        for fcat in fake_cats:
            sub_df = valid_df[(valid_df["category"].str.startswith("real")) | (valid_df["category"] == fcat)]
            if len(sub_df["is_fake"].unique()) > 1:
                try:
                    auc = roc_auc_score(sub_df["is_fake"], sub_df[sig])
                    row_auc[fcat] = f"{auc:.4f}"
                    if auc >= 0.60:
                        useful_any = True
                except Exception:
                    row_auc[fcat] = "N/A"
            else:
                row_auc[fcat] = "N/A"
                
        row_auc["Status"] = "Useful" if useful_any else "Not Useful (AUC < 0.60)"
        auc_data.append(row_auc)
        
    auc_df = pd.DataFrame(auc_data)
    report_lines.append(auc_df.to_markdown(index=False))
    report_lines.append("")
    
    # 3. Fused Score Recall at <= 5% FPR
    report_lines.append("## 3. Fused Score Recall @ 5% FPR")
    rec_data = []
    for fcat in fake_cats:
        sub_df = df_results[(df_results["category"].str.startswith("real")) | (df_results["category"] == fcat)]
        rec, thresh = calc_recall_at_fpr(sub_df["is_fake"], sub_df["fused_p_fake"], target_fpr=0.05)
        try:
            auc = roc_auc_score(sub_df["is_fake"], sub_df["fused_p_fake"])
        except Exception:
            auc = 0.0
        rec_data.append({"Category": fcat, "AUC": f"{auc:.4f}", "Recall@5%FPR": f"{rec*100:.2f}%", "Threshold": f"{thresh:.4f}"})
    report_lines.append(pd.DataFrame(rec_data).to_markdown(index=False))
    report_lines.append("")
    
    # 4. Verdict Confusion Matrix
    report_lines.append("## 4. Verdict Distribution & Confusion Matrix")
    conf_table = pd.crosstab(df_results["category"], df_results["verdict"])
    report_lines.append(conf_table.to_markdown())
    report_lines.append("")
    
    # 5. Top 10 Worst Mistakes
    report_lines.append("## 5. Top 10 Worst Mistakes")
    df_results["error_severity"] = np.where(
        df_results["category"].str.startswith("real"),
        df_results["fused_p_fake"],
        1.0 - df_results["fused_p_fake"]
    )
    worst = df_results.sort_values("error_severity", ascending=False).head(10)
    
    worst_data = []
    for idx, r in worst.iterrows():
        sig_scores = [(col, r[col]) for col in signal_cols if pd.notnull(r[col])]
        sig_scores.sort(key=lambda x: abs(x[1] - 0.5), reverse=True)
        top3 = ", ".join([f"{name}:{score:.2f}" for name, score in sig_scores[:3]])
        worst_data.append({
            "Path": r["path"],
            "Category": r["category"],
            "Fused_Score": f"{r['fused_p_fake']:.3f}",
            "Verdict": r["verdict"],
            "Top-3 Signals": top3
        })
    report_lines.append(pd.DataFrame(worst_data).to_markdown(index=False))
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    with open(output_report_path, "w") as f:
        f.write(report_content)
        
    print(f"Report written to {output_report_path}")
    return report_content

if __name__ == "__main__":
    run_evaluation()
