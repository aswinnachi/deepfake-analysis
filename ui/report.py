"""Offline HTML and JSON report generator for forensic analysis."""
import os
import io
import json
import base64
import datetime
import numpy as np
from PIL import Image

def image_to_base64(img_array_or_pil) -> str:
    """Converts numpy RGB array or PIL Image into a base64 data URI."""
    if img_array_or_pil is None:
        return ""
    try:
        if isinstance(img_array_or_pil, np.ndarray):
            pil_img = Image.fromarray(img_array_or_pil.astype(np.uint8))
        elif isinstance(img_array_or_pil, Image.Image):
            pil_img = img_array_or_pil
        else:
            return ""
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception:
        return ""

def generate_html_report(data: dict) -> str:
    """Generates a standalone, beautifully styled offline HTML report."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    filename = data.get("filename", "unknown_file")
    sha256 = data.get("file_sha", "N/A")
    elapsed = data.get("elapsed_sec", 0.0)
    verdict = data.get("verdict", "INCONCLUSIVE")
    real_pct = data.get("real_pct", 50.0)
    fake_pct = round(100.0 - real_pct, 1)
    badge_color = data.get("badge_color", "#f59e0b")
    bg_color = data.get("bg_color", "rgba(245, 158, 11, 0.12)")
    icon = data.get("icon", "🔍")
    verdict_note = data.get("verdict_note", "")
    evidence_strength = data.get("evidence_strength", "Moderate")
    region_caption = data.get("region_caption", "No single region stands out")
    
    overlay_img = data.get("overlay")
    b64_overlay = image_to_base64(overlay_img)
    
    # Top 3 reasons
    top_3_html = ""
    for r in data.get("top_3_reasons", []):
        top_3_html += f"""
        <li style="margin-bottom: 8px;">
            <strong>{r['badge']} ({r['name']}):</strong> {r['detail']}
        </li>
        """
        
    # Full signals table
    signals_rows = ""
    for grp_name, sig_list in data.get("active_groups", {}).items():
        for s in sig_list:
            signals_rows += f"""
            <tr>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{s['name']}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{grp_name}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155; font-weight: 600; color: {s['bar_color']};">{s['score']:.3f}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155;">{s['weight']}</td>
                <td style="padding: 8px 12px; border-bottom: 1px solid #334155; font-size: 0.85rem; color: #94a3b8;">{s['detail']}</td>
            </tr>
            """

    heatmap_section = ""
    if b64_overlay:
        heatmap_section = f"""
        <div style="margin-top: 24px; padding: 20px; background: #1e293b; border-radius: 12px; border: 1px solid #334155;">
            <h3 style="margin-top: 0; color: #f8fafc;">Manipulation Heatmap & Regional Localization</h3>
            <p style="color: #cbd5e1; font-size: 0.9rem;"><strong>Region focus:</strong> {region_caption}</p>
            <div style="text-align: center; margin-top: 16px;">
                <img src="{b64_overlay}" alt="Manipulation Heatmap" style="max-width: 100%; max-height: 500px; border-radius: 8px; border: 1px solid #475569;" />
            </div>
            <p style="color: #94a3b8; font-size: 0.8rem; margin-top: 8px; text-align: center;">Legend: Red/Warm = most suspicious region · White box = detected facial bounds</p>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Forensic Analysis Report - {filename}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f172a;
            color: #e2e8f0;
            margin: 0;
            padding: 32px 20px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background-color: #1e293b;
            border-radius: 16px;
            border: 1px solid #334155;
            padding: 32px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        }}
        .header {{
            border-bottom: 1px solid #334155;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            color: #f8fafc;
            font-size: 1.8rem;
        }}
        .meta-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            font-size: 0.85rem;
            color: #94a3b8;
        }}
        .meta-val {{
            color: #f1f5f9;
            font-family: monospace;
            word-break: break-all;
        }}
        .verdict-card {{
            background: {bg_color};
            border: 1px solid {badge_color};
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .verdict-title {{
            font-size: 1.6rem;
            font-weight: 800;
            color: {badge_color};
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }}
        .gauge-bar {{
            height: 12px;
            background: #334155;
            border-radius: 6px;
            overflow: hidden;
            position: relative;
            margin: 16px 0 8px 0;
        }}
        .gauge-fill {{
            height: 100%;
            width: {real_pct}%;
            background: linear-gradient(90deg, #22c55e, #14b8a6);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 16px;
        }}
        th {{
            background-color: #0f172a;
            color: #94a3b8;
            font-weight: 600;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #475569;
        }}
        .limits-box {{
            margin-top: 32px;
            padding: 16px 20px;
            background: rgba(245, 158, 11, 0.08);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 10px;
            font-size: 0.85rem;
            color: #fde68a;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ Digital Forensics & Authenticity Report</h1>
            <div class="meta-grid">
                <div><strong>File:</strong> <span class="meta-val">{filename}</span></div>
                <div><strong>Timestamp:</strong> <span class="meta-val">{now_str}</span></div>
                <div><strong>Runtime:</strong> <span class="meta-val">{elapsed}s</span></div>
                <div><strong>SHA-256:</strong> <span class="meta-val">{sha256[:16]}...</span></div>
            </div>
        </div>

        <div class="verdict-card">
            <div class="verdict-title">
                <span>{icon}</span> {verdict}
            </div>
            <div style="font-size: 1rem; color: #f1f5f9; margin-bottom: 12px;">{verdict_note}</div>
            <div style="font-size: 0.9rem; color: #94a3b8;">
                <strong>Evidence Strength:</strong> {evidence_strength}
            </div>
            
            <div class="gauge-bar">
                <div class="gauge-fill"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; font-weight: 600;">
                <span style="color: #22c55e;">Authenticity: {real_pct}% Likely Real</span>
                <span style="color: #ef4444;">{fake_pct}% Fake / Manipulated</span>
            </div>
        </div>

        <div style="margin-bottom: 24px;">
            <h3 style="color: #f8fafc; margin-bottom: 12px;">Why we reached this verdict</h3>
            <ul style="padding-left: 20px; color: #cbd5e1;">
                {top_3_html}
            </ul>
        </div>

        {heatmap_section}

        <div style="margin-top: 32px;">
            <h3 style="color: #f8fafc; margin-bottom: 8px;">Detailed Signal Breakdown</h3>
            <table>
                <thead>
                    <tr>
                        <th>Signal</th>
                        <th>Category</th>
                        <th>Score</th>
                        <th>Weight</th>
                        <th>Findings</th>
                    </tr>
                </thead>
                <tbody>
                    {signals_rows}
                </tbody>
            </table>
        </div>

        <div class="limits-box">
            <strong>⚠️ Limitations & Research Context:</strong>
            <p style="margin: 6px 0 0 0;">
                This system runs locally on independent classical and deep learning forensic cues. 
                Images with heavy compression or resizing may fall into the INCONCLUSIVE band. 
                Absence of manipulation signals is not an absolute certificate of realness. 
                Results represent probabilistic assessments based on empirical benchmarks.
            </p>
        </div>
    </div>
</body>
</html>
"""
    return html

def export_report_file(data: dict, out_path: str = "results/report.html") -> str:
    """Writes HTML report to file and returns absolute path."""
    html = generate_html_report(data)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return os.path.abspath(out_path)
