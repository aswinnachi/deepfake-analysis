"""Gradio web UI:  python app.py

Premium forensics dashboard for the hackathon demo.
"""
import gradio as gr
import numpy as np
import json
import os
from forensics.pipeline import analyze, to_markdown

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: #0a0a0f !important;
}

#header-block {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 28px 32px !important;
    margin-bottom: 16px !important;
    text-align: center !important;
}

#header-block h1 {
    background: linear-gradient(90deg, #00d2ff, #7b2ff7, #ff6b6b) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    margin-bottom: 4px !important;
}

#header-block p {
    color: rgba(255,255,255,0.55) !important;
    font-size: 0.95rem !important;
    font-weight: 400 !important;
}

.upload-area {
    border: 2px dashed rgba(123, 47, 247, 0.3) !important;
    border-radius: 16px !important;
    background: rgba(123, 47, 247, 0.04) !important;
    transition: all 0.3s ease !important;
}
.upload-area:hover {
    border-color: rgba(123, 47, 247, 0.6) !important;
    background: rgba(123, 47, 247, 0.08) !important;
}

#analyze-btn {
    background: linear-gradient(135deg, #7b2ff7 0%, #00d2ff 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(123, 47, 247, 0.35) !important;
}
#analyze-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(123, 47, 247, 0.5) !important;
}

#verdict-block {
    border-radius: 16px !important;
    padding: 20px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}

#heatmap-img {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

#signals-json {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: rgba(255,255,255,0.02) !important;
}

.result-card {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

footer { display: none !important; }
"""


def build_verdict_html(res):
    """Create a rich HTML verdict card from the analysis result."""
    verdict = res["verdict"]
    real_pct = res["authenticity_real_pct"]
    fake_pct = round(100 - real_pct, 1)
    confidence = res["confidence"]
    mode = res.get("mode", "heuristic fusion")
    degradation = res.get("degradation", 0)

    # Colour based on verdict
    if "REAL" in verdict:
        accent = "#22c55e"
        glow = "rgba(34,197,94,0.15)"
        icon = "🛡️"
        bar_color = "#22c55e"
    elif "FAKE" in verdict:
        accent = "#ef4444"
        glow = "rgba(239,68,68,0.15)"
        icon = "⚠️"
        bar_color = "#ef4444"
    else:
        accent = "#f59e0b"
        glow = "rgba(245,158,11,0.15)"
        icon = "🔍"
        bar_color = "#f59e0b"

    # Build signal bars
    signal_bars = ""
    for sig in res.get("signals", []):
        s = sig["fake_score"]
        name = sig["name"].replace("_", " ").title()
        w = sig["weight"]
        if s > 0.6:
            sc = "#ef4444"
            label = "FAKE-leaning"
        elif s < 0.4:
            sc = "#22c55e"
            label = "REAL-leaning"
        else:
            sc = "#f59e0b"
            label = "Neutral"
        bar_w = int(s * 100)
        signal_bars += f"""
        <div style="margin-bottom:10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;">
            <span style="font-weight:600;font-size:0.82rem;color:#e2e8f0;">{name}</span>
            <span style="font-size:0.72rem;color:{sc};font-weight:600;">{label} ({s:.2f}) · w={w}</span>
          </div>
          <div style="height:6px;background:rgba(255,255,255,0.06);border-radius:3px;overflow:hidden;">
            <div style="height:100%;width:{bar_w}%;background:{sc};border-radius:3px;transition:width 0.6s ease;"></div>
          </div>
        </div>"""

    # Build evidence list
    evidence = ""
    for line in res.get("explanation", []):
        if line.startswith("[FAKE"):
            dot = "#ef4444"
        elif line.startswith("[REAL"):
            dot = "#22c55e"
        elif line.startswith("[neutral"):
            dot = "#f59e0b"
        else:
            dot = "#94a3b8"
        evidence += f"""<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:6px;">
          <div style="min-width:6px;height:6px;border-radius:50%;background:{dot};margin-top:6px;"></div>
          <span style="font-size:0.78rem;color:#cbd5e1;line-height:1.4;">{line}</span>
        </div>"""

    html = f"""
    <div style="font-family:'Inter',sans-serif;">
      <!-- Verdict Header -->
      <div style="background:{glow};border:1px solid {accent}33;border-radius:14px;padding:20px 24px;margin-bottom:16px;">
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
          <span style="font-size:1.6rem;">{icon}</span>
          <div>
            <div style="font-size:1.3rem;font-weight:800;color:{accent};letter-spacing:-0.3px;">{verdict}</div>
            <div style="font-size:0.82rem;color:rgba(255,255,255,0.5);margin-top:2px;">
              via {mode} · degradation {degradation:.0%} · confidence: {confidence}
            </div>
          </div>
        </div>
        <!-- Score bar -->
        <div style="display:flex;align-items:center;gap:12px;">
          <div style="flex:1;">
            <div style="display:flex;justify-content:space-between;font-size:0.75rem;margin-bottom:4px;">
              <span style="color:#22c55e;font-weight:600;">REAL {real_pct}%</span>
              <span style="color:#ef4444;font-weight:600;">FAKE {fake_pct}%</span>
            </div>
            <div style="height:10px;background:rgba(255,255,255,0.06);border-radius:5px;overflow:hidden;position:relative;">
              <div style="position:absolute;left:0;top:0;height:100%;width:{real_pct}%;background:linear-gradient(90deg,#22c55e,{bar_color});border-radius:5px;transition:width 0.8s ease;"></div>
            </div>
          </div>
          <div style="font-size:1.5rem;font-weight:800;color:{accent};min-width:60px;text-align:right;">{real_pct}%</div>
        </div>
      </div>

      <!-- Signal Breakdown -->
      <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:18px 20px;margin-bottom:16px;">
        <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:14px;letter-spacing:0.3px;">
          📊 SIGNAL BREAKDOWN
        </div>
        {signal_bars}
      </div>

      <!-- Evidence -->
      <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:18px 20px;">
        <div style="font-size:0.88rem;font-weight:700;color:#e2e8f0;margin-bottom:12px;letter-spacing:0.3px;">
          🔎 EVIDENCE (strongest first)
        </div>
        {evidence}
      </div>
    </div>
    """
    return html


def run(file_obj, progress=gr.Progress()):
    if file_obj is None:
        return None, "<div style='text-align:center;color:#94a3b8;padding:40px;'>Upload an image, video, or audio file to begin analysis.</div>", {}

    path = file_obj  # Gradio File with type="filepath" gives us a path string
    progress(0.1, desc="Loading file...")

    try:
        progress(0.3, desc="Running forensic signals...")
        res = analyze(path)
    except Exception as e:
        return None, f"<div style='color:#ef4444;padding:20px;'>❌ Error: {e}</div>", {}

    progress(0.8, desc="Building report...")
    verdict_html = build_verdict_html(res)

    # Build compact JSON for the raw panel
    raw = {
        "authenticity_real_pct": res["authenticity_real_pct"],
        "verdict": res["verdict"],
        "confidence": res["confidence"],
        "mode": res.get("mode", ""),
        "degradation": res.get("degradation", 0),
        "signals": res["signals"],
    }

    progress(1.0, desc="Done!")
    return res.get("overlay"), verdict_html, raw


def build_ui():
    with gr.Blocks(
        title="Multimodal Deepfake Forensics",
        css=CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0a0a0f",
            body_background_fill_dark="#0a0a0f",
            block_background_fill="rgba(255,255,255,0.02)",
            block_background_fill_dark="rgba(255,255,255,0.02)",
            block_border_color="rgba(255,255,255,0.06)",
            block_border_color_dark="rgba(255,255,255,0.06)",
            block_label_text_color="#94a3b8",
            block_label_text_color_dark="#94a3b8",
            block_title_text_color="#e2e8f0",
            block_title_text_color_dark="#e2e8f0",
            body_text_color="#e2e8f0",
            body_text_color_dark="#e2e8f0",
            input_background_fill="rgba(255,255,255,0.04)",
            input_background_fill_dark="rgba(255,255,255,0.04)",
        ),
    ) as demo:
        # Header
        gr.HTML("""
        <div id="header-block">
            <h1>Multimodal Deepfake &amp; Digital Forensics</h1>
            <p>Upload any image, video, or audio — get an authenticity score, evidence, and manipulation heatmap.<br/>
            10 independent forensic signals fused · No single artefact decides · Generalises to unseen techniques</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.File(
                    label="📁 Drop your file here",
                    type="filepath",
                    file_types=["image", "video", "audio"],
                    elem_classes=["upload-area"],
                )
                btn = gr.Button(
                    "🔬  Analyze",
                    variant="primary",
                    elem_id="analyze-btn",
                    size="lg",
                )
                img = gr.Image(
                    label="🗺️ Manipulation Heatmap",
                    elem_id="heatmap-img",
                    show_label=True,
                )

            with gr.Column(scale=1):
                verdict_html = gr.HTML(
                    value="<div style='text-align:center;color:#94a3b8;padding:60px 20px;font-size:0.95rem;'>Upload a file and click <b>Analyze</b> to start forensic analysis.</div>",
                    elem_id="verdict-block",
                )

        with gr.Accordion("📦 Raw Signal Data (JSON)", open=False):
            js = gr.JSON(label="Raw signals", elem_id="signals-json")
            
        with gr.Accordion("🛠️ God Mode (Train Model)", open=False):
            gr.HTML("""
                <div style="font-size:0.9rem; color:#94a3b8; margin-bottom:12px;">
                Is the model wrong? Train the internal CNN directly on the uploaded image to fix it instantly!
                </div>
            """)
            with gr.Row():
                train_real_btn = gr.Button("✅ Train as REAL", variant="secondary")
                train_fake_btn = gr.Button("🚨 Train as FAKE", variant="primary")
            train_status = gr.Markdown("Status: Ready to train.")

        # Footer info
        gr.HTML("""
        <div style="text-align:center;padding:16px;margin-top:8px;">
            <p style="color:rgba(255,255,255,0.25);font-size:0.72rem;letter-spacing:0.5px;">
                HNX26PSI10 · ELA · Fourier · Noise · Face · Metadata · CNN · Temporal · Audio · AV-Sync · Compression
            </p>
        </div>
        """)

        def handle_train_real(path):
            if not path:
                return "Upload an image first!"
            from forensics.god_mode import train_image
            return train_image(path, 0.0)

        def handle_train_fake(path):
            if not path:
                return "Upload an image first!"
            from forensics.god_mode import train_image
            return train_image(path, 1.0)

        btn.click(run, inp, [img, verdict_html, js])
        train_real_btn.click(handle_train_real, inp, train_status)
        train_fake_btn.click(handle_train_fake, inp, train_status)
    
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", share=True)
