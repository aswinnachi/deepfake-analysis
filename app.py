"""Gradio web UI for Multimodal Deepfake & Digital Forensics.

Run:
    python app.py            # Standard judge / hackathon demo mode
    python app.py --admin    # Includes God Mode internal training panel
"""
import sys
import os
import time
import json
import argparse
import traceback
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

import gradio as gr

# UI helpers
from ui.styles import APP_CSS
from ui.adapter import (
    validate_file,
    analyze_cached,
    start_warmup,
    is_models_ready,
    get_file_sha256,
    IMAGE_EXTS,
    VIDEO_EXTS,
    AUDIO_EXTS
)
from ui.report import generate_html_report
from ui.components import (
    build_header_html,
    build_empty_state_html,
    build_verdict_card_html,
    build_top_reasons_html,
    build_signal_breakdown_html,
    build_how_it_works_html,
    build_about_and_limits_html
)

# Start background model preloading immediately
start_warmup()

# Ensure reports directory exists
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

DEMO_DIR = os.path.join(os.path.dirname(__file__), "samples", "demo")

# Store state for opacity slider: sha256 -> (orig_rgb, overlay_rgb)
_IMAGE_STORE: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}


def blend_images(orig: Optional[np.ndarray], overlay: Optional[np.ndarray], alpha: float) -> Optional[np.ndarray]:
    """Blends original and overlay images dynamically based on opacity slider."""
    if overlay is None:
        return orig
    if orig is None:
        return overlay
    try:
        # Match dimensions if needed
        if orig.shape[:2] != overlay.shape[:2]:
            from PIL import Image as PILImage
            ov_pil = PILImage.fromarray(overlay.astype(np.uint8)).resize((orig.shape[1], orig.shape[0]))
            overlay = np.array(ov_pil)
            
        blended = (1.0 - alpha) * orig.astype(np.float32) + alpha * overlay.astype(np.float32)
        return np.clip(blended, 0, 255).astype(np.uint8)
    except Exception:
        return overlay


def sanitize_for_json(obj: Any) -> Any:
    """Recursively converts ndarrays and special types to JSON-safe representations."""
    if isinstance(obj, np.ndarray):
        return f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
    elif hasattr(obj, "to_dict") and callable(obj.to_dict):
        return sanitize_for_json(obj.to_dict())
    elif hasattr(obj, "__dict__"):
        d = {k: v for k, v in obj.__dict__.items() if k != "overlay" and k != "heatmap"}
        return sanitize_for_json(d)
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items() if k != "overlay" and k != "heatmap" and k != "_signals"}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def run_analysis(file_path: Optional[str], progress=gr.Progress()):
    """Executes forensic pipeline with friendly errors, progress reporting, and caching."""
    if not file_path:
        return (
            None, # orig_img
            None, # heatmap_img
            "<div style='text-align: center; color: #94a3b8; padding: 40px;'>Please upload an image, video, or audio file to begin analysis.</div>", # verdict_html
            "", # top_reasons_html
            "", # signal_breakdown_html
            "", # legend_and_caption_html
            gr.update(visible=False), # opacity_slider
            gr.update(visible=False), # all_evidence_acc
            "", # all_evidence_text
            gr.update(visible=False), # json_acc
            {}, # raw_json
            gr.update(visible=False), # download_html_btn
            gr.update(visible=False), # download_json_btn
            "" # active_sha state
        )

    # 1. Validation
    progress(0.05, desc="Validating file format & integrity...")
    valid, err_msg = validate_file(file_path)
    if not valid:
        error_card = f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 14px; padding: 24px; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">⚠️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f87171; margin-bottom: 6px;">Unable to Process File</div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">{err_msg}</div>
        </div>
        """
        return (
            None, None, error_card, "", "", "",
            gr.update(visible=False), gr.update(visible=False), "",
            gr.update(visible=False), {"error": err_msg},
            gr.update(visible=False), gr.update(visible=False), ""
        )

    # 2. Execution with step-by-step progress
    try:
        progress(0.2, desc="Finding faces & analyzing boundary contours...")
        time.sleep(0.05)
        progress(0.4, desc="Running forensic checks (ELA, FFT, Noise PRNU)...")
        time.sleep(0.05)
        progress(0.65, desc="Running AI detectors & ViT models...")
        time.sleep(0.05)
        progress(0.85, desc="Building heatmap & calibrated fusion...")
        
        adapted = analyze_cached(file_path)
        progress(1.0, desc="Forensic analysis complete!")
    except Exception as e:
        # Honest error without exposing raw tracebacks in UI
        traceback.print_exc()
        error_card = f"""
        <div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 14px; padding: 24px; text-align: center;">
            <div style="font-size: 2rem; margin-bottom: 8px;">⚠️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f87171; margin-bottom: 6px;">Analysis Interrupted</div>
            <div style="color: #cbd5e1; font-size: 0.9rem; max-width: 500px; margin: 0 auto;">
                The system encountered an unexpected internal issue while processing media tracks.
                Please ensure the file is not corrupted or try a different sample.
            </div>
        </div>
        """
        return (
            None, None, error_card, "", "", "",
            gr.update(visible=False), gr.update(visible=False), "",
            gr.update(visible=False), {"error": str(e)},
            gr.update(visible=False), gr.update(visible=False), ""
        )

    # 3. Prepare visual outputs
    ext = os.path.splitext(file_path)[1].lower()
    orig_img = None
    if ext in IMAGE_EXTS:
        try:
            with Image.open(file_path) as im:
                orig_img = np.array(im.convert("RGB"))
        except Exception:
            orig_img = None

    overlay_img = adapted.get("overlay")
    file_sha = adapted.get("file_sha", get_file_sha256(file_path))
    _IMAGE_STORE[file_sha] = (orig_img, overlay_img)

    # 4. Verdict & cards
    verdict_html = build_verdict_card_html(adapted)
    top_reasons_html = build_top_reasons_html(adapted.get("top_3_reasons", []))
    signals_html = build_signal_breakdown_html(adapted.get("active_groups", {}))
    
    # Legend & localization caption
    region_caption = adapted.get("region_caption", "No single region stands out")
    legend_html = f"""
    <div style="margin-top: 8px; padding: 8px 12px; background: rgba(255,255,255,0.02); border-radius: 8px; font-size: 0.8rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; border: 1px solid rgba(255,255,255,0.05);">
        <div><strong>Legend:</strong> <span style="color: #ef4444; font-weight: 600;">■ Red = most suspicious region</span> · <span style="color: #f8fafc; border: 1px solid #94a3b8; padding: 0 4px; border-radius: 2px;">□ White box = detected face</span></div>
        <div style="color: #e2e8f0;"><strong>Region focus:</strong> {region_caption}</div>
    </div>
    """

    # All evidence list
    all_evidence_lines = adapted.get("all_evidence", [])
    all_evidence_str = "\n".join([f"• {line}" for line in all_evidence_lines])

    # Export report files
    html_report_path = os.path.join(REPORTS_DIR, f"report_{file_sha[:10]}.html")
    html_content = generate_html_report(adapted)
    with open(html_report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    json_safe_raw = sanitize_for_json(adapted.get("raw", {}))
    json_report_path = os.path.join(REPORTS_DIR, f"report_{file_sha[:10]}.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(json_safe_raw, f, indent=2)

    has_overlay = overlay_img is not None
    slider_update = gr.update(value=1.0, visible=has_overlay)

    return (
        orig_img,
        overlay_img,
        verdict_html,
        top_reasons_html,
        signals_html,
        legend_html,
        slider_update,
        gr.update(visible=True), # all_evidence_acc
        all_evidence_str,
        gr.update(visible=True), # json_acc
        json_safe_raw,
        gr.update(value=html_report_path, visible=True),
        gr.update(value=json_report_path, visible=True),
        file_sha
    )


def handle_opacity_change(alpha: float, file_sha: str):
    """Dynamically recalculates blended overlay without re-analyzing."""
    if not file_sha or file_sha not in _IMAGE_STORE:
        return None
    orig_img, overlay_img = _IMAGE_STORE[file_sha]
    return blend_images(orig_img, overlay_img, float(alpha))


def build_app(enable_admin: bool = False):
    """Builds the judge-ready Gradio interface."""
    with gr.Blocks(
        title="Multimodal Deepfake & Digital Forensics",
        css=APP_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.indigo,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#090d16",
            body_background_fill_dark="#090d16",
            block_background_fill="rgba(22, 30, 49, 0.7)",
            block_background_fill_dark="rgba(22, 30, 49, 0.7)",
            block_border_color="rgba(255, 255, 255, 0.08)",
            block_border_color_dark="rgba(255, 255, 255, 0.08)",
            block_label_text_color="#94a3b8",
            block_label_text_color_dark="#94a3b8",
            body_text_color="#f8fafc",
            body_text_color_dark="#f8fafc",
            input_background_fill="rgba(255, 255, 255, 0.04)",
            input_background_fill_dark="rgba(255, 255, 255, 0.04)",
        )
    ) as demo:

        # Active file sha state
        active_sha = gr.State("")

        # 1. Slim Header
        gr.HTML(build_header_html())

        # 2. Main Tabs
        with gr.Tabs():
            with gr.Tab("🔬 Analyze"):
                with gr.Row():
                    # Left Column: Upload & Controls
                    with gr.Column(scale=5):
                        inp_file = gr.File(
                            label="📁 Upload Media (Image, Video, Audio)",
                            type="filepath",
                            file_types=["image", "video", "audio"],
                            elem_classes=["upload-area"]
                        )
                        
                        gr.HTML("""
                        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b; margin-top: -6px; margin-bottom: 12px; padding: 0 4px;">
                            <span>Accepted: JPG, PNG, WEBP, MP4, MOV, WAV, MP3</span>
                            <span>Max size: 200 MB</span>
                        </div>
                        """)

                        analyze_btn = gr.Button(
                            "🔬  Run Forensic Analysis",
                            variant="primary",
                            elem_classes=["analyze-button-primary"],
                            size="lg"
                        )

                        # Demo Examples Row
                        if os.path.exists(DEMO_DIR):
                            demo_files = sorted(os.listdir(DEMO_DIR))
                            # Filter files with valid extensions
                            valid_demos = [
                                f for f in demo_files 
                                if os.path.splitext(f)[1].lower() in (IMAGE_EXTS | VIDEO_EXTS | AUDIO_EXTS)
                            ]
                            
                            if valid_demos:
                                gr.HTML("""
                                <div style="margin-top: 20px; margin-bottom: 8px; font-size: 0.82rem; font-weight: 600; color: #94a3b8; display: flex; align-items: center; gap: 6px;">
                                    <span>💡 Try an example:</span>
                                </div>
                                """)
                                
                                # Render sample buttons in small grids
                                with gr.Row():
                                    sample_btns = []
                                    # Pick the top key demonstration files
                                    priority_names = ["real_portrait.jpg", "woman.jpg", "fake_faceswap.jpg", "guy.png", "sample_video.mp4", "sample_audio.wav"]
                                    ordered_demos = [f for f in priority_names if f in valid_demos]
                                    for f in valid_demos:
                                        if f not in ordered_demos and len(ordered_demos) < 8:
                                            ordered_demos.append(f)

                                    for f in ordered_demos:
                                        btn_lbl = f
                                        if f == "woman.jpg":
                                            btn_lbl = "woman.jpg (AI)"
                                        elif f == "real_portrait.jpg":
                                            btn_lbl = "portrait.jpg (Real)"
                                        elif f == "fake_faceswap.jpg":
                                            btn_lbl = "faceswap.jpg"
                                        elif f == "guy.png":
                                            btn_lbl = "guy.png (Cutout)"
                                            
                                        s_btn = gr.Button(btn_lbl, size="sm", elem_classes=["sample-pill-btn"])
                                        sample_btns.append((s_btn, os.path.join(DEMO_DIR, f)))

                        # Export Report Buttons
                        gr.HTML("<div style='margin-top: 24px;'></div>")
                        with gr.Row():
                            dl_report_btn = gr.DownloadButton(
                                "📄 Download Report (HTML)",
                                variant="secondary",
                                size="sm",
                                visible=False
                            )
                            dl_json_btn = gr.DownloadButton(
                                "📦 Export Signals (JSON)",
                                variant="secondary",
                                size="sm",
                                visible=False
                            )

                    # Right Column: Results
                    with gr.Column(scale=7):
                        verdict_card = gr.HTML(build_empty_state_html())
                        top_reasons = gr.HTML("")

                        # Visual Heatmap & Original side-by-side
                        with gr.Row(elem_id="preview-row"):
                            orig_preview = gr.Image(
                                label="Original Frame",
                                show_label=True,
                                interactive=False
                            )
                            heatmap_preview = gr.Image(
                                label="Manipulation Heatmap",
                                show_label=True,
                                interactive=False
                            )

                        legend_and_caption = gr.HTML("")

                        opacity_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            label="Heatmap Overlay Opacity",
                            visible=False
                        )

                        signal_breakdown = gr.HTML("")

                        with gr.Accordion("📋 All Forensic Evidence (Raw Strings)", open=False, visible=False) as all_evidence_acc:
                            all_evidence_view = gr.Markdown("")

                        with gr.Accordion("📦 Raw Signal Data (JSON)", open=False, visible=False) as json_acc:
                            raw_json_view = gr.JSON(label="Pipeline Output")

            with gr.Tab("📖 How it works"):
                gr.HTML(build_how_it_works_html())

            with gr.Tab("📊 About & limits"):
                gr.HTML(build_about_and_limits_html())

        # Admin God Mode (Visible ONLY when started with --admin)
        if enable_admin:
            with gr.Accordion("🛠️ God Mode (Train Model) [Admin Only]", open=False):
                gr.HTML("""
                    <div style="font-size: 0.88rem; color: #94a3b8; margin-bottom: 12px;">
                    Administrator fine-tuning override: Fine-tunes the local ResNet patch classifier on the uploaded image.
                    </div>
                """)
                with gr.Row():
                    train_real_btn = gr.Button("✅ Train as REAL", variant="secondary")
                    train_fake_btn = gr.Button("🚨 Train as FAKE", variant="primary")
                train_status = gr.Markdown("Status: Ready to train.")

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

                train_real_btn.click(handle_train_real, inp_file, train_status)
                train_fake_btn.click(handle_train_fake, inp_file, train_status)

        # Footer Status
        gr.HTML(f"""
        <div class="app-footer">
            <div>
                <span class="status-dot"></span>
                <span>Forensic Engine Ready · Calibrated Fusion Active</span>
            </div>
            <div>
                <span>Defense-in-depth forensics · Defense against single-model bias</span>
            </div>
        </div>
        """)

        # Event Handlers
        analyze_outputs = [
            orig_preview,
            heatmap_preview,
            verdict_card,
            top_reasons,
            signal_breakdown,
            legend_and_caption,
            opacity_slider,
            all_evidence_acc,
            all_evidence_view,
            json_acc,
            raw_json_view,
            dl_report_btn,
            dl_json_btn,
            active_sha
        ]

        analyze_btn.click(
            run_analysis,
            inputs=[inp_file],
            outputs=analyze_outputs
        )

        # Connect opacity slider
        opacity_slider.change(
            handle_opacity_change,
            inputs=[opacity_slider, active_sha],
            outputs=[heatmap_preview]
        )

        # Connect example buttons
        if os.path.exists(DEMO_DIR) and 'sample_btns' in locals():
            for btn, sample_path in sample_btns:
                # Clicking sample sets input file and executes analysis
                btn.click(
                    lambda p=sample_path: p,
                    inputs=[],
                    outputs=[inp_file]
                ).then(
                    run_analysis,
                    inputs=[inp_file],
                    outputs=analyze_outputs
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Multimodal Deepfake & Digital Forensics Web UI")
    parser.add_argument("--admin", action="store_true", help="Enable internal God Mode training panel")
    parser.add_argument("--port", type=int, default=7860, help="Server port (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    args, _ = parser.parse_known_args()

    demo = build_app(enable_admin=args.admin)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
