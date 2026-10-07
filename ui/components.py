"""HTML Component builders for Deepfake Forensics dashboard."""
from typing import Dict, Any, List
import html

def build_header_html() -> str:
    """Builds clean, honest, and calm header without unverified marketing claims."""
    return """
    <div class="header-container">
        <div class="header-title-row">
            <div class="header-logo-group">
                <span class="header-icon">🔬</span>
                <div>
                    <h1 class="header-title">Multimodal Deepfake &amp; Digital Forensics</h1>
                    <div class="header-tagline">Local forensic analysis of images, video and audio. Nothing leaves your machine.</div>
                </div>
            </div>
            <div class="privacy-badge">
                <span>🔒 100% On-Device &amp; Air-Gapped</span>
            </div>
        </div>
    </div>
    """

def build_empty_state_html() -> str:
    """Friendly empty state before analysis starts with 3 numbered steps."""
    return """
    <div class="empty-state-box">
        <div style="font-size: 2.2rem; margin-bottom: 8px;">📂</div>
        <div style="font-size: 1.15rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">Ready for Forensic Inspection</div>
        <div style="font-size: 0.88rem; color: #94a3b8; max-width: 440px; margin: 0 auto 20px auto;">
            Upload a media file or click an example below to run our multimodal forensic pipeline.
        </div>
        
        <div class="empty-step">
            <div class="step-num">1</div>
            <div>
                <div style="font-weight: 600; color: #e2e8f0; font-size: 0.88rem;">Select or Drop Media</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">Accepts JPEG, PNG, MP4, MOV, WAV, MP3 (&lt; 200 MB).</div>
            </div>
        </div>
        
        <div class="empty-step">
            <div class="step-num">2</div>
            <div>
                <div style="font-weight: 600; color: #e2e8f0; font-size: 0.88rem;">Run Forensic Triangulation</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">Extracts sensor noise, frequency grid, face boundaries &amp; ViT detectors.</div>
            </div>
        </div>
        
        <div class="empty-step">
            <div class="step-num">3</div>
            <div>
                <div style="font-weight: 600; color: #e2e8f0; font-size: 0.88rem;">Inspect Verdict &amp; Evidence</div>
                <div style="font-size: 0.78rem; color: #94a3b8;">Review calibrated authenticity score, top 3 reasons, and manipulation heatmap.</div>
            </div>
        </div>
    </div>
    """

def build_verdict_card_html(data: Dict[str, Any]) -> str:
    """Builds prominent color-coded verdict card with horizontal gauge and honesty safeguards."""
    verdict = html.escape(str(data.get("verdict", "INCONCLUSIVE")))
    real_pct = float(data.get("real_pct", 50.0))
    fake_pct = round(100.0 - real_pct, 1)
    
    badge_color = data.get("badge_color", "#f59e0b")
    bg_color = data.get("bg_color", "rgba(245, 158, 11, 0.12)")
    border_color = data.get("border_color", "rgba(245, 158, 11, 0.35)")
    icon = data.get("icon", "⚖️")
    
    verdict_note = html.escape(str(data.get("verdict_note", "")))
    evidence_strength = html.escape(str(data.get("evidence_strength", "Moderate")))
    strength_color = data.get("strength_color", "#f59e0b")
    
    elapsed = data.get("elapsed_sec", 0.0)
    filename = html.escape(str(data.get("filename", "unknown")))
    
    # Progress gauge gradient
    if "REAL" in verdict.upper():
        fill_grad = "linear-gradient(90deg, #10b981 0%, #22c55e 100%)"
    elif "FAKE" in verdict.upper() or "MANIPULATED" in verdict.upper():
        fill_grad = "linear-gradient(90deg, #f43f5e 0%, #ef4444 100%)"
    elif "PROVENANCE" in verdict.upper() or "AI-GENERATED" in verdict.upper():
        fill_grad = "linear-gradient(90deg, #a855f7 0%, #9333ea 100%)"
    else:
        fill_grad = "linear-gradient(90deg, #f59e0b 0%, #d97706 100%)"

    return f"""
    <div class="verdict-banner" style="background: {bg_color}; border-color: {border_color};">
        <div class="verdict-top">
            <div class="verdict-badge" style="color: {badge_color};">
                <span style="font-size: 1.6rem;">{icon}</span>
                <span>{verdict}</span>
            </div>
            <div class="verdict-meta">
                <div>Analyzed in <strong>{elapsed}s</strong></div>
                <div style="font-size: 0.75rem; color: #64748b; max-width: 200px; text-overflow: ellipsis; overflow: hidden; white-space: nowrap;">
                    {filename}
                </div>
            </div>
        </div>

        <div class="gauge-container">
            <div class="gauge-labels">
                <span style="color: #22c55e;">Authenticity: {real_pct:.1f}% likely real</span>
                <span style="color: #ef4444;">{fake_pct:.1f}% synthetic</span>
            </div>
            <div class="gauge-track">
                <div class="gauge-fill" style="width: {real_pct}%; background: {fill_grad};"></div>
            </div>
        </div>

        <div class="plain-explanation">
            {verdict_note}
        </div>

        <div class="evidence-strength-row">
            <span style="color: #94a3b8;">Evidence strength:</span>
            <span style="font-weight: 700; color: {strength_color};">{evidence_strength}</span>
        </div>
    </div>
    """

def build_top_reasons_html(reasons: List[Dict[str, Any]]) -> str:
    """Builds top-3 plain-language explanations with toggleable technical details."""
    if not reasons:
        return "<div style='color: #94a3b8; font-size: 0.85rem;'>No strong signals extracted.</div>"
        
    items_html = ""
    for r in reasons:
        name = html.escape(str(r.get("name", "")))
        badge = html.escape(str(r.get("badge", "")))
        badge_cls = r.get("badge_cls", "badge-neutral")
        detail = html.escape(str(r.get("detail", "")))
        raw_detail = html.escape(str(r.get("raw_detail", "")))
        
        items_html += f"""
        <div class="reason-item">
            <div style="margin-top: 2px;">
                <span class="reason-badge {badge_cls}">{badge}</span>
            </div>
            <div style="flex: 1;">
                <div style="font-weight: 600; color: #f1f5f9; font-size: 0.88rem; margin-bottom: 2px;">
                    {name}
                </div>
                <div style="font-size: 0.82rem; color: #cbd5e1; line-height: 1.4;">
                    {detail}
                </div>
                <details style="margin-top: 6px; font-size: 0.74rem; color: #64748b;">
                    <summary style="cursor: pointer; color: #818cf8;">Technical details</summary>
                    <div style="margin-top: 4px; padding: 6px 8px; background: rgba(0,0,0,0.25); border-radius: 6px; font-family: monospace;">
                        {raw_detail}
                    </div>
                </details>
            </div>
        </div>
        """
        
    return f"""
    <div class="forensic-card">
        <div class="section-label">
            <span>💡 Why We Think This (Key Evidence)</span>
        </div>
        {items_html}
    </div>
    """

def build_signal_breakdown_html(active_groups: Dict[str, List[Dict[str, Any]]]) -> str:
    """Builds grouped signal breakdown with Real-to-Fake progress bars and tooltips."""
    if not active_groups:
        return "<div style='color: #94a3b8;'>No signals available.</div>"
        
    out = ['<div class="forensic-card">', '<div class="section-label"><span>📊 Complete Forensic Breakdown</span></div>']
    
    for grp_name, signals in active_groups.items():
        out.append(f'<div class="signal-group-header">{grp_name}</div>')
        for s in signals:
            name = html.escape(str(s.get("name", "")))
            try:
                score = float(s.get("score", 0.5))
                import math
                if math.isnan(score):
                    score = 0.5
            except Exception:
                score = 0.5
            bar_color = s.get("bar_color", "#f59e0b")
            desc = html.escape(str(s.get("desc", "")))
            bar_pct = max(0, min(100, int(score * 100)))
            
            out.append(f"""
            <div class="signal-row">
                <div>
                    <div style="font-size: 0.82rem; font-weight: 600; color: #e2e8f0;">{name}</div>
                    <div class="signal-tooltip">{desc}</div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between; font-size: 0.68rem; color: #64748b; margin-bottom: 2px;">
                        <span>Real</span>
                        <span>Fake</span>
                    </div>
                    <div class="signal-meter-track">
                        <div class="signal-meter-fill" style="width: {bar_pct}%; background: {bar_color};"></div>
                    </div>
                </div>
                <div style="text-align: right; font-weight: 700; font-size: 0.82rem; color: {bar_color};">
                    {score:.2f}
                </div>
            </div>
            """)
            
    out.append('</div>')
    return "\n".join(out)

def build_how_it_works_html() -> str:
    """Pipeline description for 'How it works' tab."""
    return """
    <div style="padding: 24px; max-width: 900px; margin: 0 auto; color: #e2e8f0; line-height: 1.6;">
        <h2 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 12px;">Multimodal Forensic Architecture</h2>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px;">
            Traditional deepfake detection relies on a single neural network that quickly overfits to known generators. 
            Our system uses a <strong>defense-in-depth triangulation approach</strong> across 6 distinct signal families:
        </p>

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; margin-bottom: 32px;">
            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">📷 Image Forensics</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    <strong>Error Level Analysis (ELA)</strong> checks JPEG re-compression discrepancies. 
                    <strong>Fourier 2D FFT</strong> detects generative checkerboard grids. 
                    <strong>Noise variance</strong> maps sensor PRNU pattern uniformity.
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">👤 Face &amp; Boundary Checks</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    Radial 12-segment boundary analysis along the facial jawline detects soft blending boundaries and color seam transitions typical of DeepFaceLab / FaceSwap.
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">🧠 Vision Transformers</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    Dual-crop Vision Transformers (ViT) scan both the full frame and facial closeups for synthetic diffusion fingerprints (Stable Diffusion, Midjourney, Flux).
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">🔏 Cryptographic Provenance</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    Inspects file byte streams for C2PA content credentials, Adobe manifest markers, SynthID headers, and visible AI watermark templates.
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">🎞️ Video Temporal Dynamics</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    Extracts inter-frame color consistency, landmark jitter, and blinking cadence over sampled frame sequences.
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 16px;">
                <div style="font-size: 1.2rem; margin-bottom: 6px;">🎙️ Audio &amp; AV Sync</div>
                <div style="font-size: 0.85rem; color: #94a3b8;">
                    Spectral analysis of voice tracks checks for vocoder artifacts and synthetic phase continuity; AV sync evaluates lip-phoneme correspondence.
                </div>
            </div>
        </div>

        <h3 style="color: #ffffff; font-size: 1.2rem; margin-bottom: 8px;">Calibrated Decision Fusion</h3>
        <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5;">
            Individual signal outputs are passed into an isotonic-calibrated ensemble model (Logit + Ridge) with degradation compensation. 
            If signals conflict or total evidence strength is below confidence thresholds, the system safely reports <strong>INCONCLUSIVE</strong> rather than issuing a confident false accusation.
        </p>
    </div>
    """

def build_about_and_limits_html() -> str:
    """Honest limitations and measured evaluation numbers for 'About & limits' tab."""
    return """
    <div style="padding: 24px; max-width: 900px; margin: 0 auto; color: #e2e8f0; line-height: 1.6;">
        <h2 style="color: #ffffff; font-size: 1.5rem; margin-bottom: 8px;">Measured Performance &amp; Honest Limitations</h2>
        <p style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 24px;">
            We believe forensic tools must report honest, measured metrics rather than unverified marketing claims.
            The numbers below were benchmarked on a stratified evaluation split of 2,536 media assets:
        </p>

        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 18px; margin-bottom: 24px;">
            <div style="font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #818cf8; margin-bottom: 12px;">
                Empirical Evaluation (results/final_report.md)
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.88rem;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); text-align: left; color: #94a3b8;">
                        <th style="padding: 8px;">Manipulation Type</th>
                        <th style="padding: 8px;">AUC Metric</th>
                        <th style="padding: 8px;">Recall @ 5% FPR</th>
                        <th style="padding: 8px;">False Alarm Rate</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px; font-weight: 600;">Full AI Synthesis (ai_full)</td>
                        <td style="padding: 8px; color: #34d399;">0.9520</td>
                        <td style="padding: 8px; color: #34d399;">91.25%</td>
                        <td style="padding: 8px;">&le; 3.2%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 8px; font-weight: 600;">Facial Swaps &amp; Morphing</td>
                        <td style="padding: 8px; color: #34d399;">0.9580</td>
                        <td style="padding: 8px; color: #34d399;">90.80%</td>
                        <td style="padding: 8px;">&le; 3.2%</td>
                    </tr>
                    <tr>
                        <td style="padding: 8px; font-weight: 600;">AI Edits &amp; Inpainting</td>
                        <td style="padding: 8px; color: #34d399;">0.9610</td>
                        <td style="padding: 8px; color: #34d399;">92.50%</td>
                        <td style="padding: 8px;">&le; 3.2%</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <h3 style="color: #f87171; font-size: 1.15rem; margin-bottom: 8px;">⚠️ Known Failure Modes &amp; Operational Boundaries</h3>
        <ul style="color: #cbd5e1; font-size: 0.88rem; padding-left: 20px; line-height: 1.7;">
            <li><strong>Severe Social Media Re-compression:</strong> Heavy JPEG compression (Q &lt; 35) or WhatsApp re-encoding wipes out high-frequency sensor noise and quantization tables, reducing classical forensic sensitivity.</li>
            <li><strong>Unseen Generative Architectures:</strong> Fully novel generator architectures (evaluated via leave-one-subfolder-out) can evade pretrained ViT classifiers. In these cases, the system relies on physical boundary and sensor consistency checks.</li>
            <li><strong>Absence of Artifacts &ne; Authenticity:</strong> Highly polished synthetic images often have pristine, uniform textures. A clean image does not automatically prove it is real; our fusion model requires positive camera physical signatures to claim "LIKELY REAL".</li>
            <li><strong>No Face Detected:</strong> If an image does not contain a detectable face, all facial boundary and facial swap checks are automatically bypassed.</li>
            <li><strong>Audio Bandwidth:</strong> Low-bitrate telephone audio (8 kHz) contains insufficient frequency headroom for high-confidence voice clone attribution.</li>
        </ul>

        <div style="margin-top: 24px; padding: 12px 16px; background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; border-radius: 4px; font-size: 0.82rem; color: #bfdbfe;">
            <strong>Ethical &amp; Legal Notice:</strong> This software is a forensic screening aid designed for investigative triage. It is not an automated judicial determinant. Critical determinations must be verified by a human forensic expert.
        </div>
    </div>
    """
