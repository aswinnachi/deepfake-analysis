"""Design system and CSS styles for Deepfake Forensics dashboard."""

APP_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-dark: #090d16;
    --card-bg: rgba(22, 30, 49, 0.7);
    --border-color: rgba(255, 255, 255, 0.08);
    --border-hover: rgba(99, 102, 241, 0.3);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --primary-blue: #3b82f6;
    --primary-purple: #8b5cf6;
    --accent-teal: #14b8a6;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
}

* { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; 
    box-sizing: border-box;
}

code, pre, .font-mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: var(--bg-dark) !important;
    padding: 16px 24px !important;
}

/* Header Styling */
.header-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 20px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px -1px rgba(0, 0, 0, 0.3);
}

.header-title-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}

.header-logo-group {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-icon {
    font-size: 1.8rem;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    padding: 6px 12px;
    border-radius: 10px;
}

.header-title {
    font-size: 1.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
}

.header-tagline {
    font-size: 0.92rem;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 400;
}

.privacy-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(20, 184, 166, 0.1);
    color: #2dd4bf;
    border: 1px solid rgba(20, 184, 166, 0.25);
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Primary Button */
.analyze-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%) !important;
    border: none !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.02rem !important;
    padding: 12px 28px !important;
    border-radius: 12px !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(79, 70, 229, 0.35) !important;
    cursor: pointer !important;
}

.analyze-button-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(79, 70, 229, 0.5) !important;
}

/* Sample pills */
.sample-pill-btn {
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 4px 10px !important;
    background: rgba(255, 255, 255, 0.04) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    color: #cbd5e1 !important;
    transition: all 0.2s ease !important;
}

.sample-pill-btn:hover {
    background: rgba(99, 102, 241, 0.15) !important;
    border-color: rgba(99, 102, 241, 0.35) !important;
    color: #ffffff !important;
}

/* Cards & Containers */
.forensic-card {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    backdrop-filter: blur(8px);
}

.section-label {
    font-size: 0.8rem;
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Verdict Block */
.verdict-banner {
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}

.verdict-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}

.verdict-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-size: 1.45rem;
    font-weight: 800;
    letter-spacing: -0.01em;
}

.verdict-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    text-align: right;
}

.gauge-container {
    margin: 14px 0 10px 0;
}

.gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 6px;
}

.gauge-track {
    height: 12px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    overflow: hidden;
    position: relative;
}

.gauge-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}

.plain-explanation {
    font-size: 0.92rem;
    color: #e2e8f0;
    line-height: 1.45;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

.evidence-strength-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 0.82rem;
}

/* Reason List */
.reason-item {
    background: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}

.reason-badge {
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
}

.badge-fake {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-real {
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.badge-neutral {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* Signal category breakdown */
.signal-group-header {
    font-size: 0.85rem;
    font-weight: 700;
    color: #cbd5e1;
    margin-top: 14px;
    margin-bottom: 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.signal-row {
    display: grid;
    grid-template-columns: 180px 1fr 60px;
    align-items: center;
    gap: 12px;
    padding: 6px 0;
}

@media (max-width: 768px) {
    .signal-row {
        grid-template-columns: 140px 1fr 50px;
        gap: 8px;
    }
}

.signal-meter-track {
    height: 7px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}

.signal-meter-fill {
    height: 100%;
    border-radius: 4px;
}

/* Tooltip text */
.signal-tooltip {
    font-size: 0.73rem;
    color: #94a3b8;
    margin-top: 2px;
}

/* Empty State */
.empty-state-box {
    text-align: center;
    padding: 48px 24px;
    background: rgba(255, 255, 255, 0.015);
    border: 1px dashed rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    color: #94a3b8;
}

.empty-step {
    display: flex;
    align-items: center;
    gap: 12px;
    text-align: left;
    margin: 14px auto;
    max-width: 420px;
    background: rgba(255, 255, 255, 0.03);
    padding: 12px 16px;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.step-num {
    background: #4f46e5;
    color: #ffffff;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
    flex-shrink: 0;
}

/* Footer status */
.app-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 8px;
    margin-top: 24px;
    border-top: 1px solid var(--border-color);
    font-size: 0.8rem;
    color: #64748b;
}

.status-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #10b981;
    margin-right: 6px;
    box-shadow: 0 0 8px #10b981;
}

/* Hide default Gradio footer */
footer { display: none !important; }
"""
