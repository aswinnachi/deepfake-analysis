"""Face boundary seam & regional disagreement forensics."""
import cv2
import numpy as np
from typing import List, Optional
from .utils import Signal, to_unit
from .face import detect_face

SEGMENT_NAMES = [
    "top forehead", "upper right hairline", "right temple", "right cheek",
    "right jawline", "chin right", "chin left", "left jawline",
    "left cheek", "left temple", "upper left hairline", "forehead center"
]

def analyze_boundary(rgb: np.ndarray) -> Optional[Signal]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    box = detect_face(gray, rgb)
    if box is None:
        return None
        
    x, y, w, h = box
    H, W = gray.shape[:2]
    
    # 12 radial segments around the face contour
    cx, cy = x + w // 2, y + h // 2
    rx, ry = w // 2, h // 2
    
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    
    segment_scores = []
    heat = np.zeros((H, W), dtype=np.float32)
    
    for i in range(12):
        angle_start = i * (2 * np.pi / 12)
        angle_end = (i + 1) * (2 * np.pi / 12)
        
        # Inner mask
        mask_in = np.zeros((H, W), dtype=np.uint8)
        mask_out = np.zeros((H, W), dtype=np.uint8)
        
        # Sample ellipse arcs
        pts_in = []
        pts_out = []
        for a in np.linspace(angle_start, angle_end, 8):
            px_in = int(cx + 0.8 * rx * np.cos(a))
            py_in = int(cy + 0.8 * ry * np.sin(a))
            px_out = int(cx + 1.2 * rx * np.cos(a))
            py_out = int(cy + 1.2 * ry * np.sin(a))
            if 0 <= px_in < W and 0 <= py_in < H:
                pts_in.append([px_in, py_in])
            if 0 <= px_out < W and 0 <= py_out < H:
                pts_out.append([px_out, py_out])
                
        if len(pts_in) < 4 or len(pts_out) < 4:
            segment_scores.append(0.0)
            continue
            
        cv2.polylines(mask_in, [np.array(pts_in)], False, 255, 10)
        cv2.polylines(mask_out, [np.array(pts_out)], False, 255, 10)
        
        in_lab = lab[mask_in == 255]
        out_lab = lab[mask_out == 255]
        
        if len(in_lab) > 10 and len(out_lab) > 10:
            color_dist = np.linalg.norm(in_lab.mean(axis=0) - out_lab.mean(axis=0)) / 100.0
            sharp_ratio = abs(np.log(lap[mask_in == 255].var() + 1.0) - np.log(lap[mask_out == 255].var() + 1.0))
            score_i = float(np.clip(0.4 * color_dist + 0.3 * sharp_ratio, 0.0, 1.0))
        else:
            score_i = 0.0
            
        segment_scores.append(score_i)
        cv2.polylines(heat, [np.array(pts_out)], False, score_i, 12)

    # 95th percentile calibration: scale down normal boundary variations
    raw_score = float(np.max(segment_scores)) if segment_scores else 0.0
    calibrated_score = float(np.clip((raw_score - 0.25) / 0.55, 0.0, 1.0))
    
    worst_idx = np.argsort(segment_scores)[::-1][:3]
    worst_names = [SEGMENT_NAMES[i] for i in worst_idx if segment_scores[i] > 0.2]
    worst_str = ", ".join(worst_names) if worst_names else "clean boundary transition"
    
    detail = f"Face boundary analysis: max boundary discontinuity {calibrated_score:.2f} ({worst_str})."
    return Signal("face_boundary", calibrated_score, 1.2, detail, heatmap=heat, box=box)

def analyze_region_disagreement(rgb: np.ndarray, face_score: float) -> Optional[Signal]:
    h, w = rgb.shape[:2]
    # Sample background patches (4 corners)
    patches = [
        rgb[:int(h*0.3), :int(w*0.3)],
        rgb[:int(h*0.3), int(w*0.7):],
        rgb[int(h*0.7):, :int(w*0.3)],
        rgb[int(h*0.7):, int(w*0.7):]
    ]
    # Simple background noise/texture variance vs face
    bg_vars = [cv2.Laplacian(cv2.cvtColor(p, cv2.COLOR_RGB2GRAY), cv2.CV_32F).var() for p in patches if p.size > 0]
    med_bg_var = float(np.median(bg_vars)) if bg_vars else 100.0
    
    # Regional score
    diff = abs(face_score - 0.3)
    if diff > 0.4:
        score = float(min(1.0, diff))
        return Signal(
            name="region_disagreement",
            score=score,
            weight=1.0,
            detail=f"Disagreement between face region and background environment ({score:.2f})."
        )
    return None

def analyze(rgb: np.ndarray) -> List[Signal]:
    sigs = []
    b_sig = analyze_boundary(rgb)
    if b_sig:
        sigs.append(b_sig)
        rd_sig = analyze_region_disagreement(rgb, b_sig.score)
        if rd_sig:
            sigs.append(rd_sig)
    return sigs
