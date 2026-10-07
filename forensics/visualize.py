import numpy as np
import cv2


def overlay(rgb, signals, alpha=0.45):
    """Blend all localisation heatmaps (ELA, noise, face, CNN) onto the image."""
    h, w = rgb.shape[:2]
    maps = []
    for s in signals:
        if s is not None and s.heatmap is not None:
            hm = cv2.resize(s.heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
            hm = hm / (hm.max() + 1e-6)
            maps.append(hm * (0.5 + s.score))
    if not maps:
        return rgb
    comb = np.mean(maps, axis=0)
    comb = np.clip(comb / (np.percentile(comb, 99.5) + 1e-6), 0, 1)
    colour = cv2.cvtColor(cv2.applyColorMap((comb * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    out = (rgb * (1 - alpha) + colour * alpha).astype(np.uint8)
    for s in signals:
        if s is not None and s.box is not None:
            x, y, bw, bh = s.box
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
    return out
