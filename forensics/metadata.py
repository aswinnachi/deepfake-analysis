"""Provenance cues: EXIF / PNG text chunks. Easy to strip, so weighted low
unless an explicit AI-generator marker is present."""
from PIL import Image
from .utils import Signal

AI_MARKERS = ["stable diffusion", "midjourney", "dall-e", "dall·e", "firefly", "comfyui",
              "automatic1111", "novelai", "leonardo", "imagen", "sdxl", "flux", "openai", "c2pa"]


def analyze(path):
    try:
        im = Image.open(path)
    except Exception:
        return None
    blob = " ".join(str(v) for v in im.info.values() if isinstance(v, (str, bytes))).lower()
    exif = im.getexif()
    text = blob + " " + str(exif.get(305, "")).lower()
    make, model = str(exif.get(271, "")), str(exif.get(272, ""))
    hits = [m for m in AI_MARKERS if m in text]
    if hits:
        # Strong indicator if metadata explicitly names an AI tool
        return Signal("metadata", 0.97, 2.0, f"Metadata names an AI generator ({', '.join(hits)}).")
    if make or model:
        # Standard camera EXIF cue
        return Signal("metadata", 0.25, 1.0,
                      f"Camera EXIF present ({(make + ' ' + model).strip()}): consistent with a real capture (EXIF can be forged).")
    # Missing EXIF is weak evidence because re-shared photos strip metadata; keep weight modest
    return Signal("metadata", 0.55, 0.5,
                  "No camera EXIF / provenance data (typical of generated or re-shared images; weak evidence).")
