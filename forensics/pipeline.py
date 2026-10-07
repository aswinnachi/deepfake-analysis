"""One entry point: analyze(path) for image, video or audio files."""
import os
from . import ela, frequency, noise, face, metadata, cnn, compression, audio, av_sync, video, hf_detector
from .fusion import fuse
from .utils import load_rgb
from .visualize import overlay

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUD_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # one failing module must not kill the report
        print(f"[warn] {getattr(fn, '__module__', fn)} failed: {e}")
        return None


def analyze_array(rgb, path=None, use_learned=True):
    deg, note = compression.degradation(rgb)
    sigs = [_safe(metadata.analyze, path) if path else None]
    sigs += [_safe(m.analyze, rgb) for m in (ela, frequency, noise, face, cnn)]
    hf_sigs = _safe(hf_detector.analyze, rgb)
    if hf_sigs:
        sigs += hf_sigs
    res = fuse(sigs, deg, use_learned)
    res["explanation"].append(note)
    res["overlay"] = overlay(rgb, sigs)
    res["_signals"] = sigs
    return res


def analyze_image(path, use_learned=True):
    return analyze_array(load_rgb(path), path, use_learned)


def analyze_video(path):
    frames = video.sample_frames(path)
    if not frames:
        raise ValueError("Could not read video frames")
    sigs, worst = video.analyze_frames(frames)
    sigs.append(_safe(video.temporal_face, path))
    y = sr = None
    try:
        y, sr = audio.load_audio(path)
    except Exception:
        pass
    if y is not None:
        sigs += _safe(audio.analyze, y, sr) or []
        sigs.append(_safe(av_sync.analyze, path, y, sr))
    deg, note = compression.degradation(frames[len(frames) // 2])
    res = fuse(sigs, deg, use_learned=False)
    res["explanation"].append(note)
    if y is None:
        res["explanation"].append("No audio track found: lip-sync / voice checks skipped.")
    res["overlay"] = overlay(worst, sigs)
    return res


def analyze_audio(path):
    y, sr = audio.load_audio(path)
    sigs = audio.analyze(y, sr)
    res = fuse(sigs, 0.0, use_learned=False)
    res["overlay"] = None
    return res


def analyze(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in IMG_EXT:
        return analyze_image(path)
    if ext in VID_EXT:
        return analyze_video(path)
    if ext in AUD_EXT:
        return analyze_audio(path)
    raise ValueError(f"Unsupported file type: {ext}")


def to_markdown(res):
    lines = [f"## {res['verdict']}",
             f"**Authenticity: {res['authenticity_real_pct']}% likely real** "
             f"({100 - res['authenticity_real_pct']:.1f}% fake) - confidence: {res['confidence']}",
             "", "### Evidence (strongest first)"]
    lines += [f"- {l}" for l in res["explanation"]]
    return "\n".join(lines)
