"""Voice-clone / synthetic-speech cues from the audio track (heuristic priors).
Each cue is weak alone; they are fused with video cues."""
import numpy as np
from .utils import Signal, to_unit


def load_audio(path, max_sec=30):
    import librosa
    y, sr = librosa.load(str(path), sr=None, mono=True, duration=max_sec)
    return y, sr


def analyze(y, sr):
    import librosa
    if y is None or len(y) < sr * 1.0:
        return []
    out = []
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]

    # 1) digital silence / unnaturally clean gaps (TTS pads with exact zeros)
    dig = float((rms < 1e-4).mean())
    quiet = rms[rms < np.percentile(rms, 20)]
    floor_cv = float(quiet.std() / (quiet.mean() + 1e-9)) if len(quiet) else 0
    s = 0.7 * to_unit(dig, 0.01, 0.15) + 0.3 * (1 - to_unit(floor_cv, 0.3, 1.5))
    out.append(Signal("audio_silence", s, 0.8,
                      f"Silence analysis: {dig*100:.1f}% digital-zero frames, noise-floor variation {floor_cv:.2f} -> "
                      f"{'unnaturally clean gaps (synthetic-speech trait)' if s > 0.5 else 'natural room/mic noise in pauses'}."))

    # 2) spectral flatness + band-limit cutoff
    flat = float(librosa.feature.spectral_flatness(y=y).mean())
    spec = np.abs(librosa.stft(y, n_fft=1024)).mean(axis=1)
    db = 20 * np.log10(spec / (spec.max() + 1e-9) + 1e-9)
    idx = np.where(db > -70)[0]
    cutoff = float(idx.max() * sr / 1024) if len(idx) else 0.0
    band_limited = sr >= 32000 and cutoff < 9000
    s = 0.5 * (1.0 if band_limited else 0.2) + 0.5 * to_unit(flat, 0.01, 0.08)
    out.append(Signal("audio_spectral", float(np.clip(s, 0, 1)), 0.8,
                      f"Spectrum: flatness {flat:.3f}, energy cutoff ~{cutoff/1000:.1f} kHz at {sr/1000:.1f} kHz sampling -> "
                      f"{'band-limited / vocoder-like spectrum' if s > 0.5 else 'full natural bandwidth'}."))

    # 3) prosody: cloned voices are often too monotone
    try:
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=2048, hop_length=512)
        r2 = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0][: len(f0)]
        voiced = f0[: len(r2)][r2 > np.percentile(r2, 50)]
        semis = 12 * np.log2(voiced / np.median(voiced))
        sd = float(np.std(semis))
        s = 1 - to_unit(sd, 1.0, 4.0)
        out.append(Signal("audio_prosody", s, 0.7,
                          f"Pitch variation {sd:.2f} semitones -> "
                          f"{'flat, monotone delivery (possible voice clone)' if s > 0.5 else 'natural pitch movement'}."))
    except Exception:
        pass
    return out
