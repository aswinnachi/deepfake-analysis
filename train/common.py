"""Dataset layout expected (any image files, nested folders allowed):
    data/real/**.jpg|png
    data/fake/<generator_or_method_name>/**.jpg|png     e.g. data/fake/stylegan, data/fake/sd, data/fake/faceswap
Video datasets (FaceForensics++, DFDC): extract frames first with scripts/extract_frames.py
"""
import os, glob
EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def list_images(folder):
    return [p for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True) if p.lower().endswith(EXT)]


def load_dataset(root="data"):
    real = list_images(os.path.join(root, "real"))
    fakes = {}
    fake_root = os.path.join(root, "fake")
    for g in sorted(os.listdir(fake_root)) if os.path.isdir(fake_root) else []:
        d = os.path.join(fake_root, g)
        if os.path.isdir(d):
            fakes[g] = list_images(d)
    # images placed directly in data/fake/ count as generator "misc"
    direct = [p for p in glob.glob(os.path.join(fake_root, "*")) if p.lower().endswith(EXT)]
    if direct:
        fakes["misc"] = direct
    return real, fakes
