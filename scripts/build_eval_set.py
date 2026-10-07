"""Build degraded evaluation set of real photos for robustness and false-positive testing.

Creates degraded copies in data/real_degraded/ applying:
 - JPEG compression (quality 30)
 - Half resolution (50% downscaling)
 - 80% center crop
 - Gaussian blur
 - Combined social-media style degradation (all transformations applied)

Usage:
    python scripts/build_eval_set.py [input_dir] [--output data/real_degraded]
"""
import os
import sys
import argparse
import glob
import io
import cv2
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forensics.utils import load_rgb

VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def apply_jpeg(rgb: np.ndarray, quality: int = 30) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def apply_half_res(rgb: np.ndarray) -> np.ndarray:
    return cv2.resize(rgb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def apply_center_crop_80(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    return rgb[int(0.1 * h):int(0.9 * h), int(0.1 * w):int(0.9 * w)]


def apply_gaussian_blur(rgb: np.ndarray, sigma: float = 1.2) -> np.ndarray:
    return cv2.GaussianBlur(rgb, (0, 0), sigma)


def apply_combined(rgb: np.ndarray) -> np.ndarray:
    c = apply_center_crop_80(rgb)
    h = apply_half_res(c)
    b = apply_gaussian_blur(h, 1.2)
    return apply_jpeg(b, quality=30)


def save_image(rgb: np.ndarray, path: str, quality: int = 95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(rgb).save(path, quality=quality)


def build_degraded_dataset(input_dir: str, output_dir: str = "data/real_degraded", max_images: int = None):
    os.makedirs(output_dir, exist_ok=True)
    
    # Gather image files
    candidates = []
    if os.path.isfile(input_dir):
        candidates = [input_dir]
    elif os.path.isdir(input_dir):
        for root, _, files in os.walk(input_dir):
            for f in files:
                if f.lower().endswith(VALID_EXTS):
                    candidates.append(os.path.join(root, f))
    else:
        print(f"Error: input path not found: {input_dir}")
        return

    if not candidates:
        print(f"No image files found in {input_dir}")
        return

    if max_images:
        candidates = candidates[:max_images]

    print(f"Processing {len(candidates)} images from '{input_dir}' into '{output_dir}'...")

    count = 0
    for idx, path in enumerate(candidates, 1):
        try:
            rgb = load_rgb(path)
            stem = os.path.splitext(os.path.basename(path))[0]

            # 1. JPEG quality 30
            img_jpeg30 = apply_jpeg(rgb, quality=30)
            save_image(img_jpeg30, os.path.join(output_dir, f"{stem}_jpeg30.jpg"), quality=30)

            # 2. Half resolution
            img_half = apply_half_res(rgb)
            save_image(img_half, os.path.join(output_dir, f"{stem}_half_res.jpg"))

            # 3. 80% center crop
            img_crop = apply_center_crop_80(rgb)
            save_image(img_crop, os.path.join(output_dir, f"{stem}_crop80.jpg"))

            # 4. Gaussian blur
            img_blur = apply_gaussian_blur(rgb)
            save_image(img_blur, os.path.join(output_dir, f"{stem}_blur.jpg"))

            # 5. Combined social-media degradation (crop + half-res + blur + jpeg30)
            img_comb = apply_combined(rgb)
            save_image(img_comb, os.path.join(output_dir, f"{stem}_combined.jpg"), quality=30)

            count += 1
            if idx % 10 == 0 or idx == len(candidates):
                print(f"  Processed {idx}/{len(candidates)} images...")
        except Exception as e:
            print(f"  [warn] Failed to process {path}: {e}")

    print(f"Successfully generated degraded copies for {count} images in '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Create degraded copies of real photos.")
    parser.add_argument("input", nargs="?", default="data/real", help="Folder containing real photos (default: data/real)")
    parser.add_argument("--output", default="data/real_degraded", help="Output directory (default: data/real_degraded)")
    parser.add_argument("--max", type=int, default=None, help="Maximum number of input images to process")
    args = parser.parse_args()

    build_degraded_dataset(args.input, args.output, args.max)


if __name__ == "__main__":
    main()
