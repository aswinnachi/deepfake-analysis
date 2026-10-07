"""Generate sample images for demo and testing.

Creates realistic-looking test images and their manipulated versions
so you have something to demo without needing external datasets.

    python scripts/generate_samples.py
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import io

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "samples")
DATA_REAL = os.path.join(os.path.dirname(__file__), "..", "data", "real")
DATA_FAKE = os.path.join(os.path.dirname(__file__), "..", "data", "fake", "synthetic")


def make_dirs():
    for d in [SAMPLES_DIR, DATA_REAL, DATA_FAKE]:
        os.makedirs(d, exist_ok=True)


def create_natural_scene(seed=42):
    """Create a realistic-looking natural photograph (landscape/texture)."""
    rng = np.random.default_rng(seed)
    h, w = 640, 960

    # Sky gradient
    sky = np.zeros((h, w, 3), np.float32)
    for y in range(h // 2):
        t = y / (h / 2)
        sky[y] = [135 + 50 * t, 180 + 30 * t, 235 - 20 * t]

    # Ground with texture
    for y in range(h // 2, h):
        t = (y - h // 2) / (h / 2)
        base = np.array([34 + 60 * t, 85 + 40 * (1 - t), 25 + 30 * t])
        noise = rng.normal(0, 8, (w, 3))
        sky[y] = base + noise

    # Add some cloud-like blobs
    for _ in range(5):
        cx, cy = rng.integers(100, w - 100), rng.integers(30, h // 3)
        rx, ry = rng.integers(40, 120), rng.integers(20, 50)
        Y, X = np.ogrid[-cy:h - cy, -cx:w - cx]
        mask = (X ** 2 / (rx ** 2 + 1)) + (Y ** 2 / (ry ** 2 + 1)) < 1
        alpha = np.exp(-((X ** 2 / (rx ** 2 + 1)) + (Y ** 2 / (ry ** 2 + 1))) * 2)
        for c in range(3):
            sky[:, :, c] = sky[:, :, c] * (1 - alpha * mask) + 245 * alpha * mask

    # Add sensor noise (realistic)
    sky += rng.normal(0, 3.5, sky.shape)
    img = np.clip(sky, 0, 255).astype(np.uint8)

    # Save as JPEG with realistic quality (adds authentic compression artifacts)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=92, exif=b"")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def create_portrait_scene(seed=123):
    """Create a scene with a face-like oval region (different texture stats)."""
    rng = np.random.default_rng(seed)
    h, w = 640, 480

    # Background
    bg = rng.normal(120, 15, (h, w, 3)).clip(0, 255).astype(np.float32)
    bg = cv2.GaussianBlur(bg, (5, 5), 2)

    # Add a face-like oval with similar texture to background (real)
    cx, cy = w // 2, h // 3
    Y, X = np.ogrid[-cy:h - cy, -cx:w - cx]
    face_mask = ((X / 80) ** 2 + (Y / 110) ** 2) < 1
    face_region = rng.normal(165, 12, (h, w, 3)).clip(0, 255).astype(np.float32)
    face_region = cv2.GaussianBlur(face_region, (3, 3), 1.2)

    for c in range(3):
        bg[:, :, c] = np.where(face_mask, face_region[:, :, c], bg[:, :, c])

    # Add uniform sensor noise across entire image (consistent = real)
    bg += rng.normal(0, 3.0, bg.shape)
    img = np.clip(bg, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=90)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def create_splice_fake(real_img, seed=77):
    """Create a spliced image: paste a region from a differently-compressed source."""
    rng = np.random.default_rng(seed)
    h, w = real_img.shape[:2]
    fake = real_img.copy()

    # Create an alien patch (different noise profile, different compression history)
    patch_h, patch_w = h // 3, w // 3
    px, py = rng.integers(w // 4, w // 2), rng.integers(h // 4, h // 2)

    # This patch comes from a "different image" (different noise, blur, compression)
    patch = rng.normal(140, 20, (patch_h, patch_w, 3)).clip(0, 255).astype(np.uint8)
    patch = cv2.GaussianBlur(patch, (7, 7), 3)  # Different blur = different noise stats

    # Compress the patch separately (different JPEG history → ELA will detect this)
    buf = io.BytesIO()
    Image.fromarray(patch).save(buf, "JPEG", quality=60)
    buf.seek(0)
    patch = np.array(Image.open(buf).convert("RGB"))

    # Paste with slight feathering
    fake[py:py + patch_h, px:px + patch_w] = patch
    return fake


def create_smooth_fake(seed=99):
    """Create an image that mimics GAN output: unnaturally smooth, different frequency profile."""
    rng = np.random.default_rng(seed)
    h, w = 512, 512

    # Very smooth gradients (GANs are often too smooth)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        base = rng.uniform(80, 200)
        for y in range(h):
            for x_block in range(0, w, 64):
                img[y, x_block:min(x_block + 64, w), c] = base + 30 * np.sin(y / 50.0 + c) + 15 * np.cos(x_block / 40.0)

    # Heavy blur = unnaturally clean (noise analysis will catch this)
    img = cv2.GaussianBlur(img, (9, 9), 4)

    # Add very faint noise (much less than a real camera)
    img += rng.normal(0, 0.8, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # No JPEG artifacts (PNG-like, no compression history → ELA sees uniform)
    return img


def create_faceswap_fake(real_portrait, seed=55):
    """Simulate a face swap: replace the face region with different-textured content."""
    rng = np.random.default_rng(seed)
    h, w = real_portrait.shape[:2]
    fake = real_portrait.copy()

    cx, cy = w // 2, h // 3
    Y, X = np.ogrid[-cy:h - cy, -cx:w - cx]
    face_mask = ((X / 80) ** 2 + (Y / 110) ** 2) < 1

    # Swapped face: very different noise characteristics (sharper, different blur)
    swap_face = rng.normal(175, 8, (h, w, 3)).clip(0, 255).astype(np.float32)
    # Much sharper than surroundings (face.py will detect sharpness mismatch)
    swap_face = cv2.GaussianBlur(swap_face, (3, 3), 0.5)
    # Different noise level
    swap_face += rng.normal(0, 1.0, swap_face.shape)

    for c in range(3):
        fake[:, :, c] = np.where(face_mask, swap_face[:, :, c].clip(0, 255).astype(np.uint8), fake[:, :, c])

    return fake


def create_upsampled_fake(seed=33):
    """Create an image with upsampling artifacts (frequency analysis will catch periodic spikes)."""
    rng = np.random.default_rng(seed)
    # Start small
    small = rng.normal(128, 25, (128, 128, 3)).clip(0, 255).astype(np.uint8)
    small = cv2.GaussianBlur(small, (3, 3), 1)

    # Upsample with nearest-neighbor then smooth (creates periodic frequency artifacts)
    big = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
    big = cv2.GaussianBlur(big, (3, 3), 0.8)
    return big


def add_metadata_markers(img_path, generator_name="Stable Diffusion XL"):
    """Add AI generator metadata to an image."""
    im = Image.open(img_path)
    from PIL.PngImagePlugin import PngInfo
    # For JPEG, we'll use a simple approach
    info = im.info.copy()
    info["Software"] = generator_name
    im.save(img_path, quality=95)


def main():
    make_dirs()
    print("Generating sample images...")

    # ── Real images ──
    real_landscape = create_natural_scene(seed=42)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "real_landscape.jpg"),
                cv2.cvtColor(real_landscape, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(DATA_REAL, "real_landscape.jpg"),
                cv2.cvtColor(real_landscape, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("  ✓ real_landscape.jpg")

    real_portrait = create_portrait_scene(seed=123)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "real_portrait.jpg"),
                cv2.cvtColor(real_portrait, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(DATA_REAL, "real_portrait.jpg"),
                cv2.cvtColor(real_portrait, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  ✓ real_portrait.jpg")

    real2 = create_natural_scene(seed=100)
    cv2.imwrite(os.path.join(DATA_REAL, "real_scene2.jpg"),
                cv2.cvtColor(real2, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])

    real3 = create_natural_scene(seed=200)
    cv2.imwrite(os.path.join(DATA_REAL, "real_scene3.jpg"),
                cv2.cvtColor(real3, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 93])

    real4 = create_portrait_scene(seed=300)
    cv2.imwrite(os.path.join(DATA_REAL, "real_portrait2.jpg"),
                cv2.cvtColor(real4, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 91])
    print("  ✓ 3 more real images for data/real/")

    # ── Fake: splice ──
    splice = create_splice_fake(real_landscape, seed=77)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "fake_splice.jpg"),
                cv2.cvtColor(splice, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    cv2.imwrite(os.path.join(DATA_FAKE, "fake_splice.jpg"),
                cv2.cvtColor(splice, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("  ✓ fake_splice.jpg (spliced region with different compression)")

    # ── Fake: smooth/GAN-like ──
    smooth = create_smooth_fake(seed=99)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "fake_smooth_gan.png"),
                cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DATA_FAKE, "fake_smooth_gan.png"),
                cv2.cvtColor(smooth, cv2.COLOR_RGB2BGR))
    print("  ✓ fake_smooth_gan.png (unnaturally smooth, GAN-like noise profile)")

    # ── Fake: face swap simulation ──
    faceswap = create_faceswap_fake(real_portrait, seed=55)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "fake_faceswap.jpg"),
                cv2.cvtColor(faceswap, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(os.path.join(DATA_FAKE, "fake_faceswap.jpg"),
                cv2.cvtColor(faceswap, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("  ✓ fake_faceswap.jpg (face region with mismatched texture stats)")

    # ── Fake: upsampled ──
    upsampled = create_upsampled_fake(seed=33)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "fake_upsampled.png"),
                cv2.cvtColor(upsampled, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(DATA_FAKE, "fake_upsampled.png"),
                cv2.cvtColor(upsampled, cv2.COLOR_RGB2BGR))
    print("  ✓ fake_upsampled.png (upsampling artifacts in frequency domain)")

    # More fakes for data/
    for i, seed in enumerate([111, 222, 333]):
        smooth_i = create_smooth_fake(seed=seed)
        cv2.imwrite(os.path.join(DATA_FAKE, f"fake_smooth_{i}.png"),
                    cv2.cvtColor(smooth_i, cv2.COLOR_RGB2BGR))
    for i, seed in enumerate([444, 555]):
        up_i = create_upsampled_fake(seed=seed)
        cv2.imwrite(os.path.join(DATA_FAKE, f"fake_upsampled_{i}.png"),
                    cv2.cvtColor(up_i, cv2.COLOR_RGB2BGR))
    print("  ✓ 5 more fake images for data/fake/synthetic/")

    print(f"\n✅ Done! Generated:")
    print(f"   samples/ : {len(os.listdir(SAMPLES_DIR))} demo files")
    print(f"   data/real/: {len([f for f in os.listdir(DATA_REAL) if not f.startswith('.')])} images")
    print(f"   data/fake/synthetic/: {len(os.listdir(DATA_FAKE))} images")
    print(f"\nRun:  python app.py   → upload files from samples/ to test")


if __name__ == "__main__":
    main()
