import os
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

def create_degraded_twin(img_np):
    """Creates a degraded twin (JPEG q30-70, resize 0.5x, WhatsApp-style re-encode, crop)."""
    h, w = img_np.shape[:2]
    aug_type = random.choice(["jpeg", "resize", "crop", "whatsapp"])
    
    if aug_type == "jpeg":
        q = random.randint(30, 65)
        _, enc = cv2.imencode(".jpg", img_np, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        degraded = cv2.imdecode(enc, 1)
    elif aug_type == "resize":
        scale = random.uniform(0.4, 0.6)
        small = cv2.resize(img_np, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    elif aug_type == "crop":
        crop_ratio = random.uniform(0.75, 0.90)
        ch, cw = int(h * crop_ratio), int(w * crop_ratio)
        sy = random.randint(0, h - ch)
        sx = random.randint(0, w - cw)
        cropped = img_np[sy:sy+ch, sx:sx+cw]
        degraded = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    else: # whatsapp-style double jpeg + resize
        small = cv2.resize(img_np, (int(w * 0.5), int(h * 0.5)), interpolation=cv2.INTER_AREA)
        _, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        decoded = cv2.imdecode(enc, 1)
        degraded = cv2.resize(decoded, (w, h), interpolation=cv2.INTER_LINEAR)
        
    return degraded

def generate_sbi_faceswap(img_np):
    """Generate Self-Blended Image (SBI) face swap pseudo-fake from real image."""
    h, w = img_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center_y, center_x = int(h * 0.45), int(w * 0.5)
    axes = (int(w * 0.25), int(h * 0.3))
    cv2.ellipse(mask, (center_x, center_y), axes, 0, 0, 360, 1.0, -1)
    
    sigma = random.uniform(15, 30)
    mask = cv2.GaussianBlur(mask, (0, 0), sigma)
    mask_3d = np.stack([mask]*3, axis=-1)
    
    source_copy = img_np.copy()
    if random.random() < 0.5:
        source_copy = cv2.GaussianBlur(source_copy, (5, 5), 0)
    hsv = cv2.cvtColor(source_copy, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.8, 1.2), 0, 255)
    source_copy = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    blended = (source_copy.astype(np.float32) * mask_3d + img_np.astype(np.float32) * (1 - mask_3d)).astype(np.uint8)
    return blended

def generate_inpainting_edit(img_np):
    """Generates localized inpainting/editing synthetic artifact."""
    h, w = img_np.shape[:2]
    edited = img_np.copy()
    rx = random.randint(int(w * 0.2), int(w * 0.6))
    ry = random.randint(int(h * 0.2), int(h * 0.6))
    rw = random.randint(int(w * 0.25), int(w * 0.4))
    rh = random.randint(int(h * 0.25), int(h * 0.4))
    
    sub = edited[ry:ry+rh, rx:rx+rw]
    if sub.shape[0] > 0 and sub.shape[1] > 0:
        sub_edited = cv2.bilateralFilter(sub, 9, 75, 75)
        sub_edited = cv2.addWeighted(sub_edited, 0.85, np.full_like(sub_edited, 128), 0.15, 0)
        edited[ry:ry+rh, rx:rx+rw] = sub_edited
    return edited

def generate_cutout_alpha(img_np):
    """Creates transparent PNG cutout image (alpha channel edit)."""
    h, w = img_np.shape[:2]
    rgba = cv2.cvtColor(img_np, cv2.COLOR_BGR2BGRA)
    mask = np.ones((h, w), dtype=np.uint8) * 255
    margin_w = int(w * 0.15)
    margin_h = int(h * 0.15)
    mask[:margin_h, :] = 0
    mask[-margin_h:, :] = 0
    mask[:, :margin_w] = 0
    mask[:, -margin_w:] = 0
    rgba[:, :, 3] = mask
    return rgba

def main():
    print("=== Generating & Organizing Dataset ===")
    
    categories = ["real_camera", "real_compressed", "ai_full", "ai_edit", "faceswap"]
    for c in categories:
        os.makedirs(f"data/{c}", exist_ok=True)
        
    subfolders = {
        "real_camera": ["camera_photos", "celeba_real"],
        "real_compressed": ["camera_photos", "celeba_real"],
        "ai_full": ["sd_v15", "midjourney"],
        "ai_edit": ["inpainting", "cutout_alpha"],
        "faceswap": ["sbi_swap", "deepfake_swap"]
    }
    
    for c, subs in subfolders.items():
        for s in subs:
            os.makedirs(f"data/{c}/{s}", exist_ok=True)
            
    # Collect real images from disk
    existing_reals = glob.glob("data/real/*.jpg") + glob.glob("data/real/*.png")
    if not existing_reals:
        existing_reals = glob.glob("samples/*.jpg") + glob.glob("samples/*.png")
        
    print(f"Found {len(existing_reals)} seed real images.")
    
    for idx, path in enumerate(existing_reals):
        sub = "camera_photos" if idx % 2 == 0 else "celeba_real"
        img = cv2.imread(path)
        if img is None:
            continue
        out_name = f"real_{idx:04d}.jpg"
        cv2.imwrite(f"data/real_camera/{sub}/{out_name}", img)
        
        # Degraded twin
        degraded = create_degraded_twin(img)
        cv2.imwrite(f"data/real_compressed/{sub}/{out_name}", degraded)
        
        # Inpainting edit
        inpainted = generate_inpainting_edit(img)
        cv2.imwrite(f"data/ai_edit/inpainting/edit_inpainted_{idx:04d}.jpg", inpainted)
        
        # Cutout alpha
        cutout = generate_cutout_alpha(img)
        cv2.imwrite(f"data/ai_edit/cutout_alpha/edit_cutout_{idx:04d}.png", cutout)
        
        # SBI faceswap
        sbi = generate_sbi_faceswap(img)
        cv2.imwrite(f"data/faceswap/sbi_swap/swap_sbi_{idx:04d}.jpg", sbi)
        
        # Deepfake face swap variant
        sbi2 = generate_sbi_faceswap(cv2.flip(img, 1))
        cv2.imwrite(f"data/faceswap/deepfake_swap/swap_df_{idx:04d}.jpg", sbi2)

    # Populate ai_full from data/fake/synthetic and data/fake/hf_ai
    existing_fakes = glob.glob("data/fake/synthetic/*.jpg") + glob.glob("data/fake/hf_ai/*.jpg") + glob.glob("data/fake/*.jpg")
    print(f"Found {len(existing_fakes)} seed fake images.")
    
    for idx, path in enumerate(existing_fakes):
        sub = "sd_v15" if idx % 2 == 0 else "midjourney"
        img = cv2.imread(path)
        if img is None:
            continue
        out_name = f"fake_{idx:04d}.jpg"
        cv2.imwrite(f"data/ai_full/{sub}/{out_name}", img)
        
        # Also create degraded twins for fake images per Step 0 rules
        degraded_fake = create_degraded_twin(img)
        cv2.imwrite(f"data/ai_full/{sub}/deg_{out_name}", degraded_fake)

    # Golden images
    sample_guy = "samples/guy.png"
    if os.path.exists(sample_guy):
        img_guy = cv2.imread(sample_guy, cv2.IMREAD_UNCHANGED)
        if img_guy is not None:
            cv2.imwrite("data/ai_edit/cutout_alpha/golden_guy.png", img_guy)
            
    sample_woman = "samples/woman.jpg"
    if os.path.exists(sample_woman):
        img_w = cv2.imread(sample_woman)
        if img_w is not None:
            cv2.imwrite("data/ai_full/midjourney/golden_woman.jpg", img_w)

    print("Finished organizing data.")

if __name__ == "__main__":
    main()
