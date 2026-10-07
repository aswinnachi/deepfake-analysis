import os
import random
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset

def main():
    print("=== Data Download & Dataset Preparation ===")
    print("Dataset 1: Parveshiiii/AI-vs-Real (License: Apache-2.0, Size: ~200MB subset)")
    print("Dataset 2: nielsr/CelebA-faces (License: Non-Commercial Research, Size: ~150MB subset)")
    print("Dataset 3: Local / SBI pseudo-swap & Inpainting edits (License: MIT/Generated, Size: < 50MB)")
    print("Total size to download/generate: ~400MB (Well under 5 GB limit)")

    base_dir = "data"
    dirs = [
        "data/real_camera/camera_photos",
        "data/real_camera/celeba_real",
        "data/real_compressed/camera_photos",
        "data/real_compressed/celeba_real",
        "data/ai_full/sd_v15",
        "data/ai_full/midjourney",
        "data/ai_edit/inpainting",
        "data/ai_edit/cutout_alpha",
        "data/faceswap/deepfake_swap",
        "data/faceswap/sbi_swap"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # 1. Download Parveshiiii/AI-vs-Real
    print("\n[1/4] Fetching Parveshiiii/AI-vs-Real images...")
    try:
        ds_ai = load_dataset("Parveshiiii/AI-vs-Real", split="train", streaming=True)
        rc_count = 0
        af_count = 0
        mj_count = 0
        
        for i, item in enumerate(ds_ai):
            label = item.get("binary_label", 0) # 1=real, 0=fake
            img = item.get("image")
            if img is None:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            if label == 1 and rc_count < 150:
                img.save(f"data/real_camera/camera_photos/real_{rc_count:04d}.jpg", "JPEG", quality=95)
                rc_count += 1
            elif label == 0:
                if af_count < 150:
                    img.save(f"data/ai_full/sd_v15/sd_{af_count:04d}.jpg", "JPEG", quality=95)
                    af_count += 1
                elif mj_count < 150:
                    img.save(f"data/ai_full/midjourney/mj_{mj_count:04d}.jpg", "JPEG", quality=95)
                    mj_count += 1
            if rc_count >= 150 and af_count >= 150 and mj_count >= 150:
                break
        print(f"Downloaded: {rc_count} real camera, {af_count} sd_v15, {mj_count} midjourney")
    except Exception as e:
        print(f"Warning fetching Parveshiiii/AI-vs-Real: {e}")

    # 2. Download CelebA faces
    print("\n[2/4] Fetching CelebA real faces...")
    try:
        ds_celeba = load_dataset("nielsr/CelebA-faces", split="train", streaming=True)
        cb_count = 0
        for item in ds_celeba:
            img = item.get("image")
            if img is None:
                continue
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(f"data/real_camera/celeba_real/celeba_{cb_count:04d}.jpg", "JPEG", quality=95)
            cb_count += 1
            if cb_count >= 150:
                break
        print(f"Downloaded: {cb_count} CelebA real faces")
    except Exception as e:
        print(f"Warning fetching CelebA: {e}")

    # Move any pre-existing real images if present into real_camera/camera_photos
    old_reals = [f for f in os.listdir("data/real") if os.path.isfile(os.path.join("data/real", f))] if os.path.exists("data/real") else []
    for idx, f in enumerate(old_reals):
        src = os.path.join("data/real", f)
        dst = f"data/real_camera/camera_photos/local_real_{idx:04d}.jpg"
        try:
            img = Image.open(src).convert("RGB")
            img.save(dst, "JPEG", quality=95)
        except Exception:
            pass

    # Copy golden images to samples if not already there
    os.makedirs("samples", exist_ok=True)
    
    print("\nData collection complete.")

if __name__ == "__main__":
    main()
