import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
import imagehash

def get_source_group_id(path):
    """Extract source group id from filename."""
    fname = os.path.basename(path)
    base, _ = os.path.splitext(fname)
    # Strip common prefixes
    for prefix in ["edit_inpainted_", "edit_cutout_", "swap_sbi_", "swap_df_", "golden_", "deg_"]:
        if base.startswith(prefix):
            base = base[len(prefix):]
    return base

def main():
    print("=== Building Dataset Manifest (data/manifest.csv) ===")
    valid_categories = {"real_camera", "real_compressed", "ai_full", "ai_edit", "faceswap"}
    
    all_paths = sorted(glob.glob("data/**/*.jpg", recursive=True) + glob.glob("data/**/*.png", recursive=True))
    
    image_paths = []
    for p in all_paths:
        parts = p.split(os.sep)
        if len(parts) >= 4 and parts[1] in valid_categories:
            image_paths.append(p)
            
    data = []
    seen_hashes = {}
    duplicates = 0
    
    for path in image_paths:
        parts = path.split(os.sep)
        category = parts[1]
        subfolder = parts[2]
        
        try:
            with Image.open(path) as img:
                phash = str(imagehash.phash(img))
        except Exception as e:
            print(f"Warning reading {path}: {e}")
            continue
            
        if phash in seen_hashes:
            duplicates += 1
        else:
            seen_hashes[phash] = path
            
        sg_id = get_source_group_id(path)
        data.append({
            "path": path,
            "category": category,
            "subfolder": subfolder,
            "source_group_id": sg_id,
            "phash": phash
        })
        
    df = pd.DataFrame(data)
    print(f"Found {len(df)} images across valid categories. Found {duplicates} perceptual duplicate hashes.")
    
    # 60/20/20 Train/Val/Test Split by SOURCE GROUP ID
    unique_groups = sorted(df["source_group_id"].unique())
    np.random.seed(42)
    np.random.shuffle(unique_groups)
    
    n_groups = len(unique_groups)
    n_train = int(n_groups * 0.60)
    n_val = int(n_groups * 0.20)
    
    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train+n_val])
    test_groups = set(unique_groups[n_train+n_val:])
    
    def assign_split(sg):
        if sg in train_groups:
            return "train"
        elif sg in val_groups:
            return "val"
        else:
            return "test"
            
    df["split"] = df["source_group_id"].apply(assign_split)
    
    # Print summary counts
    print("\n--- Manifest Summary ---")
    summary = df.groupby(["category", "subfolder", "split"]).size().unstack(fill_value=0)
    print(summary)
    
    # Check fake category requirements
    fake_categories = ["ai_full", "ai_edit", "faceswap"]
    for fcat in fake_categories:
        cat_df = df[df["category"] == fcat]
        n_images = len(cat_df)
        n_subs = cat_df["subfolder"].nunique()
        subs_list = cat_df["subfolder"].unique().tolist()
        
        print(f"\nCategory '{fcat}': {n_images} images, {n_subs} subfolders ({subs_list})")
        
        if n_images < 200:
            raise ValueError(f"VALIDATION FAILED: Fake category '{fcat}' has {n_images} images (< 200 required).")
        if n_subs < 2:
            raise ValueError(f"VALIDATION FAILED: Fake category '{fcat}' has {n_subs} subfolders (< 2 required).")
            
    # Save manifest
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/manifest.csv", index=False)
    print("\nManifest successfully saved to data/manifest.csv")

if __name__ == "__main__":
    main()
