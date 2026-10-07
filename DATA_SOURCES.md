# Data Sources & Dataset Layout

This document describes the datasets and data layout used for training, validation, and evaluation in `deepfake-forensics`.

## Category Layout (`data/`)
Images are organized into 5 categories, each containing subfolders representing specific generators, tools, or dataset sources:

```
data/
├── real_camera/
│   ├── camera_photos/       # Real camera photos (local + FFHQ/CelebA subsets)
│   └── celeba_real/         # Real face images from CelebA dataset
├── real_compressed/
│   ├── camera_photos/       # JPEG q30-70, 0.5x resize, crop degraded twins
│   └── celeba_real/         # Degraded twins for CelebA real faces
├── ai_full/
│   ├── sd_v15/              # Full diffusion images (Stable Diffusion v1.5)
│   └── midjourney/          # Full diffusion images (Midjourney / GenImage-style)
├── ai_edit/
│   ├── inpainting/          # Locally edited / inpainted real images
│   └── cutout_alpha/        # PNG transparent cutouts & subject edits
└── faceswap/
    ├── sbi_swap/            # Self-Blended Images (SBI) pseudo-face swaps
    └── deepfake_swap/       # Deepfake / FaceSwap public dataset samples
```

## Public Datasets Used

1. **Parveshiiii/AI-vs-Real**
   - **Hugging Face ID**: `Parveshiiii/AI-vs-Real`
   - **License**: Apache-2.0 / Open Access
   - **Size**: ~1.0 GB total (sampled subset: ~200 MB)
   - **Description**: Contains real human photos and AI-generated images across various diffusion models.

2. **CelebA-faces**
   - **Hugging Face ID**: `nielsr/CelebA-faces`
   - **License**: Non-commercial research license
   - **Size**: ~1.2 GB total (sampled subset: ~150 MB)
   - **Description**: Real face dataset widely used for facial forensics benchmarks.

3. **FaceForensics++ (FF++)**
   - **Access**: Restricted access (requires signed access request form to TUM / FF++ authors).
   - **Status**: Documented per requirements. Not downloaded automatically without signed permission. Synthetic SBI swaps used locally instead.

4. **Self-Blended Images (SBI)**
   - **Source**: Algorithmic pseudo-fakes generated from real face images per Li et al. (CVPR 2022).
   - **Description**: Blends target face features with source face augmentations using deformed Gaussian masks.
