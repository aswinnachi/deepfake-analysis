# Multimodal Deepfake & Digital Forensics (HNX26PSI10)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Framework: Gradio](https://img.shields.io/badge/UI-Gradio%204.44-orange.svg)](https://gradio.app/)
[![Privacy: 100% Local](https://img.shields.io/badge/privacy-100%25%20On--Device-teal.svg)](#privacy--external-apis)

An end-to-end multimodal digital forensics platform engineered for hackathon evaluation and forensic screening. The system inspects **images, video, and audio** to determine authenticity by triangulating physical sensor traces, radial facial boundary contours, dual-crop Vision Transformers, and cryptographic provenance credentials.

Every verdict is delivered with an **isotonic-calibrated authenticity percentage**, **plain-English evidence explanations**, and an **explainable spatial heatmap** highlighting suspicious regions and facial bounding boxes.

---

## Table of Contents
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Mandatory Disclosures & Resources Used](#mandatory-disclosures--resources-used)
  - [External APIs](#1-external-apis-policy)
  - [Datasets](#2-datasets)
  - [Pre-Trained Models](#3-pre-trained-models)
  - [Open-Source Components](#4-open-source-libraries)
  - [Forensic Standards & Assets](#5-standards-and-forensic-assets)
- [Empirical Evaluation Benchmarks](#empirical-evaluation-benchmarks)
- [Quick Start](#quick-start)
- [CLI & Automation](#cli--automation)
- [Operational Boundaries & Limitations](#operational-boundaries--limitations)

---

## Key Features

- **Multimodal Triangulation:** Supports images (`.jpg`, `.png`, `.webp`, `.tiff`), videos (`.mp4`, `.mov`, `.avi`), and audio tracks (`.wav`, `.mp3`).
- **Defense-in-Depth Fusion:** Blends 15+ independent forensic signals across classical signal processing, deep neural classifiers, and metadata signatures.
- **Explainable Heatmap & Localization:** Provides side-by-side original and manipulation heatmaps with interactive opacity blending and region localization callouts.
- **Honest & Calibrated Decision Logic:** Isotonic calibration maps raw model logits to real-world posterior probabilities. Conflicting or degraded signals trigger a safe **INCONCLUSIVE** rating rather than issuing a confident false accusation.
- **Offline HTML & JSON Export:** Generates standalone, self-contained forensic reports embedding base64 heatmaps, SHA-256 media checksums, signal breakdowns, and audit timestamps.
- **100% Air-Gapped Operation:** All neural networks and signal analyzers execute locally on CPU/MPS/CUDA. Zero data leaves your machine.

---

## System Architecture

```
[ Uploaded Media: Image / Video / Audio ]
                   │
    ┌──────────────┴───────────────────────────┐
    ▼                                          ▼
[ Classical Forensics ]               [ Learned Neural Probes ]
├─ Error Level Analysis (ELA)         ├─ HF ViT AI Image (Full & Face)
├─ 2D Fourier FFT Spectrum            ├─ HF ViT Deepfake (Full & Face)
├─ Noise PRNU Variance Residuals      ├─ CommunityForensics ViT
├─ 12-Segment Facial Boundary         ├─ Linear CLIP Vision Probe (ViT-B/32)
├─ Face vs Background Consistency     ├─ Native-Resolution Patch ResNet-18
└─ Audio Vocoder / AV-Sync            └─ Boundary Morphing ResNet-18
    │                                          │
    └──────────────┬───────────────────────────┘
                   ▼
       [ Provenance & Metadata ]
       ├─ EXIF / PNG Chunk Parsing
       ├─ C2PA / SynthID Byte Markers
       └─ Visible AI Watermark Template Matching
                   │
                   ▼
    [ Calibrated Decision Fusion ]
    ├─ Degradation Compensation (JPEG / Blur / Resizing)
    ├─ Logit + Ridge Fusion
    └─ Isotonic Probability Calibration
                   │
                   ▼
[ Verdict Card + Top 3 Explanations + Heatmap + HTML Report ]
```

---

## Mandatory Disclosures & Resources Used

In accordance with competition rules and academic integrity standards, all datasets, pre-trained models, libraries, and open-source components utilized in this project are explicitly detailed below:

### 1. External APIs Policy
* **Zero External Runtime APIs:** This project uses **NO cloud APIs, third-party inference endpoints, or external web services** at runtime (no OpenAI API, no Google Cloud Vision, no AWS Rekognition, no Replicate).
* **Privacy & Air-Gap Compliance:** All neural inference, signal processing, and report compilation occur entirely on the local host machine. Media never leaves local memory.

### 2. Datasets
The system was evaluated and calibrated on a structured, deduplicated benchmark manifest of **2,536 media assets** (`data/manifest.csv`) partitioned into a 60/20/20 Train/Val/Test split by source group using perceptual hash (pHash) deduplication:

| Dataset / Source | Modality | Samples | Role in Project | Reference / License |
| :--- | :--- | :---: | :--- | :--- |
| **CelebA** (`nielsr/CelebA-faces`) | Image | 1,000 | Authentic facial baseline | Non-commercial research |
| **FFHQ (Flickr-Faces-HQ)** | Image | 500 | High-resolution authentic facial textures | Creative Commons BY-NC-SA 4.0 |
| **Real Camera Photographs** | Image | 300+ | Uncompressed sensor noise & EXIF baseline | Authentic ground truth |
| **Stable Diffusion v1.5** | Image | 500+ | Full-image synthetic generation evaluation | CreativeML OpenRAIL-M |
| **Midjourney v5** | Image | 350+ | Photorealistic generative diffusion evaluation | Research crawl |
| **FaceForensics++ (Deepfakes / FaceSwap)** | Video/Image | 650+ | Facial swap & manipulation boundary evaluation | Technical University of Munich |
| **Synthetic Blended Images (SBI)** | Image | 350+ | Controlled self-blended boundary artifacts | Research publication |
| **Stable Diffusion Inpainting** | Image | 350+ | Localized contextual splice / inpaint evaluation | CreativeML OpenRAIL-M |
| **Alpha Cutout Composites** | Image | 350+ | Segmented subject insertion evaluation | Synthetic generation |
| **AI vs Real Face Detector Data** (`sahilSpit`) | Image | 500+ | Multi-generator facial verification | Hugging Face Hub |

### 3. Pre-Trained Models

The system integrates open-source pretrained backbones for visual representation and facial localization:

1. **Vision Transformers (Hugging Face Hub):**
   - **`umm-maybe/AI-image-detector`**: Pre-trained Vision Transformer (ViT) fine-tuned for distinguishing synthetic diffusion imagery from real camera captures. Run dual-crop (full image and cropped face).
   - **`dima806/deepfake_vs_real_image_detection`**: Pre-trained ViT specialized in face swap and morphing artifacts.
   - **`buildborderless/CommunityForensics-DeepfakeDet-ViT`**: Community forensic model trained on diverse digital manipulation artifacts.
2. **OpenAI CLIP (`openai/clip-vit-base-patch32`):**
   - Frozen vision transformer backbone used to extract 512-dimensional semantic image representations, paired with our trained linear forensic probing head (`models/clip_probe.pth`).
3. **OpenCV Caffe SSD Face Detector:**
   - Architecture: ResNet-10 SSD (`deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel`).
   - Purpose: Detects facial bounding boxes and landmarks under extreme angles and partial occlusions.
4. **Local Custom Models (Trained in this project):**
   - **Patch ResNet-18 (`models/patch_cnn.pth`)**: Custom patch-level classifier trained on native-resolution 128×128 image patches to prevent downsampling artifact loss.
   - **Face Swap Boundary CNN (`models/face_swap_cnn.pth`)**: ResNet-18 model trained specifically on facial perimeter morphing boundaries.
   - **Calibrated Decision Fusion (`models/fusion.joblib`, `models/calibrator.joblib`)**: Logistic Ridge regression combiner with Isotonic calibration fitted on validation splits.

### 4. Open-Source Libraries

- **UI & Deployment:** [Gradio 4.44.1](https://github.com/gradio-app/gradio) (Apache 2.0)
- **Deep Learning:** [PyTorch](https://pytorch.org/) & [torchvision](https://pytorch.org/vision/) (BSD-style), [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache 2.0), [timm](https://github.com/huggingface/pytorch-image-models) (Apache 2.0)
- **Computer Vision:** [OpenCV (`opencv-python`)](https://opencv.org/) (Apache 2.0 / BSD)
- **Audio & Signal Processing:** [librosa](https://librosa.org/) (ISC), [soundfile](https://python-soundfile.readthedocs.io/) (BSD-3-Clause), [SciPy](https://scipy.org/) (BSD-new)
- **Machine Learning & Calibration:** [scikit-learn](https://scikit-learn.org/) (BSD-3-Clause)
- **Imaging & I/O:** [Pillow (PIL)](https://python-pillow.org/) (HPND)
- **Scientific Computing:** [NumPy](https://numpy.org/) (BSD-new)

### 5. Standards and Forensic Assets
- **C2PA Standard:** Scans byte streams for Coalition for Content Provenance and Authenticity (`c2pa`), `jumb`, and SynthID digital signatures.
- **Watermark Templates:** Edge matching against visible generative watermark signatures (`assets/watermark_template.png`) with normalized multi-scale template correlation.

---

## Empirical Evaluation Benchmarks

All performance claims are strictly backed by empirical evaluations documented in [results/final_report.md](results/final_report.md) across held-out splits:

### 1. Classification Performance on Held-Out Test Split
| Category | Evaluated Models | Baseline AUC | **Our System AUC** | **Recall @ 5% FPR** | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Full AI Synthesis (`ai_full`)** | Midjourney v5, SD v1.5 | 0.5908 | **0.9520** | **91.25%** | **PASSED** |
| **Facial Swaps (`faceswap`)** | FaceForensics++, SBI | 0.4666 | **0.9580** | **90.80%** | **PASSED** |
| **AI Edits (`ai_edit`)** | Inpainting, Alpha Cutouts | 0.4450 | **0.9610** | **92.50%** | **PASSED** |
| **Real Camera & Compressed** | Camera photos, JPEG Q30 | - | - | **&le; 3.2% FPR** | **PASSED** |

### 2. Generalization on Unseen Techniques (Leave-One-Subfolder-Out)
When evaluated on novel generator architectures withheld during training, individual neural classifiers degrade toward baseline. This demonstrates why **multimodal triangulation** (relying on ELA, FFT, PRNU noise variance, and facial boundary seams) is mathematically required to prevent single-model blind spots.

---

## Quick Start

### 1. Installation
Clone the repository and install the dependencies in a virtual environment:

```bash
git clone https://github.com/aswinnachi/deepfake-analysis.git
cd deepfake-analysis

python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
Start the local Gradio interface:

```bash
python app.py
```
Open **`http://127.0.0.1:7860`** in your browser.

*To create a temporary public share link (for remote devices / judges):*
```bash
python app.py --share
```

*To enable the internal model fine-tuning panel:*
```bash
python app.py --admin
```

---

## CLI & Automation

Analyze any image, video, or audio file directly from the terminal:

```bash
# Analyze an image
python analyze.py samples/demo/woman.jpg

# Analyze a video file
python analyze.py samples/demo/sample_video.mp4

# Analyze an audio recording
python analyze.py samples/demo/sample_audio.wav

# Run smoke test suite
python -m tests.test_smoke
```

---

## Operational Boundaries & Limitations

In compliance with forensic ethics and transparency:
1. **Severe Social Media Re-compression:** Heavy JPEG compression (Q &lt; 35) or WhatsApp re-encoding wipes out high-frequency sensor noise and quantization tables, lowering confidence.
2. **Novel Generative Architectures:** Brand-new generator architectures with no prior training can evade ViT models; in these scenarios, the system safely triggers **INCONCLUSIVE** rather than issuing a confident false verdict.
3. **Absence of Artifacts &ne; Authenticity:** Fully synthetic images often have pristine, uniform textures. Absence of artifacts does not prove authenticity; our pipeline demands positive physical camera signatures to claim **LIKELY REAL**.
4. **Intended Use:** This tool is designed as an investigative triage assistant and screening aid. It is **not** an automated judicial determinant.

---

## Authors & Acknowledgments
- **Team:** HNX26PSI10 (Multimodal Deepfake & Digital Forensics)
- Developed for the National Hackathon Challenge.
