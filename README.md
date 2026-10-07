# Multimodal Deepfake & Digital Forensics (HNX26PSI10)

Detects fake / manipulated **images, videos and audio**. Every verdict comes with an
**authenticity score**, the **evidence** behind it, and a **heatmap** showing *where* the
manipulation is. No single artefact decides: independent signals are fused.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/generate_samples.py                     # create demo images (real + fake)
python app.py                                          # web UI at http://127.0.0.1:7860
python analyze.py samples/fake_splice.jpg              # or CLI (also video / audio)
python -m tests.test_smoke                             # sanity check
```
Works immediately with the built-in forensic signals (no training needed). Training the
optional models below improves accuracy.

## Signals (what gets fused)
| Modality | Signal | Catches |
|---|---|---|
| Image | Error Level Analysis (`ela.py`) | splices, inpainting, edited regions (+ heatmap) |
| Image | Fourier spectrum (`frequency.py`) | GAN / diffusion up-sampling fingerprints |
| Image | Noise residual consistency (`noise.py`) | composited or over-smooth synthetic content (+ heatmap) |
| Image | Face vs context statistics (`face.py`) | face swaps / blended faces (+ face box) |
| Image | Metadata / provenance (`metadata.py`) | AI-generator tags in EXIF / PNG text |
| Image | Learned CNN on patches (`cnn.py`, optional) | generator textures, patch-level localisation |
| Video | Per-frame forensics + temporal flicker (`video.py`) | frame-to-frame instability of swapped faces |
| Audio | Silence, bandwidth, prosody (`audio.py`) | voice cloning / TTS |
| A+V | Mouth motion vs speech energy (`av_sync.py`) | lip-sync fakes, dubbed or cloned audio |
| All | Compression estimate (`compression.py`) | lowers confidence on heavily re-encoded media |

`fusion.py` combines signals in logit space (or a learned logistic regression if
`models/fusion.joblib` exists), shrinks the score toward 50% when media is heavily compressed,
and reports a confidence level based on signal agreement and coverage.

## Face detection
The system uses **OpenCV DNN (Caffe SSD)** for face detection by default, which handles
angled, profile, and partially occluded faces far better than Haar cascades. The DNN model
files are included in `models/`. Falls back to Haar if the DNN files are missing.

## Datasets
Put images here (video: run `scripts/extract_frames.py` first):
```
data/real/                      # real photos / frames (FFHQ, CelebA, COCO, FF++ originals)
data/fake/<method_name>/        # one folder PER fake method (StyleGAN, SD, Deepfakes, FaceSwap, ...)
```
Suggested: FaceForensics++, DFDC, CIFAKE, GenImage, ArtiFact. Keep one folder per generator:
that is what makes the unseen-technique evaluation possible.

Quick start with synthetic samples:
```bash
python scripts/generate_samples.py                     # generates demo images for testing
```

## Train and evaluate
```bash
python scripts/extract_frames.py FF++/manipulated/Deepfakes data/fake/deepfakes --every 15
python -m train.train_fusion --max-per-class 400   # prints leave-one-generator-out AUC (unseen techniques)
python -m train.train_cnn --epochs 5 --holdout stylegan   # optional, GPU recommended; prints unseen-generator AUC
python -m scripts.evaluate --max-per-class 200     # AUC/ACC/false-positive rate under JPEG q30, half-res, crop, blur
```
`train_fusion` holds each fake method out in turn, so the reported AUC is on techniques the
model never saw. Report those numbers (and `results/robustness.json`) to the judges.

## Mapping to the judging criteria
- **Unseen fakes:** low-level signals (ELA, spectrum, noise, face consistency) are not tied to one generator; leave-one-generator-out evaluation proves it.
- **Localisation:** heatmap overlay + face box (`forensics/visualize.py`).
- **Real content not flagged:** report the false-positive rate from `scripts/evaluate.py`; the INCONCLUSIVE band (35-65%) avoids forced calls.
- **Compression / crop / re-encode:** degradation estimate lowers confidence; CNN trained with random JPEG/resize/blur augmentation.
- **Audio-video match:** `av_sync.py`.
- **Explanations:** every signal contributes a sentence, sorted by influence.

## Resources used
- **OpenCV** (BSD) – image processing, DNN face detection
- **OpenCV DNN face model** (`res10_300x300_ssd_iter_140000.caffemodel`) – pre-trained face detector from OpenCV samples
- **PyTorch / torchvision** (BSD) – optional CNN training (ResNet-18 backbone, ImageNet weights)
- **librosa** (ISC) – audio feature extraction
- **scikit-learn** (BSD) – logistic regression for fusion model
- **Gradio** (Apache 2.0) – web UI
- **PIL/Pillow** (MIT-like) – image I/O and metadata parsing

## Honest limitations
- Heuristic thresholds are priors; calibrate with `train_fusion` on your own data before demoing.
- Audio cues are weak alone; a pretrained anti-spoofing model (e.g. AASIST / wav2vec2-based) can be added as another `Signal`.
- A score is evidence, not proof. Heavily compressed or tiny media gives low-confidence output by design.

## Extending
Add a module returning `Signal(name, score_0_to_1, weight, detail, heatmap)` and include it in
`forensics/pipeline.py`. The fusion, explanation and heatmap layers pick it up automatically.
