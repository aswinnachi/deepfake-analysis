"""Check Hugging Face detector models on an image and verify label mappings.

Usage:
    python scripts/check_hf.py <image_path>
"""
import sys
import os

# Ensure project root is in sys.path when run directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from PIL import Image
from forensics.utils import load_rgb
from forensics.hf_detector import analyze, MODELS, _get_pipeline, _is_fake_label


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_hf.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Error: File not found: {path}")
        sys.exit(1)

    print(f"Analyzing: {path}")
    print("=" * 60)

    rgb = load_rgb(path)
    pil_img = Image.fromarray(rgb)

    print("\n--- Raw Pipeline Predictions & Label Mapping ---")
    for name, model_id in MODELS.items():
        try:
            pipe = _get_pipeline(name)
            if pipe is None:
                print(f"[{name}] Could not load model: {model_id}")
                continue
            preds = pipe(pil_img, top_k=None)
            print(f"\n[Model: {name} ({model_id})]")
            for p in preds:
                is_fake = _is_fake_label(p["label"])
                tag = "-> [FAKE keyword match]" if is_fake else "-> [REAL / capture]"
                print(f"  • Label: {p['label']:<15} Score: {p['score']:.4f}  {tag}")

            fake_score = sum(float(p["score"]) for p in preds if _is_fake_label(p["label"]))
            print(f"  Total Fake Probability: {fake_score:.4f} ({fake_score * 100:.1f}% fake)")
        except Exception as e:
            print(f"[{name}] Error running inference: {e}")

    print("\n--- Signals returned by forensics.hf_detector.analyze ---")
    signals = analyze(rgb)
    for s in signals:
        print(f"  Signal name : {s.name}")
        print(f"  Score (fake): {s.score:.4f}")
        print(f"  Weight      : {s.weight}")
        print(f"  Detail      : {s.detail}")
        print("-" * 40)


if __name__ == "__main__":
    main()
