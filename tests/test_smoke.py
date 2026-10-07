import sys, os, tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np, cv2
from forensics.pipeline import analyze, to_markdown

def test_image_pipeline():
    rng = np.random.default_rng(0)
    img = (rng.normal(128, 30, (480, 640, 3))).clip(0, 255).astype(np.uint8)
    cv2.rectangle(img, (200, 120), (400, 360), (180, 140, 120), -1)
    p = os.path.join(tempfile.mkdtemp(), "t.jpg"); cv2.imwrite(p, img)
    res = analyze(p)
    assert 0 <= res["authenticity_real_pct"] <= 100 and res["explanation"]
    print(to_markdown(res))

if __name__ == "__main__":
    test_image_pipeline()
