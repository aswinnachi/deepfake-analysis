"""Turn FaceForensics++ / DFDC / your own videos into training images.
    python scripts/extract_frames.py path/to/videos data/fake/deepfakes --every 15
"""
import sys, os, glob, cv2

src, dst = sys.argv[1], sys.argv[2]
every = int(sys.argv[sys.argv.index("--every") + 1]) if "--every" in sys.argv else 15
os.makedirs(dst, exist_ok=True)
for v in glob.glob(os.path.join(src, "**", "*.*"), recursive=True):
    if not v.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    cap = cv2.VideoCapture(v); i = 0; base = os.path.splitext(os.path.basename(v))[0]
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            cv2.imwrite(os.path.join(dst, f"{base}_{i:05d}.png"), f)
        i += 1
    cap.release()
print("done")
