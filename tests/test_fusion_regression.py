import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forensics.utils import Signal
from forensics.fusion import fuse

def test_case_a():
    signals = [
        Signal("metadata", 0.55, 0.5, "meta"),
        Signal("ela", 0.00, 1.0, "ela"),
        Signal("frequency", 0.23, 1.0, "freq"),
        Signal("noise", 0.22, 0.8, "noise"),
        Signal("face_consistency", 0.10, 0.5, "face"),
        Signal("cnn", 0.71, 0.5, "cnn")
    ]
    res = fuse(signals, degradation=0.45, use_learned=False)
    assert res["verdict"] != "LIKELY FAKE / MANIPULATED", f"Failed Case A: {res['verdict']}"
    assert res["authenticity_real_pct"] >= 50.0, f"Failed Case A real pct: {res['authenticity_real_pct']}"
    print("test_case_a passed")

def test_case_b():
    signals = [
        Signal("metadata", 0.55, 0.5, "meta"),
        Signal("ela", 0.00, 1.0, "ela"),
        Signal("frequency", 0.17, 1.0, "freq"),
        Signal("noise", 0.04, 0.8, "noise"),
        Signal("face_consistency", 0.00, 0.5, "face"),
        Signal("cnn", 0.56, 0.5, "cnn"),
        Signal("hf_ai_image", 0.79, 2.0, "hf1"),
        Signal("hf_deepfake", 0.99, 2.0, "hf2")
    ]
    res = fuse(signals, degradation=0.38, use_learned=False)
    assert res["authenticity_real_pct"] < 45.0, f"Failed Case B real pct: {res['authenticity_real_pct']}"
    assert res["verdict"] != "LIKELY REAL", f"Failed Case B verdict: {res['verdict']}"
    print("test_case_b passed")

def test_case_c():
    signals = [
        Signal("hf_ai_image", 0.10, 2.0, "hf1"),
        Signal("hf_deepfake", 0.08, 2.0, "hf2"),
        Signal("ela", 0.00, 1.0, "ela"),
        Signal("frequency", 0.20, 1.0, "freq"),
        Signal("noise", 0.20, 0.8, "noise"),
        Signal("face_consistency", 0.10, 0.5, "face"),
        Signal("cnn", 0.40, 0.5, "cnn"),
        Signal("metadata", 0.55, 0.5, "meta")
    ]
    res = fuse(signals, degradation=0.2, use_learned=False)
    assert res["verdict"] == "LIKELY REAL", f"Failed Case C verdict: {res['verdict']}"
    print("test_case_c passed")

def test_case_d():
    signals = [
        Signal("ela", 0.90, 1.0, "ela"),
        Signal("noise", 0.80, 0.8, "noise"),
        Signal("face_consistency", 0.70, 0.5, "face"),
        Signal("hf_ai_image", 0.50, 2.0, "hf1"),
        Signal("hf_deepfake", 0.50, 2.0, "hf2"),
        Signal("frequency", 0.50, 1.0, "freq"),
        Signal("cnn", 0.50, 0.5, "cnn"),
        Signal("metadata", 0.50, 0.5, "meta")
    ]
    res = fuse(signals, degradation=0.0, use_learned=False)
    assert res["authenticity_real_pct"] < 50.0, f"Failed Case D real pct: {res['authenticity_real_pct']}"
    print("test_case_d passed")

if __name__ == "__main__":
    test_case_a()
    test_case_b()
    test_case_c()
    test_case_d()
    print("All tests passed!")
