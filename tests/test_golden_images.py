import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from forensics.pipeline import analyze

def test_golden_images():
    ai_face_path = "samples/ai_face.jpg"
    real_phone_path = "samples/real_phone.jpg"
    
    if os.path.exists(ai_face_path):
        print("\n--- AI FACE ---")
        res = analyze(ai_face_path)
        for sig in res.get("_signals", []):
            if sig:
                print(f"{sig.name}: {sig.score:.2f}")
        print(f"Verdict: {res['verdict']}")
        print(f"Real Pct: {res['authenticity_real_pct']}%")
        assert res["verdict"] != "LIKELY REAL", f"Failed on ai_face, got: {res['verdict']}"
        
    if os.path.exists(real_phone_path):
        print("\n--- REAL PHONE ---")
        res = analyze(real_phone_path)
        for sig in res.get("_signals", []):
            if sig:
                print(f"{sig.name}: {sig.score:.2f}")
        print(f"Verdict: {res['verdict']}")
        print(f"Real Pct: {res['authenticity_real_pct']}%")
        assert res["verdict"] != "LIKELY FAKE / MANIPULATED", f"Failed on real_phone, got: {res['verdict']}"
        
    print("All golden tests passed!")

if __name__ == "__main__":
    test_golden_images()
