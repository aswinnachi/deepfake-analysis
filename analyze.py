"""CLI:  python analyze.py path/to/file [--json out.json]"""
import sys, json
from forensics.pipeline import analyze, to_markdown

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    res = analyze(sys.argv[1])
    print(to_markdown(res))
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        json.dump({k: v for k, v in res.items() if k not in ("overlay", "_signals")}, open(out, "w"), indent=2)
