"""Translate descriptions with Opus-MT (Helsinki-NLP, Apache 2.0), run with ONNX Runtime (whenframe/translate.py).

Used to measure how the search copes with French descriptions: the second and twelfth rounds translate the
benchmarks' descriptions into French by machine, greedily, a batch at a time (Xenova/opus-mt-en-fr's ONNX export).

    python scripts/translate.py --model-dir DIR "a dog jumps into the pool"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whenframe.translate import Translator  # noqa: E402, F401  (scripts import it from here)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    for line in Translator(a.model_dir)(a.text):
        print(line)
