"""English to French with Opus-MT (Helsinki-NLP, Apache 2.0), run with ONNX Runtime.

Used to measure how the search copes with French descriptions: the DiDeMo
descriptions are translated by machine, greedily, a batch at a time. The ONNX
export is Xenova/opus-mt-en-fr on the Hugging Face Hub.

    python scripts/translate.py --model-dir DIR "a dog jumps into the pool"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # see whenframe/encoder.py

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


class Translator:
    def __init__(self, folder: str | Path):
        folder = Path(folder)
        cfg = json.loads((folder / "config.json").read_text())
        self.start, self.eos, self.pad = cfg["decoder_start_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]
        spec = json.loads((folder / "tokenizer.json").read_text())
        if (spec.get("normalizer") or {}).get("precompiled_charsmap", "") is None:
            spec["normalizer"] = None  # an empty "Precompiled" normaliser does nothing, and the library refuses it
        self.tokenizer = Tokenizer.from_str(json.dumps(spec))
        opts = ort.SessionOptions()
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.encoder = ort.InferenceSession(str(folder / "onnx" / "encoder_model.onnx"), opts)
        self.decoder = ort.InferenceSession(str(folder / "onnx" / "decoder_model.onnx"), opts)

    def __call__(self, texts: list[str], batch: int = 16, max_tokens: int = 64) -> list[str]:
        out = []
        for i in range(0, len(texts), batch):
            encoded = self.tokenizer.encode_batch(texts[i : i + batch])
            n, length = len(encoded), max(len(e.ids) for e in encoded)
            ids = np.full((n, length), self.pad, dtype=np.int64)
            mask = np.zeros((n, length), dtype=np.int64)
            for j, e in enumerate(encoded):
                ids[j, : len(e.ids)] = e.ids
                mask[j, : len(e.ids)] = 1
            hidden = self.encoder.run(["last_hidden_state"], {"input_ids": ids, "attention_mask": mask})[0]
            words = np.full((n, 1), self.start, dtype=np.int64)
            done = np.zeros(n, dtype=bool)
            for _ in range(max_tokens):
                feed = {"input_ids": words, "encoder_hidden_states": hidden, "encoder_attention_mask": mask}
                logits = self.decoder.run(["logits"], feed)[0][:, -1]
                logits[:, self.pad] = -np.inf  # the padding token is never generated
                nxt = np.where(done, self.pad, logits.argmax(axis=1))
                words = np.concatenate([words, nxt[:, None]], axis=1)
                done |= nxt == self.eos
                if done.all():
                    break
            for row in words[:, 1:].tolist():
                row = row[: row.index(self.eos)] if self.eos in row else row
                out.append(self.tokenizer.decode([t for t in row if t != self.pad], skip_special_tokens=True))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("text", nargs="+")
    a = ap.parse_args()
    for line in Translator(a.model_dir)(a.text):
        print(line)
