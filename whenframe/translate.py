"""Descriptions written in French, searched in English: Opus-MT (Helsinki-NLP, Apache 2.0) run with ONNX Runtime.

SigLIP 2 read many languages in training; PE core B/16's text tower and CLAP read English. A description the tool
takes for French (`is_french`) is translated into English on the computer, greedily, a sentence at a time, and
searched as an English one (docs/PROTOCOL.md, twelfth round). Nothing leaves the computer: the translator is
downloaded once, like the models.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")  # see encoder.py

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from .encoder import cache_dir, download  # noqa: E402

# Opus-MT fr-en as Xenova exports it to ONNX (Helsinki-NLP/opus-mt-fr-en, Apache 2.0), at the revision measured in
# the twelfth round, in the precision chosen there on the tuning sets; each file's SHA-256 is checked
REPO = "Xenova/opus-mt-fr-en"
REVISION = "6b166a182780e118c997879d0ad5be4b53671644"
ENCODER, DECODER = "encoder_model.onnx", "decoder_model.onnx"  # full precision (its 8-bit export found less)
FILES = {
    "config.json": "6c2851f154d7b88c7767e4fdcf8a5694a36a456bfa6cbcde1b13fdbade3b56f6",
    "tokenizer.json": "8391785c1a2139e7af4678571ccd8dc654ecbb72e4be186940f65d7c604f0246",
    f"onnx/{ENCODER}": "6e18e05c0a458a99296c526b6b2a989158ecf890a86072f9b3960df52bfc99a7",
    f"onnx/{DECODER}": "1844d3c6f595df70cd0745084afa8fd9e98227bce7ac8fb9a3f1ddd41bcd171b",
}

# The marks of a French description, and of an English one (docs/PROTOCOL.md, twelfth round): French's function
# words, and words frequent in descriptions of videos and photos that English does not have; English's function
# words. Words common to both languages count for neither (a, on, son, pour, plus). Each accented letter and each
# elision (l', d', qu', s'...) is a French mark too. Fixed on the tuning sets' descriptions in both languages: no
# English one has any of these French marks.
FRENCH = frozenset(
    """le la les un une des du de au aux et ou en y dans sur sous avec sans par qui que quoi dont où
est sont était ont fait font va vont peut il elle ils elles lui eux nous vous je tu moi toi sa ses leur leurs ce cet
cette ces mon ma mes ton ta tes notre nos votre vos se ne pas rien très aussi encore tout tous toute toutes comme
mais quand pendant depuis contre chaque plusieurs beaucoup peu trop bien vers devant derrière puis après avant chez
entre autour dessus dessous gauche droite droit quelque quelques chose choses homme hommes femme femmes personne
personnes enfant enfants fille filles gens deux trois quatre cinq petit petits jeune jeunes vieux vieille plage
neige mer montagne montagnes chien chiens voiture voitures anniversaire mariage vacances jardin maison piscine soleil
nuit ciel arbre arbres fleur fleurs oiseau oiseaux bateau cheval chevaux poisson poissons ville feu feux chanson
repas""".split()
)
ENGLISH = frozenset(
    """the an of in at with without for by who which is are was were he she it they we you his
her its their this these those my your to from and or not very into over under then after before while there up
down out off around through""".split()
)
ACCENTS = frozenset("àâçéèêëîïôûùüÿœæ")
LETTERS = "a-zàâçéèêëîïôûùüÿœæ"
WORD = re.compile(f"[{LETTERS}]+")
ELISION = re.compile(f"(?<![{LETTERS}])(?:l|d|j|m|n|s|t|c|qu|jusqu|lorsqu|puisqu)['’](?=[{LETTERS}h])")


def is_french(text: str) -> bool:
    """Whether a description has more French marks than English ones (a tie, or no mark, is English)."""
    low = text.lower()
    words = WORD.findall(low)
    french = sum(w in FRENCH for w in words) + sum(ch in ACCENTS for ch in low) + len(ELISION.findall(low))
    return french > sum(w in ENGLISH for w in words)


class Translator:
    """An Opus-MT model as Xenova exports it to ONNX: tokenizer.json, config.json and onnx/ENCODER, onnx/DECODER."""

    def __init__(self, folder: str | Path, encoder: str = "encoder_model.onnx", decoder: str = "decoder_model.onnx",
                 threads: int | None = None):  # fmt: skip
        folder = Path(folder)
        cfg = json.loads((folder / "config.json").read_text())
        self.start, self.eos, self.pad = cfg["decoder_start_token_id"], cfg["eos_token_id"], cfg["pad_token_id"]
        spec = json.loads((folder / "tokenizer.json").read_text())
        if (spec.get("normalizer") or {}).get("precompiled_charsmap", "") is None:
            spec["normalizer"] = None  # an empty "Precompiled" normaliser does nothing, and the library refuses it
        self.tokenizer = Tokenizer.from_str(json.dumps(spec))
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.encoder = ort.InferenceSession(str(folder / "onnx" / encoder), opts, providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(folder / "onnx" / decoder), opts, providers=["CPUExecutionProvider"])

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


def fetch() -> Path:
    """The translator's files, downloaded once into the cache (about 430 MB)."""
    folder = cache_dir() / "models" / REPO.replace("/", "--")
    for name, digest in FILES.items():
        download(f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}", folder / name, digest)
    return folder


_translator: Translator | None = None
_lock = threading.Lock()


def to_english(text: str) -> str:
    """A description as it is searched: in English, translated on this computer when it is taken for French; as
    typed otherwise. The translator is downloaded the first time a description needs it."""
    if not is_french(text):
        return text
    global _translator
    with _lock:
        if _translator is None:
            _translator = Translator(fetch(), ENCODER, DECODER)
        out = _translator([text], batch=1)[0].strip()
    return out or text
