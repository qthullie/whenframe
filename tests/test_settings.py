"""Each model is searched with the speech settings and the sound weight chosen with it on the tuning sets
(docs/RESULTS.md, sixth, seventh and tenth rounds), as the benchmarks searched it."""

import json
import sys
from pathlib import Path

import numpy as np

from whenframe import Library, SearchConfig, Searcher, Settings, temporal
from whenframe.encoder import FUSED, MODELS
from whenframe.search import chosen
from whenframe.sounds import SoundConfig
from whenframe.speech import SpeechConfig

ROOT = Path(__file__).resolve().parents[1]
TUNED = json.loads((ROOT / "results" / "compare_tuned.json").read_text())
# what was chosen for the temporal model the package ships: v3, the seventh round's
SHIPPED = json.loads((ROOT / "results" / "round7_tuned.json").read_text())["v3"]
# what was chosen for v3 and v4 searched together, the tenth round's fusion tested in the eleventh
FUSION = json.loads((ROOT / "results" / "round11_tuned.json").read_text())["fusion"]
BOTH = ("siglip2-base+pe-core-b16", "temporal-siglip2-base+temporal-pe-core-b16")


def library(folder: Path, model: str, temporal_model: str = "") -> Library:
    return Library(folder, Settings(model=model, temporal=temporal_model), encoder=object())


def test_each_image_model_has_the_settings_chosen_with_it():
    for name, model in (("base", "siglip2-base"), ("b32", "siglip2-b32"), ("so400m", "siglip2-so400m")):
        spec, t = MODELS[model], TUNED[name]
        assert (spec.speech_pad, spec.sound_weight) == (t["speech"]["pad"], t["sound_weight"]), model
        assert (t["speech"]["weight"], t["speech"]["normalise"]) == (SpeechConfig().weight, SpeechConfig().normalise)


def test_the_shipped_model_carries_the_settings_chosen_with_it():
    assert temporal.load()["search"] == {"speech": SHIPPED["speech"], "sound_weight": SHIPPED["sound_weight"]}


def test_a_search_uses_the_settings_of_its_librarys_model(tmp_path):
    cases = {
        ("siglip2-base", ""): (0.0, 2.0),  # `--no-temporal`
        ("siglip2-base", "temporal-siglip2-base"): (SHIPPED["speech"]["pad"], SHIPPED["sound_weight"]),
        ("siglip2-b32", ""): (5.0, 1.5),
        ("siglip2-so400m", ""): (0.0, 1.5),
        ("siglip2-base+siglip2-b32", ""): (0.0, 2.0),  # several models together: none were chosen, the defaults
        BOTH: (FUSION["speech"]["pad"], FUSION["sound_weight"]),  # but for those measured together
    }
    for i, ((model, temporal_model), (pad, weight)) in enumerate(cases.items()):
        c = Searcher(library(tmp_path / str(i), model, temporal_model)).config
        assert (c.speech.pad, c.sound.weight) == (pad, weight), (model, temporal_model)


def test_settings_given_to_a_search_are_kept(tmp_path):
    given = SearchConfig(speech=SpeechConfig(weight=1.0), sound=SoundConfig(weight=0.5))
    c = Searcher(library(tmp_path, "siglip2-b32"), given).config
    assert (c.speech, c.sound) == (SpeechConfig(weight=1.0), SoundConfig(weight=0.5))


def test_without_the_temporal_models_settings_the_image_models_count(tmp_path):
    missing = library(tmp_path / "a", "siglip2-so400m", "temporal-not-installed")
    assert chosen(missing) == (SpeechConfig(), SoundConfig(weight=1.5))
    older = library(tmp_path / "b", "siglip2-b32", "temporal-siglip2-base")
    older._temporal = {"layers": 2, "heads": 4}  # weights from a file without settings
    assert chosen(older) == (SpeechConfig(pad=5.0), SoundConfig(weight=1.5))


def test_the_fusion_has_the_settings_chosen_with_it(tmp_path):
    c = Searcher(library(tmp_path, *BOTH)).config
    sp = FUSION["speech"]
    assert (c.speech.weight, c.speech.pad, c.speech.normalise, c.sound.weight) == (
        sp["weight"], sp["pad"], sp["normalise"], FUSION["sound_weight"])  # fmt: skip
    assert FUSED[BOTH[0]].weights == (1.0, FUSION["fuse"][2])  # PE's weight, 0.75


def test_the_fusion_ranks_frames_as_the_benchmark_measured_it(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import compare

    rng = np.random.default_rng(0)
    a, b = rng.normal(0.10, 0.05, 500), rng.normal(0.20, 0.03, 500)
    s = Searcher(library(tmp_path, *BOTH), SearchConfig(bank=0))  # the corrections are the same either way
    got = s.ranking_scores({"siglip2-base": a, "pe-core-b16": b})
    assert np.allclose(got, compare.fused(a[:, None], b[:, None], FUSION["fuse"][2])[:, 0])
    assert s._on_first_scale  # what is said is added with its weight as it is, as with one model
