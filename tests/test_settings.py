"""Each model is searched with the speech settings and the sound weight chosen with it on the tuning sets
(docs/RESULTS.md, sixth and seventh rounds), as the benchmarks searched it."""

import json
from pathlib import Path

from whenframe import Library, SearchConfig, Searcher, Settings, temporal
from whenframe.encoder import MODELS
from whenframe.search import chosen
from whenframe.sounds import SoundConfig
from whenframe.speech import SpeechConfig

ROOT = Path(__file__).resolve().parents[1]
TUNED = json.loads((ROOT / "results" / "compare_tuned.json").read_text())
SHIPPED = TUNED["v2"]  # what was chosen for the temporal model the package ships


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
