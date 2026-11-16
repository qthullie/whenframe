"""Searching what is said: the query's words, the segments' scores, and the library's transcripts."""

import shutil
import sqlite3

import numpy as np
import pytest
from test_library import ToyEncoder, make_video

from whenframe import Library, SearchConfig, Searcher, Settings, speech


def test_match_expression():
    assert speech.match_expression("Add the onions to the pan, the onions!") == '"add" OR "onions" OR "pan"'
    assert speech.match_expression("ajoute les oignons dans la poêle") == '"ajoute" OR "oignons" OR "poêle"'
    assert speech.match_expression("the and of a") is None


def test_segment_and_frame_scores():
    db = sqlite3.connect(":memory:")
    db.executescript(speech.SCHEMA)
    db.executemany("insert into speech (video_id, start, end, text) values (?, ?, ?, ?)",
                   [(7, 2.0, 4.0, "now we chop the onions"), (7, 9.0, 10.0, "and fry them"),
                    (8, 0.0, 3.0, "Chopping garlic")])  # fmt: skip
    hits = speech.segment_scores(db, "chopped onions")
    assert sorted(h[4] for h in hits) == ["Chopping garlic", "now we chop the onions"]  # stemmed: chop, chopping
    best = max(hits, key=lambda h: h[3])
    assert best[4] == "now we chop the onions"
    times = np.arange(12, dtype=np.float32)
    ranges = {7: (0, 12)}
    s = speech.frame_scores(times, ranges, hits, speech.SpeechConfig(pad=0.0))
    assert np.flatnonzero(s).tolist() == [2, 3] and np.isclose(s[3], best[3])  # the seconds 2-3 and 3-4
    padded = speech.frame_scores(times, ranges, hits, speech.SpeechConfig(pad=2.0, normalise="best"))
    assert np.flatnonzero(padded).tolist() == [0, 1, 2, 3, 4, 5] and np.isclose(padded.max(), 1.0)
    sparse = speech.frame_scores(np.array([0.0, 5.0, 10.0]), {7: (0, 3)}, hits, speech.SpeechConfig())
    assert np.flatnonzero(sparse).tolist() == [0]  # frame 0 stands for 0-5 s, when the segment is said


class FakeTranscriber:
    def __init__(self, said):
        self.said, self.calls = said, []

    def __call__(self, path, duration):
        self.calls.append(path)
        return self.said.get(path.rsplit("/", 1)[-1], [])


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_library_searches_what_is_said(tmp_path):
    make_video(str(tmp_path / "a.mp4"), [("grey", 8), ("grey", 8)])
    make_video(str(tmp_path / "b.mp4"), [("grey", 16)])
    said = {"a.mp4": [(11.0, 13.0, "happy birthday to you")], "b.mp4": [(1.0, 2.0, "hello")]}
    fake = FakeTranscriber(said)
    lib = Library(tmp_path / "lib", Settings(model="toy"), encoder=ToyEncoder(), transcriber=fake)
    for name in ("a", "b"):
        lib.add(tmp_path / f"{name}.mp4")
    assert fake.calls == []  # speech is off until asked for
    lib.enable_speech("base")
    assert lib.needs_transcript((tmp_path / "a.mp4").resolve())
    for name in ("a", "b"):
        lib.add(tmp_path / f"{name}.mp4")  # frames unchanged: only transcribed
    assert len(fake.calls) == 2 and lib.stats()["speech_segments"] == 2
    assert Library(tmp_path / "lib", encoder=ToyEncoder()).settings.speech == "base"

    search = Searcher(lib, SearchConfig(window=4.0, speech=speech.SpeechConfig(weight=1.0)))
    grey = ToyEncoder().encode_texts(["grey"])[0]
    _, embs, _ = lib.snapshot()
    hits = search.search("birthday", k=3, scores={"toy": embs["toy"] @ grey})  # every frame is grey: speech decides
    assert hits[0].path.endswith("a.mp4") and hits[0].start <= 11.0 <= hits[0].end
    assert hits[0].said == "happy birthday to you"

    lib.remove(tmp_path / "a.mp4")
    assert lib.stats()["speech_segments"] == 1
