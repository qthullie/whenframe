import numpy as np
import pytest

from whenframe.moments import on_grid, smooth, spans, windows


def test_on_grid_holds_each_value_until_the_next_sample():
    times = np.array([0.0, 2.0, 5.0])
    values = np.array([1.0, 2.0, 3.0])
    grid, held = on_grid(times, values, duration=6.0, step=1.0)
    assert grid.tolist() == [0, 1, 2, 3, 4, 5]
    assert held.tolist() == [1, 1, 2, 2, 2, 3]


def test_smooth_keeps_a_constant_curve():
    assert np.allclose(smooth(np.full(20, 0.3), sigma=2.0), 0.3)


def test_a_moment_grows_around_its_peak_and_stops_at_the_background():
    grid = np.arange(0, 30, 1.0)
    curve = np.full(30, 0.1)
    curve[10:15] = [0.3, 0.6, 0.8, 0.6, 0.3]
    (s,) = spans(grid, curve, step=1.0, alpha=0.5, count=1)
    assert (s.start, s.end) == (11.0, 14.0)
    assert s.peak == 0.8 and s.floor == 0.1


def test_further_moments_do_not_overlap():
    grid = np.arange(0, 40, 1.0)
    curve = np.full(40, 0.0)
    curve[5:8] = 1.0
    curve[25:30] = 0.7
    found = spans(grid, curve, step=1.0, alpha=0.5, count=3)
    assert [(s.start, s.end) for s in found[:2]] == [(5.0, 8.0), (25.0, 30.0)]
    for a in found:
        for b in found:
            if a is not b:
                assert a.end <= b.start or b.end <= a.start


def test_short_moments_are_widened_to_the_minimum_length():
    grid = np.arange(0, 10, 0.5)
    curve = np.zeros(20)
    curve[8] = 1.0
    (s,) = spans(grid, curve, step=0.5, alpha=0.9, count=1, min_length=2.0)
    assert s.end - s.start == 2.0 and s.start <= 4.0 < s.end


def test_windows_are_centred_on_the_best_frames_and_do_not_repeat_them():
    times = np.arange(30.0)
    values = np.zeros(30)
    values[[4, 6, 20]] = [1.0, 0.9, 0.8]
    first, second = windows(times, values, duration=30.0, length=10.0, count=2)
    assert (first.start, first.end, first.peak) == (0.0, 10.0, 1.0)  # 4.5 +- 5, kept inside the video
    assert (second.start, second.end, second.peak) == (15.5, 25.5, 0.8)  # 6 is too close to 4


def test_the_search_defaults_are_the_tuned_ones():
    import json
    from pathlib import Path

    from whenframe import SearchConfig, Searcher
    from whenframe.encoder import MODELS

    tuned = Path(__file__).resolve().parents[1] / "results" / "didemo_tuned.json"
    if not tuned.exists():
        pytest.skip("no tuning run")
    t = json.loads(tuned.read_text())
    windows, spans = t["whenframe_dedup"]["params"], t["whenframe_bank"]["params"]
    c = SearchConfig()
    assert (c.window, c.per_video) == (windows["window"], windows["per_video"])
    assert (c.step, c.sigma, c.alpha, c.contrast, c.min_length) == tuple(
        spans[k] for k in ("step", "sigma", "alpha", "contrast", "min_length")
    )
    spec = MODELS["siglip2-base"]
    assert (spec.bank_beta, spec.bank_lambda) == (windows["beta"], windows["lam"])
    assert (spec.bank_beta, spec.spans_lambda) == (spans["beta"], spans["lam"])

    class Bare(Searcher):  # only the parameters, no library
        def __init__(self, config):
            self.config = config

    assert Bare(SearchConfig(moments="windows"))._tuned("siglip2-base") == (50.0, 1.0)
    assert Bare(SearchConfig(moments="spans"))._tuned("siglip2-base") == (50.0, 0.5)


def test_auto_places_spans_in_libraries_of_long_videos():
    from whenframe import SearchConfig, Searcher
    from whenframe.library import Video

    class Shelf:  # only what the choice reads
        def __init__(self, durations):
            self.videos = [Video(i, f"v{i}", d, 0, 1) for i, d in enumerate(durations)]

        def snapshot(self):
            return self.videos, {}, np.zeros(0)

    class Bare(Searcher):
        def __init__(self, library):
            self.config, self.library = SearchConfig(), library

    assert Bare(Shelf([20.0, 30.0, 300.0])).moments == "windows"  # mostly short clips: DiDeMo's kind
    assert Bare(Shelf([45.0, 289.0, 600.0])).moments == "spans"  # mostly long videos: YouCook2's kind
    assert Bare(Shelf([0.0, 0.0, 0.0, 120.0])).moments == "spans"  # photos (no duration) do not count
    assert Bare(Shelf([45.0, 289.0]))._tuned("siglip2-base") == (50.0, 0.5)  # with the spans' bank strength


def test_every_model_a_library_is_built_with_has_the_query_bank_it_was_measured_with():
    from whenframe import bank
    from whenframe.encoder import DEFAULT_LIBRARY

    # a new library's, and those of libraries built in the modes of earlier versions
    for model in (*DEFAULT_LIBRARY.split("+"), "siglip2-base", "siglip2-b32", "siglip2-so400m"):
        b = bank.load(model)
        assert b is not None and b.shape[0] == 4096, model
