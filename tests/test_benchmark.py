"""The evaluation code on a corpus small enough to know every answer by hand."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark import Frozen, iou, run  # noqa: E402

PARAMS = {"step": 0.5, "sigma": 0.0, "alpha": 0.5, "contrast": 0.0, "min_length": 1.0, "per_video": 1}


def corpus():
    """Three 10-second videos at one frame per second; directions 1 and 2 stand for two descriptions."""
    e = np.eye(3)
    background = 0.2 * e[1] + e[0]
    videos = [np.tile(background, (10, 1)) for _ in range(3)]
    videos[0][3:6] = e[1] + 0.3 * e[0]  # description 1 happens in video 0, from 3 s to 6 s
    videos[1][7] = e[2]  # description 2 happens in video 1, from 7 s to 8 s
    emb = np.concatenate(videos)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return Frozen(
        names=["a", "b", "c"], emb=emb, times=np.tile(np.arange(10.0), 3), starts=np.array([0, 10, 20]),
        counts=np.array([10, 10, 10]), durations=np.full(3, 10.0), keep=np.ones(30, bool), cuts=[[], [], []],
    )  # fmt: skip


QUERIES = [{"text": "one", "video": 0, "moment": (3.0, 6.0)}, {"text": "two", "video": 1, "moment": (7.0, 8.0)}]
TEXT = np.eye(3)[[1, 2]]


def test_iou():
    assert iou((0, 10), (5, 15)) == pytest.approx(5 / 15)
    assert iou((0, 1), (2, 3)) == 0.0


def test_moments_found_first_count_at_every_k():
    r = run(corpus(), QUERIES, TEXT, "whenframe", PARAMS)
    assert r["corpus"]["R@1 IoU>=0.5"] == 1.0
    assert r["corpus"]["R@1 IoU>=0.7"] == 1.0
    assert r["single"]["mIoU"] == pytest.approx(1.0)
    assert r["video"]["R@1"] == 1.0


def test_a_better_match_elsewhere_pushes_the_right_moment_down_one_rank():
    f = corpus()
    f.emb[20:22] = np.eye(3)[1]  # video 2 shows description 1 even more clearly, in its first two seconds
    r = run(f, QUERIES, TEXT, "whenframe", {**PARAMS, "contrast": 1.0})
    assert r["corpus"]["R@1 IoU>=0.5"] == 0.5  # description 1 now comes second
    assert r["corpus"]["R@5 IoU>=0.5"] == 1.0
    assert r["video"]["R@1"] == 0.5
    assert r["single"]["R1 IoU>=0.5"] == 1.0  # inside its own video it is still placed right


def test_selecting_frames_keeps_videos_apart():
    f = corpus()
    sub = f.select(f.every(2))
    assert sub.counts.tolist() == [5, 5, 5] and sub.starts.tolist() == [0, 5, 10]
    assert sub.times[:5].tolist() == [0, 2, 4, 6, 8]


def test_a_second_encoder_identical_to_the_first_changes_nothing():
    f = corpus()
    f.bank = np.eye(3)[[0, 1]]
    f.extra = {"mobileclip-s2": f.emb.copy()}
    f.extra_banks = {"mobileclip-s2": f.bank.copy()}
    texts = {"siglip2-base": TEXT, "mobileclip-s2": TEXT}
    p = {"window": 3.0, "per_video": 2, "beta": 10.0, "lam": 0.5}
    ref = run(f, QUERIES, texts, "frame_window_bank", p)
    for method, extra in (("mobileclip_bank", {}), ("ensemble_bank", {"beta2": 10.0, "lam2": 0.5, "weight": 1.0})):
        r = run(f, QUERIES, texts, method, {**p, **extra})
        assert (r["corpus"], r["single"], r["video"]) == (ref["corpus"], ref["single"], ref["video"])
    sub = f.select(f.every(2))
    assert len(sub.extra["mobileclip-s2"]) == len(sub.emb) == 15
