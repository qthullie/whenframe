import sys
from pathlib import Path

import numpy as np
import pytest

from whenframe import temporal

DIM, WIDTH, LAYERS, HEADS = 16, 8, 2, 2


def params(rng: np.random.Generator, out_scale: float = 0.1, dim: int = DIM) -> dict:
    """Random weights of the model's shapes, small: the parity and the windows are tested, not the training."""

    def w(*shape, scale=0.3):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    d, h, f = WIDTH, WIDTH // 2, 2 * len(temporal.PERIODS)
    p = {
        "in_ln_g": 1 + w(dim),
        "in_ln_b": w(dim),
        "in_w": w(dim, d),
        "in_b": w(d),
        "time_w": w(f, d),
        "out_ln_g": 1 + w(d),
        "out_ln_b": w(d),
        "out_w": w(d, dim, scale=out_scale),
        "out_b": w(dim, scale=0.0),
    }
    for i in range(LAYERS):
        p |= {
            f"l{i}_ln1_g": 1 + w(d),
            f"l{i}_ln1_b": w(d),
            f"l{i}_qkv_w": w(d, 3 * d),
            f"l{i}_qkv_b": w(3 * d),
            f"l{i}_out_w": w(d, d),
            f"l{i}_out_b": w(d),
            f"l{i}_ln2_g": 1 + w(d),
            f"l{i}_ln2_b": w(d),
            f"l{i}_fc1_w": w(d, 2 * d),
            f"l{i}_fc1_b": w(2 * d),
            f"l{i}_fc2_w": w(2 * d, d),
            f"l{i}_fc2_b": w(d),
        }
    p |= {"scale": np.float32(np.log(10.0)), "bias": np.float32(-2.0), "lam": np.float32(1.0),
          "beta": np.float32(50.0), "h_f_ln_g": 1 + w(dim), "h_f_ln_b": w(dim), "h_f_w": w(dim, h),
          "h_q_w": w(dim, h), "h_q_b": 1 + w(h), "h_m_w": w(h, h), "h_m_b": w(h), "h_time_w": w(f, h),
          "h_ln1_g": 1 + w(h), "h_ln1_b": w(h), "h_qkv_w": w(h, 3 * h), "h_qkv_b": w(3 * h), "h_out_w": w(h, h),
          "h_out_b": w(h), "h_ln2_g": 1 + w(h), "h_ln2_b": w(h), "h_s_w": w(h), "h_s_b": np.float32(0.1),
          "h_d_w": w(h, 2), "h_d_b": w(2)}  # fmt: skip
    p["layers"], p["heads"] = LAYERS, HEADS
    return p


def script():
    """scripts/temporal.py, where the head tried in the fifth round lives."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import temporal as script

    return script


def frames(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    x = rng.standard_normal((n, DIM)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True), np.arange(n, dtype=np.float64) * 1.5


def test_untrained_model_leaves_the_embeddings_as_they_are():
    rng = np.random.default_rng(0)
    p = params(rng, out_scale=0.0)  # the output projection starts at zero, as in training
    x, t = frames(rng, 70)
    assert np.allclose(temporal.apply(p, x, t), x, atol=1e-6)


@pytest.mark.parametrize("n", [1, 5, temporal.WINDOW, temporal.WINDOW + 1, 100])
def test_any_number_of_frames_gives_unit_vectors(n):
    rng = np.random.default_rng(n)
    p = params(rng)
    x, t = frames(rng, n)
    out = temporal.apply(p, x, t)
    assert out.shape == x.shape
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert not np.allclose(out, x)


def test_a_short_video_is_one_window():
    rng = np.random.default_rng(1)
    p = params(rng)
    x, t = frames(rng, 20)
    out = x + temporal.forward(p, x, t)
    assert np.allclose(temporal.apply(p, x, t), out / np.linalg.norm(out, axis=1, keepdims=True), atol=1e-6)


def test_frames_far_apart_do_not_see_each_other():
    rng = np.random.default_rng(2)
    p = params(rng)
    x, t = frames(rng, 200)
    changed = x.copy()
    changed[0] = -changed[0]
    a, b = temporal.apply(p, x, t), temporal.apply(p, changed, t)
    assert not np.allclose(a[1 : temporal.WINDOW], b[1 : temporal.WINDOW])
    assert np.allclose(a[2 * temporal.WINDOW :], b[2 * temporal.WINDOW :], atol=1e-6)


def test_times_count_from_the_window_not_from_the_video():
    rng = np.random.default_rng(3)
    p = params(rng)
    x, t = frames(rng, 20)
    assert np.allclose(temporal.forward(p, x, t), temporal.forward(p, x, t + 600.0), atol=1e-4)


def test_the_head_without_correction_is_the_search_score():
    rng = np.random.default_rng(4)
    p = params(rng)
    p["h_s_w"][:] = 0.0
    x, t = frames(rng, 12)
    q = x[3] + 0.1
    off = rng.uniform(0, 0.1, 12).astype(np.float32)
    fg, reach = script().numpy_head(p, x, t, q, off)
    expected = np.exp(p["scale"]) * (x @ q - p["lam"] * off) + p["bias"] + p["h_s_b"]
    assert np.allclose(fg, expected, atol=1e-5)
    assert reach.shape == (12, 2) and (reach >= 0).all()


def test_moments_start_at_the_best_frames_and_do_not_overlap():
    t = np.arange(30, dtype=np.float64)
    fg = np.zeros(30, np.float32)
    fg[[5, 6, 20]] = [3.0, 2.0, 1.0]
    reach = np.full((30, 2), 2.0, np.float32)
    found = script().numpy_moments(fg, reach, t, duration=30.0, count=3)
    assert found[0] == (3.5, 7.5, 5)  # frame 6 is inside the first: it starts no moment
    assert found[1] == (18.5, 22.5, 20)
    assert len(found) == 3
    for a, b, _ in found:
        assert 0.0 <= a < b <= 30.0


def test_moments_stay_inside_the_video():
    t = np.arange(5, dtype=np.float64)
    fg = np.array([0, 0, 0, 0, 1], np.float32)
    reach = np.full((5, 2), 10.0, np.float32)
    ((a, b, i),) = script().numpy_moments(fg, reach, t, duration=5.0, count=1)
    assert (a, b, i) == (0.0, 5.0, 4)


def test_numpy_runs_the_model_as_the_training_does():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    training = script()
    rng = np.random.default_rng(5)
    p = params(rng)
    x, t = frames(rng, 24)
    jp = {k: jnp.asarray(v) for k, v in p.items() if k not in ("layers", "heads")}
    mask = np.ones((1, 24), bool)
    new = np.asarray(training.forward(jp, x[None], t[None].astype(np.float32), mask, LAYERS, HEADS))[0]
    assert np.allclose(new, temporal.apply(p, x, t), atol=1e-5)
    q = x[2] + 0.1
    off = rng.uniform(0, 0.1, 24).astype(np.float32)
    fg, reach = training.head(jp, new[None], t[None].astype(np.float32), mask, q[None], off[None], HEADS)
    fg_np, reach_np = training.numpy_head(p, new, t, q, off)
    assert np.allclose(np.asarray(fg)[0], fg_np, atol=1e-4)
    assert np.allclose(np.asarray(reach)[0], reach_np, atol=1e-4)
