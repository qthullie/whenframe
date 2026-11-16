import numpy as np

from whenframe.moments import on_grid, smooth, spans


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
