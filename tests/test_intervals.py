import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from ml import intervals

TABLE = {
    "model": "fake",
    "coverage": 0.8,
    "bin_edges": [0, 15, 30, 60, 100, 1e9],
    "half_width": [4.0, 6.0, 10.0, 20.0, 12.0],
    "n_per_bin": [100, 100, 100, 100, 100],
}


def use_fake_table(monkeypatch):
    monkeypatch.setitem(intervals._cache, "fake", TABLE)


def test_interval_uses_the_bin_of_the_prediction(monkeypatch):
    use_fake_table(monkeypatch)
    near_failure = intervals.add_interval(10.0, "fake")
    healthy = intervals.add_interval(80.0, "fake")

    assert (near_failure["rul_lower"], near_failure["rul_upper"]) == (6.0, 14.0)
    assert (healthy["rul_lower"], healthy["rul_upper"]) == (60.0, 100.0)
    assert near_failure["interval_coverage"] == 0.8


def test_lower_bound_never_below_zero(monkeypatch):
    use_fake_table(monkeypatch)
    assert intervals.add_interval(2.0, "fake")["rul_lower"] == 0.0


def test_missing_table_gives_none(monkeypatch):
    monkeypatch.setitem(intervals._cache, "nomodel", None)
    assert intervals.add_interval(20.0, "nomodel") == {
        "rul_lower": None,
        "rul_upper": None,
        "interval_coverage": None,
    }


def test_half_widths_cover_about_the_target():
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 140, 20000)
    true = pred + rng.normal(0, 8, pred.size)

    widths, counts = intervals.half_widths(pred, true)
    assert sum(counts) == pred.size

    err = np.abs(true - pred)
    for lo, hi, w in zip(intervals.BIN_EDGES[:-1], intervals.BIN_EDGES[1:], widths):
        in_bin = (pred >= lo) & (pred < hi)
        assert 0.77 <= (err[in_bin] <= w).mean() <= 0.84
