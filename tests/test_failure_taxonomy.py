import importlib.util
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "failure_taxonomy", Path(__file__).resolve().parents[1] / "scripts" / "failure_taxonomy.py")
ft = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft)


def test_modes_follow_the_documented_thresholds():
    assert ft.failure_mode(150.0, 0.0, 0.0) == "ok"
    assert ft.failure_mode(1500.0, 0.2, 0.25) == "A"
    assert ft.failure_mode(3000.0, 1.0, 0.0) == "B"
    assert ft.failure_mode(500.0, 1.0, 0.5) == "C"
    assert ft.failure_mode(500.0, 1.0, 0.9) == "D"
    assert ft.failure_mode(500.0, 1.0, None) == "D"


def test_share_inside_counts_points_within_the_radius():
    pts = np.array([[52.0, 13.0], [52.0, 13.0 + 0.02], [52.0, 13.0 + 0.2]])  # 0, ~1.4 km, ~13.7 km
    assert ft.share_inside(pts, 52.0, 13.0, 2000.0) == 2 / 3
