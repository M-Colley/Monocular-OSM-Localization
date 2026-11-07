import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "metro_ring_paired", Path(__file__).resolve().parents[1] / "scripts" / "metro_ring_paired.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)

RING = [[float(t), 45.6, -73.7, 0.3] for t in range(0, 40, 4)]
ORDINARY = [[float(t), 45.5, -73.57, 0.12] for t in range(0, 40, 4)]


def _result(kept):
    return {"vpr_prior": {"track": RING},
            "answer": {"route_latlon": [[45.6, -73.7]] * 10, "mode": "track"},
            "metro_ring_retrieval": {"kept": kept, "baseline_track": ORDINARY}}


def test_shadow_restores_the_ordinary_track_and_answer():
    out = mp.ordinary_result(_result("ring+refine"))
    assert out["vpr_prior"]["track"] == ORDINARY
    assert out["answer"]["route_latlon"] == [[45.5, -73.57]] * 10
    assert out["answer"]["mode"] == "track"


def test_shadow_leaves_untouched_runs_alone():
    result = _result("baseline")
    assert mp.ordinary_result(result) is result
