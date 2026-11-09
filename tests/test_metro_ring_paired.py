import json
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


def test_compare_lists_a_clip_that_never_ran(tmp_path, capsys):
    summary = {"clips": {"ran": {"name": "Ran", "methods": {"answer": {"symmetric_mean_m": {"median": 50.0}}}},
                         "unrun": {"name": "Unrun", "methods": {}}}}
    for run in ("ring", "ordinary"):
        (tmp_path / run).mkdir()
        (tmp_path / run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    rep = tmp_path / "ring" / "ran" / "rep0"
    rep.mkdir(parents=True)
    (rep / "status.json").write_text(json.dumps({"wall_s": 12.0}), encoding="utf-8")
    (tmp_path / "ring" / "unrun" / "rep0").mkdir(parents=True)  # argv.json only, no status
    mp.compare(tmp_path / "ring", tmp_path / "ordinary")
    out = capsys.readouterr().out
    assert "| Ran |" in out and "| Unrun | - | not run |" in out

