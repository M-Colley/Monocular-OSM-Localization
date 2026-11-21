"""The fleet probe freezes every plan/failure before opening route labels."""

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest
from shapely.geometry import Polygon

from scripts import probe_municipality_fleet as M


def _fixture(tmp_path):
    waypoints = [{"t_sec": 0, "lat": 0, "lon": 0},
                 {"t_sec": 10, "lat": .001, "lon": .001}]
    rows = [[float(w["t_sec"]), float(w["lat"]), float(w["lon"])] for w in waypoints]
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()
    manifest = {"clips": [{"id": cid, "city": city, "split": split, "available_locally": available,
                            "gt": f"ground_truth/{cid}.json", "gt_waypoints_sha256": digest}
                           for cid, city, split, available in [
                               ("local", "Small city", "dev", True),
                               ("unavailable", "Oversized city", "dev", False),
                               ("heldout", "Held-out city", "test", True)]]}
    M._write(tmp_path / "benchmarks/manifest.json", manifest)
    sources = []
    for name in ("metadata", "corridors"):
        path = tmp_path / name / "plans.json"
        M._write(path, {"records": []})
        (path.parent / "plans.sha256").write_text(M.previous.sha(path), encoding="utf-8")
        sources.append(path.parent)
    args = Namespace(root=tmp_path, out=tmp_path / "probe", metadata_source=sources[0],
                     corridor_source=sources[1], radius_m=3000, max_cells=64,
                     allow_network=False, max_network_requests=0, step_m=20)
    return args, waypoints


def test_plan_preserves_unavailable_and_overbudget_city_without_reading_gt(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path)

    def boundary(self, city):
        extent = .01 if city == "Small city" else 1
        return Polygon([(-extent, -extent), (extent, -extent),
                        (extent, extent), (-extent, extent)]), {
                            "synthetic": True, "class": "boundary", "type": "administrative"}

    monkeypatch.setattr(M.BoundaryInputs, "__call__", boundary)
    assert not (tmp_path / "ground_truth").exists()
    assert M.plan(args) == 0
    frozen = M.previous.read(args.out / "plans.json")
    assert frozen["ground_truth_read"] is False
    assert {c["clip"] for c in frozen["clips"]} == {"local", "unavailable"}
    cities = {c["city"]: c for c in frozen["cities"]}
    assert cities["Small city"]["status"] == "ok"
    assert cities["Oversized city"]["status"] == "refused_over_budget"
    assert cities["Oversized city"]["regions"] == []
    assert cities["Oversized city"]["required_cells"] > 64
    assert not (tmp_path / "ground_truth").exists()
    with pytest.raises(SystemExit, match="already frozen"):
        M.plan(args)


def test_separate_score_retains_refused_city_and_rejects_plan_mutation(tmp_path, monkeypatch):
    args, waypoints = _fixture(tmp_path)

    def boundary(self, city):
        extent = .01 if city == "Small city" else 1
        return Polygon([(-extent, -extent), (extent, -extent),
                        (extent, extent), (-extent, extent)]), {"class": "boundary", "type": "administrative"}

    monkeypatch.setattr(M.BoundaryInputs, "__call__", boundary)
    M.plan(args)
    for cid in ("local", "unavailable"):
        M._write(tmp_path / f"ground_truth/{cid}.json", {"waypoints": waypoints})
    assert M.score(args) == 0
    scores = {r["clip"]: r for r in M.previous.read(args.out / "scores.json")["records"]}
    assert scores["local"]["coverage"]["municipality_union"]["arc_fraction_inside"] == 1
    assert scores["unavailable"]["gt_status"] == "ok"
    assert scores["unavailable"]["city_plan_status"] == "refused_over_budget"
    assert scores["unavailable"]["coverage"] == {}
    path = args.out / "plans.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen plan changed"):
        M.score(args)


def test_geocoding_http_budget_counts_failed_attempt_without_retry(tmp_path, monkeypatch):
    import requests

    calls = []

    class Rejected:
        status_code = 429

        def raise_for_status(self):
            raise RuntimeError("rate limited")

    def get(url, **kwargs):
        calls.append(url)
        return Rejected()

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(M.time, "sleep", lambda _: None)
    boundary = M.BoundaryInputs(tmp_path, tmp_path / "isolated", allow_network=True,
                                max_network_requests=1)
    with pytest.raises(RuntimeError, match="rate limited"):
        boundary("First city")
    with pytest.raises(RuntimeError, match="budget exhausted"):
        boundary("Second city")
    assert len(calls) == len(boundary.requests) == 1
    assert boundary.requests[0]["status_code"] == 429
