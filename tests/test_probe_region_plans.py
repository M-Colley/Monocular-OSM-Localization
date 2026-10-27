"""Corridor probe preserves the original frozen plan and separates GT scoring."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("probe_regions_tested", ROOT / "scripts/probe_region_plans.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def _source(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    regions = [{"name": "seed", "center": [40.0, -80.0], "radius_m": 8000, "origin": "seed"},
               {"name": "named", "center": [40.2, -80.0], "radius_m": 3000, "origin": "video_named"}]
    plan = {"exploratory": True, "ground_truth_read": False, "settings": {"max_regions": 6},
            "records": [{"clip": "clipA", "production_regions": regions}],
            "manifest_sha256": "original-manifest"}
    (source / "plans.json").write_text(json.dumps(plan), encoding="utf-8")
    (source / "plans.sha256").write_text(M.sha(source / "plans.json"), encoding="utf-8")
    (source / "geocodes.json").write_text('{"old query": null}', encoding="utf-8")
    return source, regions


def test_corridor_variant_uses_frozen_regions_without_gt_or_new_geocodes(tmp_path, monkeypatch):
    import monocular_osm.regional_retrieval as resolver

    source, original_regions = _source(tmp_path)
    old_bytes = {p.name: p.read_bytes() for p in source.iterdir()}
    out = tmp_path / "corridors"
    calls = []

    def fill(regions, *, max_regions, radius_m):
        calls.append((regions, max_regions, radius_m))
        return regions + [{"name": "corridor", "center": [40.1, -80],
                           "radius_m": radius_m, "origin": "corridor"}]

    monkeypatch.setattr(resolver, "fill_region_corridors", fill, raising=False)
    import monocular_osm.location_prior as location_prior
    monkeypatch.setattr(location_prior, "_default_geocode", lambda *_: pytest.fail("no new geocoding"))
    original_read = M.read
    reads = []

    def guarded_read(path):
        path = Path(path)
        assert path == source / "plans.json", "planning must not read GT, manifest or new inputs"
        reads.append(path)
        return original_read(path)

    monkeypatch.setattr(M, "read", guarded_read)
    args = argparse.Namespace(source=source, out=out, max_regions=6, radius_m=3000)
    assert M.plan_corridors(args) == 0
    result = json.loads((out / "plans.json").read_text(encoding="utf-8"))
    assert result["ground_truth_read"] is False and len(reads) == 1
    assert calls == [(original_regions, 6, 3000)]
    assert result["records"][0]["pre_corridor_regions"] == original_regions
    assert result["records"][0]["production_regions"][:2] == original_regions
    assert len(result["records"][0]["production_regions"]) == 3
    assert (out / "geocodes.json").read_bytes() == old_bytes["geocodes.json"]
    assert M.sha(out / "plans.json") == (out / "plans.sha256").read_text().strip()
    assert {p.name: p.read_bytes() for p in source.iterdir()} == old_bytes


def test_corridor_variant_refuses_changed_source_plan(tmp_path):
    source, _ = _source(tmp_path)
    (source / "plans.json").write_text("{}", encoding="utf-8")
    args = argparse.Namespace(source=source, out=tmp_path / "out", max_regions=6, radius_m=3000)
    with pytest.raises(ValueError, match="source frozen plan fingerprint changed"):
        M.plan_corridors(args)


def test_corridor_variant_refuses_to_overwrite_an_existing_frozen_variant(tmp_path):
    source, _ = _source(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "plans.json").write_text("old frozen result", encoding="utf-8")
    args = argparse.Namespace(source=source, out=out, max_regions=6, radius_m=3000)
    with pytest.raises(SystemExit, match="is frozen"):
        M.plan_corridors(args)
    assert (out / "plans.json").read_text(encoding="utf-8") == "old frozen result"
