import hashlib
import json
from pathlib import Path

import pytest

from scripts import extract_road_geometry as driver


def write(path, value):
    path.write_text(json.dumps(value)); return driver.core.file_sha(path)


def prepared(tmp_path, monkeypatch):
    source = {"commit": "a"*40, "files": {"script": "b"*64}}
    monkeypatch.setattr(driver, "source_identity", lambda _: source)
    family = tmp_path/"family"; family.mkdir()
    original = tmp_path/"source.pbf"; original.write_bytes(b"publicinput")
    config = family/"osmconf.ini"; config.write_text(driver.core.OSM_CONFIG, encoding="utf-8", newline="\n")
    registration = {"source": source, "source_inputs": {str(original): driver.core.file_sha(original)},
        "pbf": str(original), "reader": {"identity": 1}, "profile_sha256": driver.core.PROFILE_SHA256,
        "osm_config_sha256": driver.core.file_sha(config), "bbox": None}
    write(family/"registration.json", registration)
    for name in ("profile.json", "extract_ledger.json", "extract.poly"):
        (family/name).write_text("fixture")
    frozen = {"status": "prepared", "source_commit": "a"*40, "profile_sha256": driver.core.PROFILE_SHA256,
        "artifacts": {p.name: driver.core.file_sha(p) for p in family.iterdir()}}
    input_sha = write(family/"inputs_frozen.json", frozen)
    return family, input_sha


def test_prepared_binding_rejects_external_sha_before_registration(tmp_path, monkeypatch):
    family, accepted = prepared(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="accepted input"):
        driver.verified_prepared(family, "a"*40, "f"*64)
    assert driver.verified_prepared(family, "a"*40, accepted)[0] == family


def test_original_mutation_refuses_before_extraction(tmp_path, monkeypatch):
    family, accepted = prepared(tmp_path, monkeypatch)
    (tmp_path/"source.pbf").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="original input"):
        driver.verified_prepared(family, "a"*40, accepted)


def test_config_mutation_refuses(tmp_path, monkeypatch):
    family, accepted = prepared(tmp_path, monkeypatch)
    (family/"osmconf.ini").write_text("bbox=forbidden")
    with pytest.raises(ValueError, match="prepared artifact"):
        driver.verified_prepared(family, "a"*40, accepted)


@pytest.mark.parametrize("same", ["input", "source"])
def test_output_cannot_modify_input_family_or_source_checkout(tmp_path, monkeypatch, same):
    original = tmp_path/"input"/"file.pbf"; original.parent.mkdir(); original.write_bytes(b"x")
    monkeypatch.setattr(driver, "ROOT", tmp_path/"source")
    output = original.parent/"child" if same == "input" else driver.ROOT/"output"
    with pytest.raises(ValueError): driver.outside_inputs(output, [original])


def test_complete_and_partial_outputs_never_overwritten(tmp_path, monkeypatch):
    family, accepted = prepared(tmp_path, monkeypatch)
    (family/"frozen_road_geometry.json").write_text("old")
    with pytest.raises(ValueError, match="no implicit"):
        driver.extract(family, "a"*40, accepted)
    assert (family/"frozen_road_geometry.json").read_text() == "old"


def test_resource_failure_preserves_failure_not_complete(tmp_path, monkeypatch):
    family, accepted = prepared(tmp_path, monkeypatch)
    (family/"temporary").mkdir()
    class Reader:
        identity = {"identity": 1}
        def rows(self, *args):
            raise ValueError("headroom intentionally refused")
    class Guard:
        def __init__(self, *args): pass
        def check(self, **kwargs): pass
    monkeypatch.setattr(driver.core, "GdalWays", Reader)
    monkeypatch.setattr(driver.core, "ResourceGuard", Guard)
    monkeypatch.setattr(driver.core, "source_way_ledger", lambda *args: {})
    with pytest.raises(ValueError, match="headroom"):
        driver.extract(family, "a"*40, accepted)
    assert json.loads((family/"failure.json").read_text())["status"] == "failed"
    assert not (family/"frozen_road_geometry.json").exists()


def synthetic_family(tmp_path, monkeypatch):
    from tests.test_road_geometry import pbf_file
    public = tmp_path/"public"; public.mkdir()
    source = public/"tiny.pbf"; pbf_file(source)
    polygon = public/"extract.poly"
    polygon.write_text("fixture\n1\n -84 41\n -82 41\n -82 43\n -84 43\n -84 41\nEND\nEND\n")
    ledger = public/"extract_ledger.json"
    write(ledger, {"verified": True, "sha256": driver.core.file_sha(source), "bytes": source.stat().st_size,
        "license": "https://example.test/public-license", "source_url": "https://example.test/tiny.pbf"})
    identity = {"commit": "a"*40, "files": {"testfixture": "b"*64}}
    monkeypatch.setattr(driver, "source_identity", lambda _: identity)
    class Guard:
        def __init__(self, *args): pass
        def check(self, **kwargs): pass
        def record(self): return {"synthetic": True}
    monkeypatch.setattr(driver.core, "ResourceGuard", Guard)
    family = tmp_path/"family"
    driver.prepare(source, driver.core.file_sha(source), polygon, driver.core.file_sha(polygon), ledger, family, "a"*40)
    return family, driver.core.file_sha(family/"inputs_frozen.json")


def test_actual_prepare_extract_tiny_pbf_complete_freeze(tmp_path, monkeypatch):
    pytest.importorskip("pyogrio")
    family, accepted = synthetic_family(tmp_path, monkeypatch)
    master = driver.extract(family, "a"*40, accepted)
    assert master["status"] == "complete"
    assert set(master["artifacts"]) == {"roads.sqlite", "source_ways.sqlite", "summary.json", "execution.json", "inputs_frozen.json"}
    assert all(driver.core.file_sha(family/name) == expected for name, expected in master["artifacts"].items())
    summary = json.loads((family/"summary.json").read_text())
    assert summary["counts"]["ways"] == 1 and summary["source_way_counts_and_tags_verified"]
    assert summary["topology"] is False and master["projection_run"] is False


def test_end_headroom_refusal_keeps_partial_without_complete_master(tmp_path, monkeypatch):
    pytest.importorskip("pyogrio")
    family, accepted = synthetic_family(tmp_path, monkeypatch)
    class Guard:
        def __init__(self, *args): pass
        def check(self, **kwargs):
            if (family/"summary.json").exists(): raise ValueError("end headroom refused")
        def record(self): return {"synthetic": True}
    monkeypatch.setattr(driver.core, "ResourceGuard", Guard)
    with pytest.raises(ValueError, match="end headroom"):
        driver.extract(family, "a"*40, accepted)
    assert (family/"failure.json").is_file() and not (family/"frozen_road_geometry.json").exists()


@pytest.mark.parametrize("rebless", [False, True])
def test_final_callback_original_or_reblessed_input_cannot_complete(tmp_path, monkeypatch, rebless):
    pytest.importorskip("pyogrio")
    family, accepted = synthetic_family(tmp_path, monkeypatch)
    class Guard:
        def __init__(self, *args): self.mutated = False
        def check(self, **kwargs):
            if (family/"summary.json").exists() and not self.mutated:
                self.mutated = True
                path = family/"registration.json"; registration = json.loads(path.read_text())
                original = Path(registration["pbf"]); original.write_bytes(b"changed after final callback")
                if rebless:
                    registration["source_inputs"][str(original)] = driver.core.file_sha(original)
                    path.write_text(json.dumps(registration))
                    frozen_path = family/"inputs_frozen.json"; frozen = json.loads(frozen_path.read_text())
                    frozen["artifacts"]["registration.json"] = driver.core.file_sha(path)
                    frozen_path.write_text(json.dumps(frozen))
        def record(self): return {"synthetic": True}
    monkeypatch.setattr(driver.core, "ResourceGuard", Guard)
    with pytest.raises(ValueError, match="original input|input freeze|artifact changed"):
        driver.extract(family, "a"*40, accepted)
    assert (family/"failure.json").is_file() and not (family/"frozen_road_geometry.json").exists()
