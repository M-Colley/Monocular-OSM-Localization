"""Run provenance must identify the code and settings — and never a secret."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from monocular_osm.pipeline import PipelineConfig
from monocular_osm.provenance import collect_provenance, config_snapshot, git_state


def _cfg(tmp_path: Path) -> PipelineConfig:
    return PipelineConfig(url="https://example.com/v", city="Ulm, Germany",
                          data_dir=tmp_path / "d", output_dir=tmp_path / "o")


def test_secret_values_are_never_recorded(tmp_path, monkeypatch) -> None:
    secret = "MLY|0000|this-must-not-appear"
    monkeypatch.setenv("MLY_TOKEN", secret)
    prov = collect_provenance(_cfg(tmp_path))
    dumped = json.dumps(prov)
    assert secret not in dumped
    assert "this-must-not-appear" not in dumped
    assert prov["env"]["mly_token_set"] is True


def test_unset_secret_is_recorded_as_unset(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MLY_TOKEN", raising=False)
    assert collect_provenance(_cfg(tmp_path))["env"]["mly_token_set"] is False


def test_config_snapshot_is_complete_and_json_safe(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    snap = config_snapshot(cfg)
    json.dumps(snap)                                   # must not raise
    assert snap["city"] == "Ulm, Germany"
    assert snap["data_dir"] == str(tmp_path / "d")
    assert set(snap) == set(vars(cfg))                 # every field recorded


def test_provenance_records_code_identity(tmp_path) -> None:
    from monocular_osm import __version__

    prov = collect_provenance(_cfg(tmp_path), extra={"vo_algo_version": 3})
    assert prov["package_version"] == __version__
    assert prov["vo_algo_version"] == 3
    assert "numpy" in prov["libraries"] and prov["libraries"]["numpy"]
    assert set(prov["git"]) == {"commit", "branch", "dirty"}


def test_git_state_outside_a_repo_degrades_to_none(monkeypatch) -> None:
    import monocular_osm.provenance as provenance

    monkeypatch.setattr(provenance, "_git", lambda *a: None)
    assert git_state() == {"commit": None, "branch": None, "dirty": None}


@pytest.mark.parametrize("value", [Path("a/b"), (1, 2), {"k": Path("x")}])
def test_jsonable_handles_paths_and_containers(value) -> None:
    from monocular_osm.provenance import _jsonable

    json.dumps(_jsonable(value))
