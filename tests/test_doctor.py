"""Offline setup checks catch broken imports without loading weights or tokens."""

from types import SimpleNamespace

import pytest

from monocular_osm import doctor


def _diagnose(**kwargs):
    return doctor.collect_diagnostics(
        finder=lambda name: object(),
        importer=lambda name: SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
        versioner=lambda name: "1.0", **kwargs)


def test_cpu_installation_is_ready_with_a_token_and_no_models_loaded():
    report = _diagnose(environ={"MLY_TOKEN": "test-secret"})
    assert report["ready"] and report["cuda_available"] is False
    assert all(p["status"] == "ok" for p in report["checks"].values())
    assert "test-secret" not in str(report)
    assert report["model_weights"] == "not checked; no models loaded"


def test_missing_token_only_blocks_requested_fresh_mapillary_retrieval():
    assert not _diagnose(environ={})["ready"]
    assert _diagnose(environ={}, source="kartaview")["ready"]
    assert _diagnose(environ={}, required=["core"])["ready"]


def test_installed_but_broken_torchvision_is_not_reported_ready(capsys):
    token = "MLY|private-test-token"

    def importer(name):
        if name == "torchvision":
            print(token)
            raise RuntimeError(f"missing operator; debug token {token}")
        return SimpleNamespace()

    report = doctor.collect_diagnostics(
        finder=lambda name: object(), importer=importer, versioner=lambda name: "1.0",
        environ={"MLY_TOKEN": token})
    assert not report["ready"]
    assert report["checks"]["torchvision"]["status"] == "failed"
    assert "RuntimeError" in report["checks"]["torchvision"]["error"]
    assert token not in str(report) and token not in doctor.format_diagnostics(report)
    assert token not in capsys.readouterr().out


def test_optional_absence_does_not_block_default_requirements():
    report = doctor.collect_diagnostics(
        finder=lambda name: None if name in {"rapidocr", "fast_alpr"} else object(),
        importer=lambda name: SimpleNamespace(), versioner=lambda name: "1.0",
        environ={"MLY_TOKEN": "set"})
    assert report["ready"]
    assert not report["capabilities"]["ocr"]["ready"]
    assert report["checks"]["fast_alpr"]["status"] == "missing"


def test_strict_exit_status_and_json_output(monkeypatch, capsys):
    report = _diagnose(environ={})
    monkeypatch.setattr(doctor, "collect_diagnostics", lambda **kw: report)
    assert doctor.main(["--strict", "--json"]) == 1
    import json
    assert not json.loads(capsys.readouterr().out)["ready"]
    assert doctor.main([]) == 0


def test_diagnostics_do_not_import_the_pipeline():
    import subprocess
    import sys
    probe = subprocess.run([sys.executable, "-c",
                            "import sys; import monocular_osm.doctor; "
                            "assert 'monocular_osm.pipeline' not in sys.modules; "
                            "assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert probe.returncode == 0, probe.stderr


def test_unknown_required_capability_is_rejected():
    with pytest.raises(ValueError, match="unknown capabilities"):
        _diagnose(required=["typo"], environ={})


def test_missing_namespace_parent_is_reported_as_missing():
    def finder(name):
        if name == "google.protobuf":
            raise ModuleNotFoundError("No module named 'google'", name="google")
        return object()

    report = doctor.collect_diagnostics(finder=finder, importer=lambda name: SimpleNamespace(),
                                        versioner=lambda name: "1.0", environ={})
    assert report["checks"]["google.protobuf"]["status"] == "missing"
