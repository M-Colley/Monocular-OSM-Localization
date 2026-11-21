"""The assisted children's CPU thread count: configured, validated, audited."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from monocular_osm.pipeline import PipelineConfig

ROOT = Path(__file__).resolve().parents[1]


def _audit():
    spec = importlib.util.spec_from_file_location("assisted_goal_audit_threads", ROOT / "scripts/assisted_goal_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(tmp_path, **kw):
    return PipelineConfig(url="u", city="Ulm, Germany", data_dir=tmp_path / "d", output_dir=tmp_path / "o", **kw)


def test_default_is_the_four_threads_every_assisted_result_ran_with(tmp_path):
    assert _cfg(tmp_path).assisted_cpu_threads == 4
    assert _cfg(tmp_path, assisted_cpu_threads=16).assisted_cpu_threads == 16


@pytest.mark.parametrize("bad", [0, 65, -1, True, 4.0, "8"])
def test_invalid_thread_counts_are_refused(tmp_path, bad):
    with pytest.raises(ValueError, match="assisted_cpu_threads"):
        _cfg(tmp_path, assisted_cpu_threads=bad)


def test_audit_expects_the_recorded_count_and_four_for_older_runs():
    audit = _audit()
    assert audit.expected_cpu_threads({}) == 4                       # recorded before the option
    assert audit.expected_cpu_threads(None) == 4
    assert audit.expected_cpu_threads({"assisted_cpu_threads": 12}) == 12
    for bad in (0, 65, True, 8.0):
        with pytest.raises(ValueError):
            audit.expected_cpu_threads({"assisted_cpu_threads": bad})


def test_cli_passes_the_option_through():
    from monocular_osm import cli
    args = cli.build_arg_parser().parse_args(["--url", "https://www.youtube.com/watch?v=x", "--city", "Ulm, Germany",
                              "--assisted-cpu-threads", "8"])
    assert args.assisted_cpu_threads == 8
