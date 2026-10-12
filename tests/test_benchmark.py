"""scripts/benchmark.py: leak guard, credential handling, run bookkeeping,
cold-VPR cache handling and offline aggregation on synthetic run dirs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    spec = importlib.util.spec_from_file_location(
        "benchmark_under_test", ROOT / "scripts" / "benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = _load_benchmark()
SECRET = "MLY|secret-token-0123456789"


def _entry(**over) -> dict:
    e = {"id": "clipA", "name": "Clip A", "fleet": "core", "split": "dev",
         "gt": "ground_truth/clip_a.json", "video": {"url": "https://www.youtube.com/watch?v=abc"},
         "vo_segment": "0:300", "city": "Ulm, Germany",
         "city_alternatives": ["Neu-Ulm, Germany"], "available_locally": True}
    e.update(over)
    return e


# ---------------------------------------------------------------------------
# leak guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [
    ["--osm-around", "48.4,9.9,800"],
    ["--osm-around=48.4,9.9,800"],
    ["--osm", "1"],                                 # argparse prefix abbreviation
    ["--ground-truth-waypoints", "x.json"],
    ["--ground-truth", "Olgastrasse"],
    ["--ground", "x"],
    ["--estimated-length-m", "2500"],
    ["--ocr-video", "ground_truth/ulm.json"],      # GT path smuggled as a value
    ["--foo", "48.40593,9.98368"],                 # coordinates
])
def test_assert_leak_free_rejects(bad) -> None:
    with pytest.raises(M.LeakError):
        M.assert_leak_free(["python", "-m", "monocular_osm.cli", *bad])


def test_assert_leak_free_checks_city_and_gt_name() -> None:
    e = _entry()
    ok = M.build_pipeline_argv(e, city="Ulm, Germany", output_dir=Path("o"),
                               data_dir=Path("data"), python="py")
    M.assert_leak_free(ok, e)
    with pytest.raises(M.LeakError):
        M.assert_leak_free(ok[:-6] + ["--city", "Somewhere Else", "--vo-segment", "0:1"], e)
    with pytest.raises(M.LeakError):
        M.assert_leak_free(["--ocr-video", "clip_a.json"], e)


def test_build_pipeline_argv_is_minimal() -> None:
    e = _entry(video={"path": "data/kitti/drive_0009.mp4"})
    argv = M.build_pipeline_argv(e, city="Ulm, Germany", output_dir=Path("out/x"),
                                 data_dir=Path("data"), python="py")
    assert argv == ["py", "-m", "monocular_osm.cli", "--video", "data/kitti/drive_0009.mp4",
                    "--city", "Ulm, Germany", "--vo-segment", "0:300",
                    "--data-dir", "data", "--output-dir", "out/x"]
    with pytest.raises(M.LeakError):
        M.build_pipeline_argv(e, city="Ulm, Germany", output_dir=Path("o"),
                              data_dir=Path("data"), extra=["--osm-around", "1,2,3"])


@pytest.mark.parametrize("variant", [None, 0, 1])
def test_argv_for_every_committed_clip_never_carries_ground_truth(variant) -> None:
    manifest = M.load_manifest()
    for c in manifest["clips"]:
        city = M.city_for(c, variant)
        if city is None:
            continue
        argv = M.build_pipeline_argv(c, city=city, output_dir=Path("output/x"),
                                     data_dir=Path("data"), python="python")
        joined = " ".join(argv)
        for flag in M.FORBIDDEN_FLAGS:
            assert flag not in joined
        assert "ground_truth" not in joined and Path(c["gt"]).name not in joined
        assert argv[argv.index("--city") + 1] == city
        assert argv[argv.index("--vo-segment") + 1] == c["vo_segment"]
        assert argv[1:3] == ["-m", "monocular_osm.cli"]
        # Exactly: video source, city, window, data/output roots.
        flags = [t for t in argv if t.startswith("--")]
        assert flags in (["--url", "--city", "--vo-segment", "--data-dir", "--output-dir"],
                         ["--video", "--city", "--vo-segment", "--data-dir", "--output-dir"])


def test_passthrough_rejects_runner_and_gt_flags() -> None:
    assert M.check_passthrough(["--enable-ocr-anchor"]) == ["--enable-ocr-anchor"]
    for bad in (["--city", "X"], ["--data-dir", "d"], ["--vo-seg", "0:1"],
                ["--analyze-minutes", "3"]):
        with pytest.raises(SystemExit):
            M.check_passthrough(bad)
    with pytest.raises(M.LeakError):
        M.check_passthrough(["--osm-around", "1,2,3"])


def test_city_variant_selection() -> None:
    e = _entry()
    assert M.city_for(e, None) == "Ulm, Germany"
    assert M.city_for(e, 0) == "Neu-Ulm, Germany"
    assert M.city_for(e, 1) is None


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------


def test_child_env_reads_registry_and_never_records_the_value() -> None:
    env, rec = M.child_env({"PATH": "x"}, registry=lambda name: SECRET)
    assert env["MLY_TOKEN"] == SECRET
    assert rec == {"mly_token_set": True, "mly_token_origin": "registry"}
    assert SECRET not in json.dumps(rec)
    env2, rec2 = M.child_env({"MLY_TOKEN": SECRET}, registry=lambda n: None)
    assert rec2["mly_token_origin"] == "environ"
    env3, rec3 = M.child_env({}, registry=lambda n: None)
    assert "MLY_TOKEN" not in env3 and rec3["mly_token_set"] is False


def test_redact_file(tmp_path: Path) -> None:
    p = tmp_path / "log.txt"
    p.write_text(f"token={SECRET} again {SECRET}\n", encoding="utf-8")
    assert M.redact_file(p, SECRET) == 2
    assert SECRET not in p.read_text(encoding="utf-8")
    assert M.redact_file(p, None) == 0


# ---------------------------------------------------------------------------
# run: bookkeeping, resumability, no token / GT anywhere
# ---------------------------------------------------------------------------


def _write_manifest(root: Path, clips: list[dict]) -> None:
    (root / "benchmarks").mkdir(parents=True, exist_ok=True)
    (root / "benchmarks" / "manifest.json").write_text(
        json.dumps({"clips": clips}), encoding="utf-8")


def _run_args(**over) -> argparse.Namespace:
    ns = M.build_parser().parse_args(["run", "--run-name", "r1"])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_cmd_run_end_to_end_with_fake_pipeline(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_git_state", lambda: {"commit": "c0ffee", "dirty": False})
    monkeypatch.setenv("MLY_TOKEN", SECRET)
    _write_manifest(tmp_path, [_entry(), _entry(id="clipB", fleet="overlay",
                                                gt="ground_truth/clip_b.json")])
    calls = []

    def fake_run_one(argv, log_path, env, timeout):
        calls.append(list(argv))
        assert env["MLY_TOKEN"] == SECRET         # the child does get the token
        out = tmp_path / argv[argv.index("--output-dir") + 1] / "some-slug"
        out.mkdir(parents=True)
        (out / "result.json").write_text(json.dumps({"position": {}}), encoding="utf-8")
        Path(log_path).write_text(f"oops the child printed {SECRET}\n", encoding="utf-8")
        return {"rc": 0, "timed_out": False, "wall_s": 1.0,
                "started_utc": "t0", "finished_utc": "t1"}

    monkeypatch.setattr(M, "run_one", fake_run_one)
    assert M.cmd_run(_run_args(repeats=2)) == 0
    assert len(calls) == 2                                    # core fleet only, 2 reps
    for argv in calls:
        M.assert_leak_free(argv)
        assert "--osm-around" not in argv and "--ground-truth-waypoints" not in argv

    run_dir = tmp_path / "output" / "benchmark_runs" / "r1"
    for k in (0, 1):
        rep = run_dir / "clipA" / f"rep{k}"
        status = json.loads((rep / "status.json").read_text(encoding="utf-8"))
        assert status["rc"] == 0 and status["result_found"] and status["mly_token_set"]
        assert status["city"] == "Ulm, Germany"
        assert (rep / "result.json").exists()
        for f in ("status.json", "argv.json", "log.txt", "result.json"):
            assert SECRET not in (rep / f).read_text(encoding="utf-8"), f
        assert status["token_redactions"] == 1
    assert SECRET not in capsys.readouterr().out
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert cfg["config"]["cold_vpr"] is False and SECRET not in json.dumps(cfg)

    # Resumable: completed reps are skipped.
    assert M.cmd_run(_run_args(repeats=2)) == 0
    assert len(calls) == 2
    # One more repeat runs only rep2.
    assert M.cmd_run(_run_args(repeats=3)) == 0
    assert len(calls) == 3
    # A run directory cannot silently mix configurations.
    with pytest.raises(SystemExit):
        M.cmd_run(_run_args(cold_vpr=True))


def test_cmd_run_skips_unavailable_and_missing_local_video(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(M, "ROOT", tmp_path)
    monkeypatch.setattr(M, "_git_state", lambda: {})
    _write_manifest(tmp_path, [
        _entry(id="gone", available_locally=False),
        _entry(id="local", video={"path": "data/missing.mp4"}),
    ])
    monkeypatch.setattr(M, "run_one", lambda *a, **k: pytest.fail("must not run"))
    assert M.cmd_run(_run_args()) == 0


def test_split_test_is_opt_in() -> None:
    manifest = {"clips": [_entry(id="d"), _entry(id="t", split="test")]}
    assert [c["id"] for c in M.select_clips(manifest, split="dev")] == ["d"]
    assert [c["id"] for c in M.select_clips(manifest, split="all")] == ["d"]
    assert [c["id"] for c in M.select_clips(manifest, split="test")] == ["t"]


# ---------------------------------------------------------------------------
# cold VPR: aside, then restore; fresh caches archived, nothing deleted
# ---------------------------------------------------------------------------


def test_cold_vpr_moves_aside_and_restores(tmp_path: Path) -> None:
    slug = tmp_path / "data" / "abc-some-title-ulm-germany"
    (slug / "mapillary").mkdir(parents=True)
    (slug / "mapillary" / "ref_img_store.npz").write_bytes(b"warm")
    (slug / "mapillary_c2f").mkdir()
    (slug / "trajectory_v3.npz").write_bytes(b"vo")          # not a VPR cache

    moved = M.set_vpr_caches_aside(slug)
    assert sorted(moved) == ["mapillary", "mapillary_c2f"]
    assert not (slug / "mapillary").exists()
    assert (slug / "trajectory_v3.npz").exists()

    # The cold run writes a fresh cache...
    (slug / "mapillary").mkdir()
    (slug / "mapillary" / "ref_img_store.npz").write_bytes(b"fresh")
    arch = slug / M.COLD_ARCHIVE_DIR / "r1" / "clipA" / "rep0"
    M.restore_vpr_caches(slug, arch)
    assert (slug / "mapillary" / "ref_img_store.npz").read_bytes() == b"warm"
    assert (arch / "mapillary" / "ref_img_store.npz").read_bytes() == b"fresh"
    assert (slug / "mapillary_c2f").exists()
    assert not (slug / M.ASIDE_DIR).exists()


def test_cold_vpr_recovers_from_a_crashed_run(tmp_path: Path) -> None:
    slug = tmp_path / "abc-x"
    (slug / M.ASIDE_DIR / "kartaview").mkdir(parents=True)          # leftover warm
    (slug / "kartaview").mkdir()                                     # leftover fresh
    (slug / "kartaview" / "f").write_bytes(b"fresh")
    M.set_vpr_caches_aside(slug)
    # Warm cache is aside again, the orphaned fresh one archived (not deleted).
    assert (slug / M.ASIDE_DIR / "kartaview").exists()
    archived = list((slug / M.COLD_ARCHIVE_DIR).rglob("f"))
    assert archived and archived[0].read_bytes() == b"fresh"


def test_candidate_slug_dirs_globs_by_video_id(tmp_path: Path) -> None:
    (tmp_path / "abc-title-ulm-germany").mkdir()
    (tmp_path / "abc-title-neu-ulm-germany").mkdir()
    (tmp_path / "xyz-other").mkdir()
    dirs = M.candidate_slug_dirs(_entry(), "Ulm, Germany", tmp_path)
    assert {d.name for d in dirs} == {"abc-title-ulm-germany", "abc-title-neu-ulm-germany"}


# ---------------------------------------------------------------------------
# aggregate on synthetic run dirs
# ---------------------------------------------------------------------------

LAT0, LON0 = 48.40, 9.99
_CRS = CRS.from_proj4(f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +datum=WGS84 +units=m")
_INV = Transformer.from_crs(_CRS, "EPSG:4326", always_xy=True)


def _ll(xy) -> list[list[float]]:
    xy = np.asarray(xy, dtype=float)
    lon, lat = _INV.transform(xy[:, 0], xy[:, 1])
    return np.column_stack([lat, lon]).tolist()


GT_XY = np.array([[x, 0.0] for x in range(0, 1001, 100)], dtype=float)


def _setup_root(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "repo"
    (root / "ground_truth").mkdir(parents=True)
    (root / "data").mkdir()
    clips = []
    for cid, shift in (("clipA", 0.0), ("clipB", 5000.0), ("clipC", 10000.0)):
        wps = [{"t_sec": float(i * 30), "lat": la, "lon": lo}
               for i, (la, lo) in enumerate(_ll(GT_XY + [shift, 0.0]))]
        gt = {"vo_segment": "0:300", "waypoints": wps}
        (root / "ground_truth" / f"{cid}.json").write_text(json.dumps(gt), encoding="utf-8")
        clips.append(_entry(id=cid, name=cid.upper(), gt=f"ground_truth/{cid}.json",
                            city=f"City {cid}", city_alternatives=[],
                            gt_waypoints_sha256=M.gt_waypoints_digest(gt)))
    # City centroids 2 km north of each clip's start.
    snap = {f"City {cid}": _ll([[shift, 2000.0]])[0]
            for cid, shift in (("clipA", 0.0), ("clipB", 5000.0), ("clipC", 10000.0))}
    (root / "data" / "geocode_cache.json").write_text(json.dumps(snap), encoding="utf-8")
    return root, {"clips": clips}


def _fake_result(shift_x: float, offset_m: float, *, commit="abc123", dirty=False,
                 token=True, argv=None) -> dict:
    route = _ll(GT_XY + [shift_x, offset_m])
    return {
        "position": {"source": "anchor_primary_vpr", "route_latlon": route,
                     "hypotheses": [{"rank": 1, "latitude": route[0][0] + 0.01,
                                     "longitude": route[0][1]},
                                    {"rank": 2, "latitude": route[0][0],
                                     "longitude": route[0][1]}]},
        "matcher_position": {"source": "matcher",
                             "route_latlon": _ll(GT_XY + [shift_x, 900.0])},
        "vpr_prior": {"lat": route[5][0], "lon": route[5][1], "origin": "vpr"},
        "provenance": {
            "git": {"commit": commit, "dirty": dirty, "branch": "b"},
            "env": {"mly_token_set": token},
            "config": {"ocr_engine": "rapidocr", "ground_truth_waypoints": None,
                       "ground_truth_streets": []},
            "vo_algo_version": 3, "package_version": "0.1.2",
            "argv": argv or ["-m", "monocular_osm.cli", "--city", "X"],
        },
    }


def _write_rep(run_dir: Path, clip: str, k: int, result: dict | None, rc: int = 0,
               city: str | None = None) -> None:
    d = run_dir / clip / f"rep{k}"
    d.mkdir(parents=True)
    (d / "status.json").write_text(json.dumps(
        {"rc": rc, "timed_out": False, "wall_s": 10.0 + k,
         "city": city or f"City {clip}"}), encoding="utf-8")
    if result is not None:
        (d / "result.json").write_text(json.dumps(result), encoding="utf-8")


def _make_run(tmp_path: Path, **result_kw) -> tuple[Path, dict, Path]:
    root, manifest = _setup_root(tmp_path)
    run_dir = tmp_path / "runs" / "r1"
    offsets = {"clipA": [20.0, 30.0, 40.0], "clipB": [100.0, 150.0, 200.0],
               "clipC": [300.0, 300.0, None]}           # clipC rep2 crashed
    shifts = {"clipA": 0.0, "clipB": 5000.0, "clipC": 10000.0}
    for cid, offs in offsets.items():
        for k, off in enumerate(offs):
            if off is None:
                _write_rep(run_dir, cid, k, None, rc=1)
            else:
                _write_rep(run_dir, cid, k, _fake_result(shifts[cid], off, **result_kw))
    return run_dir, manifest, root


def test_aggregate_scores_methods_and_baselines(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=999)

    a = s["clips"]["clipA"]["methods"]
    assert a["pipeline"]["symmetric_mean_m"]["median"] == pytest.approx(30.0, abs=0.5)
    assert a["pipeline"]["symmetric_mean_m"]["q25"] == pytest.approx(25.0, abs=0.5)
    assert a["pipeline"]["symmetric_mean_m"]["q75"] == pytest.approx(35.0, abs=0.5)
    assert a["pipeline"]["start_error_m"]["min"] == pytest.approx(20.0, abs=0.5)
    assert a["matcher_only"]["symmetric_mean_m"]["median"] == pytest.approx(900.0, abs=1)
    # Pinned report absent in every rep -> unanswered, counted as failures.
    assert a["pinned_only"]["n_answered"] == 0
    assert math.isinf(a["pinned_only"]["symmetric_mean_m"]["median"])
    # City centroid 2 km north of the start: GT->pred mean = mean dist to the point.
    cc = a["city_centroid"]
    expect = float(np.mean(np.hypot(GT_XY[:, 0], 2000.0)))
    assert cc["gt_to_pred_mean_m"]["median"] == pytest.approx(expect, abs=1.0)
    assert a["shortlist_oracle@5"]["start_error_m"]["median"] == pytest.approx(30.0, abs=0.5)
    assert a["pipeline"]["success_rate"]["start_error_m"]["25"] == pytest.approx(1 / 3)

    # The crashed rep is a failure, not a missing value.
    c = s["clips"]["clipC"]
    assert c["n_ok"] == 2 and c["n_reps"] == 3
    assert c["methods"]["pipeline"]["success_rate"]["symmetric_mean_m"]["1000"] == \
        pytest.approx(2 / 3)
    assert math.isinf(c["methods"]["pipeline"]["symmetric_mean_m"]["max"])


def test_aggregate_fleet_bootstrap_ci_shape(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=999)
    fleet = s["fleets"]["core"]
    assert fleet["clips"] == ["clipA", "clipB", "clipC"]
    ci = fleet["methods"]["pipeline"]["symmetric_mean_m"]["median_of_clip_medians"]
    assert set(ci) >= {"estimate", "low", "high", "n", "confidence", "method"}
    assert ci["n"] == 3 and ci["confidence"] == 0.95
    assert ci["low"] <= ci["estimate"] <= ci["high"]
    sr = fleet["methods"]["pipeline"]["success_rate"]["start_error_m"]
    assert list(sr) == ["25", "50", "100", "250", "500", "1000"]
    assert 0.0 <= sr["100"]["low"] <= sr["100"]["estimate"] <= sr["100"]["high"] <= 1.0
    paired = fleet["paired_vs_pipeline"]["city_centroid"]["symmetric_mean_m"]
    assert paired["pipeline_wins"] == 3 and paired["pipeline_losses"] == 0
    assert 0.0 < paired["sign_test_p"] <= 1.0

    md = M.render_markdown(s)
    assert "| Clip |" in md and "City centroid" in md and "Shortlist oracle@5" in md
    blob = json.dumps(M._jsonable(s))
    assert "Infinity" not in blob and "NaN" not in blob


def test_aggregate_refuses_mixed_provenance(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    _write_rep(run_dir, "clipA", 3, _fake_result(0.0, 25.0, commit="other"))
    with pytest.raises(SystemExit, match="mixed"):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99,
                    allow_mixed=True)
    assert s["provenance"]["consistent"] is False


@pytest.mark.parametrize("kw,match", [({"dirty": True}, "dirty"),
                                      ({"token": False}, "MLY_TOKEN")])
def test_aggregate_refuses_dirty_or_token_mismatch(tmp_path, kw, match) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    _write_rep(run_dir, "clipB", 3, _fake_result(5000.0, 25.0, **kw))
    with pytest.raises(SystemExit, match=match):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)


def test_aggregate_refuses_a_leaked_run_even_with_allow_mixed(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    leaked = _fake_result(0.0, 5.0, argv=["main.py", "--osm-around", "48.4,9.9,800"])
    _write_rep(run_dir, "clipA", 3, leaked)
    with pytest.raises(SystemExit, match="ground truth reached the pipeline"):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99,
                    allow_mixed=True)


def test_aggregate_refuses_gt_drift_and_foreign_city(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    manifest["clips"][0]["gt_waypoints_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="hash"):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    run_dir2, manifest2, root2 = _make_run(tmp_path / "b")
    _write_rep(run_dir2, "clipA", 3, _fake_result(0.0, 25.0), city="Daly City, CA")
    with pytest.raises(SystemExit, match="not in the manifest"):
        M.aggregate(run_dir2, manifest2, root=root2, allow_network=False, n_resamples=99)


def test_cmd_aggregate_writes_summary_files(tmp_path: Path, monkeypatch) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(M, "ROOT", root)
    args = M.build_parser().parse_args([
        "aggregate", "--run-name", "r1", "--runs-dir", str(tmp_path / "runs"),
        "--manifest", str(mpath), "--offline", "--n-resamples", "99"])
    assert M.cmd_aggregate(args) == 0
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["provenance"]["commits"] == ["abc123"]
    assert summary["provenance"]["ocr_engines"] == ["rapidocr"]
    assert (run_dir / "summary.md").read_text(encoding="utf-8").startswith("# Benchmark run")


@pytest.mark.parametrize("source,leaks", [
    ("video_title", False),   # derived from the video: deployable
    ("user", True),           # set by hand: may encode ground truth
    (None, True),             # unknown origin (pre-provenance result)
])
def test_leak_audit_trusts_only_video_derived_discs(source, leaks) -> None:
    prov = {"argv": [], "osm_around_cfg": [48.4, 9.99, 3000.0],
            "osm_around_source_cfg": source}
    assert bool(M.leak_audit(prov)) is leaks
    assert M.leak_audit({"argv": [], "osm_around_cfg": None}) == []
