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
    # --overpass-wait 0: unit tests must neither probe nor wait on the network.
    ns = M.build_parser().parse_args(["run", "--run-name", "r1", "--overpass-wait", "0",
                                          "--min-free-commit-gb", "0",
                                          "--min-free-vram-gb", "0"])
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
        _entry(id="local", video={"path": "data/missing.mp4"},
               city_basis="dataset documentation"),
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


# --- run health -----------------------------------------------------------------

def _healthy_extras() -> dict:
    return {"n_frames": 400, "n_valid_poses": 380,
            "channels": {"vpr": {"status": "ok"}, "coarse_from_frames": {"status": "inactive"}},
            "vpr_prior": {"lat": 1.0, "lon": 1.0, "origin": "vpr", "source": "mapillary",
                          "backbone": {"name": "megaloc"}}}


def test_health_of_reads_recorded_keys_only() -> None:
    h = M._health_of(_healthy_extras())
    assert h["vpr_status"] == "ok" and h["vpr_source"] == "mapillary"
    assert h["vpr_backbone"] == "megaloc" and h["valid_pose_frac"] == 0.95
    assert M._health_of(None) == {} and M.degraded_reasons({}, token_set=True) == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda r: None, []),
    (lambda r: r["channels"].update(vpr={"status": "inactive", "reason": "only 0 refs"}),
     ["retrieval inactive (only 0 refs)"]),
    (lambda r: r["vpr_prior"].update(source="kartaview"),
     ["retrieval used 'kartaview' although a token was set"]),
    (lambda r: r["vpr_prior"]["backbone"].update(name="eigenplaces"),
     ["retrieval backbone 'eigenplaces', not megaloc"]),
    (lambda r: r["channels"].update(tile3d={"status": "failed"}),
     ["failed channels: tile3d"]),
    (lambda r: r.update(n_valid_poses=100), ["only 25% of VO poses valid"]),
])
def test_degraded_reasons(mutate, expect) -> None:
    r = _healthy_extras()
    mutate(r)
    assert M.degraded_reasons(M._health_of(r), token_set=True) == expect


def test_kartaview_without_a_token_is_not_degraded() -> None:
    r = _healthy_extras()
    r["vpr_prior"]["source"] = "kartaview"
    assert M.degraded_reasons(M._health_of(r), token_set=False) == []


def test_aggregate_flags_degraded_runs_and_renders_health(tmp_path: Path) -> None:
    root, manifest = _setup_root(tmp_path)
    run_dir = tmp_path / "runs" / "h"
    good = {**_fake_result(0.0, 20.0), **_healthy_extras()}
    bad = {**_fake_result(0.0, 30.0), **_healthy_extras()}
    bad["channels"] = {"vpr": {"status": "inactive", "reason": "only 0 mapillary refs"}}
    _write_rep(run_dir, "clipA", 0, good)
    _write_rep(run_dir, "clipA", 1, bad)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    h = s["clips"]["clipA"]["health"]
    assert h["n_degraded"] == 1 and "retrieval inactive" in h["degraded"][0]
    assert any("DEGRADED" in w and "clipA" in w for w in s["warnings"])
    md = M.render_markdown(s)
    assert "### Run health" in md and "**1**" in md and "megaloc" in md


# --- infrastructure-failure classification and retry --------------------------------

def _log(tmp_path: Path, text: str, enc: str = "utf-8") -> Path:
    p = tmp_path / "log.txt"
    p.write_bytes(text.encode(enc))
    return p


@pytest.mark.parametrize("rc,text,expect_infra", [
    (3221227274, "", True),                                       # 0xC000070A: native termination
    (137, "", True),                                              # SIGKILL
    (1, "cv2.error: Unknown C++ exception from OpenCV code", True),
    (1, "numpy._core._exceptions._ArrayMemoryError: Unable to allocate 8.3 GiB", True),
    (1, "ValueError: Could not reach Overpass to fetch the OSM road graph", True),
    (1, "yt_dlp.utils.DownloadError: ERROR: unable to download video data: HTTP Error 403", True),
    (1, "ValueError: Could not fetch an OSM road graph for 'Qwertyville'", False),   # bad city
    (1, "AssertionError: pipeline invariant", False),                                # real bug
    (1, "TruncatedVideoError: input.mp4 stops decoding at 44 s", False),             # real input problem
    (0, "Unknown C++ exception", False),                                             # success wins
])
def test_infra_failure_reason(tmp_path, rc, text, expect_infra) -> None:
    why = M.infra_failure_reason({"rc": rc, "timed_out": False}, _log(tmp_path, text))
    assert (why is not None) is expect_infra


def test_timeouts_are_never_infra_failures(tmp_path) -> None:
    out = {"rc": None, "timed_out": True}
    assert M.infra_failure_reason(out, _log(tmp_path, "bad_alloc")) is None


def test_infra_classifier_reads_utf16_logs(tmp_path) -> None:
    p = _log(tmp_path, "cv2.error: Unknown C++ exception from OpenCV code", enc="utf-16")
    assert M.infra_failure_reason({"rc": 1, "timed_out": False}, p) is not None


def test_cmd_run_retries_infrastructure_failures_only(tmp_path, monkeypatch, capsys) -> None:
    """An externally-killed attempt is rerun and recorded; a genuine failure
    is not retried and stays a failure."""
    root, manifest = _setup_root(tmp_path)
    monkeypatch.setattr(M, "ROOT", root)
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    calls: dict[str, int] = {}

    def fake_run_one(argv, log_path, env, timeout):
        clip = next(a for a in argv if "clip" in str(a).lower() and "rep" in str(a))
        key = "A" if "clipA" in clip else "B"
        calls[key] = calls.get(key, 0) + 1
        if key == "A" and calls[key] == 1:
            Path(log_path).write_text("killed", encoding="utf-8")
            return {"rc": 3221227274, "timed_out": False, "wall_s": 1.0,
                    "started_utc": "t", "finished_utc": "t"}
        if key == "B":
            Path(log_path).write_text("AssertionError: pipeline invariant", encoding="utf-8")
            return {"rc": 1, "timed_out": False, "wall_s": 1.0,
                    "started_utc": "t", "finished_utc": "t"}
        Path(log_path).write_text("ok", encoding="utf-8")
        return {"rc": 0, "timed_out": False, "wall_s": 1.0,
                "started_utc": "t", "finished_utc": "t"}

    monkeypatch.setattr(M, "run_one", fake_run_one)
    monkeypatch.setattr(M.time, "sleep", lambda s: None)
    monkeypatch.setattr(M, "child_env", lambda *a, **k: ({}, {"mly_token_set": False,
                                                             "mly_token_origin": None}))
    args = M.build_parser().parse_args([
        "run", "--run-name", "t", "--manifest", "benchmarks/manifest.json",
        "--clips", "clipA", "clipB", "--repeats", "1", "--infra-retries", "2",
        "--infra-wait", "0", "--overpass-wait", "0", "--min-free-commit-gb", "0",
        "--min-free-vram-gb", "0", "--include-unavailable"])
    M.cmd_run(args)
    st_a = json.loads((root / M.RUNS_DIR / "t" / "clipA" / "rep0" / "status.json").read_text())
    st_b = json.loads((root / M.RUNS_DIR / "t" / "clipB" / "rep0" / "status.json").read_text())
    assert calls == {"A": 2, "B": 1}
    assert st_a["rc"] == 0 and len(st_a["infra_retries"]) == 1
    assert "abnormal native termination" in st_a["infra_retries"][0]["reason"]
    assert st_b["rc"] == 1 and st_b["infra_retries"] == []


# --- Overpass preflight ---------------------------------------------------------------

def test_wait_for_overpass_polls_until_reachable() -> None:
    answers = iter([None, None, "65.109.112.52"])
    slept, said = [], []
    ok = M.wait_for_overpass(600, 120, probe=lambda: next(answers),
                             sleep=slept.append, announce=said.append)
    assert ok is True and slept == [120, 120]
    assert len(said) == 2 and "unreachable" in said[0]


def test_wait_for_overpass_gives_up_after_its_budget() -> None:
    slept, said = [], []
    ok = M.wait_for_overpass(240, 120, probe=lambda: None,
                             sleep=slept.append, announce=said.append)
    assert ok is False and slept == [120, 120]
    assert "launching anyway" in said[-1]


def test_wait_for_overpass_is_immediate_when_reachable() -> None:
    slept = []
    assert M.wait_for_overpass(600, probe=lambda: "1.2.3.4", sleep=slept.append) is True
    assert slept == []


# --- commit gate ----------------------------------------------------------------------

def test_wait_for_commit_polls_until_there_is_room() -> None:
    free = iter([2.0, 6.0, 20.0])
    slept, said = [], []
    ok = M.wait_for_commit(16, 600, 60, probe=lambda: next(free),
                           sleep=slept.append, announce=said.append)
    assert ok is True and slept == [60, 60] and "2.0 GB commit free" in said[0]


def test_wait_for_commit_gives_up_after_its_budget() -> None:
    slept, said = [], []
    ok = M.wait_for_commit(16, 120, 60, probe=lambda: 1.0,
                           sleep=slept.append, announce=said.append)
    assert ok is False and slept == [60, 60] and "launching anyway" in said[-1]


def test_wait_for_commit_proceeds_when_it_cannot_measure() -> None:
    assert M.wait_for_commit(16, 600, probe=lambda: None, sleep=lambda s: 1 / 0) is True


def test_available_commit_gb_is_a_plausible_number() -> None:
    v = M.available_commit_gb()
    assert v is None or 0.0 < v < 100_000.0


def test_a_method_that_never_ran_is_not_scored_as_a_failure(tmp_path: Path) -> None:
    """The OCR-anchor pin is off by default, so no clip has a pinned_position.
    Scoring it as +inf printed a 0% 'failure' column for a channel that was
    never run."""
    run_dir, manifest, root = _make_run(tmp_path)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    fs = next(iter(s["fleets"].values()))
    assert "pinned_only" in fs["methods_not_run"] and "pinned_only" not in fs["methods"]
    md = M.render_markdown(s)
    assert "Not run (no clip produced an answer" in md and "Anchor-pinned only" in md
    assert "| Anchor-pinned only |" not in md          # no row in the success table


def test_a_method_answered_in_some_clips_stays_a_per_clip_failure(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    # Give clipA's first rep a pinned route: the method now exists in the fleet.
    p = run_dir / "clipA" / "rep0" / "result.json"
    res = json.loads(p.read_text(encoding="utf-8"))
    res["pinned_position"] = {"source": "ocr_anchor_pin",
                              "route_latlon": res["position"]["route_latlon"]}
    p.write_text(json.dumps(res), encoding="utf-8")
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    fs = next(iter(s["fleets"].values()))
    assert "pinned_only" in fs["methods"] and "pinned_only" not in fs["methods_not_run"]
    assert s["clips"]["clipB"]["methods"]["pinned_only"]["n_answered"] == 0   # still a miss


def test_vram_gate_uses_the_gpu_probe_and_its_own_label() -> None:
    said = []
    ok = M.wait_for_commit(8, 0, probe=lambda: 1.3, announce=said.append,
                           what="GPU memory")
    assert ok is False and "1.3 GB GPU memory free" in said[-1]


def test_available_vram_gb_is_none_or_plausible() -> None:
    v = M.available_vram_gb()
    assert v is None or 0.0 <= v < 1000.0


def test_provenance_compares_package_trees_not_commit_hashes(monkeypatch) -> None:
    """Changing scripts/ or tests/ between reps must not make a run
    'inconsistent': results depend only on monocular_osm/."""
    trees = {"c1": "tree-A", "c2": "tree-A", "c3": "tree-B"}
    monkeypatch.setattr(M, "package_tree", lambda c, root=None: trees.get(c))

    def rep(commit):
        return {"result_found": True, "provenance": {
            "present": True, "commit": commit, "dirty": False, "mly_token_set": True,
            "ocr_engine": "rapidocr", "vo_algo_version": 3, "package_version": "x",
            "argv": [], "gt_waypoints_cfg": None, "gt_streets_cfg": [],
            "osm_around_cfg": None}, "clip": "c", "rep": 0}

    same_code = M.check_provenance([rep("c1"), rep("c2")])
    assert same_code["consistent"] and same_code["commits"] == ["c1", "c2"]
    assert same_code["package_trees"] == ["tree-A"]
    other_code = M.check_provenance([rep("c1"), rep("c3")])
    assert not other_code["consistent"]
    assert any("different pipeline code" in p for p in other_code["problems"])


def test_cold_archive_metadata_only_drops_arrays_keeps_id_lists(tmp_path: Path) -> None:
    slug = tmp_path / "abc-x"
    (slug / "mapillary").mkdir(parents=True)
    M.set_vpr_caches_aside(slug)
    fresh = slug / "mapillary"
    fresh.mkdir()
    (fresh / "ref_img_store.npz").write_bytes(b"x" * 1000)
    (fresh / "ref_emb_megaloc_abc.npz").write_bytes(b"e")
    (fresh / "mly_ref_meta_abc.json").write_text("{}")
    arch = slug / M.COLD_ARCHIVE_DIR / "r1" / "clipA" / "rep0"
    moved = M.restore_vpr_caches(slug, arch, keep_arrays=False)
    assert (arch / "mapillary" / "mly_ref_meta_abc.json").exists()
    assert not list((arch / "mapillary").glob("*.npz"))
    assert any("dropped" in m for m in moved)
    assert (slug / "mapillary").exists()                    # warm cache restored


def test_aggregate_scores_the_preregistered_selective_rule(tmp_path: Path) -> None:
    """pipeline_selective answers with the pipeline when its retrieval
    confidence clears the committed tau, else with the city centre; the rule
    is read from benchmarks/preregistered_rules.json, never fitted."""
    root, manifest = _setup_root(tmp_path)
    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "preregistered_rules.json").write_text(json.dumps(
        {"rules": {"pipeline_selective": {
            "tau": 0.15, "fallback": "city_centroid",
            "confidence_signal": ["vpr_coarse_to_fine.pass2_sim",
                                  "vpr_prior.track_sim_median"]}}}), encoding="utf-8")
    run_dir = tmp_path / "runs" / "r1"
    # clipA: confident and right; clipB: unconfident and 3 km off; clipC: no signal.
    for cid, shift, off, sim in (("clipA", 0.0, 20.0, 0.25), ("clipB", 5000.0, 3000.0, 0.10),
                                 ("clipC", 10000.0, 3000.0, None)):
        res = _fake_result(shift, off)
        if sim is not None:
            res["vpr_coarse_to_fine"] = {"pass1_sim": sim, "pass2_sim": sim}
        _write_rep(run_dir, cid, 0, res)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)

    ms = {c: s["clips"][c]["methods"] for c in ("clipA", "clipB", "clipC")}
    sym = lambda c, m: ms[c][m]["symmetric_mean_m"]["median"]          # noqa: E731
    assert sym("clipA", "pipeline_selective") == pytest.approx(sym("clipA", "pipeline"))
    assert sym("clipB", "pipeline_selective") == pytest.approx(sym("clipB", "city_centroid"))
    assert sym("clipC", "pipeline_selective") == pytest.approx(sym("clipC", "city_centroid"))
    assert s["clips"]["clipA"]["selective"]["n_answered"] == 1
    assert s["clips"]["clipC"]["selective"]["confidence"] == [None]
    sel = s["fleets"]["core"]["selective"]
    assert sel["tau"] == 0.15 and sel["coverage"] == pytest.approx(1 / 3)
    assert sel["table"] == {"answer_win": 1, "answer_loss": 0,
                            "abstain_win": 0, "abstain_loss": 2}
    assert sel["in_sample_clips"] == []                     # rule file names no dev clips
    hy = sel["hypotheses"]
    # Selective ties the pipeline on clipA and replaces two 3 km misses by the centre.
    assert (hy["H1"]["selective_wins"], hy["H1"]["selective_losses"], hy["H1"]["ties"]) == (2, 0, 1)
    assert hy["H1"]["direction_holds"] is True
    assert hy["H2"] == {"clips_answered": ["clipA"], "pipeline_beats_centre": 1, "holds": True}
    assert hy["H3"]["direction_holds"] is True
    assert hy["H3"]["fisher_one_sided_p"] == pytest.approx(1 / 3)
    md = M.render_markdown(s)
    assert "**H1** selective beats the pipeline" in md
    assert "Selective prediction (pre-registered τ = 0.15)" in md
    assert "Selective (pre-registered τ)" in md


def test_aggregate_without_a_rule_file_has_no_selective_method(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    assert "pipeline_selective" not in s["clips"]["clipA"]["methods"]
    assert "selective" not in s["fleets"]["core"]


# ---------------------------------------------------------------------------
# hardening after v2: kills, disk, devices, per-fleet provenance, fair cities
# ---------------------------------------------------------------------------


def test_a_process_killed_by_session_logoff_is_infrastructure(tmp_path: Path) -> None:
    log = tmp_path / "log.txt"
    log.write_text("[4a] VPR coarse prior\n", encoding="utf-8")
    assert "killed" in M.infra_failure_reason({"rc": 0x40010004}, log)      # DBG_TERMINATE_PROCESS
    assert "native" in M.infra_failure_reason({"rc": 0xC000026B}, log)      # DLL init failed, logoff
    assert M.infra_failure_reason({"rc": 1}, log) is None                   # a real failure stays one


def test_a_rep_never_starts_on_a_nearly_full_disk(monkeypatch, tmp_path: Path) -> None:
    import collections

    du = collections.namedtuple("du", "total used free")
    monkeypatch.setattr(M.shutil, "disk_usage", lambda p: du(100e9, 95e9, 5e9))
    assert "5.0 GB free disk" in M.disk_problem(10.0, tmp_path)
    assert M.disk_problem(0.0, tmp_path) is None
    monkeypatch.setattr(M.shutil, "disk_usage", lambda p: du(100e9, 50e9, 50e9))
    assert M.disk_problem(10.0, tmp_path) is None


def test_city_strings_must_be_named_in_the_title_or_documented() -> None:
    yt = _entry(city="Chicago, Illinois, USA")
    assert M.city_fairness_problem(yt, "Chicago | Night Drive", None) is None
    assert M.city_fairness_problem(yt, None, None) is None          # metadata not cached yet
    assert "not named" in M.city_fairness_problem(
        _entry(city="Daly City, California, USA"), "Driving to work, Bay Area", "")
    assert M.city_fairness_problem(_entry(city="Málaga, Spain"), "Malaga drive", None) is None
    nc = _entry(city="Newcastle upon Tyne, England, UK")
    assert M.city_fairness_problem(nc, "Newcastle Driving - Tyne Bridge", None) is None
    assert "not named" in M.city_fairness_problem(
        _entry(city="San Francisco, USA"), "San Jose commute", None)     # 'San' is not enough
    assert M.city_fairness_problem(_entry(city="Den Haag, Netherlands"),
                                   "ritje denhaag 10-06-2021", None) is None
    local = _entry(video={"path": "x.mp4"}, city="Karlsruhe, Germany")
    assert "city_basis" in M.city_fairness_problem(local, None, None)
    assert M.city_fairness_problem({**local, "city_basis": "KITTI docs"}, None, None) is None


def test_provenance_is_checked_within_each_fleet_and_reported_between(tmp_path: Path) -> None:
    run_dir, manifest, root = _make_run(tmp_path)
    for c in manifest["clips"]:
        if c["id"] == "clipC":
            c["fleet"] = "overlay"
    for k in range(3):                                   # clipC on other code: fine
        d = run_dir / "clipC" / f"rep{k}"
        if (d / "result.json").exists():
            res = json.loads((d / "result.json").read_text(encoding="utf-8"))
            res["provenance"]["git"]["commit"] = "other"
            (d / "result.json").write_text(json.dumps(res), encoding="utf-8")
    s = M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)
    assert s["provenance"]["consistent"] is True
    assert any("different pipeline code" in x for x in s["provenance"]["cross_fleet"])
    # ...but two code states inside one fleet are still refused.
    _write_rep(run_dir, "clipA", 3, _fake_result(0.0, 25.0, commit="third"))
    with pytest.raises(SystemExit, match="core: reps ran different pipeline code"):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)


def test_retrieval_device_mixing_within_a_fleet_is_a_provenance_problem(tmp_path: Path) -> None:
    root, manifest = _setup_root(tmp_path)
    run_dir = tmp_path / "runs" / "r1"
    for k, dev in enumerate(("cuda", "cpu")):
        res = _fake_result(0.0, 20.0)
        res["vpr_prior"]["backbone"] = {"name": "megaloc", "device": dev}
        _write_rep(run_dir, "clipA", k, res)
    with pytest.raises(SystemExit, match="different devices"):
        M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)


def _summary_for(tmp_path: Path, name: str, offsets: dict, city=None) -> dict:
    root, manifest = _setup_root(tmp_path / name)
    run_dir = tmp_path / name / "runs" / "r"
    shifts = {"clipA": 0.0, "clipB": 5000.0, "clipC": 10000.0}
    for cid, offs in offsets.items():
        for k, off in enumerate(offs):
            _write_rep(run_dir, cid, k, _fake_result(shifts[cid], off),
                       city=(city or {}).get(cid))
    if city:
        for c in manifest["clips"]:
            if c["id"] in city:
                c["city_alternatives"] = [city[c["id"]]]
    return M.aggregate(run_dir, manifest, root=root, allow_network=False, n_resamples=99)


def test_compare_pairs_matching_clips_and_refuses_the_rest(tmp_path: Path) -> None:
    base = _summary_for(tmp_path, "base", {"clipA": [100, 110], "clipB": [400, 420],
                                           "clipC": [300, 310]})
    new = _summary_for(tmp_path, "new", {"clipA": [30, 35], "clipB": [200, 210],
                                         "clipC": [300, 310]},
                       city={"clipC": "City clipC alt"})
    cmp = M.compare_runs(base, new)
    assert [r["clip"] for r in cmp["rows"]] == ["clipA", "clipB"]
    assert cmp["new_better"] == 2 and cmp["new_worse"] == 0
    assert cmp["excluded"][0]["clip"] == "clipC" and "city differs" in cmp["excluded"][0]["reason"]
    assert any("only 2 paired clips" in n for n in cmp["notes"])
    md = M.render_compare(cmp, "base", "new")
    assert "Not compared: clipC" in md and "New better on 2 clips" in md
