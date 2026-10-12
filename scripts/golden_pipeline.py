"""Behaviour-preservation check for refactoring the pipeline.

``run_pipeline`` is ~2,500 lines of stages sharing dozens of locals, and has
no end-to-end test: its unit tests cover helpers, not the wiring between
stages. Restructuring it therefore needs an oracle. This script runs the
full DEFAULT pipeline on one clip from a COPY of its warm caches (video,
VO trajectory, road graphs, Mapillary references and embeddings, geocodes —
so no network and no re-fetch variance), normalises ``result.json`` and
``fusion_features.json``, and either records that as the golden digest or
compares against a recorded one and prints every differing path.

    # once, on the code you trust:
    python scripts/golden_pipeline.py record --golden output/golden/before.json
    # after each refactoring step:
    python scripts/golden_pipeline.py compare --golden output/golden/before.json
    # are two consecutive runs identical at all? (noise floor)
    python scripts/golden_pipeline.py determinism

The clip is KITTI 0009 (data/kitti/drive_0009.mp4, 46 s). The warm-cache
source is ``--cache-src`` (default ``data/smoke/<slug>``, produced by running
the pipeline once with an MLY_TOKEN set). Nothing is written there.

What a default run does NOT exercise, and how to reach it offline:

* ``--extra-args "<cli flags>"`` passes flags verbatim to the CLI, so a second
  and third digest can switch on optional stages (sliding windows, IPM,
  ground-flow scale, loop closure, a staged trajectory, ground truth...).
* Without a Mapillary token the VPR channel reports "unavailable" (the warm
  cache does not cover the city-extent query), which silently skips everything
  VPR drives: the re-rank penalty, coarse-to-fine, the anchor-primary
  placement. ``--stub-vpr`` replaces the retrieval call by a deterministic,
  seeded, noisy track along the ground-truth route, so those stages run
  offline and reproducibly. ``--mly-token`` instead uses the real token (live
  downloads; slow, not reproducible; for ad-hoc checks only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CITY = "Karlsruhe, Germany"
SEGMENT = "0:46"
DEFAULT_DATA_ROOT = ROOT / "data"

# result.json keys that legitimately differ between runs / checkouts.
VOLATILE_PATHS = {
    ("provenance", "created_utc"), ("provenance", "git"), ("provenance", "argv"),
    ("provenance", "platform"), ("provenance", "vo_cache"),
    ("provenance", "osm_graph", "cache"), ("provenance", "osm_graph", "created_date"),
    ("provenance", "config", "data_dir"), ("provenance", "config", "output_dir"),
    ("provenance", "config", "video_path"),
    ("provenance", "package_version"),
}
FLOAT_DIGITS = 3


def _normalise(obj, path=()):
    """JSON-safe copy with volatile keys dropped, floats rounded, paths made
    run-independent."""
    if isinstance(obj, dict):
        return {k: _normalise(v, path + (k,)) for k, v in sorted(obj.items())
                if path + (k,) not in VOLATILE_PATHS}
    if isinstance(obj, (list, tuple)):
        return [_normalise(v, path) for v in obj]
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else round(obj, FLOAT_DIGITS)
    if isinstance(obj, str):
        return obj.replace("\\", "/").replace(str(ROOT).replace("\\", "/"), "<root>")
    return obj


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


STUB_BOOT = r'''
"""Run the CLI with the VPR retrieval replaced by a deterministic synthetic track."""
import json
import sys

import numpy as np

sys.path.insert(0, {root!r})
import monocular_osm.kartaview_vpr as kv

_GT = json.load(open({gt!r}, encoding="utf-8"))["waypoints"]
_T = np.array([w["t_sec"] for w in _GT], dtype=float)
_LAT = np.array([w["lat"] for w in _GT], dtype=float)
_LON = np.array([w["lon"] for w in _GT], dtype=float)
_CALLS = {{"n": 0}}


def _stub_track(frames_bgr, center, radius_m=3000.0, *, n_query=80, **kwargs):
    """(query_indices, latlons, sims) ~60 m noisy around the GT route; the noise
    seed depends on the call number so pass 1 and the coarse-to-fine pass differ."""
    _CALLS["n"] += 1
    n = len(frames_bgr)
    idx = np.unique(np.linspace(0, n - 1, min(n_query, n)).round().astype(int))
    tt = idx / max(n - 1, 1) * _T[-1]
    lat, lon = np.interp(tt, _T, _LAT), np.interp(tt, _T, _LON)
    rng = np.random.RandomState(1234 + _CALLS["n"])
    sig = 60.0
    lat = lat + rng.randn(len(idx)) * sig / 111320.0
    lon = lon + rng.randn(len(idx)) * sig / (111320.0 * np.cos(np.radians(lat.mean())))
    sims = 0.6 + 0.05 * np.sin(np.arange(len(idx)))
    return idx, np.c_[lat, lon], sims


kv.kartaview_vpr_track = _stub_track
from monocular_osm.cli import main

sys.argv = ["osm-localize"] + sys.argv[1:]
main()
'''


def _mly_token() -> str | None:
    """MLY_TOKEN from the environment, else the Windows user environment (an
    agent / IDE shell does not always inherit it). Never printed or logged."""
    tok = os.environ.get("MLY_TOKEN")
    if tok:
        return tok
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return winreg.QueryValueEx(k, "MLY_TOKEN")[0] or None
    except (ImportError, OSError):
        return None


def run_once(cache_src: Path, workdir: Path, keep: bool, video: Path,
             extra: list[str] | None = None, stub_vpr: bool = False,
             use_token: bool = False) -> dict:
    """One hermetic run; returns the normalised digest dict."""
    if workdir.exists():
        shutil.rmtree(workdir)
    data, out = workdir / "data", workdir / "out"
    slug_dir = next(p for p in cache_src.iterdir() if p.is_dir())
    shutil.copytree(slug_dir, data / slug_dir.name)
    for name in ("geocode_cache.json", "reverse_geocode_cache.json"):
        if (cache_src / name).exists():
            shutil.copy2(cache_src / name, data / name)
    args = ["--video", str(video), "--city", CITY, "--vo-segment", SEGMENT,
            "--data-dir", str(data), "--output-dir", str(out), *(extra or [])]
    if stub_vpr:
        gt = ROOT / "ground_truth" / "kitti_drive_0009.json"
        boot = workdir / "stub_boot.py"
        boot.write_text(STUB_BOOT.format(root=str(ROOT), gt=str(gt)), encoding="utf-8")
        cmd = [sys.executable, str(boot), *args]
    else:
        cmd = [sys.executable, "-m", "monocular_osm.cli", *args]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    env.pop("MLY_TOKEN", None)              # hermetic unless --mly-token asks otherwise
    token = _mly_token() if use_token else None
    if token:
        env["MLY_TOKEN"] = token
    log = workdir / "run.log"
    with log.open("w", encoding="utf-8") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"pipeline exited {rc}; see {log}")
    result_json = next(out.glob("*/result.json"))
    digest = {"result": _normalise(json.loads(result_json.read_text(encoding="utf-8")))}
    feats = result_json.parent / "fusion_features.json"
    if feats.exists():
        digest["fusion_features"] = _normalise(
            json.loads(feats.read_text(encoding="utf-8")))
    digest["output_files"] = sorted(
        p.name for p in result_json.parent.iterdir() if p.is_file())
    if not keep:
        shutil.rmtree(data, ignore_errors=True)
    return digest


def diff(a: dict, b: dict, limit: int = 40) -> list[str]:
    fa, fb = dict(_flatten(a)), dict(_flatten(b))
    lines = []
    for k in sorted(set(fa) | set(fb)):
        if fa.get(k, "<missing>") != fb.get(k, "<missing>"):
            lines.append(f"  {k}: {fa.get(k, '<missing>')!r} -> {fb.get(k, '<missing>')!r}")
    return lines[:limit] + ([f"  ... {len(lines) - limit} more"] if len(lines) > limit else [])


def _sha(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("record", "compare", "determinism"))
    ap.add_argument("--golden", type=Path, default=ROOT / "output" / "golden" / "golden.json")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                    help="directory holding kitti/drive_0009.mp4 and smoke/<slug>/ "
                         "(point a worktree at the main checkout's data/). The "
                         "cache slug hashes the video's RESOLVED path, so the "
                         "video must be the same file the caches were built from.")
    ap.add_argument("--cache-src", type=Path, default=None,
                    help="warm-cache source (default: <data-root>/smoke)")
    ap.add_argument("--workdir", type=Path, default=ROOT / "output" / "golden" / "work")
    ap.add_argument("--keep", action="store_true", help="keep the working data copy")
    ap.add_argument("--extra-args", default="",
                    help="extra CLI flags, passed verbatim to monocular_osm.cli "
                         "(e.g. '--enable-sliding-window --use-vpr-sequence'); "
                         "lets a further digest exercise optional stages")
    ap.add_argument("--stub-vpr", action="store_true",
                    help="replace VPR retrieval by a deterministic synthetic track "
                         "so the VPR-driven stages run offline (see module doc)")
    ap.add_argument("--mly-token", action="store_true",
                    help="use the real Mapillary token (live downloads; not "
                         "reproducible; ad-hoc checks only)")
    args = ap.parse_args(argv)
    # the pipeline runs with cwd=ROOT, so every path handed to it must be absolute
    args.workdir = args.workdir.resolve()
    video = args.data_root.resolve() / "kitti" / "drive_0009.mp4"
    cache_src = args.cache_src or args.data_root / "smoke"
    if not video.exists():
        raise SystemExit(f"{video} missing (python scripts/fetch_kitti_raw.py ... --render)")

    extra = shlex.split(args.extra_args)
    opts = dict(extra=extra, stub_vpr=args.stub_vpr, use_token=args.mly_token)
    cur = run_once(cache_src, args.workdir, args.keep, video, **opts)
    if args.mode == "record":
        args.golden.parent.mkdir(parents=True, exist_ok=True)
        args.golden.write_text(json.dumps(cur, indent=1, sort_keys=True), encoding="utf-8")
        print(f"recorded {args.golden}  digest={_sha(cur)}")
        return 0
    if args.mode == "determinism":
        again = run_once(cache_src, args.workdir, args.keep, video, **opts)
        d = diff(cur, again)
        print("IDENTICAL" if not d else "NON-DETERMINISTIC fields:\n" + "\n".join(d))
        return 0 if not d else 1
    gold = json.loads(args.golden.read_text(encoding="utf-8"))
    d = diff(gold, cur)
    if not d:
        print(f"MATCH  digest={_sha(cur)}")
        return 0
    print(f"DIFFERS from {args.golden}:\n" + "\n".join(d))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
