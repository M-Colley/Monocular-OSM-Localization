"""Reproducible, leak-free benchmark of the GPS-free pipeline.

The committed definition of the benchmark is ``benchmarks/manifest.json``
(one entry per clip: video, analysis window, the ``--city`` string a user
would type, which GT file scores it and where that GT came from). This
script regenerates/validates that manifest, runs the pipeline on it and
scores the runs OFFLINE.

    python scripts/benchmark.py manifest [--check]
    python scripts/benchmark.py geocode-snapshot          # one-off, needs network
    python scripts/benchmark.py run --run-name r1 --fleet core --repeats 3
    python scripts/benchmark.py run --run-name r1-cold --repeats 3 --cold-vpr
    python scripts/benchmark.py run --run-name r1-metro --city-variant 0
    python scripts/benchmark.py aggregate --run-name r1

Leak-freedom is by construction: ``run`` gives the pipeline ONLY the video,
the manifest's city string, the analysis window and the data/output roots.
No ground-truth file, GT-derived disc (``--osm-around``) or route-length
prior ever reaches the pipeline — :func:`assert_leak_free` checks every argv
before launch, and ``aggregate`` re-audits the argv/config each result.json
recorded. Scoring needs the GT and therefore happens only in ``aggregate``.

Runs live under ``output/benchmark_runs/<run_name>/<clip_id>/rep<k>/``
(inside the git-ignored ``output/`` tree): ``argv.json``, ``log.txt``,
``status.json`` and a copy of the pipeline's ``result.json``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from build_manifest import (  # noqa: E402  (benchmarks/build_manifest.py)
    MANIFEST_PATH,
    _read_json,
    check_manifest,
    geocode_snapshot,
    gt_waypoints_digest,
    write_manifest,
)

RUNS_DIR = Path("output") / "benchmark_runs"
DEFAULT_DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# 1. The manifest (defined in benchmarks/build_manifest.py)
# ---------------------------------------------------------------------------


def load_manifest(path: Path | None = None) -> dict:
    return _read_json(path or (ROOT / MANIFEST_PATH))


def select_clips(manifest: dict, *, fleet: str = "core", clips: Iterable[str] | None = None,
                 split: str = "dev", include_unavailable: bool = False,
                 log=print) -> list[dict]:
    """Filter manifest clips. ``split='test'`` must be asked for explicitly."""
    wanted = set(clips) if clips else None
    known = {c["id"] for c in manifest["clips"]}
    if wanted and wanted - known:
        raise SystemExit(f"unknown clip id(s): {', '.join(sorted(wanted - known))}")
    out = []
    for c in manifest["clips"]:
        if wanted is not None and c["id"] not in wanted:
            continue
        if fleet != "all" and c["fleet"] != fleet and wanted is None:
            continue
        if split != "all" and c["split"] != split:
            continue
        if split == "all" and c["split"] != "dev":
            continue  # never by accident
        if not c.get("available_locally", True) and not include_unavailable:
            log(f"SKIP {c['id']}: available_locally=false "
                f"(--include-unavailable to try anyway)")
            continue
        out.append(c)
    return out


# ---------------------------------------------------------------------------
# 2. Leak guard and argv construction
# ---------------------------------------------------------------------------

#: Flags that carry ground truth (or GT-derived knowledge) into the pipeline.
FORBIDDEN_FLAGS: tuple[str, ...] = (
    "--osm-around",              # GT-centred graph disc
    "--ground-truth-waypoints",  # the GT track itself
    "--ground-truth",            # GT street names
    "--estimated-length-m",      # route-length prior (only known from GT)
)
#: Flags the runner sets itself; a passthrough may not override them.
RUNNER_FLAGS: tuple[str, ...] = (
    "--url", "--video", "--video-path", "--city", "--vo-segment", "--analyze-minutes",
    "--data-dir", "--output-dir",
)
_COORD_RE = re.compile(r"-?\d{1,3}\.\d+\s*,\s*-?\d{1,3}\.\d+")


class LeakError(ValueError):
    """An argv would hand ground truth to the pipeline."""


def _matches_flag(token: str, flags: Iterable[str]) -> str | None:
    """The flag ``token`` would select, honouring argparse's unambiguous
    prefix abbreviations (``--osm`` selects ``--osm-around``)."""
    if not token.startswith("--") or token == "--":
        return None
    name = token.split("=", 1)[0]
    for f in flags:
        if f == name or (len(name) > 2 and f.startswith(name)):
            return f
    return None


def assert_leak_free(argv: list[str], entry: dict | None = None) -> None:
    """Raise :class:`LeakError` if ``argv`` could carry ground truth."""
    for tok in argv:
        tok = str(tok)
        hit = _matches_flag(tok, FORBIDDEN_FLAGS)
        if hit:
            raise LeakError(f"forbidden GT-derived flag {tok!r} ({hit}) in pipeline argv")
        low = tok.replace("\\", "/").lower()
        if "ground_truth" in low:
            raise LeakError(f"argv token {tok!r} references a ground-truth file")
        if _COORD_RE.search(tok):
            raise LeakError(f"argv token {tok!r} looks like a coordinate")
        if entry is not None:
            gt_name = Path(entry["gt"]).name.lower()
            if gt_name in low:
                raise LeakError(f"argv token {tok!r} names the GT file {gt_name}")
    if entry is not None:
        # Every occurrence: argparse keeps the LAST one.
        allowed = {entry["city"], *entry.get("city_alternatives", [])}
        for i, tok in enumerate(argv):
            tok = str(tok)
            if _matches_flag(tok, ("--city",)) is None:
                continue
            city = tok.split("=", 1)[1] if "=" in tok else (
                argv[i + 1] if i + 1 < len(argv) else None)
            if city not in allowed:
                raise LeakError(f"--city {city!r} is not the manifest city or a listed "
                                f"alternative for {entry['id']}")


def check_passthrough(extra: list[str]) -> list[str]:
    """Validate ``--pipeline-arg`` values: no GT flags, no runner-owned flags."""
    for tok in extra:
        if _matches_flag(tok, RUNNER_FLAGS):
            raise SystemExit(f"--pipeline-arg {tok!r} would override a runner-owned flag")
    assert_leak_free(list(extra))
    return list(extra)


def city_for(entry: dict, variant: int | None) -> str | None:
    """The manifest city, or alternative #variant (None if it has none)."""
    if variant is None:
        return entry["city"]
    alts = entry.get("city_alternatives") or []
    return alts[variant] if 0 <= variant < len(alts) else None


def build_pipeline_argv(entry: dict, *, city: str, output_dir: Path, data_dir: Path,
                        python: str = sys.executable,
                        extra: Iterable[str] = ()) -> list[str]:
    """The ONLY way the benchmark launches the pipeline."""
    video = entry["video"]
    extra = list(extra)
    if "url" in video:
        src = ["--url", video["url"]]
        if "local_path" in video and any(_matches_flag(t, ("--skip-download",)) for t in extra):
            registered = video["local_path"]
            if not isinstance(registered, str) or not registered.strip():
                raise ValueError(f"{entry['id']}: video.local_path must name the registered local video")
            path = Path(registered)
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file():
                raise FileNotFoundError(f"{entry['id']}: registered video.local_path missing: {path}")
            src += ["--video-path", path.resolve().as_posix()]
    elif "path" in video:
        src = ["--video", Path(video["path"]).as_posix()]
    else:
        raise ValueError(f"{entry['id']}: video needs 'url' or 'path'")
    argv = [python, "-m", "monocular_osm.cli", *src,
            "--city", city,
            "--vo-segment", entry["vo_segment"],
            "--data-dir", Path(data_dir).as_posix(),
            "--output-dir", Path(output_dir).as_posix(),
            *extra]
    assert_leak_free(argv, entry)
    return argv


# ---------------------------------------------------------------------------
# 3. Credentials (never logged)
# ---------------------------------------------------------------------------

SECRET_ENV = "MLY_TOKEN"


def _registry_user_env(name: str) -> str | None:
    """Persistent Windows environment value (HKCU\\Environment, then the
    machine-wide one): set via `setx`, it is missing from os.environ of
    shells started before it was set. None elsewhere / when absent."""
    if sys.platform != "win32":
        return None
    import winreg
    for hive, path in (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (winreg.HKEY_LOCAL_MACHINE,
         r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
    ):
        try:
            with winreg.OpenKey(hive, path) as key:
                value, _ = winreg.QueryValueEx(key, name)
        except OSError:
            continue
        if value:
            return str(value)
    return None


def child_env(base: dict | None = None, registry=_registry_user_env) -> tuple[dict, dict]:
    """(env for the pipeline child, a loggable token record without the value)."""
    env = dict(os.environ if base is None else base)
    origin = None
    if env.get(SECRET_ENV):
        origin = "environ"
    else:
        tok = registry(SECRET_ENV)
        if tok:
            env[SECRET_ENV] = tok
            origin = "registry"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env, {"mly_token_set": origin is not None, "mly_token_origin": origin}


def redact_file(path: Path, secret: str | None) -> int:
    """Replace any occurrence of ``secret`` in ``path``; returns the count."""
    if not secret or len(secret) < 8 or not Path(path).exists():
        return 0
    data = Path(path).read_bytes()
    n = data.count(secret.encode("utf-8"))
    if n:
        Path(path).write_bytes(data.replace(secret.encode("utf-8"), b"<redacted>"))
    return n


# ---------------------------------------------------------------------------
# 4. Cold VPR: move warm reference caches aside, never delete them
# ---------------------------------------------------------------------------

#: Cache sub-directories kartaview_vpr / the pipeline create under data/<slug>/.
VPR_CACHE_DIRS: tuple[str, ...] = (
    "mapillary", "kartaview", "panoramax", "union",
    "mapillary_c2f", "kartaview_c2f", "panoramax_c2f", "union_c2f",
    "mapillary_outer", "kartaview_outer", "panoramax_outer",
    "mapillary_regions", "kartaview_regions", "panoramax_regions", "union_regions",
    "mapillary_city", "kartaview_city", "panoramax_city", "union_city",
)
ASIDE_DIR = "_benchmark_vpr_warm_aside"
COLD_ARCHIVE_DIR = "_benchmark_cold_vpr"


def _checked_cache_path(root: Path, path: Path) -> Path:
    """Refuse cache moves/deletions through links outside the intended root."""
    resolved_root, resolved_path = Path(root).resolve(), Path(path).resolve()
    if resolved_path == resolved_root or not resolved_path.is_relative_to(resolved_root):
        raise ValueError(f"cache path escapes its root: {path}")
    return Path(path)


def _regional_slug_root(slug_dir: Path, regional_root: Path) -> Path:
    return _checked_cache_path(regional_root, Path(regional_root) / Path(slug_dir).name)


def _vpr_cache_names(root: Path, *, external_regions: bool = False) -> list[str]:
    if not external_regions:
        return list(VPR_CACHE_DIRS)
    prefixes = tuple(f"{source}_{scope}_" for source in ("mapillary", "kartaview", "panoramax", "union")
                     for scope in ("region", "city"))
    return sorted(p.name for p in Path(root).glob("*") if p.name.startswith(prefixes))


def _assisted_cache_leaves(slug_dir: Path) -> list[Path]:
    """Local child caches retain the parent slug below assisted/<scope>/."""
    nested = _checked_cache_path(slug_dir, Path(slug_dir) / "assisted")
    if not nested.exists():
        return []
    leaves = []
    for scope in sorted(nested.iterdir()):
        if scope.is_dir():
            _checked_cache_path(slug_dir, scope)
            leaf = _checked_cache_path(slug_dir, scope / Path(slug_dir).name)
            if leaf.is_dir():
                leaves.append(leaf)
    return leaves


def _regional_cache_root_from_argv(argv: list[str]) -> Path | None:
    """Resolve the child's effective external regional cache root, if supplied."""
    value = None
    for i, token in enumerate(argv):
        if _matches_flag(str(token), ("--vpr-region-cache-dir",)):
            value = (str(token).split("=", 1)[1] if "=" in str(token)
                     else argv[i + 1] if i + 1 < len(argv) else None)
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _fold(text: str) -> str:
    import unicodedata
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "")
                   if not unicodedata.combining(ch)).casefold()


def city_fairness_problem(entry: dict, title: str | None,
                          description: str | None) -> str | None:
    """Is ``entry['city']`` what a user would type, or route knowledge?

    The 0.1.x table used "Daly City" and "Vaughan", narrower than anything a
    viewer could know. A YouTube clip's city must be named in its title or
    description; a dataset clip needs a documented ``city_basis``. Returns
    None when fair or when the metadata is not cached yet (checked again at
    the next run)."""
    city = entry["city"]
    head = _fold(city.split(",")[0].strip())
    if "url" in entry.get("video", {}):
        if title is None and description is None:
            return None
        text = _fold(" ".join(t for t in (title, description) if t))
        # The full name, or its distinctive first word: a title saying
        # "Newcastle" lets a user type "Newcastle upon Tyne".
        first = head.split()[0] if head.split() else head
        squash = lambda t: re.sub(r"[\s\-_]+", "", t)          # noqa: E731
        named = (head in text or squash(head) in squash(text)    # "denhaag" names Den Haag
                 or (len(first) >= 5 and re.search(
                     rf"(?<!\w){re.escape(first)}(?!\w)", text) is not None))
        if not named:
            return (f"{entry['id']}: city {city!r} is not named in the video's title or "
                    f"description, so a user could not know it")
        return None
    if not str(entry.get("city_basis") or "").strip():
        return f"{entry['id']}: dataset clip without a documented city_basis"
    return None


def city_fairness_problems(clips: list[dict], data_dir: Path) -> list[str]:
    from monocular_osm.download import load_cached_metadata

    out = []
    for entry in clips:
        meta = None
        if "url" in entry.get("video", {}):
            try:
                meta = load_cached_metadata(Path(data_dir), entry["video"]["url"])
            except Exception:
                meta = None
        why = city_fairness_problem(entry, getattr(meta, "title", None),
                                    getattr(meta, "description", None))
        if why:
            out.append(why)
    return out


def candidate_slug_dirs(entry: dict, city: str, data_dir: Path) -> list[Path]:
    """data/<slug>/ directories the pipeline may use for this clip+city."""
    from monocular_osm.city_inference import slugify_submission
    from monocular_osm.download import load_cached_metadata, local_video_metadata

    data_dir = Path(data_dir)
    out: list[Path] = []
    video = entry["video"]
    try:
        if "path" in video:
            meta = local_video_metadata(ROOT / video["path"])
        else:
            meta = load_cached_metadata(data_dir, video["url"])
        if meta is not None:
            out.append(data_dir / slugify_submission(
                meta.video_id, meta.title, city, fallback_seed=meta.url))
    except Exception:
        pass
    if "url" in video:
        # No cached metadata: every dir of this video id (any title/city).
        vid = re.sub(r".*[?&]v=", "", video["url"]).split("&")[0]
        id_slug = slugify_submission(vid, fallback_seed=video["url"])
        out.extend(sorted(data_dir.glob(f"{id_slug}-*")))
    seen, uniq = set(), []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def restore_vpr_caches(slug_dir: Path, archive_to: Path | None = None, *,
                       keep_arrays: bool = True, region_cache_root: Path | None = None,
                       _external_regions: bool = False) -> list[str]:
    """Undo :func:`set_vpr_caches_aside`. Fresh caches created by the cold
    run are MOVED to ``archive_to`` (inside the same clip data dir); then the
    warm caches go back where they were.

    ``keep_arrays=False`` archives only the reference metadata (the JSON lists
    of image ids, positions and capture times, ~3 % of the bytes) and drops the
    run's own ``.npz`` arrays: the downloaded pixels and their embeddings,
    ~1.2 GB per clip and rep, which are regenerable from those ids. Keeping
    them filled the benchmark disk after ~50 reps (a full disk then silently
    disabled retrieval and truncated a road-graph cache)."""
    slug_dir = Path(slug_dir)
    assisted = _assisted_cache_leaves(slug_dir) if not _external_regions else []
    if archive_to is not None:
        archive_to = _checked_cache_path(slug_dir, Path(archive_to))
    moved: list[str] = []
    aside = _checked_cache_path(slug_dir, slug_dir / ASIDE_DIR)
    for name in _vpr_cache_names(slug_dir, external_regions=_external_regions):
        fresh = _checked_cache_path(slug_dir, slug_dir / name)
        if fresh.exists() and archive_to is not None:
            dropped = 0
            if not keep_arrays:
                for f in sorted(fresh.rglob("*.npz")):
                    _checked_cache_path(fresh, f)
                    dropped += f.stat().st_size
                    f.unlink()
            archive_to.mkdir(parents=True, exist_ok=True)
            dest = _checked_cache_path(slug_dir, archive_to / name)
            if dest.exists():
                dest = archive_to / f"{name}.{int(time.time())}"
            shutil.move(str(fresh), str(dest))
            moved.append(f"{name} -> {dest}"
                         + (f" (dropped {dropped / 1e9:.2f} GB of .npz arrays)"
                            if dropped else ""))
    if aside.exists():
        for child in sorted(aside.iterdir()):
            _checked_cache_path(slug_dir, child)
            target = _checked_cache_path(slug_dir, slug_dir / child.name)
            if target.exists():
                # A crashed cold run left a fresh cache here: archive it first.
                arch = _checked_cache_path(
                    slug_dir, slug_dir / COLD_ARCHIVE_DIR / f"orphan_{int(time.time())}")
                arch.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(arch / child.name))
            shutil.move(str(child), str(target))
            moved.append(f"restored {child.name}")
        aside.rmdir()
    if region_cache_root is not None:
        regional = _regional_slug_root(slug_dir, Path(region_cache_root))
        regional_archive = (regional / archive_to.resolve().relative_to(slug_dir.resolve())
                            if archive_to is not None else None)
        moved.extend("external: " + item for item in restore_vpr_caches(
            regional, regional_archive, keep_arrays=keep_arrays, _external_regions=True))
    for leaf in assisted:
        child_archive = (leaf / archive_to.resolve().relative_to(slug_dir.resolve())
                         if archive_to is not None else None)
        moved.extend(f"assisted/{leaf.parent.name}: " + item for item in restore_vpr_caches(
            leaf, child_archive, keep_arrays=keep_arrays))
    return moved


def set_vpr_caches_aside(slug_dir: Path, *, region_cache_root: Path | None = None,
                         _external_regions: bool = False) -> list[str]:
    """Move existing VPR caches of one clip data dir into ``ASIDE_DIR``."""
    slug_dir = Path(slug_dir)
    assisted = _assisted_cache_leaves(slug_dir) if not _external_regions else []
    if (slug_dir / ASIDE_DIR).exists():
        restore_vpr_caches(slug_dir, slug_dir / COLD_ARCHIVE_DIR / "recovered",
                           _external_regions=_external_regions)
    moved = []
    _checked_cache_path(slug_dir, slug_dir / ASIDE_DIR)
    for name in _vpr_cache_names(slug_dir, external_regions=_external_regions):
        src = _checked_cache_path(slug_dir, slug_dir / name)
        if src.exists():
            (slug_dir / ASIDE_DIR).mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(slug_dir / ASIDE_DIR / name))
            moved.append(name)
    if region_cache_root is not None:
        regional = _regional_slug_root(slug_dir, Path(region_cache_root))
        moved.extend("external: " + item for item in set_vpr_caches_aside(
            regional, _external_regions=True))
    for leaf in assisted:
        moved.extend(f"assisted/{leaf.parent.name}: " + item for item in set_vpr_caches_aside(leaf))
    return moved


# ---------------------------------------------------------------------------
# 5. run
# ---------------------------------------------------------------------------


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _git_state() -> dict:
    from monocular_osm.provenance import git_state
    return git_state()


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
    else:
        try:
            os.killpg(proc.pid, 9)
        except OSError:
            proc.kill()


def run_one(argv: list[str], log_path: Path, env: dict, timeout: float) -> dict:
    """Run one pipeline child; returns rc / wall time / timeout flag."""
    t0 = time.time()
    started = _now()
    from monocular_osm.runtime_budget import DEADLINE_ENV, positive
    deadline = time.monotonic() + positive(timeout, "timeout")
    child_env = dict(env)
    # A nested caller may shorten this deadline, never extend it.
    if DEADLINE_ENV in child_env:
        deadline = min(deadline, positive(float(child_env[DEADLINE_ENV]), DEADLINE_ENV))
    child_env[DEADLINE_ENV] = repr(deadline)
    kw: dict = {}
    if os.name != "nt":
        kw["start_new_session"] = True
    with open(log_path, "wb") as fh:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                env=child_env, **kw)
        try:
            rc = proc.wait(timeout=max(0., deadline - time.monotonic()))
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            proc.wait()
            rc, timed_out = None, True
    return {"rc": rc, "timed_out": timed_out, "wall_s": round(time.time() - t0, 1),
            "started_utc": started, "finished_utc": _now()}


#: POSIX exit codes of a process killed by a signal from outside (SIGKILL /
#: SIGTERM). On Windows an abnormal NATIVE termination shows up as an NTSTATUS
#: code (>= 0xC0000000: e.g. 0xC000070A seen on a killed benchmark child) -
#: never as a Python failure, which always exits 1 with a traceback.
# 0x40010004 = DBG_TERMINATE_PROCESS: Windows' code for a process ended by
# another (TerminateProcess) or by the user session logging off - it killed a
# v2 overlay rep mid-retrieval and was scored as a pipeline failure.
_EXTERNAL_KILL_RCS = {-9, -15, 137, 143, 0x40010004}
_NTSTATUS_ERROR_FLOOR = 0xC0000000
#: Log patterns of a failure caused by the machine or the network, not by the
#: pipeline: allocation failures under memory pressure (OpenCV reports a failed
#: allocation as "Unknown C++ exception"), and network errors that survived the
#: pipeline's own retries.
_INFRA_LOG_PATTERNS = (
    "Unknown C++ exception from OpenCV", "bad_alloc", "MemoryError",
    "Unable to allocate", "out of memory", "CUDA error: out of memory",
    "Could not reach Overpass", "ConnectTimeout", "HTTP Error 403", "HTTP Error 429",
    "HTTP Error 503", "Read timed out",
)


def infra_failure_reason(outcome: dict, log_path: Path, tail_bytes: int = 60000) -> str | None:
    """Why a failed run is the environment's fault, or None for a genuine
    failure (which must stay recorded as one - a bad city, an assertion, a
    pipeline bug). Successful runs and timeouts are never classified."""
    rc = outcome.get("rc")
    if rc in (0, None) or outcome.get("timed_out"):
        return None
    if rc in _EXTERNAL_KILL_RCS:
        return f"process killed by a signal (rc={rc})"
    if isinstance(rc, int) and rc & 0xFFFFFFFF >= _NTSTATUS_ERROR_FLOOR:
        return f"abnormal native termination (rc=0x{rc & 0xFFFFFFFF:08X}); killed or crashed outside Python"
    try:
        with open(log_path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - tail_bytes))
            raw = fh.read()
    except OSError:
        return None
    # The log may be UTF-16 (PowerShell redirection) or UTF-8.
    text = raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8",
                      errors="replace").replace("\x00", "")
    for pat in _INFRA_LOG_PATTERNS:
        if pat in text:
            return f"log shows {pat!r}"
    return None


def available_commit_gb() -> float | None:
    """System-wide memory that can still be COMMITTED, in GB (Windows: the
    commit limit minus the commit charge; elsewhere: available RAM).

    This, not free RAM, is what a pipeline child runs out of: with a local LLM
    server and a dozen worker processes holding ~120 of 123 GB, free RAM was
    still 11-18 GB while allocations failed (OpenCV 'Unknown C++ exception'
    in cvtColor) and children exited with a native NTSTATUS code.
    """
    if os.name == "nt":
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullAvailPageFile / 1e9
    try:
        import psutil

        return psutil.virtual_memory().available / 1e9
    except Exception:
        return None


def available_disk_gb(*paths) -> float | None:
    """Free space (GB) on the fullest of the drives holding ``paths``."""
    free = []
    for path in paths:
        try:
            free.append(shutil.disk_usage(Path(path)).free / 1e9)
        except (OSError, ValueError):
            continue
    return min(free) if free else None


def disk_problem(min_gb: float, *paths) -> str | None:
    """Why a rep must not start for lack of disk, or None. A full disk does
    not fail loudly: in v2 it disabled retrieval ("No space left on device"),
    truncated a road-graph cache that every later run of the clip loaded, and
    stopped Windows growing its pagefile (MemoryError, access violations)."""
    if min_gb <= 0:
        return None
    free = available_disk_gb(*paths)
    if free is None or free >= min_gb:
        return None
    return (f"only {free:.1f} GB free disk (< {min_gb:.0f} GB): a full disk corrupts "
            f"caches and silently disables retrieval. Free space and re-run "
            f"(completed reps are kept).")


def available_vram_gb() -> float | None:
    """Free GPU memory in GB of the first GPU (via nvidia-smi), or None.

    The image-retrieval model runs on the GPU. When another process (here a
    local LLM server that loads and unloads multi-GB models) leaves it nearly
    full, the retrieval stage dies without a traceback: the driver logs
    'Graphics FECS Exception' and the child exits with an NTSTATUS code.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=True).stdout
        return float(out.strip().splitlines()[0]) / 1024.0
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def wait_for_commit(min_gb: float, max_wait_s: float, poll_s: float = 60.0, *,
                    probe=None, sleep=None, announce=print, what: str = "commit") -> bool:
    """Block until at least ``min_gb`` of ``what`` (default: committable
    memory) is free (or the budget is spent); True when there is enough.
    Launching a 9-12 GB pipeline into an exhausted commit limit, or the
    retrieval model into a full GPU, kills it minutes in; waiting at the gate
    does not. ``probe`` returning None (cannot measure) never blocks."""
    probe = probe or available_commit_gb
    sleep = sleep or time.sleep
    waited = 0.0
    while True:
        free = probe()
        if free is None or free >= min_gb:
            return True
        if waited >= max_wait_s:
            announce(f"  -> only {free:.1f} GB {what} free after {waited:.0f} s "
                     f"(wanted {min_gb:.0f} GB); launching anyway")
            return False
        announce(f"  -> only {free:.1f} GB {what} free (wanted {min_gb:.0f} GB); "
                 f"waiting {poll_s:.0f} s ({waited:.0f}/{max_wait_s:.0f} s)")
        sleep(poll_s)
        waited += poll_s


def wait_for_overpass(max_wait_s: float, poll_s: float = 120.0, *,
                      probe=None, sleep=None, announce=print) -> bool:
    """Block until an Overpass machine accepts a TCP connection (or the budget
    is spent); True when reachable.

    A run needs the road graph at the END of 10+ minutes of VO and image
    retrieval, so launching into an Overpass outage wastes the whole attempt
    (a 15-clip benchmark lost every rep that way). Waiting at the gate is
    cheap. Runs whose graph is already cached do not need Overpass, but there
    is no way to know in advance, and a short wait is harmless then.
    """
    if probe is None:
        from monocular_osm.net import reachable_address

        probe = lambda: reachable_address("overpass-api.de", 443, probe_timeout=6.0)
    sleep = sleep or time.sleep
    waited = 0.0
    while probe() is None:
        if waited >= max_wait_s:
            announce(f"  -> Overpass still unreachable after {waited:.0f} s; launching anyway")
            return False
        announce(f"  -> Overpass unreachable; waiting {poll_s:.0f} s "
                 f"({waited:.0f}/{max_wait_s:.0f} s)")
        sleep(poll_s)
        waited += poll_s
    return True


def _find_result(pipeline_out: Path) -> Path | None:
    hits = sorted(Path(pipeline_out).glob("*/result.json"))
    return hits[0] if len(hits) == 1 else (max(hits, key=lambda p: p.stat().st_mtime)
                                            if hits else None)


def _run_config(args, manifest_path: Path) -> dict:
    return {
        "city_variant": args.city_variant,
        "cold_vpr": bool(args.cold_vpr),
        "pipeline_args": list(args.pipeline_arg or []),
        "skip_download": bool(args.skip_download),
        "data_dir": Path(args.data_dir).as_posix(),
        "manifest": manifest_path.as_posix(),
    }


def cmd_run(args) -> int:
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    manifest_path = Path(args.manifest)
    manifest = load_manifest(ROOT / manifest_path if not manifest_path.is_absolute()
                             else manifest_path)
    clips = select_clips(manifest, fleet=args.fleet, clips=args.clips, split=args.split,
                         include_unavailable=args.include_unavailable)
    extra = check_passthrough(list(args.pipeline_arg or []))
    if args.skip_download:
        extra = ["--skip-download", *extra]

    run_dir = ROOT / RUNS_DIR / args.run_name
    cfg = _run_config(args, manifest_path)
    cfg_path = run_dir / "run_config.json"
    if args.dry_run:
        pass
    elif cfg_path.exists():
        prev = _read_json(cfg_path)
        if (prev.get("manifest_sha256") is not None
                and prev["manifest_sha256"] != _sha256_file(ROOT / manifest_path)):
            raise SystemExit(f"{cfg_path}: manifest changed since this run was created; "
                             "use a new --run-name")
        if prev.get("config") != cfg:
            raise SystemExit(
                f"{cfg_path} was created with a different configuration:\n"
                f"  before: {prev.get('config')}\n  now:    {cfg}\n"
                f"Use a new --run-name (runs must be internally consistent).")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps({
            "created_utc": _now(), "config": cfg,
            "manifest_sha256": _sha256_file(ROOT / manifest_path),
            "runner_git": _git_state(),
        }, indent=2), encoding="utf-8")

    unfair = ([] if args.city_variant is not None
              else city_fairness_problems(clips, ROOT / Path(args.data_dir)))
    if unfair:
        raise SystemExit("REFUSING: city strings a user could not know:\n  - "
                         + "\n  - ".join(unfair))

    if not args.dry_run:
        # Save the denominator before starting any expensive work. Merge when
        # resuming/adding a fleet or repeats, so an interrupted final clip can
        # never disappear from a later aggregate merely because no dir exists.
        saved = _read_json(cfg_path)
        plan = saved.setdefault("execution_plan", {})
        skipped = saved.setdefault("not_scheduled", {})
        for entry in clips:
            city = city_for(entry, args.city_variant)
            video = entry["video"]
            reason = (f"no city alternative #{args.city_variant}" if city is None else
                      f"local video {video['path']} missing"
                      if "path" in video and not (ROOT / video["path"]).exists() else None)
            if reason:
                skipped[entry["id"]] = reason
                continue
            old_reps = (plan.get(entry["id"]) or {}).get("reps", [])
            plan[entry["id"]] = {
                "city": city, "vo_segment": entry["vo_segment"],
                "reps": sorted(set(old_reps) | set(range(args.repeats))),
            }
            skipped.pop(entry["id"], None)
        tmp_config = cfg_path.with_suffix(".tmp")
        tmp_config.write_text(json.dumps(saved, indent=2), encoding="utf-8")
        tmp_config.replace(cfg_path)

    env, token_rec = child_env()
    secret = env.get(SECRET_ENV)
    print(f"MLY_TOKEN set: {token_rec['mly_token_set']} "
          f"(origin: {token_rec['mly_token_origin']})", flush=True)

    data_dir = Path(args.data_dir)
    n_done = n_skip = 0
    for entry in clips:
        city = city_for(entry, args.city_variant)
        if city is None:
            print(f"SKIP {entry['id']}: no city alternative #{args.city_variant}")
            continue
        video = entry["video"]
        if "path" in video and not (ROOT / video["path"]).exists():
            print(f"SKIP {entry['id']}: local video {video['path']} missing"
                  + (f" ({entry['acquire']})" if entry.get("acquire") else ""))
            continue
        for k in range(args.repeats):
            rep_dir = run_dir / entry["id"] / f"rep{k}"
            status_path = rep_dir / "status.json"
            if status_path.exists():
                st = _read_json(status_path)
                if st.get("rc") == 0 or not args.retry_failed:
                    n_skip += 1
                    continue
            pipeline_out = rep_dir / "pipeline_out"
            argv = build_pipeline_argv(
                entry, city=city, output_dir=pipeline_out.relative_to(ROOT),
                data_dir=data_dir, extra=extra)
            regional_cache_root = _regional_cache_root_from_argv(argv)
            if args.dry_run:
                print(subprocess.list2cmdline(argv))
                continue
            rep_dir.mkdir(parents=True, exist_ok=True)
            stamp = int(time.time())
            for old in ("pipeline_out", "result.json"):
                # A retried rep must not pick up the failed attempt's output.
                if (rep_dir / old).exists():
                    (rep_dir / old).rename(rep_dir / f"{old}.attempt{stamp}")
            (rep_dir / "argv.json").write_text(json.dumps({
                "argv": argv, "cwd": ".", **token_rec}, indent=2), encoding="utf-8")
            print(f"\n=== {entry['id']} rep{k}  city={city!r}  seg={entry['vo_segment']}",
                  flush=True)

            infra_retries: list[dict] = []
            for attempt in range(1, args.infra_retries + 2):
                why_disk = disk_problem(getattr(args, "min_free_disk_gb", 0),
                                        ROOT / data_dir, ROOT)
                if why_disk:
                    raise SystemExit(f"REFUSING to start {entry['id']} rep{k}: {why_disk}")
                if getattr(args, "min_free_commit_gb", 0) > 0:
                    wait_for_commit(args.min_free_commit_gb,
                                    getattr(args, "resource_wait", 0))
                if getattr(args, "min_free_vram_gb", 0) > 0:
                    wait_for_commit(args.min_free_vram_gb,
                                    getattr(args, "resource_wait", 0),
                                    probe=available_vram_gb, what="GPU memory")
                # A successful retrieval-only run never loads a road graph.
                # Let that fast path start even while Overpass is unavailable;
                # its geometry fallback handles graph-fetch retries normally.
                retrieval_only = any(_matches_flag(token, ("--retrieval-only",))
                                     for token in argv)
                if getattr(args, "overpass_wait", 0) > 0 and not retrieval_only:
                    wait_for_overpass(args.overpass_wait)
                cold: dict = {}
                slug_dirs = (candidate_slug_dirs(entry, city, ROOT / data_dir)
                             if args.cold_vpr else [])
                try:
                    for d in slug_dirs:
                        moved = set_vpr_caches_aside(d, region_cache_root=regional_cache_root)
                        if moved:
                            cold[d.name] = {"set_aside": moved}
                    outcome = run_one(argv, rep_dir / "log.txt", env, args.timeout)
                finally:
                    if args.cold_vpr:
                        res = _find_result(pipeline_out)
                        if res is not None:
                            d = ROOT / data_dir / res.parent.name
                            if d not in slug_dirs:
                                slug_dirs.append(d)
                        for d in slug_dirs:
                            arch = (d / COLD_ARCHIVE_DIR / args.run_name / entry["id"]
                                    / f"rep{k}")
                            moved = restore_vpr_caches(
                                d, arch,
                                region_cache_root=regional_cache_root,
                                keep_arrays=getattr(args, "cold_archive", "metadata") == "full")
                            if moved:
                                cold.setdefault(d.name, {})["restored"] = moved
                why = infra_failure_reason(outcome, rep_dir / "log.txt")
                if why is None or attempt > args.infra_retries:
                    break
                # The machine or network failed, not the pipeline: wait for the
                # contention to clear and run the SAME rep again (recorded).
                infra_retries.append({"attempt": attempt, "rc": outcome["rc"],
                                      "reason": why, "wall_s": outcome["wall_s"]})
                print(f"  -> infrastructure failure ({why}); retrying in "
                      f"{args.infra_wait:.0f} s ({attempt}/{args.infra_retries})",
                      flush=True)
                time.sleep(args.infra_wait)
                stamp = int(time.time())
                if pipeline_out.exists():
                    pipeline_out.rename(rep_dir / f"pipeline_out.attempt{stamp}")

            res = _find_result(pipeline_out)
            if res is not None:
                shutil.copy2(res, rep_dir / "result.json")
                # The per-candidate fusion table lets scripts/ablate.py
                # re-score channel ablations offline for this exact rep.
                feats = res.parent / "fusion_features.json"
                if feats.exists():
                    shutil.copy2(feats, rep_dir / "fusion_features.json")
            redactions = (redact_file(rep_dir / "log.txt", secret)
                          + redact_file(rep_dir / "result.json", secret))
            status = {
                "clip": entry["id"], "rep": k, "city": city,
                "city_variant": args.city_variant, "vo_segment": entry["vo_segment"],
                **outcome, **token_rec,
                "result_found": res is not None,
                "result_source": res.relative_to(ROOT).as_posix() if res else None,
                "cold_vpr": bool(args.cold_vpr), "cold_vpr_moves": cold,
                "cold_archive": getattr(args, "cold_archive", "metadata"),
                # Which device retrieval could use: a hidden GPU ("-1") means
                # the CPU (fp32) path, the GPU path runs fp16 autocast.
                "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "token_redactions": redactions,
                "infra_retries": infra_retries,
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            n_done += 1
            print(f"  -> rc={outcome['rc']} timeout={outcome['timed_out']} "
                  f"wall={outcome['wall_s']}s result={'yes' if res else 'NO'}",
                  flush=True)
    print(f"\n{n_done} rep(s) run, {n_skip} already complete -> {run_dir}")
    return 0


# ---------------------------------------------------------------------------
# 6. aggregate (offline scoring)
# ---------------------------------------------------------------------------

METRICS: tuple[str, ...] = (
    "start_error_m", "end_error_m", "gt_to_pred_mean_m", "pred_to_gt_mean_m",
    "symmetric_mean_m", "hausdorff_m", "frechet_m",
)
_SENTINEL = 1e15


def load_gt(entry: dict, root: Path | None = None) -> tuple[np.ndarray, dict]:
    gt = _read_json((root if root is not None else ROOT) / entry["gt"])
    arr = np.array([[float(w["lat"]), float(w["lon"])] for w in gt["waypoints"]])
    return arr, gt


def _score(pred, gt_latlon: np.ndarray) -> dict:
    from monocular_osm.metrics import failed_score, score_route
    if pred is None:
        out = failed_score(len(gt_latlon))
        out["answered"] = False
        return out
    out = score_route(pred.route_latlon, gt_latlon).to_dict()
    out.update(answered=True, failed=False, source=pred.source, kind=pred.kind,
               oracle=pred.oracle, meta=pred.meta)
    return out


def _quantiles(values: list[float]) -> dict:
    """median/IQR/min/max where +inf (failed/unanswered) sorts last and any
    quantile touching a failure is itself +inf."""
    if not values:
        return {"n": 0}
    a = np.where(np.isfinite(values), values, _SENTINEL).astype(float)
    q = np.quantile(a, [0.0, 0.25, 0.5, 0.75, 1.0])
    q = [float("inf") if v >= _SENTINEL / 10 else round(float(v), 1) for v in q]
    return {"n": len(values), "min": q[0], "q25": q[1], "median": q[2],
            "q75": q[3], "max": q[4]}


def _provenance_of(result: dict | None) -> dict:
    prov = (result or {}).get("provenance") or {}
    git = prov.get("git") or {}
    cfg = prov.get("config") or {}
    backbone = (((result or {}).get("vpr_prior") or {}).get("backbone") or {})
    poi_audit = prov.get("poi_localization_audit") or {}
    assisted_audit = prov.get("assisted_audit") or {}
    if not backbone:
        backbone = next(((r.get("provenance") or {}).get("backbone") for r in (poi_audit.get("regions") or [])
                         if isinstance(r, dict) and (r.get("provenance") or {}).get("backbone")), {})
    if not backbone:
        backbone = next(((r.get("comparison") or {}).get("model") for r in
                         (assisted_audit.get("reached_stages") or []) if isinstance(r, dict)
                         and (r.get("comparison") or {}).get("model")), {})
    selected_answer = (result or {}).get("answer") or {}
    return {
        "present": bool(prov),
        "commit": git.get("commit"), "dirty": git.get("dirty"),
        "mly_token_set": (prov.get("env") or {}).get("mly_token_set"),
        "ocr_engine": cfg.get("ocr_engine"),
        "vo_algo_version": prov.get("vo_algo_version"),
        "package_version": prov.get("package_version"),
        "config": cfg,
        "graph_filter_inputs": prov.get("graph_filter_inputs"),
        "poi_inputs": prov.get("poi_inputs"),
        "poi_localization_audit": prov.get("poi_localization_audit"),
        "assisted_inputs": prov.get("assisted_inputs"),
        "assisted_audit": prov.get("assisted_audit"),
        "assisted_selected_answer": ({"mode": selected_answer.get("mode"),
            "method": selected_answer.get("method"), "source": selected_answer.get("source"),
            "route_sha256": _fingerprint(selected_answer.get("route_latlon")),
            "point_count": len(selected_answer.get("route_latlon") or [])}
            if cfg.get("answer") == "assisted" else None),
        "python": prov.get("python"),
        "libraries": prov.get("libraries") or {},
        # The actual loaded model, including the pinned hub revision. A
        # missing optional channel is tracked by run health, not as a model.
        "vpr_model": ({k: v for k, v in backbone.items() if k != "device"}
                      if backbone.get("name") else None),
        "argv": prov.get("argv") or [],
        "gt_waypoints_cfg": cfg.get("ground_truth_waypoints"),
        "gt_streets_cfg": cfg.get("ground_truth_streets") or [],
        "osm_around_cfg": cfg.get("osm_around"),
        "osm_around_source_cfg": cfg.get("osm_around_source"),
    }


# Inputs and auto-selected per-video values may legitimately differ across
# clips. Compare these within each clip instead; method knobs must match the
# whole fleet. Cache/output roots do not affect either comparison.
_CONFIG_ROOTS = {"data_dir", "output_dir"}
# The absolute process clock records execution provenance, not method identity.
# Relative budgets and reserves remain method knobs in both comparisons.
_CONFIG_OPERATIONAL_FIELDS = {"runtime_deadline_monotonic_s"}
_CONFIG_CLIP_FIELDS = {
    "url", "city", "video_path", "video_title", "video_description", "ocr_video_path",
    "vo_start_sec", "vo_end_sec",
    "frame_stride", "osm_around", "osm_around_source", "openvo_trajectory_path",
    "vggt_long_trajectory_path",
}


def _fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _effective_config(prov: dict, *, within_clip: bool = False) -> dict:
    ignored = _CONFIG_ROOTS | _CONFIG_OPERATIONAL_FIELDS
    if not within_clip:
        ignored |= _CONFIG_CLIP_FIELDS
    return {k: v for k, v in (prov.get("config") or {}).items() if k not in ignored}


def _health_of(result: dict | None) -> dict:
    """How the run actually executed - from keys result.json already records.

    A run can "succeed" (rc=0, an answer) while a channel the headline depends
    on silently did nothing: retrieval dropped out (a torch.hub trust prompt),
    fell back to a different reference source or backbone, or the VO kept only
    a fraction of its poses. Error numbers alone hide all of that.
    """
    if not result:
        return {}
    ch = result.get("channels") or {}
    vpr_prior = result.get("vpr_prior") or {}
    backbone = (vpr_prior.get("backbone") or {}).get("name")
    n_frames, n_valid = result.get("n_frames"), result.get("n_valid_poses")
    cfg = (result.get("provenance") or {}).get("config") or {}
    poi = (result.get("provenance") or {}).get("poi_localization_audit") or {}
    poi_regions = poi.get("regions") or []
    poi_requested = cfg.get("answer") == "poi"
    poi_models = [(r.get("provenance") or {}).get("backbone") or {} for r in poi_regions if isinstance(r, dict)]
    poi_complete = (poi.get("status") == "ok" and "failed_regions" in poi and not poi["failed_regions"]
                    and bool(poi_regions) and all(isinstance(r, dict) and r.get("status") == "ok" for r in poi_regions))
    assisted = (result.get("provenance") or {}).get("assisted_audit") or {}
    selected_child = next((r for r in assisted.get("reached_stages") or []
                           if isinstance(r, dict) and r.get("kind") == assisted.get("selected_candidate")), {})
    assisted_complete = (cfg.get("answer") == "assisted" and bool(assisted.get("selected_candidate"))
                          and selected_child.get("status") == "complete"
                          and selected_child.get("worker_returncode") == 0
                          and (result.get("answer") or {}).get("mode") == "assisted")
    return {
        "filter_only": (cfg.get("answer") == "filter" and cfg.get("graph_filter") is True
                        and (result.get("answer") or {}).get("mode") == "filter"),
        "poi_only": (poi_requested and (result.get("answer") or {}).get("mode") == "poi" and poi_complete
                     and (ch.get("poi_retrieval") or {}).get("status") == "ok"),
        "assisted_only": assisted_complete,
        "assisted_requested": cfg.get("answer") == "assisted",
        "assisted_selected_candidate": assisted.get("selected_candidate"),
        "assisted_child_outcomes": {r.get("kind"): r.get("status") for r in assisted.get("reached_stages") or [] if isinstance(r, dict)},
        "poi_requested": poi_requested, "poi_status": poi.get("status"), "poi_complete": poi_complete,
        "poi_retrieval_status": (ch.get("poi_retrieval") or {}).get("status"),
        "poi_source": ((poi_regions[0].get("provenance") or {}).get("source")
                       if poi_regions and isinstance(poi_regions[0], dict) else None),
        "poi_backbone": poi_models[0].get("name") if poi_models else None,
        "poi_device": poi_models[0].get("device") if poi_models else None,
        "vpr_status": (ch.get("vpr") or {}).get("status"),
        "vpr_reason": (ch.get("vpr") or {}).get("reason") or (ch.get("vpr") or {}).get("error"),
        "vpr_source": vpr_prior.get("source"),
        "vpr_origin": vpr_prior.get("origin"),
        "vpr_backbone": backbone,
        "vpr_device": (vpr_prior.get("backbone") or {}).get("device"),
        "coarse_seed": (ch.get("coarse_from_frames") or {}).get("status"),
        "valid_pose_frac": (round(n_valid / n_frames, 3)
                            if n_frames and n_valid is not None else None),
        "reference_snapshots": {
            **{f"poi_region_{i}": rp["reference_inventory"]["refs_sha256"]
               for i, region in enumerate(poi_regions) if isinstance(region, dict)
               and isinstance(rp := region.get("provenance"), dict)
               and (rp.get("reference_inventory") or {}).get("refs_sha256")},
            **{
            name: stats["snapshot"]["refs_sha256"]
            for name, stats in (result.get("vpr_listing") or {}).items()
            if isinstance(stats, dict) and (stats.get("snapshot") or {}).get("refs_sha256")
            },
        },
        "failed_channels": sorted(k for k, v in ch.items()
                                  if (v or {}).get("status") == "failed"),
    }


def degraded_reasons(health: dict, *, token_set: bool) -> list[str]:
    """Why a run is NOT the published configuration (empty = healthy)."""
    if not health:
        return []
    why = []
    independent = health.get("filter_only") or health.get("poi_only") or health.get("assisted_only")
    if health.get("assisted_requested") and not health.get("assisted_only"):
        why.append("assisted policy did not select a complete successful child")
    if health.get("poi_requested") and (not health.get("poi_complete") or health.get("poi_retrieval_status") != "ok"):
        why.append("POI localization is partial, failed or unavailable")
    if not independent and health.get("vpr_status") != "ok":
        why.append(f"retrieval {health.get('vpr_status')}"
                   + (f" ({health['vpr_reason']})" if health.get("vpr_reason") else ""))
    elif not independent and token_set and health.get("vpr_source") not in (None, "mapillary", "union"):
        why.append(f"retrieval used {health.get('vpr_source')!r} although a token was set")
    if not independent and health.get("vpr_backbone") not in (None, "megaloc"):
        why.append(f"retrieval backbone {health.get('vpr_backbone')!r}, not megaloc")
    if health.get("failed_channels"):
        why.append("failed channels: " + ", ".join(health["failed_channels"]))
    frac = health.get("valid_pose_frac")
    if frac is not None and frac < 0.5:
        why.append(f"only {frac:.0%} of VO poses valid")
    return why


def leak_audit(prov: dict) -> list[str]:
    """Post-hoc: did the run that wrote this result see ground truth?"""
    problems = []
    for tok in prov.get("argv", []):
        hit = _matches_flag(str(tok), FORBIDDEN_FLAGS)
        if hit:
            problems.append(f"argv contains {tok!r}")
        if "ground_truth" in str(tok).replace("\\", "/").lower():
            problems.append(f"argv references {tok!r}")
    if prov.get("gt_waypoints_cfg"):
        problems.append("config.ground_truth_waypoints is set")
    if prov.get("gt_streets_cfg"):
        problems.append("config.ground_truth_streets is set")
    # A search disc is deployable only when the pipeline derived it from the
    # video itself (title/description place names). Any other disc — set by
    # hand, or of unknown origin in a pre-provenance result — may encode GT.
    if prov.get("osm_around_cfg") and prov.get("osm_around_source_cfg") != "video_title":
        problems.append(
            f"config.osm_around is set (source="
            f"{prov.get('osm_around_source_cfg')!r}), not derived from the video")
    return problems


def package_tree(commit: str | None, root: Path | None = None) -> str | None:
    """Git tree hash of ``monocular_osm/`` at ``commit``: identical for two
    commits iff the pipeline code is byte-identical. Results depend on that
    code, not on tests/, scripts/ or docs, so changing the runner between reps
    must not make a run 'inconsistent'."""
    if not commit or commit == "None":
        return None
    try:
        out = subprocess.run(["git", "rev-parse", f"{commit}:monocular_osm"],
                             cwd=root or ROOT, capture_output=True, text=True,
                             timeout=20, check=True).stdout.strip()
        return out or None
    except (OSError, subprocess.SubprocessError):
        return None


def check_provenance(reps: list[dict]) -> dict:
    """Check code, effective settings, dependencies and loaded models within
    a fleet. Separate runs may intentionally change any of these."""
    provs = [r["provenance"] for r in reps if r.get("result_found")]
    commits = sorted({str(p["commit"]) for p in provs})
    trees = {c: package_tree(c) for c in commits}
    distinct_trees = sorted({t or f"unknown@{c}" for c, t in trees.items()})
    dirty = sorted({str(p["dirty"]) for p in provs})
    tokens = sorted({str(p["mly_token_set"]) for p in provs})
    devices = sorted({str(r.get("vpr_device")) for r in reps
                      if r.get("result_found") and r.get("vpr_device")})
    missing = sum(1 for p in provs if not p["present"])
    leaks = {f"{r['clip']}/rep{r['rep']}": leak_audit(r["provenance"])
             for r in reps if r.get("result_found") and leak_audit(r["provenance"])}
    problems = []
    if missing:
        problems.append(f"{missing} result(s) carry no provenance block")
    if "None" in commits:
        problems.append(f"unknown commit in at least one rep: {commits}")
    elif len(distinct_trees) > 1:
        problems.append(f"reps ran different pipeline code (monocular_osm/ differs "
                        f"between commits {commits}): {distinct_trees}")
    if dirty != ["False"] and provs:
        problems.append(f"dirty working tree in at least one rep: {dirty}")
    if len(tokens) > 1:
        problems.append(f"mixed MLY_TOKEN states: {tokens}")
    if len(devices) > 1:
        # GPU retrieval runs fp16 autocast, CPU fp32: small, but not the same run.
        problems.append(f"retrieval ran on different devices: {devices}")
    method_configs = sorted({_fingerprint(_effective_config(p)) for p in provs})
    if len(method_configs) > 1:
        problems.append("reps ran different effective method configurations")
    for clip in sorted({r["clip"] for r in reps if r.get("result_found")}):
        configs = {_fingerprint(_effective_config(r["provenance"], within_clip=True))
                   for r in reps if r.get("result_found") and r["clip"] == clip}
        if len(configs) > 1:
            problems.append(f"{clip}: reps ran different effective clip configurations")
    environments = sorted({_fingerprint({"python": p.get("python"),
                                         "libraries": p.get("libraries") or {}})
                           for p in provs})
    if len(environments) > 1:
        problems.append("reps ran different Python/library versions")
    models = sorted({_fingerprint(p["vpr_model"]) for p in provs if p.get("vpr_model")})
    if len(models) > 1:
        problems.append("reps used different VPR model fingerprints")
    return {
        "consistent": not problems and not leaks,
        "problems": problems, "leaks": leaks,
        "commits": commits, "package_trees": distinct_trees,
        "dirty": dirty, "mly_token_set": tokens, "retrieval_devices": devices,
        "ocr_engines": sorted({str(p["ocr_engine"]) for p in provs}),
        "vo_algo_versions": sorted({str(p["vo_algo_version"]) for p in provs}),
        "package_versions": sorted({str(p["package_version"]) for p in provs}),
        "method_config_fingerprints": method_configs,
        "environment_fingerprints": environments,
        "vpr_model_fingerprints": models,
        "n_results": len(provs),
    }


def bootstrap_ci(values: Iterable[float], statistic: str = "median", *,
                 n_resamples: int = 9999, seed: int = 0,
                 confidence: float = 0.95) -> dict:
    """Percentile bootstrap CI over clips (the resampling unit).

    +inf (a clip whose method failed) is kept, not dropped: dropping it
    would bias the fleet number towards the clips that worked. The
    percentile method is used because it tolerates those +inf values and
    small n; BCa's jackknife is unstable for a median of ~9 clips.
    """
    from scipy.stats import bootstrap

    x = np.asarray(list(values), dtype=float)
    fn = {"median": np.median, "mean": np.mean}[statistic]
    est = float(fn(x)) if len(x) else float("nan")
    out = {"statistic": statistic, "n": int(len(x)), "estimate": est,
           "low": None, "high": None, "confidence": confidence,
           "method": "percentile", "n_resamples": n_resamples}
    if len(x) < 2:
        return out
    if statistic == "mean" and not np.isfinite(x).all():
        out["estimate"] = float("inf")
        return out
    xs = np.where(np.isfinite(x), x, _SENTINEL)
    if np.all(xs == xs[0]):
        out["low"] = out["high"] = float(x[0])
        return out
    res = bootstrap((xs,), fn, n_resamples=n_resamples, confidence_level=confidence,
                    method="percentile", rng=np.random.default_rng(seed),
                    vectorized=True)
    lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    out["low"] = float("inf") if lo >= _SENTINEL / 10 else lo
    out["high"] = float("inf") if hi >= _SENTINEL / 10 else hi
    return out


def _clip_method_summary(scores: list[dict]) -> dict:
    """Per-clip stats of one method over its reps (missing answers = +inf)."""
    from monocular_osm.metrics import SUCCESS_METRICS, THRESHOLDS_M
    out: dict = {"n_reps": len(scores),
                 "n_answered": sum(1 for s in scores if s.get("answered"))}
    for m in METRICS:
        out[m] = _quantiles([float(s[m]) for s in scores])
    out["success_rate"] = {
        metric: {str(t): (sum(1 for s in scores if s["success"][metric][str(t)])
                          / len(scores)) if scores else None
                 for t in THRESHOLDS_M}
        for metric in SUCCESS_METRICS}
    return out


def aggregate(run_dir: Path, manifest: dict, *, allow_mixed: bool = False,
              allow_network: bool = True, k: int = 5, n_resamples: int = 9999,
              seed: int = 0, root: Path | None = None) -> dict:
    from scipy.stats import binomtest, fisher_exact

    from monocular_osm import baselines as B
    from monocular_osm.metrics import SUCCESS_METRICS, THRESHOLDS_M

    root = Path(root) if root is not None else ROOT
    # Pre-registered, never fitted here (benchmarks/preregistered_rules.json).
    rule = B.load_preregistered_rule("pipeline_selective", root=root)
    centroid_cache: dict[str, object] = {}

    def _centroid(city: str):
        if city not in centroid_cache:
            centroid_cache[city] = B.city_centroid(city, root=root,
                                                   allow_network=allow_network)
        return centroid_cache[city]
    run_dir = Path(run_dir)
    run_metadata = (_read_json(run_dir / "run_config.json")
                    if (run_dir / "run_config.json").exists() else {})
    execution_plan = run_metadata.get("execution_plan")
    planned_clips = execution_plan or {}
    by_id = {c["id"]: c for c in manifest["clips"]}
    reps: list[dict] = []
    clips_out: dict[str, dict] = {}
    warnings: list[str] = []

    observed_clips = {p.name for p in run_dir.iterdir() if p.is_dir()}
    for clip_id in sorted(observed_clips | set(planned_clips)):
        clip_dir = run_dir / clip_id
        entry = by_id.get(clip_id)
        if entry is None:
            warnings.append(f"{clip_dir.name}: not in the manifest; ignored")
            continue
        gt_latlon, gt_blob = load_gt(entry, root)
        if gt_waypoints_digest(gt_blob) != entry.get("gt_waypoints_sha256"):
            msg = f"{entry['id']}: GT waypoints differ from the manifest's hash"
            if not allow_mixed:
                raise SystemExit(msg + " (regenerate the manifest or --allow-mixed)")
            warnings.append(msg)
        clip_reps = []
        observed_reps = ({int(p.name[3:]) for p in clip_dir.iterdir()
                          if p.is_dir() and re.fullmatch(r"rep\d+", p.name)}
                         if clip_dir.exists() else set())
        clip_plan = planned_clips.get(clip_id) or {}
        for rep_number in sorted(observed_reps | set(clip_plan.get("reps", []))):
            rep_dir = clip_dir / f"rep{rep_number}"
            st_path = rep_dir / "status.json"
            completed = st_path.exists()
            if not completed:
                warnings.append(f"{entry['id']}/{rep_dir.name}: no status.json "
                                f"(not started or interrupted); counted as a failure")
            st = (_read_json(st_path) if completed else {
                "city": clip_plan.get("city") or entry["city"],
                "vo_segment": clip_plan.get("vo_segment") or entry["vo_segment"],
            })
            res_path = rep_dir / "result.json"
            result = _read_json(res_path) if res_path.exists() else None
            ok = st.get("rc") == 0 and result is not None
            rep = {"clip": entry["id"], "rep": int(rep_dir.name[3:]),
                   "completed": completed,
                   "rc": st.get("rc"), "timed_out": st.get("timed_out"),
                   "wall_s": st.get("wall_s"), "city": st.get("city"),
                   "result_found": result is not None, "ok": ok,
                   "provenance": _provenance_of(result), "scores": {}}
            preds = {
                "pipeline": B.pipeline_headline(result) if ok else None,
                "matcher_only": B.matcher_only(result) if ok else None,
                "pinned_only": B.pinned_only(result) if ok else None,
                "graph_filter_only": B.graph_filter_only(result) if ok else None,
                "poi_only": B.poi_only(result) if ok else None,
                "assisted_policy": B.assisted_policy(result) if ok else None,
                "vpr_prior_only": B.vpr_prior_only(result) if ok else None,
                "vpr_track_only": B.vpr_track_only(result) if ok else None,
                "track_checked": B.track_checked(result) if ok else None,
                "answer": B.answer(result) if ok else None,
                f"shortlist_oracle@{k}": (B.shortlist_oracle(result, gt_latlon, k)
                                          if ok else None),
            }
            if rule is not None:
                sel = (B.pipeline_selective(result, _centroid(st.get("city") or entry["city"]),
                                            rule) if ok else None)
                preds["pipeline_selective"] = sel
                rep["selective"] = dict(sel.meta) if sel is not None else None
            for name, pred in preds.items():
                rep["scores"][name] = _score(pred, gt_latlon)
            if ok:
                pos = result.get("position") or {}
                rep["headline_source"] = pos.get("source")
                rep["health"] = _health_of(result)
                # Device: recorded by the pipeline since 2026-10-16; older reps
                # fall back to what the runner exposed (a hidden GPU = CPU).
                rep["vpr_device"] = rep["health"].get("poi_device") or rep["health"].get("vpr_device") or (
                    "cpu" if st.get("cuda_visible_devices") in ("-1", "") else None)
                rep["degraded"] = degraded_reasons(
                    rep["health"], token_set=bool(rep["provenance"].get("mly_token_set")))
            rep["vo_segment"] = st.get("vo_segment")
            clip_reps.append(rep)
            reps.append(rep)

        cities = sorted({r["city"] for r in clip_reps if r.get("city")}) or [entry["city"]]
        for c in cities:
            if c not in {entry["city"], *entry.get("city_alternatives", [])}:
                raise SystemExit(f"{entry['id']}: run used city {c!r}, which is not "
                                 f"in the manifest")
        centroid = {}
        for c in cities:
            pred = B.city_centroid(c, root=root, allow_network=allow_network)
            if pred is None:
                warnings.append(f"{entry['id']}: no centroid for {c!r} (not in the "
                                f"geocode snapshots and network disabled/failed)")
            centroid[c] = {"prediction": pred.to_dict() if pred else None,
                           "score": _score(pred, gt_latlon)}
        methods = {name: _clip_method_summary([r["scores"][name] for r in clip_reps])
                   for name in (list(clip_reps[0]["scores"]) if clip_reps else [])}
        cc = centroid[cities[0]]["score"]
        methods["city_centroid"] = _clip_method_summary([cc])
        clips_out[entry["id"]] = {
            "name": entry["name"], "fleet": entry["fleet"], "split": entry["split"],
            "city": cities[0] if len(cities) == 1 else cities,
            "n_reps": len(clip_reps), "n_ok": sum(r["ok"] for r in clip_reps),
            "n_completed": sum(r["completed"] for r in clip_reps),
            "headline_sources": sorted({str(r.get("headline_source")) for r in clip_reps
                                        if r["ok"]}),
            "health": {
                "n_degraded": sum(1 for r in clip_reps if r.get("degraded")),
                "degraded": sorted({w for r in clip_reps for w in r.get("degraded", [])}),
                "vpr_sources": sorted({str(r["health"].get("poi_source") if r["health"].get("poi_requested")
                                           else r["health"].get("vpr_source")) for r in clip_reps
                                       if r.get("health")}),
                "vpr_backbones": sorted({str(r["health"].get("poi_backbone") if r["health"].get("poi_requested")
                                             else r["health"].get("vpr_backbone")) for r in clip_reps
                                         if r.get("health")}),
                "vpr_devices": sorted({str(r.get("vpr_device")) for r in clip_reps
                                       if r.get("health")}),
                "reference_snapshots": {
                    name: sorted({r["health"]["reference_snapshots"][name]
                                  for r in clip_reps if r.get("health")
                                  and name in r["health"].get("reference_snapshots", {})})
                    for name in sorted({name for r in clip_reps if r.get("health")
                                        for name in r["health"].get("reference_snapshots", {})})
                },
                "coarse_seed_fired": sum(1 for r in clip_reps
                                         if (r.get("health") or {}).get("coarse_seed") == "ok"),
                "valid_pose_frac_median": (
                    float(np.median([r["health"]["valid_pose_frac"] for r in clip_reps
                                     if (r.get("health") or {}).get("valid_pose_frac") is not None]))
                    if any((r.get("health") or {}).get("valid_pose_frac") is not None
                           for r in clip_reps) else None),
            },
            "wall_s_median": (float(np.median([r["wall_s"] for r in clip_reps
                                               if r.get("wall_s") is not None]))
                              if any(r.get("wall_s") is not None for r in clip_reps)
                              else None),
            "gt_n_waypoints": int(len(gt_latlon)),
            "gt_waypoints_sha256": gt_waypoints_digest(gt_blob),
            "vo_segments": sorted({str(r["vo_segment"]) for r in clip_reps
                                   if r.get("vo_segment")}),
            "selective": ({
                "n_answered": sum(1 for r in clip_reps
                                  if r.get("selective") and not r["selective"]["abstained"]),
                "n_reps_ok": sum(1 for r in clip_reps if r.get("selective")),
                "confidence": [r["selective"]["confidence"] for r in clip_reps
                               if r.get("selective")],
            } if rule is not None else None),
            "city_centroid": centroid,
            "methods": methods,
        }

    for cid, c in clips_out.items():
        changed_snapshots = [name for name, hashes in
                             c["health"]["reference_snapshots"].items() if len(hashes) > 1]
        if changed_snapshots:
            warnings.append(f"{cid}: reference content changed across repeats "
                            f"({', '.join(changed_snapshots)}); cold retrieval can vary")
        if c["health"]["n_degraded"]:
            warnings.append(f"{cid}: {c['health']['n_degraded']}/{c['n_reps']} rep(s) "
                            f"DEGRADED - {'; '.join(c['health']['degraded'])}")

    prov = check_provenance(reps)
    if prov["leaks"]:
        raise SystemExit("REFUSING: ground truth reached the pipeline in "
                         f"{len(prov['leaks'])} rep(s): {prov['leaks']}")
    # Fleets are summarised separately, so consistency is required WITHIN a
    # fleet; a difference BETWEEN fleets is reported, not refused (v2: core on
    # one package tree, overlay on a later, behaviour-preserving one).
    fleet_of = {cid: c["fleet"] for cid, c in clips_out.items()}
    by_fleet = {f: check_provenance([r for r in reps if fleet_of.get(r["clip"]) == f])
                for f in sorted(set(fleet_of.values()))}
    prov["by_fleet"] = {f: {k: b[k] for k in ("consistent", "problems", "commits",
                                             "package_trees", "retrieval_devices",
                                             "method_config_fingerprints",
                                             "environment_fingerprints",
                                             "vpr_model_fingerprints")}
                        for f, b in by_fleet.items()}
    prov["problems"] = [f"{f}: {q}" for f, b in by_fleet.items() for q in b["problems"]]
    prov["consistent"] = not prov["problems"]
    trees = {f: tuple(b["package_trees"]) for f, b in by_fleet.items() if b["package_trees"]}
    devs = {f: tuple(b["retrieval_devices"]) for f, b in by_fleet.items() if b["retrieval_devices"]}
    prov["cross_fleet"] = [
        *([f"fleets ran different pipeline code: {trees}"] if len(set(trees.values())) > 1 else []),
        *([f"fleets ran retrieval on different devices: {devs}"]
          if len(set(devs.values())) > 1 else [])]
    if not prov["consistent"] and not allow_mixed:
        raise SystemExit("REFUSING to aggregate inconsistent provenance:\n  - "
                         + "\n  - ".join(prov["problems"])
                         + "\n(re-run cleanly, or pass --allow-mixed to override)")

    fleets: dict[str, dict] = {}
    for fleet in sorted({c["fleet"] for c in clips_out.values()}):
        ids = [i for i, c in clips_out.items() if c["fleet"] == fleet]
        method_names = sorted({m for i in ids for m in clips_out[i]["methods"]})
        # Hide optional methods that were never requested/produced, but keep
        # mandatory or explicitly enabled methods even if every inference
        # failed. Per-clip misses of a method that ran elsewhere also count.
        fleet_provs = [r["provenance"] for r in reps if r["clip"] in ids]
        all_independent = bool(fleet_provs) and all(p.get("config", {}).get("answer") in {"filter", "poi", "assisted"} for p in fleet_provs)
        required = set() if all_independent else {"pipeline", "matcher_only"}
        if execution_plan is not None or any("answer" in p.get("config", {}) for p in fleet_provs):
            required.add("answer")
        if any(p.get("config", {}).get("use_vpr_prior") and p.get("config", {}).get("answer") not in {"filter", "poi", "assisted"}
               for p in fleet_provs):
            required.update(("vpr_prior_only", "vpr_track_only", "track_checked"))
        if any(p.get("config", {}).get("graph_filter") and p.get("config", {}).get("answer") != "poi"
               for p in fleet_provs):
            required.add("graph_filter_only")
        if any(p.get("config", {}).get("answer") == "poi" for p in fleet_provs):
            required.add("poi_only")
        if any(p.get("config", {}).get("answer") == "assisted" for p in fleet_provs):
            required.add("assisted_policy")
        not_run = [m for m in method_names if m not in required
                   if not any(clips_out[i]["methods"][m]["n_answered"]
                              for i in ids if m in clips_out[i]["methods"])]
        method_names = [m for m in method_names if m not in not_run]
        fs: dict = {"clips": ids, "methods": {}, "methods_not_run": not_run}
        for m in method_names:
            have = [i for i in ids if m in clips_out[i]["methods"]]
            ms: dict = {"n_clips": len(have)}
            for metric in ("start_error_m", "symmetric_mean_m", "gt_to_pred_mean_m"):
                med = [clips_out[i]["methods"][m][metric]["median"] for i in have]
                ms[metric] = {
                    "median_of_clip_medians": bootstrap_ci(
                        med, "median", n_resamples=n_resamples, seed=seed),
                    "mean_of_clip_medians": bootstrap_ci(
                        med, "mean", n_resamples=n_resamples, seed=seed),
                }
            ms["success_rate"] = {
                metric: {str(t): bootstrap_ci(
                    [clips_out[i]["methods"][m]["success_rate"][metric][str(t)]
                     for i in have], "mean", n_resamples=n_resamples, seed=seed)
                    for t in THRESHOLDS_M}
                for metric in SUCCESS_METRICS}
            fs["methods"][m] = ms
        # Paired, per-clip: does the pipeline beat each baseline?
        paired = {}
        for m in method_names:
            if m == "pipeline":
                continue
            row = {}
            for metric in ("start_error_m", "symmetric_mean_m"):
                w = l_ = t = 0
                for i in ids:
                    ms_ = clips_out[i]["methods"]
                    if m not in ms_ or "pipeline" not in ms_:
                        continue
                    a = ms_["pipeline"][metric]["median"]
                    b = ms_[m][metric]["median"]
                    if a == b:
                        t += 1
                    elif a < b:
                        w += 1
                    else:
                        l_ += 1
                p = (binomtest(w, w + l_, 0.5).pvalue if (w + l_) else None)
                row[metric] = {"pipeline_wins": w, "pipeline_losses": l_, "ties": t,
                               "sign_test_p": p}
            paired[m] = row
        fs["paired_vs_pipeline"] = paired
        if rule is not None:
            fs["selective"] = _selective_summary(
                [r for r in reps if r["clip"] in ids], clips_out, rule, fisher_exact)
        fleets[fleet] = fs

    return {
        "run": run_dir.name, "generated_utc": _now(),
        "aggregator_git": _git_state(),
        "run_config": run_metadata or None,
        "execution": {
            "plan_recorded": execution_plan is not None,
            "expected_reps": (sum(len(p.get("reps", [])) for p in planned_clips.values())
                              if execution_plan is not None else None),
            "n_completed": sum(r["completed"] for r in reps),
            "incomplete_reps": [f"{r['clip']}/rep{r['rep']}" for r in reps
                                if not r["completed"]],
            "complete": (all(r["completed"] for r in reps)
                         if execution_plan is not None else None),
            "not_scheduled": run_metadata.get("not_scheduled") or {},
        },
        "provenance": prov, "warnings": warnings,
        "thresholds_m": list(THRESHOLDS_M),
        "clips": clips_out, "fleets": fleets,
        "reps": [{k_: v for k_, v in r.items() if k_ != "provenance"} | {
            "commit": r["provenance"]["commit"],
            "provenance": r["provenance"]} for r in reps],
    }


def _selective_summary(fleet_reps: list[dict], clips_out: dict, rule: dict,
                       fisher_exact) -> dict:
    """Coverage and the pre-registered H3 test for ``pipeline_selective``.

    Per rep: did the rule answer (confidence >= tau), and did the pipeline
    beat the city-centroid guess (symmetric error)? H3 asks whether answering
    reps win more often than abstaining ones (one-sided Fisher exact)."""
    table = {"answer_win": 0, "answer_loss": 0, "abstain_win": 0, "abstain_loss": 0}
    for r in fleet_reps:
        sel = r.get("selective")
        pipe = (r["scores"].get("pipeline") or {}).get("symmetric_mean_m")
        cc = (((clips_out[r["clip"]]["methods"].get("city_centroid") or {})
               .get("symmetric_mean_m") or {}).get("median"))
        if sel is None or pipe is None or cc is None:
            continue
        key = ("abstain" if sel["abstained"] else "answer") + (
            "_win" if pipe < cc else "_loss")
        table[key] += 1
    n = sum(table.values())
    n_answer = table["answer_win"] + table["answer_loss"]
    p = None
    if n_answer and n - n_answer:
        p = float(fisher_exact([[table["answer_win"], table["answer_loss"]],
                                [table["abstain_win"], table["abstain_loss"]]],
                               alternative="greater")[1])
    deriv = rule.get("derivation") or {}
    dev_clips = {c for k in ("wins", "losses", "no_signal")
                 for c in (deriv.get(k) or {})}
    return {"tau": float(rule["tau"]), "n_reps": n,
            "coverage": (n_answer / n) if n else None,
            "table": table, "h3_fisher_one_sided_p": p,
            "hypotheses": _registered_hypotheses(
                sorted({r["clip"] for r in fleet_reps}), clips_out, table, p),
            # Clips tau was chosen on: there the table is the rule's fit, not a test.
            "in_sample_clips": sorted({r["clip"] for r in fleet_reps} & dev_clips)}


def _registered_hypotheses(ids: list[str], clips_out: dict, table: dict,
                           h3_p: float | None) -> dict:
    """H1-H3 of ``benchmarks/preregistered_rules.json``, computed literally.

    Per clip, errors are the median over reps of the symmetric error; a clip
    counts as answered when the rule answered in a majority of its ok reps.
    Each verdict is ``True``/``False``, or ``None`` when untestable (no clip
    or rep to decide it) — reported, never dropped."""
    from scipy.stats import binomtest

    def med(cid, m):
        q = ((clips_out[cid]["methods"].get(m) or {}).get("symmetric_mean_m") or {})
        return q.get("median")

    sel_w = sel_l = ties = 0
    answered: list[str] = []
    h2_wins = 0
    for cid in ids:
        s_, p_, c_ = med(cid, "pipeline_selective"), med(cid, "pipeline"), med(cid, "city_centroid")
        if s_ is not None and p_ is not None:
            if s_ < p_:
                sel_w += 1
            elif s_ > p_:
                sel_l += 1
            else:
                ties += 1
        cs = clips_out[cid].get("selective") or {}
        if cs.get("n_reps_ok") and 2 * cs["n_answered"] > cs["n_reps_ok"]:
            answered.append(cid)
            if p_ is not None and c_ is not None and p_ < c_:
                h2_wins += 1
    h1_p = (float(binomtest(sel_w, sel_w + sel_l, 0.5, alternative="greater").pvalue)
            if sel_w + sel_l else None)
    n_ans = table["answer_win"] + table["answer_loss"]
    n_abs = table["abstain_win"] + table["abstain_loss"]
    return {
        "H1": {"selective_wins": sel_w, "selective_losses": sel_l, "ties": ties,
               "sign_test_one_sided_p": h1_p,
               # The registration names no alpha: the direction and p are
               # reported separately, never collapsed into "significant".
               "direction_holds": (sel_w > sel_l) if sel_w + sel_l else None},
        "H2": {"clips_answered": answered, "pipeline_beats_centre": h2_wins,
               "holds": (2 * h2_wins > len(answered)) if answered else None},
        "H3": {"answer_win_rate": (table["answer_win"] / n_ans) if n_ans else None,
               "abstain_win_rate": (table["abstain_win"] / n_abs) if n_abs else None,
               "fisher_one_sided_p": h3_p,
               "direction_holds": ((table["answer_win"] / n_ans)
                                   > (table["abstain_win"] / n_abs))
                                  if n_ans and n_abs else None},
    }


# ---------------------------------------------------------------------------
# 7. Markdown
# ---------------------------------------------------------------------------

METHOD_LABEL = {
    "pipeline": "Diagnostic trajectory fit (legacy pipeline key)",
    "matcher_only": "Matcher only",
    "pinned_only": "Anchor-pinned only",
    "graph_filter_only": "Graph filter only",
    "vpr_prior_only": "VPR prior only",
    "vpr_track_only": "VPR track only",
    "track_checked": "Fit checked against track (250 m)",
    "answer": "Answer (selected primary result)",
    "poi_only": "Experimental POI alignment",
    "assisted_policy": "Experimental assisted policy",
    "pipeline_selective": "Selective (pre-registered τ)",
    "city_centroid": "City centroid",
}


def _label(m: str) -> str:
    if m.startswith("shortlist_oracle@"):
        return f"Shortlist oracle@{m.split('@')[1]} (GT-chosen, not deployable)"
    return METHOD_LABEL.get(m, m)


def _f(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, float) and math.isinf(v):
        return "fail"
    return f"{v:.0f}"


def _cell(q: dict, with_iqr: bool = True) -> str:
    if not q or not q.get("n"):
        return "n/a"
    if not with_iqr or q["n"] == 1:
        return _f(q["median"])
    return f"{_f(q['median'])} [{_f(q['q25'])}–{_f(q['q75'])}]"


def _ci(ci: dict, pct: bool = False) -> str:
    est = ci.get("estimate")
    if est is None or (isinstance(est, float) and math.isnan(est)):
        return "n/a"
    fmt = (lambda v: f"{100 * v:.0f}%") if pct else _f
    if ci.get("low") is None:
        return fmt(est)
    return f"{fmt(est)} ({fmt(ci['low'])}–{fmt(ci['high'])})"


def render_markdown(summary: dict) -> str:
    prov = summary["provenance"]
    cfg = (summary.get("run_config") or {}).get("config") or {}
    L: list[str] = []
    L.append(f"# Benchmark run `{summary['run']}`\n")
    L.append(f"- Commit(s): {', '.join(c[:10] for c in prov['commits']) or 'n/a'}; "
             f"dirty: {', '.join(prov['dirty']) or 'n/a'}; provenance consistent: "
             f"{'yes' if prov['consistent'] else 'NO — ' + '; '.join(prov['problems'])}")
    L.append(f"- MLY_TOKEN set: {', '.join(prov['mly_token_set']) or 'n/a'}; "
             f"OCR engine: {', '.join(prov['ocr_engines'])}; "
             f"VO algo version: {', '.join(prov['vo_algo_versions'])}")
    L.append(f"- City variant: {cfg.get('city_variant') if cfg.get('city_variant') is not None else 'primary'}; "
             f"cold VPR: {cfg.get('cold_vpr')}; extra pipeline args: "
             f"{' '.join(cfg.get('pipeline_args') or []) or 'none (default config)'}")
    L.append("- Pipeline input: video + `--city` + `--vo-segment` only. Ground truth is "
             "used offline for scoring; no GT disc, GT file or length prior reached any run.")
    L.append("- **Answer** is the selected primary route (`result.answer`). The diagnostic "
             "trajectory fit (`result.position`) retains the historical `pipeline` method key "
             "and its original scores.\n")
    L.append("- Errors in metres. Per-clip cells: median [IQR] over repeats; `fail` = no "
             "answer (crash/timeout/channel absent) counted as +inf. Symmetric = mean of "
             "GT→pred (legacy 'mean route error') and pred→GT (10 m arc-length "
             "resampling). Single-point baselines are scored as degenerate routes.\n")
    execution = summary.get("execution") or {}
    if execution.get("plan_recorded"):
        L.append(f"- Execution plan: {execution['n_completed']}/{execution['expected_reps']} "
                 f"planned repeats completed; "
                 f"{'complete' if execution['complete'] else 'INCOMPLETE'}. "
                 "Missing/interrupted repeats count as failures.\n")
    else:
        L.append("- No execution plan was recorded for this legacy run; missing clip "
                 "directories or never-started repeats cannot be verified.\n")
    if prov.get("cross_fleet"):
        L.append("- Between fleets (each summarised on its own): "
                 + "; ".join(prov["cross_fleet"]))
    if summary["warnings"]:
        L.append("Warnings: " + "; ".join(summary["warnings"]) + "\n")

    for fleet, fs in summary["fleets"].items():
        ids = fs["clips"]
        clips = summary["clips"]
        methods = [m for m in ("answer", "pipeline", "pipeline_selective", "city_centroid",
                               "track_checked", "vpr_prior_only", "vpr_track_only",
                               "matcher_only", "pinned_only",
                               "graph_filter_only", "poi_only", "assisted_policy") if m in fs["methods"]]
        oracle = [m for m in fs["methods"] if m.startswith("shortlist_oracle@")]
        L.append(f"## {fleet} fleet\n")
        if fs.get("methods_not_run"):
            L.append("Not run (no clip produced an answer, so no row/column): "
                     + ", ".join(_label(m) for m in fs["methods_not_run"]) + ".\n")
        L.append("### Run health\n")
        L.append("Requested stages, actual retrieval source/backbone and failed channels are recorded. "
                 "Independent filter and POI modes do not require the ordinary trajectory fit or VPR-prior stages.\n")
        hh = ["Clip", "ok/reps", "degraded", "retrieval source", "backbone", "device",
              "OCR seed fired", "valid poses (median)"]
        L.append("| " + " | ".join(hh) + " |")
        L.append("|" + "---|" * len(hh))
        for i in ids:
            c = clips[i]
            h = c.get("health") or {}
            vp = h.get("valid_pose_frac_median")
            L.append("| " + " | ".join([
                c["name"], f"{c['n_ok']}/{c['n_reps']}",
                (f"**{h['n_degraded']}**" if h.get("n_degraded") else "0"),
                ", ".join(h.get("vpr_sources") or []) or "–",
                ", ".join(h.get("vpr_backbones") or []) or "–",
                ", ".join(h.get("vpr_devices") or []) or "–",
                str(h.get("coarse_seed_fired", 0)),
                f"{vp:.0%}" if vp is not None else "–"]) + " |")
        L.append("")
        L.append("### Per clip (m)\n")
        L.append("Primary result columns: median [IQR] over repeats. Other method columns: "
                 "symmetric mean error, median over repeats; `(k/n)` = answered in k "
                 "of n repeats.\n")
        primary = "answer" if "answer" in methods else "pipeline"
        primary_label = "Answer" if primary == "answer" else "Diagnostic fit (legacy run)"
        baselines = [m for m in methods if m != primary]
        head = ["Clip", "ok/reps", f"{primary_label}: symmetric", f"{primary_label}: start",
                f"{primary_label}: GT→pred"] + [_label(m) for m in baselines]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for i in ids:
            c = clips[i]
            ms = c["methods"]
            pipe = ms.get(primary, {})
            row = [c["name"], f"{c['n_ok']}/{c['n_reps']}",
                   _cell(pipe.get("symmetric_mean_m", {})),
                   _cell(pipe.get("start_error_m", {})),
                   _cell(pipe.get("gt_to_pred_mean_m", {}))]
            for m in baselines:
                q = ms.get(m, {}).get("symmetric_mean_m", {})
                cell = _cell(q, with_iqr=False)
                cov = ms.get(m, {})
                if m not in ("city_centroid",) and cov.get("n_reps") and \
                        cov["n_answered"] < cov["n_reps"]:
                    cell += f" ({cov['n_answered']}/{cov['n_reps']})"
                row.append(cell)
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        if oracle:
            m = oracle[0]
            L.append(f"{_label(m)} — start error (m), median over repeats: " + ", ".join(
                f"{clips[i]['name']}: {_cell(clips[i]['methods'].get(m, {}).get('start_error_m', {}), False)}"
                for i in ids) + "\n")

        from monocular_osm.metrics import THRESHOLDS_M
        for metric, title in (("start_error_m", "start error"),
                              ("symmetric_mean_m", "symmetric mean error")):
            L.append(f"### Success rate: {title} ≤ threshold "
                     f"(mean over clips, 95% bootstrap CI over clips)\n")
            head = ["Method"] + [f"≤{t} m" for t in THRESHOLDS_M]
            L.append("| " + " | ".join(head) + " |")
            L.append("|" + "---|" * len(head))
            # The oracle's hypotheses are START points: only start error is
            # meaningful for it.
            for m in methods + (oracle if metric == "start_error_m" else []):
                sr = fs["methods"][m]["success_rate"][metric]
                L.append("| " + " | ".join(
                    [_label(m)] + [_ci(sr[str(t)], pct=True) for t in THRESHOLDS_M]) + " |")
            L.append("")

        L.append(f"### Fleet summary ({len(ids)} clips; 95% percentile-bootstrap CI over clips)\n")
        head = ["Method", "Median of clip medians: symmetric", "Mean of clip medians: symmetric",
                "Median of clip medians: start", "Diagnostic fit wins/losses vs this (symmetric)",
                "sign-test p"]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for m in methods + oracle:
            ms = fs["methods"][m]
            is_oracle = m in oracle
            metric = "start_error_m" if is_oracle else "symmetric_mean_m"
            pv = fs["paired_vs_pipeline"].get(m, {}).get(metric)
            wl = f"{pv['pipeline_wins']}/{pv['pipeline_losses']}" if pv else "—"
            if pv and is_oracle:
                wl += " (start)"
            p = (f"{pv['sign_test_p']:.2f}" if pv and pv["sign_test_p"] is not None
                 else "—")
            L.append("| " + " | ".join([
                _label(m),
                "—" if is_oracle else _ci(ms["symmetric_mean_m"]["median_of_clip_medians"]),
                "—" if is_oracle else _ci(ms["symmetric_mean_m"]["mean_of_clip_medians"]),
                _ci(ms["start_error_m"]["median_of_clip_medians"]),
                wl, p]) + " |")
        L.append("")
        sel = fs.get("selective")
        if sel:
            t = sel["table"]
            cov = sel["coverage"]
            L.append(f"### Selective prediction (pre-registered τ = {sel['tau']})\n")
            L.append("The historical rule selects the diagnostic trajectory fit only when "
                     "its retrieval confidence ≥ τ and "
                     "otherwise answers the city centre (rule and τ fixed in "
                     "`benchmarks/preregistered_rules.json` before the held-out clips "
                     "were run). Win = the pipeline's symmetric error is below the "
                     "city-centre guess's.\n")
            if sel.get("in_sample_clips"):
                L.append(f"**In-sample:** τ was chosen on {len(sel['in_sample_clips'])} of "
                         f"these clips ({', '.join(sel['in_sample_clips'])}), so for them "
                         f"this is the rule's fit, not evidence that it works.\n")
            L.append("| Clip | answered / reps | confidence (per rep) |")
            L.append("|---|---|---|")
            for i in ids:
                cs = clips[i].get("selective") or {}
                confs = ", ".join("none" if v is None else f"{v:.3f}"
                                  for v in cs.get("confidence", []))
                L.append(f"| {clips[i]['name']} | {cs.get('n_answered', 0)}/"
                         f"{cs.get('n_reps_ok', 0)} | {confs or '–'} |")
            L.append("")
            p3 = sel["h3_fisher_one_sided_p"]
            cov_s = "n/a" if cov is None else f"{100 * cov:.0f}%"
            p3_s = "n/a" if p3 is None else ("<0.001" if p3 < 0.001 else f"{p3:.3f}")
            L.append(f"Coverage {cov_s} of {sel['n_reps']} reps. Answered: "
                     f"{t['answer_win']} win / {t['answer_loss']} loss; abstained: "
                     f"{t['abstain_win']} win / {t['abstain_loss']} loss. One-sided "
                     f"Fisher exact p (answered reps win more often): {p3_s} (reps of "
                     f"one clip are not independent; read it per clip too).\n")
            hy = sel.get("hypotheses")
            if hy:
                def _v(x):
                    return "untestable" if x is None else ("yes" if x else "no")

                def _p(x):
                    return "n/a" if x is None else ("<0.001" if x < 0.001 else f"{x:.3f}")
                h1, h2, h3 = hy["H1"], hy["H2"], hy["H3"]
                L.append("Registered hypotheses (text in the rule file; no significance "
                         "level was registered, so direction and p are given separately):\n")
                L.append(f"- **H1** selective beats the pipeline on more clips than it loses: "
                         f"{h1['selective_wins']} vs {h1['selective_losses']} "
                         f"({h1['ties']} ties); direction {_v(h1['direction_holds'])}, "
                         f"one-sided sign test p {_p(h1['sign_test_one_sided_p'])}.")
                L.append(f"- **H2** where it answers, the pipeline beats the centre on a "
                         f"majority: {h2['pipeline_beats_centre']} of "
                         f"{len(h2['clips_answered'])} answered clips; {_v(h2['holds'])}.")
                L.append(f"- **H3** answered reps beat the centre more often than abstained "
                         f"ones: direction {_v(h3['direction_holds'])}, one-sided Fisher p "
                         f"{_p(h3['fisher_one_sided_p'])}.\n")
    return "\n".join(L) + "\n"


def _jsonable(o):
    if isinstance(o, float) and not math.isfinite(o):
        return "inf" if o > 0 else ("-inf" if o < 0 else "nan")
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return _jsonable(float(o))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def cmd_aggregate(args) -> int:
    run_dir = (Path(args.runs_dir) if args.runs_dir else ROOT / RUNS_DIR) / args.run_name
    if not run_dir.is_dir():
        raise SystemExit(f"no run directory {run_dir}")
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path if manifest_path.is_absolute()
                             else ROOT / manifest_path)
    summary = aggregate(run_dir, manifest, allow_mixed=args.allow_mixed,
                        allow_network=not args.offline, k=args.k,
                        n_resamples=args.n_resamples, seed=args.seed, root=ROOT)
    (run_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(summary)
    (run_dir / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {run_dir / 'summary.json'} and {run_dir / 'summary.md'}")
    return 0


def compare_runs(base: dict, new: dict, *, metric: str = "symmetric_mean_m",
                 fleet: str | None = None, method: str = "pipeline",
                 allow_config_changes: bool = False) -> dict:
    """Paired per-clip comparison of two aggregated runs (``aggregate()`` output).

    A clip is compared only when both runs scored it under the same city,
    analysis window, ground truth and run configuration (cold caches, extra
    pipeline flags); anything else is listed as excluded with the reason. Per
    clip the statistic is the median over reps; across clips a two-sided sign
    test (failures included). Log-ratios, their Wilcoxon signed-rank test and
    bootstrap CI use only pairs where both runs answered. Identical package
    trees are flagged, without assuming dependencies or references match."""
    from scipy.stats import binomtest, wilcoxon

    def cfg(summary):
        c = dict((summary.get("run_config") or {}).get("config") or {})
        return {k: c.get(k) for k in ("cold_vpr", "pipeline_args", "city_variant",
                                      "skip_download")}

    excluded, rows = [], []
    cfg_b, cfg_n = cfg(base), cfg(new)
    for cid in sorted(set(base["clips"]) | set(new["clips"])):
        b, n = base["clips"].get(cid), new["clips"].get(cid)
        if fleet and ((b or n)["fleet"] != fleet):
            continue
        why = None
        if b is None or n is None:
            why = f"only in the {'new' if b is None else 'base'} run"
        elif b["city"] != n["city"]:
            why = f"city differs ({b['city']!r} vs {n['city']!r})"
        elif b.get("gt_n_waypoints") != n.get("gt_n_waypoints"):
            why = "ground truth differs"
        elif b.get("gt_waypoints_sha256") != n.get("gt_waypoints_sha256"):
            why = "ground-truth waypoint fingerprint differs or is missing"
        elif (b.get("vo_segments") or None) != (n.get("vo_segments") or None):
            why = f"analysis window differs ({b.get('vo_segments')} vs {n.get('vo_segments')})"
        elif cfg_b != cfg_n and not allow_config_changes:
            why = f"run configuration differs ({cfg_b} vs {cfg_n})"
        if why is None:
            qb = ((b["methods"].get(method) or {}).get(metric) or {})
            qn = ((n["methods"].get(method) or {}).get(metric) or {})
            mb, mn = qb.get("median"), qn.get("median")
            try:
                mb, mn = float(mb), float(mn)
            except (TypeError, ValueError):
                why = "no score in one of the runs"
            else:
                if np.isnan(mb) or np.isnan(mn) or mb < 0 or mn < 0:
                    why = "invalid error score in one of the runs"
        if why:
            excluded.append({"clip": cid, "reason": why})
            continue
        cc = ((b["methods"].get("city_centroid") or {}).get(metric) or {}).get("median")
        finite_pair = np.isfinite(mb) and np.isfinite(mn)
        rows.append({"clip": cid, "name": b["name"], "base": mb, "new": mn,
                     "delta": 0.0 if mb == mn else mn - mb,
                     "log_ratio": (float(np.log(max(mn, 1e-3) / max(mb, 1e-3)))
                                   if finite_pair else None),
                     "base_failed": not np.isfinite(mb), "new_failed": not np.isfinite(mn),
                     "base_iqr": (qb.get("q25"), qb.get("q75")),
                     "new_iqr": (qn.get("q25"), qn.get("q75")),
                     "centre": cc,
                     "base_beats_centre": None if cc is None else mb < cc,
                     "new_beats_centre": None if cc is None else mn < cc})
    better = sum(1 for r in rows if r["new"] < r["base"])
    worse = sum(1 for r in rows if r["new"] > r["base"])
    lr = np.array([r["log_ratio"] for r in rows if r["log_ratio"] is not None], float)
    nz = lr[np.abs(lr) > 1e-12]
    out = {
        "metric": metric, "method": method, "fleet": fleet, "n_clips": len(rows),
        "allow_config_changes": allow_config_changes,
        "rows": rows, "excluded": excluded, "new_better": better, "new_worse": worse,
        "ties": len(rows) - better - worse,
        "n_finite_pairs": len(lr),
        "base_failures": sum(r["base_failed"] for r in rows),
        "new_failures": sum(r["new_failed"] for r in rows),
        "sign_test_p": float(binomtest(better, better + worse, 0.5).pvalue)
        if better + worse else None,
        "wilcoxon_p": float(wilcoxon(nz).pvalue) if len(nz) >= 6 else None,
        "median_ratio": (bootstrap_ci(np.exp(lr), "median", n_resamples=9999)
                         if len(lr) else None),
        "base_trees": base["provenance"].get("package_trees"),
        "new_trees": new["provenance"].get("package_trees"),
        "devices": [base["provenance"].get("retrieval_devices"),
                    new["provenance"].get("retrieval_devices")],
    }
    notes = []
    if cfg_b != cfg_n and allow_config_changes:
        notes.append(f"runner configuration differs by explicit request: base={cfg_b}, new={cfg_n}")
    if any(not base["clips"][r["clip"]].get("gt_waypoints_sha256") for r in rows):
        notes.append("legacy summaries have no ground-truth fingerprint; waypoint equality "
                     "cannot be verified for those pairs")
    if out["base_trees"] and out["base_trees"] == out["new_trees"]:
        notes.append("both runs ran the SAME pipeline code; differences may come from "
                     "configuration, dependencies, reference snapshots or run-to-run variation")
    if len(lr) != len(rows):
        notes.append(f"failures count in the sign test; log-ratio statistics use only "
                     f"the {len(lr)} pairs with finite scores in both runs")
    if out["devices"][0] and out["devices"][1] and out["devices"][0] != out["devices"][1]:
        notes.append(f"retrieval devices differ {out['devices']} (GPU fp16 vs CPU fp32)")
    for key, label in (("method_config_fingerprints", "effective method configurations"),
                       ("environment_fingerprints", "Python/library versions"),
                       ("vpr_model_fingerprints", "VPR model fingerprints")):
        b_sig, n_sig = base["provenance"].get(key), new["provenance"].get(key)
        if b_sig and n_sig and b_sig != n_sig:
            notes.append(f"runs used different {label}")
    if len(rows) < 6:
        notes.append(f"only {len(rows)} paired clips: no test can detect less than a "
                     f"near-unanimous change")
    out["notes"] = notes
    return out


def render_compare(cmp: dict, base_name: str, new_name: str) -> str:
    L = [f"# `{new_name}` vs `{base_name}` ({cmp['method']}, {cmp['metric']}, "
         f"fleet {cmp['fleet'] or 'all'})\n",
         "Per clip: median over repeats [IQR]. Compared only where city, window, "
         "and ground truth match; runner configuration must match unless "
         "--allow-config-changes was explicitly requested.\n",
         "| Clip | base | new | change | beats centre (base → new) |", "|---|---|---|---|---|"]
    for r in cmp["rows"]:
        bi, ni = r["base_iqr"], r["new_iqr"]
        if r["log_ratio"] is not None:
            change = f"{100 * (np.exp(r['log_ratio']) - 1):+.0f} %"
        elif r["base_failed"] and r["new_failed"]:
            change = "both failed"
        elif r["new_failed"]:
            change = "regressed to failure"
        else:
            change = "recovered from failure"
        L.append(f"| {r['name']} | {_f(r['base'])} [{_f(bi[0])}–{_f(bi[1])}] | "
                 f"{_f(r['new'])} [{_f(ni[0])}–{_f(ni[1])}] | "
                 f"{change} | "
                 f"{r['base_beats_centre']} → {r['new_beats_centre']} |")
    mr = cmp["median_ratio"]
    sp = cmp["sign_test_p"]
    wp = cmp["wilcoxon_p"]
    L += ["", f"New better on {cmp['new_better']} clips, worse on {cmp['new_worse']} "
          f"(sign test p = {'n/a' if sp is None else f'{sp:.3f}'}; Wilcoxon on log-ratios "
          f"p = {'n/a' if wp is None else f'{wp:.3f}'}). Median new/base ratio "
          + ("n/a" if not mr else _ci(mr)) + " (95% bootstrap CI over clips).",
          f"Failures: base {cmp['base_failures']}/{cmp['n_clips']}, "
          f"new {cmp['new_failures']}/{cmp['n_clips']}; "
          f"finite pairs for log-ratio statistics: {cmp['n_finite_pairs']}."]
    if cmp["excluded"]:
        L.append("\nNot compared: " + "; ".join(f"{e['clip']} ({e['reason']})"
                                                 for e in cmp["excluded"]) + ".")
    for note in cmp["notes"]:
        L.append(f"\n**Note:** {note}.")
    return "\n".join(L) + "\n"


def cmd_compare(args) -> int:
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path if manifest_path.is_absolute()
                             else ROOT / manifest_path)
    runs = []
    for name, d in ((args.base, args.base_runs_dir), (args.new, args.new_runs_dir)):
        run_dir = (Path(d) if d else ROOT / RUNS_DIR) / name
        if not run_dir.is_dir():
            raise SystemExit(f"no run directory {run_dir}")
        runs.append(aggregate(run_dir, manifest, allow_mixed=args.allow_mixed,
                              allow_network=not args.offline, n_resamples=999, root=ROOT))
    cmp = compare_runs(runs[0], runs[1], metric=args.metric, fleet=args.fleet,
                       method=args.method, allow_config_changes=args.allow_config_changes)
    md = render_compare(cmp, args.base, args.new)
    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(
            json.dumps(_jsonable(cmp), indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------
# 8. manifest / geocode-snapshot commands
# ---------------------------------------------------------------------------


def cmd_manifest(args) -> int:
    if args.check:
        if not check_manifest(ROOT):
            print(f"{MANIFEST_PATH.as_posix()} is stale; run: "
                  f"python scripts/benchmark.py manifest")
            return 1
        print(f"{MANIFEST_PATH.as_posix()} is up to date")
        return 0
    print(f"wrote {write_manifest(ROOT).relative_to(ROOT).as_posix()}")
    return 0


def cmd_geocode_snapshot(args) -> int:
    print(f"wrote {geocode_snapshot(ROOT).relative_to(ROOT).as_posix()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("manifest", help="regenerate benchmarks/manifest.json")
    m.add_argument("--check", action="store_true",
                   help="exit 1 if the committed manifest is stale")
    m.set_defaults(fn=cmd_manifest)

    g = sub.add_parser("geocode-snapshot",
                       help="geocode manifest city strings missing from the snapshots")
    g.set_defaults(fn=cmd_geocode_snapshot)

    r = sub.add_parser("run", help="run the pipeline on manifest clips")
    r.add_argument("--run-name", required=True, type=_run_name)
    r.add_argument("--manifest", default=MANIFEST_PATH.as_posix())
    r.add_argument("--fleet", choices=("core", "overlay", "all"), default="core")
    r.add_argument("--clips", nargs="+", default=None, metavar="ID")
    r.add_argument("--split", choices=("dev", "test", "test2", "test3", "test4", "all"), default="dev")
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--city-variant", type=int, default=None, metavar="I",
                   help="use city_alternatives[I] instead of the manifest city "
                        "(clips without one are skipped)")
    r.add_argument("--cold-vpr", action="store_true",
                   help="move each clip's warm VPR reference caches aside for every "
                        "rep (restored afterwards; the fresh caches are archived "
                        "under data/<slug>/_benchmark_cold_vpr/)")
    r.add_argument("--cold-archive", choices=("metadata", "full"), default="metadata",
                   help="what to keep of each cold rep's fresh caches: the reference "
                        "id/position lists only (default, ~40 MB per rep) or also the "
                        "downloaded pixels and embeddings (~1.2 GB per rep)")
    r.add_argument("--timeout", type=float, default=3600.0, help="seconds per run")
    r.add_argument("--retry-failed", action="store_true",
                   help="re-run reps whose rc != 0 (completed reps are always skipped)")
    # Operational, NOT part of run_config (they do not change any result):
    r.add_argument("--infra-retries", type=int, default=2,
                   help="re-run a rep up to N times when it failed because the machine "
                        "or network did (killed externally, allocation failure, network "
                        "timeout); genuine pipeline failures are never retried")
    r.add_argument("--infra-wait", type=float, default=90.0,
                   help="seconds to wait before an infrastructure retry")
    r.add_argument("--min-free-commit-gb", type=float, default=16.0,
                   help="before each attempt, wait until this much memory can still be "
                        "committed system-wide (0 = don't check); a pipeline child "
                        "needs ~9-12 GB for a 5-minute 720p window")
    r.add_argument("--min-free-vram-gb", type=float, default=8.0,
                   help="before each attempt, wait until the GPU has this much free "
                        "memory (0 = don't check): the retrieval model dies natively "
                        "when another process leaves the GPU nearly full")
    r.add_argument("--min-free-disk-gb", type=float, default=10.0,
                   help="refuse to start a rep with less free disk than this on the data "
                        "or output drive (0 = don't check); a full disk corrupts caches")
    r.add_argument("--resource-wait", type=float, default=10800.0,
                   help="max seconds to wait for --min-free-commit-gb / "
                        "--min-free-vram-gb per attempt")
    r.add_argument("--overpass-wait", type=float, default=7200.0,
                   help="before each attempt, wait up to this many seconds for "
                        "overpass-api.de to accept connections (0 = don't check)")
    r.add_argument("--include-unavailable", action="store_true",
                   help="also try clips marked available_locally=false")
    r.add_argument("--skip-download", action="store_true",
                   help="pass --skip-download (offline re-run on cached video+metadata)")
    r.add_argument("--pipeline-arg", action="append", default=[], metavar="ARG",
                   help="extra pipeline flag for a CONFIG ablation, e.g. "
                        "--pipeline-arg=--enable-ocr-anchor (GT-derived and "
                        "runner-owned flags are rejected); recorded in run_config")
    r.add_argument("--data-dir", default=DEFAULT_DATA_DIR.as_posix())
    r.add_argument("--dry-run", action="store_true", help="print argv, run nothing")
    r.set_defaults(fn=cmd_run)

    a = sub.add_parser("aggregate", help="score a run directory offline")
    a.add_argument("--run-name", required=True, type=_run_name)
    a.add_argument("--manifest", default=MANIFEST_PATH.as_posix())
    a.add_argument("--runs-dir", default=None,
                   help=f"default: {RUNS_DIR.as_posix()}")
    a.add_argument("--allow-mixed", action="store_true",
                   help="aggregate despite mixed commits/dirty tree/token states")
    a.add_argument("--offline", action="store_true",
                   help="never geocode over the network (snapshot only)")
    a.add_argument("--k", type=int, default=5, help="shortlist size for the oracle")
    a.add_argument("--n-resamples", type=int, default=9999)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(fn=cmd_aggregate)

    c = sub.add_parser("compare", help="paired comparison of two runs on matching clips")
    c.add_argument("--base", required=True, type=_run_name)
    c.add_argument("--new", required=True, type=_run_name)
    c.add_argument("--base-runs-dir", default=None)
    c.add_argument("--new-runs-dir", default=None)
    c.add_argument("--manifest", default=MANIFEST_PATH.as_posix())
    c.add_argument("--fleet", default=None, help="core / overlay (default: all)")
    c.add_argument("--metric", default="symmetric_mean_m")
    c.add_argument("--method", default="pipeline", help="method to compare (e.g. answer)")
    c.add_argument("--allow-config-changes", action="store_true",
                   help="compare intentional configuration ablations; city/window/GT must still match")
    c.add_argument("--allow-mixed", action="store_true")
    c.add_argument("--offline", action="store_true")
    c.add_argument("--out", default=None, help="write the markdown (and .json) here")
    c.set_defaults(fn=cmd_compare)
    return p


def _run_name(s: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", s) or s in (".", ".."):
        raise argparse.ArgumentTypeError("run name: letters, digits, '.', '_', '-' only")
    return s


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
