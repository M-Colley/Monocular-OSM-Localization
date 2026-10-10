"""Run provenance: everything needed to say which code and settings produced
a result.

Every published number must be traceable to a commit, a full configuration
and the library versions that computed it. Before this module, result.json
carried none of these, so two sweeps of "the same" clip disagreeing by
hundreds of metres could not be attributed to a code change, a config change
or run-to-run variance.

Secrets are never recorded: for credentials such as ``MLY_TOKEN`` only
whether they were set is stored.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import importlib.metadata as _md
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# Distributions whose versions can change results. Missing ones are recorded
# as None (an optional channel that was not installed is itself provenance).
TRACKED_DISTRIBUTIONS = (
    "numpy", "scipy", "opencv-python", "opencv-python-headless",
    "scikit-image", "shapely", "pyproj", "networkx", "osmnx", "geopandas",
    "torch", "torchvision", "transformers", "timm", "onnxruntime", "rapidocr",
    "easyocr", "yt-dlp", "pysolar", "rapidfuzz",
)

# Environment variables holding credentials: recorded as set/unset only.
SECRET_ENV_VARS = ("MLY_TOKEN",)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True,
            timeout=10, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> dict[str, Any]:
    """Commit, branch and whether tracked files differ from that commit.

    ``dirty`` ignores untracked files: data/ and output/ are untracked by
    design, while an edited tracked file means the commit alone does not
    reproduce the run.
    """
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def library_versions(names=TRACKED_DISTRIBUTIONS) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = _md.version(name)
        except _md.PackageNotFoundError:
            versions[name] = None
    return versions


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def config_snapshot(cfg: Any) -> dict[str, Any]:
    """The full configuration as plain JSON values."""
    if dataclasses.is_dataclass(cfg):
        return _jsonable(dataclasses.asdict(cfg))
    return _jsonable(vars(cfg))


def collect_provenance(cfg: Any = None, *, extra: dict | None = None) -> dict[str, Any]:
    """Provenance block for a result file."""
    from . import __version__

    prov: dict[str, Any] = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "package_version": __version__,
        "git": git_state(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "argv": list(sys.argv),
        "libraries": library_versions(),
        "env": {f"{name.lower()}_set": bool(os.environ.get(name))
                for name in SECRET_ENV_VARS},
    }
    try:
        import torch
        prov["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        prov["cuda_available"] = None
    if cfg is not None:
        prov["config"] = config_snapshot(cfg)
    if extra:
        prov.update(_jsonable(extra))
    return prov
