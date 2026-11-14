"""Offline capability checks; imports libraries but never loads models or URLs.

Independent of the pipeline so it can diagnose broken geometry imports too.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import sys
from typing import Callable, Mapping, Sequence


# Import name -> distribution name. The VPR group matches the pinned MegaLoc
# hubconf dependencies without invoking torch.hub or fetching model weights.
DEPENDENCIES = {
    "core": {"numpy": "numpy", "cv2": "opencv-python", "scipy": "scipy",
             "skimage": "scikit-image", "osmnx": "osmnx", "networkx": "networkx",
             "shapely": "shapely", "pyproj": "pyproj", "matplotlib": "matplotlib",
             "geopandas": "geopandas", "contextily": "contextily", "open3d": "open3d",
             "plotly": "plotly", "yt_dlp": "yt-dlp", "requests": "requests",
             "tenacity": "tenacity", "rapidfuzz": "rapidfuzz", "tqdm": "tqdm",
             "mercantile": "mercantile", "google.protobuf": "protobuf",
             "pysolar": "pysolar", "timezonefinder": "timezonefinder"},
    "vpr": {"torch": "torch", "torchvision": "torchvision",
            "huggingface_hub": "huggingface-hub", "safetensors": "safetensors",
            "PIL": "pillow", "requests": "requests"},
    "ocr": {"rapidocr": "rapidocr", "onnxruntime": "onnxruntime"},
    "plates": {"fast_alpr": "fast-alpr"},
    "graph-filter": {"filterpy": "filterpy"},
}


def _safe_error(exc: Exception, environ: Mapping[str, str]) -> str:
    text = f"{type(exc).__name__}: {exc}"
    for key, value in environ.items():
        if value and any(part in key.upper() for part in ("TOKEN", "KEY", "SECRET", "PASSWORD")):
            text = text.replace(value, "[redacted]")
    return text[:240]


def collect_diagnostics(*, required: Sequence[str] = ("core", "vpr"),
                        source: str = "mapillary", environ: Mapping[str, str] | None = None,
                        importer: Callable | None = None, finder: Callable | None = None,
                        versioner: Callable | None = None) -> dict:
    """Check imports and fresh-source credentials without model inference.

    ``ready`` only covers the local installation and requested credentials;
    model files, token validity and remote service availability are not tested.
    """
    if source not in {"mapillary", "kartaview", "panoramax", "union"}:
        raise ValueError(f"unknown reference source: {source}")
    requested = set(required)
    if "recommended" in requested:
        requested.remove("recommended")
        requested.update(DEPENDENCIES)
    unknown = requested.difference(DEPENDENCIES)
    if unknown:
        raise ValueError(f"unknown capabilities: {', '.join(sorted(unknown))}")
    env = os.environ if environ is None else environ
    importer = importer or importlib.import_module
    finder = finder or importlib.util.find_spec
    versioner = versioner or importlib.metadata.version
    checks, modules = {}, {}
    for deps in DEPENDENCIES.values():
        for module, distribution in deps.items():
            if module in checks:
                continue
            probe = {"distribution": distribution, "status": "missing", "version": None}
            try:
                # Keep import banners out of JSON and suppress accidental
                # credential-bearing output from broken third-party modules.
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    spec = finder(module)
                    if spec is not None:
                        modules[module] = importer(module)
                        probe["status"] = "ok"
            except ModuleNotFoundError as exc:
                missing = exc.name and (module == exc.name or module.startswith(f"{exc.name}."))
                probe.update(status="missing" if missing else "failed",
                             error=_safe_error(exc, env))
            except Exception as exc:
                probe.update(status="failed", error=_safe_error(exc, env))
            if probe["status"] == "ok":
                try:
                    probe["version"] = str(versioner(distribution))
                except importlib.metadata.PackageNotFoundError:
                    pass
            checks[module] = probe

    token_set = bool(env.get("MLY_TOKEN", "").strip())
    needs_token = source in {"mapillary", "union"}
    capabilities = {}
    for name, deps in DEPENDENCIES.items():
        problems = [f"{module}: {checks[module]['status']}" for module in deps
                    if checks[module]["status"] != "ok"]
        if name == "vpr" and needs_token and not token_set:
            problems.append("MLY_TOKEN missing for fresh Mapillary fetches")
        capabilities[name] = {"required": name in requested,
                              "ready": not problems, "problems": problems}

    cuda_available = None
    torch = modules.get("torch")
    if torch is not None:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cuda_available = bool(torch.cuda.is_available())
        except Exception:
            pass
    return {
        "python": sys.version.split()[0], "executable": sys.executable,
        "reference_source": source, "checks": checks, "capabilities": capabilities,
        "credentials": {"MLY_TOKEN": "set (not validated)" if token_set else "missing"},
        "cuda_available": cuda_available, "model_weights": "not checked; no models loaded",
        "remote_services": "not checked; no network requests",
        "ready": all(v["ready"] for v in capabilities.values() if v["required"]),
    }


def format_diagnostics(report: dict) -> str:
    lines = [f"Python {report['python']}: {report['executable']}",
             f"Reference source: {report['reference_source']}"]
    for name, cap in report["capabilities"].items():
        suffix = "required" if cap["required"] else "optional"
        lines.append(f"{name}: {'ready' if cap['ready'] else 'unavailable'} ({suffix})")
        lines.extend(f"  - {problem}" for problem in cap["problems"])
    for module, check in report["checks"].items():
        if check.get("error"):
            lines.append(f"  {module}: {check['error']}")
    lines.append(f"MLY_TOKEN: {report['credentials']['MLY_TOKEN']}")
    lines.append(f"CUDA available: {report['cuda_available']} (CPU retrieval is supported)")
    lines.append("Model weights and remote services were not checked; no models loaded or downloads made.")
    lines.append("Without MLY_TOKEN, warm Mapillary caches or tokenless sources may still work.")
    if not report["ready"]:
        lines.append('Install accuracy extras with: python -m pip install -e ".[recommended]"')
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if a requested capability or fresh-source credential is missing.")
    parser.add_argument("--require", nargs="+", choices=[*DEPENDENCIES, "recommended"],
                        default=["core", "vpr"], metavar="CAPABILITY",
                        help="Required: core, vpr, ocr, plates, graph-filter, recommended (default: core vpr).")
    parser.add_argument("--source", choices=["mapillary", "kartaview", "panoramax", "union"],
                        default="mapillary", help="Reference source for credential checks.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable diagnostics.")
    args = parser.parse_args(argv)
    report = collect_diagnostics(required=args.require, source=args.source)
    print(json.dumps(report, indent=2) if args.json else format_diagnostics(report))
    return 1 if args.strict and not report["ready"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
