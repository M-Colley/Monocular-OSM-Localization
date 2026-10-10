"""Every PipelineConfig field must be reachable from the CLI.

The CLI maps ~90 flags onto PipelineConfig by hand. Fields that were never
mapped (the scale-recovery thresholds were one) can only be changed by
editing code, so any experiment that varies them is irreproducible from the
command line, and the published configuration cannot be fully stated as a
command. Read as source so the check runs without the optional heavy deps.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Fields deliberately NOT exposed as flags, with the reason. Keep this short.
INTERNAL_FIELDS: dict[str, str] = {}


def _config_fields() -> list[str]:
    tree = ast.parse((ROOT / "monocular_osm" / "pipeline.py").read_text(encoding="utf-8"))
    cfg = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "PipelineConfig")
    return [x.target.id for x in cfg.body if isinstance(x, ast.AnnAssign)]


def _fields_set_by_cli() -> set[str]:
    src = (ROOT / "monocular_osm" / "cli.py").read_text(encoding="utf-8")
    call = src[src.index("cfg = PipelineConfig("):]
    call = call[: call.index("\n        )\n")]
    return set(re.findall(r"^\s+(\w+)=", call, re.M))


def test_every_config_field_is_set_by_the_cli() -> None:
    missing = [f for f in _config_fields()
               if f not in _fields_set_by_cli() and f not in INTERNAL_FIELDS]
    assert not missing, (
        f"PipelineConfig fields with no CLI mapping: {missing}. Add a flag in "
        f"monocular_osm/cli.py or list them in INTERNAL_FIELDS with a reason.")


def test_cli_sets_no_unknown_config_fields() -> None:
    unknown = _fields_set_by_cli() - set(_config_fields())
    assert not unknown, f"cli.py passes fields PipelineConfig does not have: {unknown}"


def test_cli_and_config_agree_on_top_k_default() -> None:
    """The CLI defaulted to 10 while the config class defaulted to 5, so the
    Python API silently ran a different pipeline from the command line."""
    tree = ast.parse((ROOT / "monocular_osm" / "pipeline.py").read_text(encoding="utf-8"))
    cfg = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "PipelineConfig")
    top_k = next(x for x in cfg.body
                 if isinstance(x, ast.AnnAssign) and x.target.id == "top_k")
    cli = (ROOT / "monocular_osm" / "cli.py").read_text(encoding="utf-8")
    m = re.search(r'"--top-k", type=int, default=(\d+)', cli)
    assert m and int(m.group(1)) == ast.literal_eval(top_k.value)
