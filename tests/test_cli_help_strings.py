"""argparse help strings must survive being formatted.

argparse runs every help string through ``%``-formatting, so a bare percent
sign raises at ``--help`` time — not at import, not in any test that only
constructs the parser. A help string reading "recovers ~61% of that" makes
``%`` + `` o`` look like the ``%o`` conversion and takes down --help for
every user of the tool.

This reads cli.py as source rather than importing it, so it still guards the
CLI in environments missing the optional heavy dependencies (open3d and
friends) that ``import monocular_osm.cli`` pulls in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CLI = Path(__file__).resolve().parents[1] / "monocular_osm" / "cli.py"


def _help_strings() -> list[tuple[int, str]]:
    tree = ast.parse(CLI.read_text(encoding="utf-8"))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                continue          # a computed help string; not our concern
            if isinstance(value, str):
                out.append((node.lineno, value))
    return out


def test_there_are_help_strings_to_check() -> None:
    # Guards the guard: an ast walk that silently finds nothing would make
    # every assertion below vacuously true.
    assert len(_help_strings()) > 50


@pytest.mark.parametrize(("lineno", "text"), _help_strings(),
                         ids=lambda v: str(v)[:40])
def test_help_string_is_percent_format_safe(lineno, text) -> None:
    try:
        text % {}
    except (TypeError, ValueError, KeyError) as e:
        pytest.fail(
            f"monocular_osm/cli.py:{lineno} help string breaks --help "
            f"({type(e).__name__}: {e}). Write a literal percent as '%%', or "
            f"reword it.\n  {text[:160]}")
