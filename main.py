"""Root CLI shim — the real logic lives in :mod:`monocular_osm.cli`.

Kept at the repo root so ``python main.py ...`` still works from a source
checkout (as the README shows throughout) and so ``import main`` in the test
suite resolves to the *same module object* the installed ``osm-localize``
console script runs — the tests monkeypatch collaborators on this module
(``setattr(main, "run_pipeline", ...)``) and expect ``main.main()`` to pick
them up.

Installed users should prefer the ``osm-localize`` console script (declared in
pyproject.toml) or ``python -m monocular_osm.cli``.
"""
from __future__ import annotations

import sys

# Match the pre-rename behaviour: force UTF-8, line-buffered stdio *before*
# importing the (heavy) pipeline modules, so a cp1252 console can't crash on
# umlauts printed during import. (monocular_osm.cli.main() repeats this for the
# console-script / -m entry points; reconfiguring twice is harmless.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

from monocular_osm import cli as _cli

if __name__ == "__main__":
    _cli.main()
else:
    # Alias this module to monocular_osm.cli so `import main` and any
    # `monkeypatch.setattr(main, name, ...)` in the tests operate on the exact
    # namespace cli.main() reads its collaborators from.
    sys.modules[__name__] = _cli
