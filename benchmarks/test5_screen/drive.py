"""Run the test5 screen and builds in the registered order until 10 clips are available.

    python benchmarks/test5_screen/drive.py

Screen until a pass, build it, repeat. Bot checks and unread frames wait
30 minutes and retry the same step unchanged. Stops at the target or when
the fixed pool is exhausted. Never localizes or scores.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAUSE_S = 1800


def attempts() -> list[dict]:
    d = HERE / "build_attempts"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []


def unbuilt_passes() -> list[str]:
    rows = json.loads((HERE / "screen_results.json").read_text(encoding="utf-8")) if (HERE / "screen_results.json").exists() else []
    final = {a["id"] for a in attempts() if a["status"] != "UNVERIFIED_INFRASTRUCTURE"}
    out = []
    for r in rows:
        if r.get("ok") and r["id"] not in final and r["id"] not in out:
            out.append(r["id"])
    return out


def run(args) -> int:
    print("$", *args, flush=True)
    return subprocess.run([sys.executable, *args]).returncode


def main() -> int:
    target = json.loads((HERE / "selection_preregistration.json").read_text(encoding="utf-8"))["target"]
    while True:
        available = sum(a["status"] == "AVAILABLE_GT_QC_PASSED" for a in attempts())
        print(f"available {available}/{target}", flush=True)
        if available >= target:
            return 0
        pending = unbuilt_passes()
        if pending:
            rc = run([str(HERE / "build_one.py"), pending[0]])
            if rc == 75:
                time.sleep(PAUSE_S)
            continue
        rc = run([str(HERE / "screen.py")])
        if rc == 2:
            print("pool exhausted", flush=True)
            return 0
        if rc not in (0, 1):
            time.sleep(PAUSE_S)


if __name__ == "__main__":
    raise SystemExit(main())
