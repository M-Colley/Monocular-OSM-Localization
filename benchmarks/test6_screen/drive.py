"""Drive test6 worldwide screening & GT building until 6-continent quotas (12 clips) are met or pool ends.

Never executes localization or scoring. Stops cleanly on anti-bot / rate-limit blocks rather than bypassing.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAUSE_S = 15
MAX_INFRA_STREAK = 20


def attempts() -> list[dict]:
    d = HERE / "build_attempts"
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(d.glob("*.json"))] if d.exists() else []


def unbuilt_passes() -> list[str]:
    res_path = HERE / "screen_results.json"
    rows = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else []
    final = {a["id"] for a in attempts() if a["status"] != "UNVERIFIED_INFRASTRUCTURE"}
    out = []
    for r in rows:
        if r.get("ok") and r["id"] not in final and r["id"] not in out:
            out.append(r["id"])
    return out


def last_row_was_bot_check() -> bool:
    res_path = HERE / "screen_results.json"
    if res_path.exists():
        rows = json.loads(res_path.read_text(encoding="utf-8"))
        if rows and rows[-1].get("bot_check"):
            return True
    atts = attempts()
    return bool(atts and atts[-1].get("bot_check"))


def run(args: list[str], timeout_s: int = 900) -> int:
    print("$", *args, flush=True)
    proc = subprocess.Popen([sys.executable, *args])
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        proc.wait()
        print(f"TIMEOUT ({timeout_s}s) running: {args}", flush=True)
        return 75


def main() -> int:
    target = json.loads((HERE / "selection_preregistration.json").read_text(encoding="utf-8"))["target"]
    infra_streak = 0
    while True:
        passed = {a["id"] for a in attempts() if a["status"] == "AVAILABLE_GT_QC_PASSED"}
        print(f"available {len(passed)}/{target}", flush=True)
        if len(passed) >= target:
            return 0
        pending = unbuilt_passes()
        if pending:
            rc = run([str(HERE / "build_one.py"), pending[0]])
            if rc == 0:
                infra_streak = 0
            elif rc == 75:
                if last_row_was_bot_check():
                    print("STOPPED: YouTube anti-bot / sign-in gate encountered (not bypassed).", flush=True)
                    return 75
                infra_streak += 1
                if infra_streak >= MAX_INFRA_STREAK:
                    print("STOPPED: repeated infrastructure errors.", flush=True)
                    return 75
                time.sleep(PAUSE_S)
            else:
                infra_streak = 0
            continue
        rc = run([str(HERE / "screen.py")])
        if rc == 2:
            print("pool/quotas exhausted", flush=True)
            return 0
        if rc in (0, 1):
            infra_streak = 0
        else:
            if last_row_was_bot_check():
                print("STOPPED: YouTube anti-bot / sign-in gate encountered (not bypassed).", flush=True)
                return 75
            infra_streak += 1
            if infra_streak >= MAX_INFRA_STREAK:
                print("STOPPED: repeated infrastructure errors.", flush=True)
                return 75
            time.sleep(PAUSE_S)


if __name__ == "__main__":
    raise SystemExit(main())
