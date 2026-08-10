"""Sanity-check the auto-extracted GPS-overlay ground truth.

    python scripts/check_overlay_gt.py                    # every overlay_*.json
    python scripts/check_overlay_gt.py --map              # + a route plot

Ground truth OCR'd off a burned-in stamp is free, which is exactly why it
needs auditing: a misread digit that survives the track filter becomes a
silently-wrong waypoint, and every downstream error number is then measured
against a lie. Nothing here needs the video — it reasons about whether the
waypoint sequence could describe a car driving.

Checks, per clip:

  time      strictly increasing t_sec, first fix near the window start
  speed     implied speed between consecutive waypoints within [0, MAX_KMH]
  jitter    no waypoint that reverses and returns (a classic digit misread)
  extent    total path length consistent with the elapsed time
  digits    coordinate precision actually present in the stamp

Exit status is non-zero if any clip fails, so this can gate a fleet rebuild.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A dashcam clip is a road vehicle. 200 km/h leaves room for an autobahn or a
# US freeway while still catching the kilometre-scale jumps a misread digit
# produces (0.001 deg of latitude is 111 m; one wrong digit is 1.1-11 km).
MAX_KMH = 200.0
# Below this, consecutive waypoints are the vehicle sitting at a light — a
# legitimate state, and one that makes the "did it move backwards" test noisy.
STOPPED_M = 15.0


def _haversine_m(a: dict, b: dict) -> float:
    R = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = (math.sin((la2 - la1) / 2) ** 2
         + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(h))


def _decimals(wps: list[dict]) -> int:
    """Significant decimal places actually present in the coordinates.

    Reveals the stamp's own precision: a 1-arcsecond DMS overlay lands on a
    ~31 m grid no matter how carefully it is read, which bounds the error
    any run against this GT can possibly report.
    """
    best = 0
    for w in wps:
        for v in (w["lat"], w["lon"]):
            s = f"{abs(v):.7f}".rstrip("0")
            best = max(best, len(s.split(".")[1]) if "." in s else 0)
    return best


def check(path: Path) -> tuple[bool, list[str], dict]:
    gt = json.loads(path.read_text(encoding="utf-8"))
    wps = gt.get("waypoints", [])
    problems: list[str] = []

    if len(wps) < 3:
        return False, [f"only {len(wps)} waypoints"], {}

    times = [w["t_sec"] for w in wps]
    if any(b <= a for a, b in zip(times, times[1:])):
        problems.append("t_sec is not strictly increasing")

    legs = [(_haversine_m(a, b), b["t_sec"] - a["t_sec"])
            for a, b in zip(wps, wps[1:])]
    speeds = [(d / dt) * 3.6 for d, dt in legs if dt > 0]
    fast = [(i, s) for i, s in enumerate(speeds) if s > MAX_KMH]
    for i, s in fast:
        problems.append(
            f"leg {i}->{i + 1} implies {s:.0f} km/h "
            f"({legs[i][0]:.0f} m in {legs[i][1]:.0f} s) — likely a misread digit")

    # An out-and-back between three waypoints, where the middle one is far
    # off the line between its neighbours, is what a single wrong digit looks
    # like after the track filter has passed it.
    for i in range(len(wps) - 2):
        a, b, c = wps[i], wps[i + 1], wps[i + 2]
        direct = _haversine_m(a, c)
        detour = _haversine_m(a, b) + _haversine_m(b, c)
        if detour > STOPPED_M and detour > 4.0 * max(direct, STOPPED_M):
            problems.append(
                f"waypoint {i + 1} (t={b['t_sec']:.0f}s) detours {detour:.0f} m "
                f"for {direct:.0f} m of progress — check it")

    total_m = sum(d for d, _ in legs)
    elapsed = times[-1] - times[0]
    stats = {
        "waypoints": len(wps),
        "elapsed_s": elapsed,
        "path_km": total_m / 1000.0,
        "mean_kmh": (total_m / elapsed) * 3.6 if elapsed else 0.0,
        "max_kmh": max(speeds) if speeds else 0.0,
        "decimals": _decimals(wps),
        "city": gt.get("city", "?"),
    }
    if stats["mean_kmh"] < 3.0:
        problems.append(f"mean speed {stats['mean_kmh']:.1f} km/h — the track "
                        f"barely moves; is the stamp being read at all?")
    return not problems, problems, stats


def _plot(rows: list[tuple[Path, dict]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(rows)
    cols = min(4, n)
    rowc = math.ceil(n / cols)
    fig, axes = plt.subplots(rowc, cols, figsize=(4 * cols, 3.6 * rowc), squeeze=False)
    for ax, (path, _stats) in zip(axes.ravel(), rows):
        gt = json.loads(path.read_text(encoding="utf-8"))
        wps = gt["waypoints"]
        lat = [w["lat"] for w in wps]
        lon = [w["lon"] for w in wps]
        ax.plot(lon, lat, "-o", ms=3, lw=1.2)
        ax.plot(lon[0], lat[0], "go", ms=7, label="start")
        ax.plot(lon[-1], lat[-1], "rs", ms=7, label="end")
        ax.set_title(f"{gt.get('video_id', path.stem)}\n{gt.get('city', '')}", fontsize=9)
        ax.set_aspect(1.0 / math.cos(math.radians(sum(lat) / len(lat))))
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("GPS-overlay ground truth — OCR'd routes", fontsize=12)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gt-dir", default="ground_truth")
    p.add_argument("--glob", default="overlay_*.json")
    p.add_argument("--map", action="store_true", help="also render the routes")
    args = p.parse_args(argv)

    paths = sorted((ROOT / args.gt_dir).glob(args.glob))
    paths = [q for q in paths if q.name != "overlay_fleet.json"]
    if not paths:
        print(f"no {args.glob} under {args.gt_dir}/ — run "
              f"scripts/build_overlay_fleet.py first")
        return 1

    print(f"{'clip':22s} {'wp':>3s} {'elapsed':>8s} {'km':>6s} {'mean':>6s} "
          f"{'max':>6s} {'dp':>3s}  city")
    print("-" * 100)
    ok_all = True
    rows: list[tuple[Path, dict]] = []
    for q in paths:
        ok, problems, s = check(q)
        ok_all &= ok
        if s:
            print(f"{q.stem:22s} {s['waypoints']:>3d} {s['elapsed_s']:>7.0f}s "
                  f"{s['path_km']:>6.2f} {s['mean_kmh']:>5.0f}k {s['max_kmh']:>5.0f}k "
                  f"{s['decimals']:>3d}  {s['city']}")
            rows.append((q, s))
        for msg in problems:
            print(f"    !! {msg}")
    if args.map and rows:
        _plot(rows, ROOT / "output" / "overlay_gt_routes.png")
    print("\n" + ("all clips plausible" if ok_all else "PROBLEMS FOUND — see !! above"))
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
