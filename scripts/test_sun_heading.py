"""Run the sun-heading capability on every available clip + validate the image side.

(A) AVAILABILITY: for each clip, does it carry a usable capture time (metadata or a
    burned-in clock)? -> sun heading available or graceful no-op.
(B) VALIDATION on Ulm 4K: it has NO timestamp, but if we ASSUME one we can check the
    image side end-to-end: the per-frame (sun-derived heading - GT heading) should be
    ~CONSTANT across the clip if the sun blob is being tracked correctly (the constant
    is just the error in the assumed capture time). A small spread => the method works
    and the only real blocker is the missing timestamp.
"""

from __future__ import annotations

import datetime
import json
import zoneinfo

import numpy as np

from src.sun_heading import estimate_heading

SP = ("C:/Users/LOCALA~1/AppData/Local/Temp/claude/"
      "C--Users-localadmin-Documents-Monocular-OSM-Localization/"
      "5aaa29d8-2db0-4cde-9d35-5898b7aa455c/scratchpad")
CLIPS = [
    ("Ulm 4K (ULl8s4qydrk)", "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4", (48.40, 9.99)),
    ("Ulm Innenstadt (aQi60unoOKw)", f"{SP}/ulm_innenstadt.mp4", (48.40, 9.99)),
    ("Erbach a.d. Donau (uKbCXuxPnZ8)", f"{SP}/erbach.mp4", (48.33, 9.89)),
]


def main():
    print("=== (A) availability across clips ===")
    for name, path, center in CLIPS:
        r = estimate_heading(path, center, list(range(20, 200, 30)))
        if r is None:
            print(f"  {name:34s}: module unavailable"); continue
        if r.get("available"):
            print(f"  {name:34s}: AVAILABLE via {r['source']} @ {r['capture_utc']} "
                  f"-> heading {r['median_heading']:.0f} deg ({r['n_used']} frames)")
        else:
            print(f"  {name:34s}: not available ({r.get('reason')})")

    print("\n=== (B) image-side validation on Ulm 4K (assumed time; check derived-GT is constant) ===")
    wps = json.load(open("ground_truth/ulm_ULl8s4qydrk.json"))["waypoints"]
    ts = np.array([w["t_sec"] for w in wps])
    la = np.array([w["lat"] for w in wps]); lo = np.array([w["lon"] for w in wps])
    times = list(np.linspace(ts.min() + 5, ts.max() - 5, 30))
    # GT compass heading per query time (bearing between consecutive interpolated fixes)
    qla = np.interp(times, ts, la); qlo = np.interp(times, ts, lo)
    dla = np.gradient(qla); dlo = np.gradient(qlo) * np.cos(np.radians(qla))
    gt_head = np.degrees(np.arctan2(dlo, dla)) % 360

    tz = zoneinfo.ZoneInfo("Europe/Berlin")
    for hh in (10, 13, 16):                         # plausible summer-2023 daytimes
        dt0 = datetime.datetime(2023, 7, 15, hh, 0, tzinfo=tz)
        r = estimate_heading(CLIPS[0][1], (48.403, 9.99), times,
                             assume_capture_utc=dt0)
        if not r.get("available"):
            print(f"  assume {hh}:00 -> {r.get('reason')}"); continue
        h = r["headings"]; good = np.isfinite(h)
        if good.sum() < 5:
            print(f"  assume {hh}:00 -> only {good.sum()} frames had a detectable sun"); continue
        diff = (h[good] - gt_head[good] + 180) % 360 - 180
        # circular std of (derived - GT): small => sun tracks heading correctly
        ang = np.radians(diff)
        cstd = np.degrees(np.sqrt(-2 * np.log(np.hypot(np.mean(np.cos(ang)), np.mean(np.sin(ang))))))
        print(f"  assume {hh}:00  sun-detected {good.sum()}/{len(times)} frames | "
              f"(derived-GT) circular-std = {cstd:.0f} deg  "
              f"{'<- TRACKS heading (works if time known)' if cstd < 35 else '(noisy)'}")


if __name__ == "__main__":
    main()
