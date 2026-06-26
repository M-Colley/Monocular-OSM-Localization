"""A/B: does GeoCalib's focal+gravity beat the confidence-sweep calibration?

For each Ulm GT waypoint window we run OrienterNet sequential fusion three ways:
  A  current   : auto FOV sweep, hardcoded gravity (0, -4)
  B+ GeoCalib  : GeoCalib per-window median focal, gravity (roll, +pitch)
  B- GeoCalib  : GeoCalib per-window median focal, gravity (roll, -pitch)
and report the metric error vs the waypoint. The +/- pitch pair also pins down
the OrienterNet gravity sign convention vs GeoCalib's.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform

from geocalib import GeoCalib
from src.orienternet_localizer import refine_route

VIDEO = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4"
NPZ = "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz"
GT = "ground_truth/ulm_ULl8s4qydrk.json"
STRIDE = 3
R, MPD = 6371000.0, 111320.0


def err_m(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def main() -> None:
    xz = np.load(NPZ)["xz"]
    wps = json.load(open(GT))["waypoints"]
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dt = STRIDE / fps
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    cl = np.cos(np.radians(lat0))
    wp_xy = np.array([[(w["lon"] - lon0) * MPD * cl, (w["lat"] - lat0) * MPD] for w in wps])
    wp_vo = np.clip([int(round(w["t_sec"] / dt)) for w in wps], 0, len(xz) - 1)
    tf = SimilarityTransform(); tf.estimate(xz[wp_vo], wp_xy)
    s = tf.scale; Rm = tf.params[:2, :2] / s

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    gc = GeoCalib().to(dev)

    def geocalib_window(frames):
        fs, rs, ps = [], [], []
        for bgr in frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float().to(dev) / 255.0
            res = gc.calibrate(img)
            fs.append(float(np.atleast_1d(res["camera"].f.detach().cpu().numpy().ravel())[0]))
            rp = res["gravity"].rp.detach().cpu().numpy().ravel()
            rs.append(float(np.degrees(rp[0]))); ps.append(float(np.degrees(rp[1])))
        return float(np.median(fs)), float(np.median(rs)), float(np.median(ps))

    rows = []
    for k in (4, 6):
        c = int(wp_vo[k])
        win = [i for i in range(c - 52, c + 53, 13) if 0 <= i < len(xz)]
        if len(win) < 4:
            continue
        geo = wp_xy[k] + s * (xz[win] - xz[c]) @ Rm.T
        prior_ll = np.c_[lat0 + geo[:, 1] / MPD, lon0 + geo[:, 0] / (MPD * cl)]
        frames = []
        for i in win:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * STRIDE))
            ok, bgr = cap.read()
            frames.append(bgr if ok else frames[-1])
        gt = np.array([wps[k]["lat"], wps[k]["lon"]])

        f_gc, roll_gc, pitch_gc = geocalib_window(frames)
        hfov_gc = np.degrees(2 * np.arctan(frames[0].shape[1] / (2 * f_gc)))

        def run(focal, grav):
            out = refine_route(frames, prior_ll, focal, tile_m=130.0, gravity=grav)
            return None if out is None else err_m(out[len(out) // 2], gt)

        eA = run(None, (0.0, -4.0))
        eBp = run(f_gc, (roll_gc, +pitch_gc))
        eBm = run(f_gc, (roll_gc, -pitch_gc))
        rows.append((wps[k]["t_sec"], hfov_gc, roll_gc, pitch_gc, eA, eBp, eBm))
        print(f"t={wps[k]['t_sec']:>4}s GeoCalib HFOV={hfov_gc:5.1f} roll={roll_gc:+.1f} "
              f"pitch={pitch_gc:+.1f} | A(sweep)={eA} B+={eBp} B-={eBm}")
    cap.release()

    if rows:
        arr = lambda j: [r[j] for r in rows if r[j] is not None]
        print("\n=== medians ===")
        print(f"  A  (sweep + pitch -4):     {np.median(arr(4)):.1f} m")
        print(f"  B+ (GeoCalib, +pitch):     {np.median(arr(5)):.1f} m")
        print(f"  B- (GeoCalib, -pitch):     {np.median(arr(6)):.1f} m")
        print(f"  GeoCalib median HFOV={np.median([r[1] for r in rows]):.1f} deg "
              f"pitch={np.median([r[3] for r in rows]):+.1f}")


if __name__ == "__main__":
    main()
