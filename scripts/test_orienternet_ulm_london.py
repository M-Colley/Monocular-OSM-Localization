"""OrienterNet on Ulm + London — clips with only SPARSE waypoints.

These dashcam clips have ~10 hand-labelled GPS waypoints, not the dense
per-frame GPS the RigidAligner fusion needs. The fix: use our cached VO
for the odometry. We anchor each waypoint's short window on the waypoint's
true position and lay the VO's (scaled, rotated) local shape around it, so
OrienterNet gets a good per-frame coarse prior + odometry without per-frame
GPS — exactly what the pipeline would do. Refined pose at the waypoint is
compared to the waypoint GT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.transform import SimilarityTransform

sys.path.insert(0, str(Path("third_party/OrienterNet")))
from maploc.models.sequential import RigidAligner  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402
from scripts.test_orienternet import load_model, prepare  # noqa: E402

R, MPD = 6371000.0, 111320.0
CLIPS = {
    "Ulm": dict(video="data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4",
                npz="data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/trajectory_v2_0-420.0_s3_fauto.npz",
                gt="ground_truth/ulm_ULl8s4qydrk.json", stride=3, fov=125.0, pitch=-4.0),
    "London": dict(video="data/london_T4wTL3LpLqU/input.mp4",
                   npz="data/local-73200bdd8068-input-london-uk/trajectory_v2_0-295.0_s3_fauto.npz",
                   gt="ground_truth/london_T4wTL3LpLqU.json", stride=3, fov=125.0, pitch=-4.0),
}


def err_m(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def run(name, spec, model, cfg, device):
    ppm = cfg.data.pixel_per_meter
    xz = np.load(spec["npz"])["xz"]
    wps = json.load(open(spec["gt"]))["waypoints"]
    cap = cv2.VideoCapture(spec["video"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nvid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dt = spec["stride"] / fps                      # seconds per VO sample
    lat0 = np.mean([w["lat"] for w in wps]); lon0 = np.mean([w["lon"] for w in wps])
    proj = Projection(lat0, lon0)
    wp_xy = np.array([proj.project(np.array([w["lat"], w["lon"]])) for w in wps])
    wp_vo = np.array([int(round(w["t_sec"] / dt)) for w in wps])
    wp_vo = np.clip(wp_vo, 0, len(xz) - 1)
    # Global VO->geo similarity (gives a consistent scale + rotation).
    tf = SimilarityTransform()
    tf.estimate(xz[wp_vo], wp_xy)
    s = tf.scale
    R2 = tf.params[:2, :2] / s                     # pure rotation

    import time

    def fetch_tiler(bbox):
        for attempt in range(6):
            try:
                return TileManager.from_bbox(proj, bbox, ppm)
            except ValueError as e:
                if "509" in str(e) and attempt < 5:
                    time.sleep(30); continue
                raise

    fov, pitch = spec["fov"], spec["pitch"]
    out = []
    for k, w in enumerate(wps):
        c = int(wp_vo[k])
        win = [i for i in range(c - 65, c + 66, 13) if 0 <= i < len(xz)]
        if len(win) < 4:
            continue
        # Anchor the window at the waypoint: geo = wp + s*R*(vo - vo_c).
        geo = wp_xy[k] + s * (xz[win] - xz[c]) @ R2.T
        lo = geo.min(0) - 140 + 30; hi = geo.max(0) + 140 + 30
        tiler = fetch_tiler(BoundaryBox(lo, hi))
        per = []
        for j, i in enumerate(win):
            vf = int(round(i * spec["stride"]))
            if vf >= nvid:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, vf)
            ok, bgr = cap.read()
            if not ok:
                continue
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, ww = image.shape[:2]
            f = ww / (2 * np.tan(np.deg2rad(fov) / 2))
            cam = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": ww, "height": h,
                                    "params": [f, ww / 2 + 0.5, h / 2 + 0.5]})
            canvas = tiler.query(BoundaryBox(geo[j] + 30, geo[j] + 30) + 110)
            data = {kk: v.to(device)[None] for kk, v in
                    prepare(image, cam, canvas, cfg, (0.0, pitch), model).items()}
            with torch.no_grad():
                lp = model(data)["log_probs"].squeeze(0)
            per.append((lp, canvas, geo[j], j))
        time.sleep(1)
        if len(per) < 3:
            continue
        gxy = np.array([p[2] for p in per]); d = np.gradient(gxy, axis=0)
        yaws = (90 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
        al = RigidAligner(num_rotations=per[0][0].shape[-1])
        order = sorted(range(len(per)), key=lambda i: abs(per[i][3] - len(win) // 2))
        for i in order:
            lp, canvas, xy, _ = per[i]
            al.update(lp, canvas, torch.tensor(xy, device=device).float(),
                      torch.tensor(float(yaws[i]), device=device).float())
        al.compute()
        # refined position of the frame nearest the waypoint
        ci = min(range(len(per)), key=lambda i: abs(per[i][3] - len(win) // 2))
        g, _ = al.transform(torch.tensor(per[ci][2], device=device).float(),
                            torch.tensor(float(yaws[ci]), device=device).float())
        pll = proj.unproject(g.cpu().numpy())
        e = err_m(pll, np.array([w["lat"], w["lon"]]))
        out.append({"t": w["t_sec"], "true": [w["lat"], w["lon"]],
                    "pred": [float(pll[0]), float(pll[1])], "err": e})
        print(f"  {name} t={w['t_sec']:>4}s: {e:6.1f} m")
    cap.release()
    if out:
        e = np.array([o["err"] for o in out])
        print(f"{name}: median {np.median(e):.1f} m  recall@10m {100*np.mean(e<=10):.0f}%  "
              f"@25m {100*np.mean(e<=25):.0f}%  ({len(e)} waypoints)\n")
    return out


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    out = {}
    for name, spec in CLIPS.items():
        try:
            out[name] = run(name, spec, model, cfg, device)
        except Exception as e:
            print(f"{name} FAILED: {type(e).__name__}: {e}")
    json.dump(out, open("output/orienternet_ulm_london.json", "w"), indent=1)


if __name__ == "__main__":
    main()
