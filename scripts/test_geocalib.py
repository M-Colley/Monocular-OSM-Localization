"""Test GeoCalib (ECCV 2024, cvg/GeoCalib) as a better camera calibrator.

Our OrienterNet head currently calibrates the camera by sweeping the FOV and
picking the value that maximises OrienterNet's own confidence (coarse, 15-deg
grid, no gravity). GeoCalib infers focal length AND gravity (roll/pitch) from a
single image with a learned Perspective Field + geometric optimisation — a
proper per-frame calibration. This script runs it on real dashcam frames and
prints HFOV + roll/pitch so we can compare against the confidence-sweep pick.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np
import torch

from geocalib import GeoCalib

CLIPS = {
    "Ulm 4K (existing)": "data/ull8s4qydrk-ulm-germany-4k-drive-ulm-germany/input.mp4",
}


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoCalib().to(dev)
    for name, path in CLIPS.items():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"{name}: cannot open {path}")
            continue
        print(f"\n=== {name} ===")
        fovs, rolls, pitches = [], [], []
        for t in (20, 60, 100, 140, 180):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, bgr = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            res = model.calibrate(img.to(dev))
            cam = res["camera"]
            f = float(np.atleast_1d(cam.f.detach().cpu().numpy().ravel())[0])
            w = rgb.shape[1]
            hfov = float(np.degrees(2 * np.arctan(w / (2 * f))))
            grav = res["gravity"]
            rp = grav.rp.detach().cpu().numpy().ravel()
            roll, pitch = float(np.degrees(rp[0])), float(np.degrees(rp[1]))
            fovs.append(hfov); rolls.append(roll); pitches.append(pitch)
            print(f"  t={t:>3}s  focal={f:7.1f}px  HFOV={hfov:5.1f} deg  "
                  f"roll={roll:+5.1f}  pitch={pitch:+5.1f}")
        cap.release()
        if fovs:
            print(f"  -> median HFOV={np.median(fovs):.1f} deg  "
                  f"roll={np.median(rolls):+.1f}  pitch={np.median(pitches):+.1f}")
            print(f"  (confidence-sweep auto-cal picked ~125-140 deg for this clip)")


if __name__ == "__main__":
    main()
