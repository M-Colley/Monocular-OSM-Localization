"""Prepare the Ulm clip for OpenVO inference (bypassing Metric3Dv2/WildCamera).

Builds the KITTI-style layout OpenVO's InferDataset expects:
  third_party/OpenVO/data/ULM/sequences/00/image_2/*.jpg     (RGB frames)
  third_party/OpenVO/data/ULM/depth/sequences/00/image_2/*.png  (DA3 metric depth, uint16 = depth_m*256)
  + an intrinsics json (DA3's estimated focal, scaled to the saved frame size)

We substitute our own DA3 metric depth for Metric3Dv2 and DA3's intrinsics for
WildCamera (the user-approved "bypass build" path).
"""

from __future__ import annotations

import glob
import json
import os
import shutil

import cv2
import numpy as np
from PIL import Image

SRC_FRAMES = "C:/vggt_frames/ulm_full"          # 1260 frames @ 3 fps (0-420 s)
OUT = "C:/Users/localadmin/Documents/Monocular-OSM-Localization/third_party/OpenVO/data/ULM"
SCENE = "00"
BATCH = 12
DEPTH_SCALE = 256.0    # uint16 = depth_m * 256  (matches OpenVO save_depth_png)


def main():
    import torch
    from src.da3_reconstruction import load_da3_model

    img_dir = f"{OUT}/sequences/{SCENE}/image_2"
    dep_dir = f"{OUT}/depth/sequences/{SCENE}/image_2"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(dep_dir, exist_ok=True)

    frames = sorted(glob.glob(f"{SRC_FRAMES}/*.jpg"))
    print(f"{len(frames)} frames", flush=True)
    W0, H0 = Image.open(frames[0]).size
    print(f"frame size {W0}x{H0}", flush=True)

    model = load_da3_model(device="cuda")
    fx_da3 = []
    for b in range(0, len(frames), BATCH):
        chunk = frames[b:b + BATCH]
        pil = [Image.open(f).convert("RGB") for f in chunk]
        pred = model.inference(pil)
        depths = np.asarray(pred.depth)                 # (N,h,w) metric meters
        K = np.asarray(pred.intrinsics)                 # (N,3,3) at proc res
        proc = np.asarray(pred.processed_images)
        ph, pw = depths.shape[1], depths.shape[2]
        for i, fpath in enumerate(chunk):
            idx = b + i
            # copy RGB into the layout (keep original resolution)
            shutil.copyfile(fpath, f"{img_dir}/{idx:06d}.jpg")
            # depth -> resize to original frame size -> uint16 png (*256)
            d = cv2.resize(depths[i], (W0, H0), interpolation=cv2.INTER_LINEAR)
            d16 = np.clip(d * DEPTH_SCALE, 0, 65535).astype(np.uint16)
            cv2.imwrite(f"{dep_dir}/{idx:06d}.png", d16)
            # DA3 focal scaled from proc width to saved width
            fx_da3.append(float(K[i, 0, 0]) * (W0 / pw))
        if b % (BATCH * 10) == 0:
            print(f"  {b+len(chunk)}/{len(frames)}", flush=True)

    fx = float(np.median(fx_da3))
    fy = fx
    cx, cy = W0 / 2.0, H0 / 2.0
    hfov = 2 * np.degrees(np.arctan((W0 / 2.0) / fx))
    print(f"DA3 median fx={fx:.1f} -> HFOV={hfov:.1f} deg ; intrinsics [{fx:.1f},{fy:.1f},{cx:.1f},{cy:.1f}]")
    intr = {SCENE: [fx, fy, cx, cy]}
    with open(f"{OUT}/ulm_da3_intrs.json", "w") as f:
        json.dump(intr, f, indent=2)
    print("wrote intrinsics json + layout under", OUT)


if __name__ == "__main__":
    main()
