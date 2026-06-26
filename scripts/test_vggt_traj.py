"""Probe VGGT camera trajectory vs VO/GT on a clip.

VGGT (feed-forward geometry transformer) predicts camera extrinsics for a
set of frames in one pass with global cross-frame attention — far less
drift than incremental monocular VO. This checks whether its top-down
camera path is faithful: on a loop (KITTI 0033) a low-drift trajectory
should CLOSE (VO leaves a 27% end-start gap).

    python scripts/test_vggt_traj.py VIDEO --n 48
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np


def vggt_trajectory(frames: list[np.ndarray], device: str = "cuda",
                    smooth: int = 0) -> np.ndarray:
    """Top-down (x, z) camera-centre path from VGGT extrinsics, (N, 2).

    ``smooth`` > 0 applies a Savitzky-Golay filter of that window to the
    path — VGGT poses are globally consistent but locally jittery on
    forward-driving footage, and the matcher is sensitive to that jitter.
    """
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for i, f in enumerate(frames):
            p = f"{td}/f{i:04d}.png"
            cv2.imwrite(p, f)
            paths.append(p)
        images = load_and_preprocess_images(paths).to(device)  # (N,3,H,W)
        model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
            pred = model(images[None])  # add batch dim
        extri, _intri = pose_encoding_to_extri_intri(
            pred["pose_enc"], images.shape[-2:])
        extri = extri[0].float().cpu().numpy()  # (N,3,4) world->cam (OpenCV)
    # Camera centre C = -R^T t; ground plane is X-Z (Y is down in OpenCV).
    centres = np.array([-(e[:, :3].T @ e[:, 3]) for e in extri])
    xy = centres[:, [0, 2]]
    if smooth and len(xy) > smooth:
        from scipy.signal import savgol_filter
        w = smooth if smooth % 2 else smooth + 1
        xy = np.column_stack([savgol_filter(xy[:, 0], w, 2),
                              savgol_filter(xy[:, 1], w, 2)])
    return xy


def _arc(xy):
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f0 = int(args.start * fps)
    f1 = int(args.end * fps) if args.end else total
    idxs = np.linspace(f0, min(f1, total) - 1, args.n).round().astype(int)
    frames = []
    for ix in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(ix))
        ok, fr = cap.read()
        if ok:
            frames.append(fr)
    cap.release()
    print(f"{len(frames)} keyframes")

    xy = vggt_trajectory(frames)
    arc = _arc(xy)
    gap = float(np.linalg.norm(xy[-1] - xy[0]))
    straight = gap
    print(f"VGGT trajectory: {len(xy)} poses")
    print(f"  arc {arc:.2f} (scale-free units)  end-start gap {gap:.2f} "
          f"({100 * gap / max(arc, 1e-9):.0f}% of arc)")
    print(f"  sinuosity {arc / max(straight, 1e-9):.2f}")
    np.save("output/vggt_traj.npy", xy)
    print("  saved output/vggt_traj.npy")


if __name__ == "__main__":
    main()
