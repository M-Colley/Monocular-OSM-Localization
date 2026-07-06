"""Compare a VGGT-Long trajectory against GT + the OpenVO cache (KITTI 0033).

VGGT-Long (ICRA'25) chunks VGGT over a long sequence and Sim3-aligns the
chunks — the candidate VO-backbone upgrade from the July-2026 sweep. This
scores its camera_poses.txt the same way the matcher consumes a trajectory:
similarity-Procrustes onto the GT OXTS track (shape RMS after optimal
scale+rotation — what candidate SELECTION sees) plus the no-rotation scale
ratio. The OpenVO cache is scored identically for a like-for-like verdict.

    python scripts/test_vggt_long_kitti.py <path/to/camera_poses.txt>
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.kitti_raw import load_oxts_track  # noqa: E402

OXTS = "data/kitti/2011_09_30/2011_09_30_drive_0033_sync"
VO_NPZ = glob.glob(
    "data/local-36a50c34107a-drive-0033-karlsruhe-germany/trajectory_v2_*.npz")


def _resample(xy: np.ndarray, n: int = 256) -> np.ndarray:
    d = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    if d[-1] <= 0:
        raise ValueError("degenerate path")
    t = np.linspace(0.0, d[-1], n)
    return np.column_stack([np.interp(t, d, xy[:, 0]), np.interp(t, d, xy[:, 1])])


def shape_rms_vs(traj: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(similarity-Procrustes RMS in m, scale ratio traj/gt by arc length)."""
    a = _resample(np.asarray(traj, float))
    b = _resample(np.asarray(gt, float))
    a0 = a - a.mean(0)
    b0 = b - b.mean(0)
    na = np.linalg.norm(a0)
    H = a0.T @ b0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    s = (S[0] + d * S[1]) / (na ** 2)
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    resid = float(np.sqrt(np.mean(np.sum((s * (R @ a0.T).T - b0) ** 2, axis=1))))
    arc = lambda p: float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))  # noqa: E731
    return resid, arc(a) / max(arc(b), 1e-9)


def main() -> None:
    poses_txt = sys.argv[1]
    mats = np.loadtxt(poses_txt).reshape(-1, 4, 4)
    vggt_xy = mats[:, [0, 2], 3]                      # camera x (right), z (fwd)

    fixes = load_oxts_track(OXTS)
    gt = np.array([[f.lat, f.lon] for f in fixes], dtype=np.float64)
    lat0 = float(np.mean(gt[:, 0]))
    gt_xy = np.column_stack([
        (gt[:, 1] - gt[0, 1]) * 111320.0 * np.cos(np.radians(lat0)),
        (gt[:, 0] - gt[0, 0]) * 111320.0,
    ])

    r, sc = shape_rms_vs(vggt_xy, gt_xy)
    print(f"VGGT-Long : shape RMS {r:7.1f} m   (n={len(vggt_xy)} poses)")

    if VO_NPZ:
        vo = np.load(VO_NPZ[0])
        vo_xy = np.asarray(vo["xz"], float)[np.asarray(vo["valid"], bool)]
        r2, _ = shape_rms_vs(vo_xy, gt_xy)
        print(f"OpenVO    : shape RMS {r2:7.1f} m   (n={len(vo_xy)} poses, "
              f"{Path(VO_NPZ[0]).name})")


if __name__ == "__main__":
    main()
