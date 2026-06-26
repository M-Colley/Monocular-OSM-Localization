"""Test OrienterNet (CVPR'23 neural OSM localization) on our KITTI frames.

OrienterNet matches a learned BEV of a ground image against the OSM raster
to predict a 3-DoF pose — reported ~3 m recall on KITTI, far below our
shape-matcher's ceiling. It REFINES a coarse prior within a tile, so it
pairs with our pipeline: shape-match -> coarse neighbourhood -> OrienterNet
-> metric pose. This probes it on real KITTI drive_0033 frames with a
deliberately offset prior (simulating a coarse GPS / our shape estimate),
measuring the residual to the true OXTS position.

    python scripts/test_orienternet.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ON = Path("third_party/OrienterNet")
sys.path.insert(0, str(ON))

from maploc.evaluation.run import pretrained_models, resolve_checkpoint_path  # noqa: E402
from maploc.models.orienternet import OrienterNet  # noqa: E402
from maploc.models.voting import argmax_xyr, fuse_gps  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402
from src.kitti_raw import load_oxts_track  # noqa: E402

DRIVE = "data/kitti/2011_09_30/2011_09_30_drive_0033_sync"
KITTI_FX = 721.5  # P_rect_02 focal for the 1242-wide image


def load_model(device):
    exp, _ = pretrained_models["OrienterNet_MGL"]
    path = resolve_checkpoint_path(exp)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]
    cfg.model.image_encoder.backbone.pretrained = False
    model = OrienterNet(cfg.model).eval()
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    return model.to(device), cfg


def prepare(image, camera, canvas, cfg, gravity, model):
    from maploc.data.image import pad_image, rectify_image, resize_image
    tfl = cfg.data.resize_image / 2
    factor = tfl / camera.f
    size = (camera.size * factor).round().int()
    im = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    roll, pitch = gravity
    im, valid = rectify_image(im, camera.float(), roll=-roll, pitch=-pitch)
    im, _, camera, *_ = resize_image(im, size.tolist(), camera=camera, valid=valid)
    stride = max(model.image_encoder.layer_strides)
    size = (torch.ceil(size / stride) * stride).int()
    im, valid, camera = pad_image(im, size.tolist(), camera, crop_and_center=True)
    return {"image": im, "map": torch.from_numpy(canvas.raster).long(),
            "camera": camera.float(), "valid": valid}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    ppm = cfg.data.pixel_per_meter
    print(f"OrienterNet loaded; pixel_per_meter={ppm}, resize={cfg.data.resize_image}")

    fixes = load_oxts_track(DRIVE)            # per-frame true lat/lon (10 Hz)
    cap = cv2.VideoCapture("data/kitti/drive_0033.mp4")

    R = 6371000.0
    M_PER_DEG = 111320.0

    def predict_latlon(fidx):
        """OrienterNet predicted lat/lon for one frame (prior ~42 m off)."""
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, bgr = cap.read()
        if not ok:
            return None
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        fx = KITTI_FX * (w / 1242.0)
        camera = Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": w, "height": h,
                                   "params": [fx, w / 2 + 0.5, h / 2 + 0.5]})
        true = fixes[fidx]
        true_ll = np.array([true.lat, true.lon])
        prior_ll = true_ll + np.array(
            [30.0 / M_PER_DEG, 30.0 / (M_PER_DEG * np.cos(np.radians(true.lat)))])
        proj = Projection(*prior_ll)
        bbox = BoundaryBox(proj.project(prior_ll), proj.project(prior_ll)) + 128
        tiler = TileManager.from_bbox(proj, bbox + 10, ppm)
        canvas = tiler.query(bbox)
        data = prepare(image, camera, canvas, cfg, (0.0, 0.0), model)
        data_ = {k: v.to(device)[None] for k, v in data.items()}
        with torch.no_grad():
            pred = model(data_)
        lp = pred["log_probs"].squeeze(0)
        lp = fuse_gps(lp, torch.from_numpy(canvas.to_uv(bbox.center)).to(lp), ppm, sigma=108)
        xyr = argmax_xyr(lp).cpu().numpy()
        return proj.unproject(canvas.to_xy(xyr[:2])), true_ll

    def err_m(a, b):
        dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
        h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2
        return float(2 * R * np.arcsin(np.sqrt(h)))

    refs = list(range(250, 1350, 90))    # ~12 reference frames along the drive
    single, fused = [], []
    for r in refs:
        out = predict_latlon(r)
        if out is None:
            continue
        pred_r, true_r = out
        single.append(err_m(pred_r, true_r))
        # Sequence fusion: predict nearby frames too, shift each back to r by
        # the known OXTS displacement (our VO supplies odometry), take median.
        ests = [pred_r]
        for k in (-6, -3, 3, 6):
            o = predict_latlon(r + k)
            if o is None:
                continue
            pred_k, true_k = o
            shift = true_r - true_k          # frame k -> r displacement (odometry)
            ests.append(pred_k + shift)
        fest = np.median(np.array(ests), axis=0)
        fused.append(err_m(fest, true_r))
        print(f"  ref {r}: single {single[-1]:5.1f} m | seq-fused({len(ests)}) {fused[-1]:5.1f} m")
    cap.release()

    def recall(e, t):
        return 100.0 * np.mean(np.array(e) <= t)
    print(f"\nOrienterNet KITTI 0033 ({len(single)} refs):")
    print(f"  single-frame: median {np.median(single):5.1f} m  recall@3m {recall(single,3):3.0f}%  @5m {recall(single,5):3.0f}%")
    print(f"  seq-fused:    median {np.median(fused):5.1f} m  recall@3m {recall(fused,3):3.0f}%  @5m {recall(fused,5):3.0f}%")


if __name__ == "__main__":
    main()
