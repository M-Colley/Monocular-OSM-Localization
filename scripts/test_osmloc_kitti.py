"""OSMLoc (Information Fusion 2026, maploc fork) vs OrienterNet A/B on KITTI.

Mirrors scripts/test_orienternet_all.py's oracle protocol EXACTLY (prior =
GT + 30 m offset, 4 chunks x 10 frames, RigidAligner sequential fusion) but
runs the OSMLoc-S checkpoint from third_party/OSMLoc — so the numbers are
directly comparable to output/orienternet_all.json (OrienterNet: 1.9 m median
on drive_0033). Self-contained: imports OSMLoc's OWN maploc fork (do NOT mix
with third_party/OrienterNet's in one process).

    python scripts/test_osmloc_kitti.py [drive_0033|drive_0009]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "OSMLoc"))
sys.path.insert(0, str(ROOT))
# OSMLoc's depth-anything encoder hub-loads its vendored DINOv2 via a path
# RELATIVE to the CWD ('maploc/models/dinov2', source='local'), so run from
# inside the repo; all data paths below are absolute to compensate.
os.chdir(ROOT / "third_party" / "OSMLoc")
from maploc.models.sequential import RigidAligner  # noqa: E402
from maploc.osm.tiling import TileManager  # noqa: E402
from maploc.utils.geo import BoundaryBox, Projection  # noqa: E402
from maploc.utils.wrappers import Camera  # noqa: E402

_VARIANT = os.environ.get("OSMLOC_VARIANT", "small")   # small | base
CKPT = str(ROOT / f"third_party/OSMLoc/checkpoints/loca_polar_{_VARIANT}.ckpt")
R, MPD = 6371000.0, 111320.0

CLIPS = {
    "drive_0033": dict(
        video=str(ROOT / "data/kitti/drive_0033.mp4"), fps=10.0, focal=721.5,
        ref_w=1242,
        oxts=str(ROOT / "data/kitti/2011_09_30/2011_09_30_drive_0033_sync")),
    "drive_0009": dict(
        video=str(ROOT / "data/kitti/drive_0009.mp4"), fps=10.0, focal=721.5,
        ref_w=1242,
        oxts=str(ROOT / "data/kitti/2011_09_26/2011_09_26_drive_0009_sync")),
    # comma2k19: the CROSS-DOMAIN case (US suburb; OrienterNet oracle fell to
    # 15.2 m median / 25% recall@5m here) — the terrain OSMLoc's cross-area
    # claim is about.
    "comma": dict(
        video=str(ROOT / "data/comma/route_148.mp4"), fps=20.0, focal=910.0,
        ref_w=1164, comma_segs=[
            str(ROOT / f"data/comma/extracted/b0c9d2329ad1606b_2018-08-17--14-55-39/{s}")
            for s in (1, 2, 3, 4)]),
}


def load_model(device):
    from maploc.models import get_model
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]
    # The released repo renamed the model module: the checkpoint says
    # 'loca_polar' (their internal name) but ships only 'osmloc'.
    try:
        cls = get_model(cfg.model.name)
    except ModuleNotFoundError:
        print(f"model '{cfg.model.name}' not in repo; falling back to 'osmloc'")
        cls = get_model("osmloc")
    model = cls(cfg.model).eval()
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"state fit: {len(missing)} missing / {len(unexpected)} unexpected "
          f"of {len(state)} ckpt keys")
    if len(missing) > 0.1 * len(state):
        raise SystemExit("checkpoint does not match the released architecture; "
                         "aborting rather than reporting numbers from a "
                         "half-loaded model")
    return model.to(device), cfg


def prepare(image, camera, canvas, cfg, gravity, model):
    from maploc.data.image import pad_image, rectify_image, resize_image
    tfl = cfg.data.resize_image / 2
    size = (camera.size * (tfl / camera.f)).round().int()
    im = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    im, valid = rectify_image(im, camera.float(), roll=-gravity[0],
                              pitch=-gravity[1])
    im, _, camera, *_ = resize_image(im, size.tolist(), camera=camera,
                                     valid=valid)
    try:
        stride = max(model.image_encoder.layer_strides)
    except Exception:
        stride = 14  # DINOv2 patch stride
    size = (np.ceil(size.numpy() / stride) * stride).astype(int)
    im, valid, camera = pad_image(im, size.tolist(), camera,
                                  crop_and_center=True)
    return {"image": im, "map": torch.from_numpy(canvas.raster).long(),
            "camera": camera.float(), "valid": valid}


def kitti_gps(drive):
    files = sorted((Path(drive) / "oxts" / "data").glob("*.txt"))

    def get(i):
        p = files[min(i, len(files) - 1)].read_text().split()
        return (float(p[0]), float(p[1]), np.degrees(float(p[3])),
                np.degrees(float(p[4])))
    return len(files), get


def comma_gps(segdirs):
    from src.comma2k19 import load_route_track
    track = load_route_track(segdirs)

    def get(i):
        f = track[min(i, len(track) - 1)]
        return f.lat, f.lon, 0.0, 0.0
    return len(track), get


def err_m(a, b):
    dlat = np.radians(a[0] - b[0]); dlon = np.radians(a[1] - b[1])
    h = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(b[0])) ** 2 * np.sin(dlon / 2) ** 2)
    return float(2 * R * np.arcsin(np.sqrt(h)))


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "drive_0033"
    spec = CLIPS[name]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(device)
    ppm = cfg.data.pixel_per_meter
    if "comma_segs" in spec:
        n_gps, get = comma_gps(spec["comma_segs"])
    else:
        n_gps, get = kitti_gps(spec["oxts"])
    cap = cv2.VideoCapture(spec["video"])
    n = min(n_gps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    step = max(1, int(round(1.3 * spec["fps"])))
    span = 10 * step
    starts = np.linspace(int(0.05 * n), int(0.95 * n) - span, 4).round().astype(int)
    out = []
    for cstart in starts:
        frames = list(range(cstart, cstart + span, step))
        if frames[-1] >= n:
            continue
        latm, lonm, *_ = get(frames[len(frames) // 2])
        proj = Projection(latm + 30 / MPD,
                          lonm + 30 / (MPD * np.cos(np.radians(latm))))
        per = []
        for fr in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, bgr = cap.read()
            if not ok:
                continue
            lat, lon, roll, pitch = get(fr)
            true_ll = np.array([lat, lon])
            xy = proj.project(true_ll)
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]
            fx = spec["focal"] * w / spec["ref_w"]
            cam = Camera.from_dict({
                "model": "SIMPLE_PINHOLE", "width": w, "height": h,
                "params": [fx, w / 2 + 0.5, h / 2 + 0.5]})
            center = proj.project(true_ll + np.array(
                [30 / MPD, 30 / (MPD * np.cos(np.radians(lat)))]))
            bbox = BoundaryBox(center, center) + 96
            canvas = TileManager.from_bbox(proj, bbox + 10, ppm).query(bbox)
            data = {k: v.to(device)[None] for k, v in
                    prepare(image, cam, canvas, cfg, (roll, pitch),
                            model).items()}
            with torch.no_grad():
                lp = model(data)["log_probs"].squeeze(0)
            per.append((lp, canvas, xy, true_ll, fr))
        if len(per) < 3:
            continue
        xys = np.array([p[2] for p in per])
        d = np.gradient(xys, axis=0)
        yaws = (90 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360
        aligner = RigidAligner(num_rotations=per[0][0].shape[-1])
        order = sorted(range(len(per)), key=lambda i: abs(i - len(per) // 2))
        for i in order:
            lp, canvas, xy, _, _ = per[i]
            aligner.update(lp, canvas, torch.tensor(xy, device=device).float(),
                           torch.tensor(float(yaws[i]), device=device).float())
        aligner.compute()
        for i, (_lp, _c, xy, true_ll, fr) in enumerate(per):
            gxy, _ = aligner.transform(
                torch.tensor(xy, device=device).float(),
                torch.tensor(float(yaws[i]), device=device).float())
            pll = proj.unproject(gxy.cpu().numpy())
            out.append({"frame": int(fr),
                        "true": [float(true_ll[0]), float(true_ll[1])],
                        "pred": [float(pll[0]), float(pll[1])],
                        "err_m": err_m(pll, true_ll)})
    cap.release()
    e = np.array([o["err_m"] for o in out]) if out else np.array([np.inf])
    print(f"OSMLoc-{_VARIANT} {name}: median {np.median(e):6.1f} m  "
          f"recall@5m {100 * np.mean(e <= 5):3.0f}%  "
          f"@10m {100 * np.mean(e <= 10):3.0f}%  ({len(out)} frames)")
    (ROOT / "output").mkdir(exist_ok=True)
    json.dump({name: out},
              open(ROOT / "output" / f"osmloc_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
