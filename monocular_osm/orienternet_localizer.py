"""OrienterNet metric localization head (neural BEV -> OSM matching).

Our shape-matcher gets the right *neighbourhood* (~100-150 m) but can't pin
the street — trajectory shape is non-unique in road networks. OrienterNet
(CVPR 2023) instead encodes each frame into a learned bird's-eye view and
matches it against the OSM raster, predicting a metric pose. With the
paper's sequential fusion over a short window it reaches ~2 m on KITTI —
including on our own data (median 1.9 m, recall@5m 100% on drive_0033).

This module wraps OrienterNet as a REFINEMENT head: given a coarse route
(our shape-match estimate, in lat/lon) + the video keyframes + the camera
focal, it fuses per-frame BEV->OSM beliefs along the route (using the
route's own relative motion as odometry) and returns refined metric
positions. The coarse route only needs to be within ~tile/2 of truth, so
the OSM tile is sized to cover the shape-matcher's error.

Heavy + optional: needs `third_party/OrienterNet` on the path, the `vggt`-
style weights download, and a GPU. Returns ``None`` if anything is missing,
so the pipeline degrades gracefully.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_MODEL = None
_CFG = None
_ON = Path(__file__).resolve().parents[1] / "third_party" / "OrienterNet"


def _ensure_on_path() -> bool:
    import sys
    if not (_ON / "maploc").exists():
        return False
    if str(_ON) not in sys.path:
        sys.path.insert(0, str(_ON))
    return True


def _load_model(device):
    global _MODEL, _CFG
    if _MODEL is not None:
        return _MODEL, _CFG
    import torch
    from maploc.evaluation.run import pretrained_models, resolve_checkpoint_path
    from maploc.models.orienternet import OrienterNet
    exp, _ = pretrained_models["OrienterNet_MGL"]
    ckpt = torch.load(resolve_checkpoint_path(exp), map_location="cpu", weights_only=False)
    cfg = ckpt["hyper_parameters"]
    cfg.model.image_encoder.backbone.pretrained = False
    model = OrienterNet(cfg.model).eval()
    state = {k[len("model."):]: v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state, strict=True)
    _MODEL, _CFG = model.to(device), cfg
    return _MODEL, _CFG


def _fetch_canvases(
    tile_manager_cls,
    boundary_box_cls,
    proj,
    xy_all: np.ndarray,
    tile_m: float,
    ppm,
    *,
    max_shared_span_m: float = 2000.0,
    retry_sleep_s: float = 20.0,
    cache_dir=None,
    fetch_deadline_s: float = 180.0,
) -> list:
    """Fetch one OSM canvas per keyframe, downloading OSM data ONCE.

    ``TileManager.from_bbox`` is a full osm.org map.json download; calling it
    per keyframe means N heavily-overlapping ~tile-sized downloads, each with
    its own HTTP-509 backoff. When the route's bbox is modest
    (span <= ``max_shared_span_m``) we fetch a single TileManager covering
    the whole route (+ tile margin) and ``query`` the per-keyframe bbox
    from it. The per-keyframe path is kept as fallback for very long
    routes and for keyframes the shared manager cannot serve.

    Two hardening layers (added after a tile-256 run sat 35+ min inside one
    request): every download runs in a daemon thread under a HARD wall-clock
    deadline — osm.org streams large responses slowly enough that urllib3's
    per-op ``timeout=10`` never fires — and ``cache_dir`` persists the raw
    OSM JSON per (origin, bbox) through ``from_bbox``'s native ``path=``
    cache, so warm reruns never touch the network at all.
    """
    import hashlib
    import queue
    import threading
    import time

    def _with_deadline(fn):
        q: queue.Queue = queue.Queue()

        def run():
            try:
                q.put(("ok", fn()))
            except Exception as e:
                q.put(("err", e))

        threading.Thread(target=run, daemon=True).start()
        try:
            kind, val = q.get(timeout=fetch_deadline_s)
        except queue.Empty:
            raise TimeoutError(
                f"OSM fetch exceeded {fetch_deadline_s:.0f} s")
        if kind == "err":
            raise val
        return val

    def _retry(fn):
        # Backoff retry: the OSM API returns HTTP 509 under bursty load.
        for attempt in range(5):
            try:
                return _with_deadline(fn)
            except ValueError as e:
                if "509" in str(e) and attempt < 4:
                    time.sleep(retry_sleep_s)
                    continue
                raise

    def _json_cache(bbox):
        if cache_dir is None:
            return None
        origin = np.round(np.asarray(proj.latlonalt[:2], dtype=float), 5)
        key = hashlib.sha1(
            f"{origin.tolist()}|{np.round(bbox.min_, 1).tolist()}|"
            f"{np.round(bbox.max_, 1).tolist()}".encode()).hexdigest()[:16]
        p = Path(cache_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / f"osm_{key}.json"

    def _manager(bbox):
        path = _json_cache(bbox)
        if path is not None and path.exists():
            try:  # the containment assert can trip on sub-metre bbox drift
                return tile_manager_cls.from_bbox(proj, bbox, ppm, path=path)
            except Exception:
                path.unlink(missing_ok=True)
        return _retry(lambda: tile_manager_cls.from_bbox(
            proj, bbox, ppm, path=path))

    mn = xy_all.min(axis=0)
    mx = xy_all.max(axis=0)
    span = float(np.max(mx - mn))

    shared = None
    if span <= max_shared_span_m:
        try:
            shared = _manager(boundary_box_cls(mn, mx) + (tile_m + 10))
        except Exception as e:
            print(f"      -> shared OSM fetch unavailable ({e}); "
                  f"falling back to per-keyframe tiles")
            shared = None

    canvases = []
    for center in xy_all:
        bbox = boundary_box_cls(center, center) + tile_m
        canvas = None
        if shared is not None:
            try:
                canvas = shared.query(bbox)
            except Exception:
                canvas = None
        if canvas is None:
            canvas = _manager(bbox + 10).query(bbox)
        canvases.append(canvas)
    return canvases


def _prepare(image, camera, canvas, cfg, gravity, model):
    import torch
    from maploc.data.image import pad_image, rectify_image, resize_image
    tfl = cfg.data.resize_image / 2
    size = (camera.size * (tfl / camera.f)).round().int()
    im = torch.from_numpy(image).permute(2, 0, 1).float().div_(255)
    im, valid = rectify_image(im, camera.float(), roll=-gravity[0], pitch=-gravity[1])
    im, _, camera, *_ = resize_image(im, size.tolist(), camera=camera, valid=valid)
    stride = max(model.image_encoder.layer_strides)
    size = (np.ceil(size.numpy() / stride) * stride).astype(int)
    im, valid, camera = pad_image(im, size.tolist(), camera, crop_and_center=True)
    return {"image": im, "map": torch.from_numpy(canvas.raster).long(),
            "camera": camera.float(), "valid": valid}


def refine_route(
    frames_bgr: list,
    prior_latlon: np.ndarray,
    focal_px: float | None = None,
    *,
    fov_deg: float | None = None,
    tile_m: float = 160.0,
    gravity: tuple | None = None,
    device=None,
    cache_dir=None,
) -> np.ndarray | None:
    """Refine a coarse per-keyframe route with OrienterNet sequential fusion.

    ``frames_bgr`` and ``prior_latlon`` (N,2) are aligned per keyframe.
    The camera is calibrated AUTOMATICALLY when neither ``focal_px`` nor
    ``fov_deg`` is given: the horizontal FOV is swept and the value that
    maximises OrienterNet's own confidence (a peaked belief = a correct
    BEV) is chosen — so unknown dashcams work with no manual calibration.
    Returns refined ``(N,2)`` lat/lon, or ``None`` if OrienterNet is
    unavailable / fails.
    """
    if not _ensure_on_path() or len(frames_bgr) < 2:
        return None
    try:
        import cv2
        import torch
        from maploc.models.sequential import RigidAligner
        from maploc.osm.tiling import TileManager
        from maploc.utils.geo import BoundaryBox, Projection
        from maploc.utils.wrappers import Camera
    except Exception:
        return None

    def _camera(image, fpx):
        h, w = image.shape[:2]
        return Camera.from_dict({"model": "SIMPLE_PINHOLE", "width": w, "height": h,
                                 "params": [fpx, w / 2 + 0.5, h / 2 + 0.5]})

    def _logprob(image, canvas, fpx, grav):
        data = {k: v.to(device)[None] for k, v in
                _prepare(image, _camera(image, fpx), canvas, cfg, grav, model).items()}
        with torch.no_grad():
            return model(data)["log_probs"].squeeze(0)

    try:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg = _load_model(device)
        ppm = cfg.data.pixel_per_meter
        prior_latlon = np.asarray(prior_latlon, dtype=np.float64)
        proj = Projection(*prior_latlon[len(prior_latlon) // 2])
        xy_all = np.array([proj.project(ll) for ll in prior_latlon])
        d = np.gradient(xy_all, axis=0)
        yaw = (90.0 - np.degrees(np.arctan2(d[:, 1], d[:, 0]))) % 360.0

        # Pre-fetch the OSM canvases (ONE Overpass download covering the
        # whole route bbox when it is small enough; per-keyframe fallback
        # otherwise) + the RGB image per keyframe once.
        canvases = _fetch_canvases(TileManager, BoundaryBox, proj, xy_all,
                                   tile_m, ppm, cache_dir=cache_dir)
        prepped = []
        for i, bgr in enumerate(frames_bgr):
            image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepped.append((image, canvases[i], xy_all[i], yaw[i]))

        # AUTO-CALIBRATE the effective FOV by maximising OrienterNet's own
        # confidence (a peaked belief = a BEV that aligns to the OSM). NB the
        # confidence-maximising FOV is the BEV's *effective* footprint, not the
        # optical FOV: GeoCalib's optically correct ~84 deg scores ~2x WORSE
        # here (25.5 vs 13.6 m on Ulm) than the ~120-135 deg this finds -- so we
        # trust OrienterNet's own objective over an external calibrator. Sweeping
        # the pitch jointly by the same criterion was also tested and did NOT
        # help (16.9 vs 16.8 m), so pitch is left at the supplied dashcam tilt.
        gravity = (0.0, -4.0) if gravity is None else gravity
        if focal_px is None and fov_deg is None:
            sample = prepped[:: max(1, len(prepped) // 4)][:4]
            best = (-1e18, 120.0)
            for fov in (75, 90, 105, 120, 135, 150):
                fpx = frames_bgr[0].shape[1] / (2 * np.tan(np.deg2rad(fov) / 2))
                conf = [float(torch.log_softmax(
                    _logprob(im, cv, fpx, gravity).flatten(), 0).max())
                    for im, cv, _xy, _yw in sample]
                if float(np.mean(conf)) > best[0]:
                    best = (float(np.mean(conf)), fov)
            fov_deg = best[1]
            print(f"      -> OrienterNet auto-calibrated camera FOV: {fov_deg:.0f} deg")
        if focal_px is None:
            focal_px = frames_bgr[0].shape[1] / (2 * np.tan(np.deg2rad(fov_deg) / 2))

        per = []
        for image, canvas, xy, yw in prepped:
            per.append((_logprob(image, canvas, focal_px, gravity), canvas, xy, yw))

        # Anchor the aligner at the MIDDLE frame: its belief lives on the
        # reference frame's canvas, so the reference must be a frame whose
        # coarse prior actually contains the truth (the middle of a route
        # is far more reliable than the loop-phase-ambiguous endpoints).
        aligner = RigidAligner(num_rotations=per[0][0].shape[-1])
        order = sorted(range(len(per)), key=lambda i: abs(i - len(per) // 2))
        for i in order:
            lp, canvas, xy, yw = per[i]
            aligner.update(lp, canvas, torch.tensor(xy, device=device).float(),
                           torch.tensor(float(yw), device=device).float())
        aligner.compute()
        out = []
        for _lp, _c, xy, yw in per:
            gxy, _ = aligner.transform(torch.tensor(xy, device=device).float(),
                                       torch.tensor(float(yw), device=device).float())
            out.append(proj.unproject(gxy.cpu().numpy()))
        return np.array(out)
    except Exception:
        return None
