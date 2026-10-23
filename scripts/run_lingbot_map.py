"""Run LingBot-Map on a KITTI raw drive and save its camera poses.

LingBot-Map (Chen et al., arXiv 2604.14141, Apache-2.0) is a streaming
feed-forward reconstruction model that predicts per-frame camera poses. This
driver uses the repo's own image loader, model builder and post-processing
(``demo.py``: ``load_images``, ``load_model``, ``postprocess``) in streaming
mode with the SDPA attention path, so it runs without FlashInfer, on the CPU
when the GPU is busy. It writes ``lingbot_<drive>.npz`` with the world-to-camera
extrinsics (OpenCV convention), intrinsics and the source frame indices;
``scripts/eval_graph_filter.py --odometry vo_lingbot`` scores them against
OXTS and runs the road-graph particle filter on them.

    python scripts/run_lingbot_map.py --drive 0009 \
        --lingbot-root E:/monocular-osm-tools/lingbot-map \
        --weights E:/monocular-osm-tools/weights/lingbot-map.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def main(argv=None) -> int:
    import torch

    from eval_graph_filter import DRIVES

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", required=True, choices=list(DRIVES))
    ap.add_argument("--lingbot-root", type=Path, default=Path("E:/monocular-osm-tools/lingbot-map"))
    ap.add_argument("--weights", type=Path,
                    default=Path("E:/monocular-osm-tools/weights/lingbot-map.pt"))
    ap.add_argument("--stride", type=int, default=1, help="use every N-th KITTI frame (10 Hz)")
    ap.add_argument("--first-k", type=int, default=None, help="timing runs: first K frames only")
    ap.add_argument("--camera-iterations", type=int, default=4, help="the model's default is 4")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--out", type=Path, default=ROOT / "output" / "graph_filter")
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.lingbot_root))
    from demo import load_images, load_model, postprocess  # the model's own pipeline

    spec = DRIVES[args.drive]
    folder = ROOT / "data" / "kitti" / "raw" / spec["date"] / spec["name"] / "image_02" / "data"
    images, paths, _ = load_images(image_folder=str(folder), stride=args.stride,
                                   first_k=args.first_k, image_size=518, patch_size=14)
    n = int(images.shape[0])
    cfg = SimpleNamespace(mode="streaming", image_size=518, patch_size=14, enable_3d_rope=True,
                          max_frame_num=1024, kv_cache_sliding_window=64, num_scale_frames=8,
                          use_sdpa=True, camera_num_iterations=args.camera_iterations,
                          model_path=str(args.weights))
    device = torch.device("cpu")
    model = load_model(cfg, device)
    # The demo's own rule: keyframes every ceil(n/320) frames past 320 (trained on 320 views).
    kfi = 1 if n <= 320 else math.ceil(n / 320)
    t0 = time.perf_counter()
    with torch.no_grad():
        preds = model.inference_streaming(images, num_scale_frames=8, keyframe_interval=kfi,
                                          output_device=device)
    runtime = time.perf_counter() - t0
    preds, _ = postprocess(preds, images)
    ext = preds["extrinsic"].numpy().astype(np.float64)     # [S, 3, 4] world-to-camera
    intr = preds["intrinsic"].numpy().astype(np.float64)
    frame_index = np.array([int(Path(p).stem) for p in paths])
    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"lingbot_{args.drive}" + (f"_s{args.stride}" if args.stride > 1 else "") + (
        f"_first{args.first_k}" if args.first_k else "")
    np.savez_compressed(args.out / f"{tag}.npz", extrinsic=ext, intrinsic=intr,
                        frame_index=frame_index, runtime_s=runtime, keyframe_interval=kfi,
                        image_hw=np.array(images.shape[-2:]))
    print(f"{n} frames in {runtime:.0f} s ({runtime / max(n, 1):.2f} s/frame, CPU, "
          f"keyframe interval {kfi}) -> {args.out / (tag + '.npz')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
