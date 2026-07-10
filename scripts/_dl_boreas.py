"""Download Boreas Glen Shields camera video + pose GT from the public S3."""
import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

SEQ = "boreas-2020-11-26-13-58"
OUT = "data/ext_raw/boreas"
s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED))
xfer = TransferConfig(use_threads=True, max_concurrency=8)
keys = [
    f"{SEQ}/raw_video.mp4",                    # forward camera, 608 MB
    f"{SEQ}/applanix/gps_post_process.csv",    # 200 Hz WGS84 (radians) + ENU
    f"{SEQ}/applanix/camera_poses.csv",        # per-image ENU pose
    f"{SEQ}/calib/P_camera.txt",
]
for k in keys:
    dst = os.path.join(OUT, k)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print(f"[boreas] downloading {k} ...", flush=True)
    s3.download_file("boreas", k, dst, Config=xfer)
    print(f"[boreas] done {k} ({os.path.getsize(dst)/1e6:.1f} MB)", flush=True)
print("[boreas] ALL DONE", flush=True)
