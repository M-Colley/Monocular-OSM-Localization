"""Bounded native-video metric preflight for the experimental assisted policy.

Post-hoc DEV. This measures VO and the road plane; it never runs a filter or
loads a trajectory cache. Call in an isolated worker: OpenCV RNG is seeded.
"""
from __future__ import annotations
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import threading
from types import MappingProxyType

import cv2
import numpy as np

from .assisted_policy import CONFIG_SHA256, policy_config
from .frame_extraction import GrayscaleFrames, extract_frames
from .plane_speed import plane_step_lengths
from .provenance import file_sha256
from .visual_odometry import VO_ALGO_VERSION, default_intrinsics, estimate_trajectory

PROFILE = {"version": "assisted-native-metric-preflight-v1", "target_hz": 10.,
    "frame_memory_mb": 128., "max_frames": None, "frame_max_width": None,
    "hfov_deg": 70., "vo_workers": 2, "enforce_planar": True, "camera_height_m": 1.5,
    "horizon": "principal", "mask_static": False, "seed": 0,
    "validation_status": "posthoc_dev_unvalidated"}
_TEMP_LOCK = threading.RLock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


PROFILE_SHA256 = _digest(PROFILE)
PROFILE = MappingProxyType(PROFILE)


def _write(path, value):
    temporary = path.with_suffix(".part.json")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _finite(value, low=0., high=None):
    if isinstance(value, (bool, str)):
        return None
    try: value = float(value)
    except (TypeError, ValueError): return None
    return value if np.isfinite(value) and value >= low and (high is None or value <= high) else None


@contextmanager
def _frame_storage(data_dir):
    """Scope the extractor's existing temporary storage to this worker's data."""
    root = Path(data_dir) / "assisted_metric_frames"
    root.mkdir(parents=True, exist_ok=True)
    with _TEMP_LOCK:
        previous = tempfile.tempdir
        tempfile.tempdir = str(root.resolve())
        try: yield
        finally: tempfile.tempdir = previous


def _sources(output):
    root = Path(__file__).resolve().parent
    hashes = {}
    for name in ("assisted_motion.py", "assisted_policy.py", "frame_extraction.py", "visual_odometry.py",
                 "plane_speed.py", "provenance.py"):
        data = (root / name).read_bytes()
        path = output / "source_snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def _auto_reasons(auto, video_sha, window):
    thresholds, reasons = policy_config(), []
    if auto.get("status") != "complete": reasons.append("auto_vo_incomplete")
    valid = _finite(auto.get("valid_fraction"), high=1)
    duration = _finite(auto.get("duration_s"))
    if valid is None or valid < thresholds["pf_vo_valid_min"]: reasons.append("auto_vo_below_fixed_pf_minimum")
    if duration is None or duration < thresholds["pf_duration_min_s"]: reasons.append("auto_duration_below_fixed_pf_minimum")
    if auto.get("video_sha256") != video_sha or auto.get("window") != window:
        reasons.append("auto_video_or_window_identity_unproven")
    digest = auto.get("timestamps_sha256")
    if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
            or _finite(auto.get("fps"), low=np.finfo(float).tiny) is None
            or isinstance(auto.get("stride"), bool) or not isinstance(auto.get("stride"), int)
            or auto["stride"] < 1):
        reasons.append("auto_grid_identity_unproven")
    counts = [auto.get(name) for name in ("n_valid", "n_frames")]
    if (any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
            or counts[1] < 2 or not 0 <= counts[0] <= counts[1]
            or valid is None or abs(counts[0] / counts[1] - valid) > 1e-12):
        reasons.append("auto_vo_counts_inconsistent")
    artifact = auto.get("vo_artifact")
    if artifact is not None:
        try:
            if file_sha256(artifact["path"]) != artifact["sha256"]: raise ValueError
            with np.load(artifact["path"], allow_pickle=False) as data:
                actual = np.asarray(data["valid"])
                if actual.shape != (counts[1],) or int(actual.sum()) != counts[0]: raise ValueError
        except (OSError, ValueError, KeyError, TypeError):
            reasons.append("auto_vo_artifact_changed_or_inconsistent")
    return reasons


def _video_grid(video, window):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened(): raise ValueError("metric preflight cannot open supplied video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(total) or total < 1:
        raise ValueError("native preflight needs actual encoded FPS and frame count")
    start, end = window
    stride = max(1, int(round(fps / PROFILE["target_hz"])))
    first, stop = int(start * fps), min(int(total), int(end * fps) if end is not None else int(total))
    count = max(0, (stop - first + stride - 1) // stride)
    if count < 2: raise ValueError("native registered video window needs at least two frames")
    return {"fps": fps, "container_frame_count": int(total), "stride": stride,
            "first_ordinal": first, "stop_ordinal_exclusive": stop, "expected_n_frames": count}


def _finish(output, summary, registration, arrays=None):
    summary["input_signature"] = _digest({"registration_sha256": file_sha256(output / "metric_registration.json"),
        "native_quality": summary["native_quality"], "plane": summary["plane"]})
    if arrays is not None:
        arrays["input_signature"] = np.array(summary["input_signature"])
        arrays["profile_sha256"] = np.array(PROFILE_SHA256)
        temporary = output / "native_metric_inputs.part.npz"
        try:
            np.savez_compressed(temporary, **arrays)
            temporary.replace(output / "native_metric_inputs.npz")
        finally:
            temporary.unlink(missing_ok=True)
    _write(output / "metric_summary.json", summary)
    names = ["metric_registration.json", "metric_summary.json"]
    if arrays is not None: names.append("native_metric_inputs.npz")
    artifacts = {name: {"path": str((output / name).resolve()), "sha256": file_sha256(output / name)} for name in names}
    _write(output / "metric_inputs_frozen.json", {"scoring_has_not_run": True,
        "profile_sha256": PROFILE_SHA256, "policy_config_sha256": CONFIG_SHA256,
        "source_sha256": registration["source_sha256"], "artifacts": artifacts})
    artifacts["freeze"] = {"path": str((output / "metric_inputs_frozen.json").resolve()),
                           "sha256": file_sha256(output / "metric_inputs_frozen.json")}
    return {"summary": summary, "artifacts": artifacts, "native_valid_fraction": summary["native_quality"].get("valid_fraction"),
            "profile_sha256": PROFILE_SHA256, "policy_config_sha256": CONFIG_SHA256,
            "source_sha256": registration["source_sha256"]}


def prepare_metric_inputs(cfg, video_path, output_dir, data_dir, *, auto_quality=None):
    """Freeze a fresh full-window native VO/plane measurement, without PF.

    auto_quality is the preceding auto-grid measurement, including actual
    duration, counts, video/window identity and (when available) its VO file.
    Missing/weak auto evidence declines before extraction. Plane measurements
    remain truthful even when the independently measured native VO is weak.
    """
    output, video = Path(output_dir), Path(video_path).resolve(strict=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("metric preflight output must be new or empty; never re-bless a freeze")
    output.mkdir(parents=True, exist_ok=True)
    start, end = getattr(cfg, "vo_start_sec", 0.), getattr(cfg, "vo_end_sec", None)
    start = _finite(start)
    end = None if end is None else _finite(end)
    if start is None or (getattr(cfg, "vo_end_sec", None) is not None and (end is None or end <= start)):
        raise ValueError("metric preflight needs finite ordered registered video limits")
    window, auto = [start, end], deepcopy(auto_quality or {})
    registration = {"experimental": True, "input_policy": {"ground_truth": False, "external_trajectory": False,
        "calibration": False, "gps_or_oxts": False, "position_prior": False, "filter_execution": False},
        "video": {"path": str(video), "sha256": file_sha256(video)}, "window": window,
        "profile": dict(PROFILE), "profile_sha256": PROFILE_SHA256, "policy_config_sha256": CONFIG_SHA256,
        "vo_algo_version": VO_ALGO_VERSION, "auto_quality": auto, "source_sha256": _sources(output)}
    registration["dependencies"] = {}
    for name in ("numpy", "opencv-python", "scipy", "scikit-image"):
        try: registration["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: registration["dependencies"][name] = None
    _write(output / "metric_registration.json", registration)
    summary = {"status": "not_run", "auto_quality": auto, "native_quality": {"status": "not_run"},
        "plane": {"status": "not_run", "measured": False, "coverage": None, "rel_spread": None},
        "eligible": False, "reasons": _auto_reasons(auto, registration["video"]["sha256"], window)}
    if summary["reasons"]:
        return _finish(output, summary, registration)
    frames, arrays = None, None
    previous_threads, random_state = cv2.getNumThreads(), np.random.get_state()
    try:
        cv2.setNumThreads(1)
        cv2.setRNGSeed(PROFILE["seed"])
        np.random.seed(PROFILE["seed"])
        grid = _video_grid(video, window)
        registration["native_grid"] = grid
        _write(output / "metric_registration.json", registration)
        with _frame_storage(data_dir):
            frames = extract_frames(video, stride=grid["stride"], max_frames=None, start_sec=start,
                end_sec=end, memory_mb=PROFILE["frame_memory_mb"], max_width=None, sample_count=None)
        arrays = {}
        _measure_native(frames, grid, summary, arrays)
        summary["status"] = "complete"
        thresholds = policy_config()
        if summary["native_quality"]["valid_fraction"] < thresholds["pf_vo_valid_min"]:
            summary["reasons"].append("native_vo_below_fixed_pf_minimum")
        if not summary["plane"]["eligible"]:
            summary["reasons"].append(summary["plane"]["reason"])
        summary["eligible"] = not summary["reasons"]
    except Exception as error:
        summary["status"] = "failed"
        if summary["plane"]["status"] == "running":
            summary["plane"].update(status="failed", measured=False, reason="road_plane_measurement_failed")
        summary["reasons"].append(f"native_preflight_failed:{type(error).__name__}")
        summary["error"] = str(error)
    finally:
        if frames is not None: frames.close()
        cv2.setNumThreads(previous_threads)
        np.random.set_state(random_state)
    if file_sha256(video) != registration["video"]["sha256"]:
        summary["status"], summary["eligible"] = "failed", False
        summary["reasons"].append("video_changed_during_native_measurement")
    return _finish(output, summary, registration, arrays or None)


def _measure_native(frames, grid, summary, arrays):
    times = np.asarray(frames.timestamps, dtype="<f8")
    n = len(frames.frames)
    if (n != grid["expected_n_frames"] or times.shape != (n,) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or abs(float(frames.fps) - grid["fps"]) > 1e-9):
        raise ValueError("native extraction must cover the complete declared frame grid and actual timebase")
    pixels = hashlib.sha256()
    shape = np.asarray(frames.frames[0]).shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("native extraction must retain the actual BGR image grid")
    for frame in frames.frames:
        frame = np.asarray(frame)
        if frame.shape != shape or frame.dtype != np.uint8:
            raise ValueError("native frame grid or pixel type changed within the video")
        frame = np.ascontiguousarray(frame)
        pixels.update(str((frame.shape, frame.dtype.str)).encode())
        pixels.update(memoryview(frame).cast("B"))
    K = default_intrinsics(shape[1], shape[0], hfov_deg=PROFILE["hfov_deg"])
    gray = GrayscaleFrames(frames.frames)
    trajectory = estimate_trajectory(gray, K, n_workers=PROFILE["vo_workers"], enforce_planar=True)
    expected = {"centers": (n, 3), "xz": (n, 2), "valid": (n,), "n_inliers": (n,),
                "rotations": (n, 3, 3), "translations": (n, 3)}
    for key, dimensions in expected.items():
        value = np.asarray(getattr(trajectory, key))
        if value.shape != dimensions or value.dtype.kind not in "bifu" or not np.isfinite(value).all():
            raise ValueError(f"native trajectory field {key} does not bind the extracted frame grid")
        if key == "valid" and value.dtype.kind != "b":
            raise ValueError("native VO validity must be actual boolean frame measurements")
        arrays[key] = value.astype(bool if key == "valid" else "<i8" if key == "n_inliers" else "<f8")
    arrays.update(timestamps=times, K=np.asarray(K, dtype="<f8"),
                  frame_ordinals=grid["first_ordinal"] + np.arange(n, dtype="<i8") * grid["stride"])
    valid = arrays["valid"]
    summary["native_quality"] = {"status": "complete", "n_valid": int(valid.sum()), "n_frames": n,
        "valid_fraction": float(valid.mean()), "stride": grid["stride"], "fps": float(frames.fps),
        "duration_s": float(times[-1] - times[0]), "first_s": float(times[0]), "last_s": float(times[-1]),
        "timestamps_sha256": _array_sha(times), "pixels_sha256": pixels.hexdigest(),
        "frame_ordinals_sha256": _array_sha(arrays["frame_ordinals"]), "native_shape": list(shape),
        "K_sha256": _array_sha(arrays["K"]), "vo_algo_version": VO_ALGO_VERSION,
        "full_declared_grid": True}
    summary["plane"].update(status="running", measured=False)
    speed = plane_step_lengths(gray, trajectory, K, camera_height_m=PROFILE["camera_height_m"],
        horizon=PROFILE["horizon"], mask_static=PROFILE["mask_static"], seed=PROFILE["seed"])
    for key in ("lengths_m", "raw_m", "inliers"):
        value = np.asarray(getattr(speed, key))
        if value.shape != (n - 1,) or value.dtype.kind not in "fiu":
            raise ValueError("road-plane arrays must bind every native VO step")
        if key != "raw_m" and (not np.isfinite(value).all() or np.any(value < 0)):
            raise ValueError("measured road-plane steps/inliers must be finite and nonnegative")
        if key == "raw_m" and np.isinf(value).any():
            raise ValueError("road-plane missing measurements must be NaN, not infinity")
        arrays["plane_" + key] = value.astype("<i8" if key == "inliers" else "<f8")
    coverage, spread = _finite(speed.coverage, high=1), _finite(speed.rel_spread)
    thresholds = policy_config()
    eligible = (coverage is not None and coverage >= thresholds["pf_plane_coverage_min"]
                and spread is not None and spread <= thresholds["pf_plane_rel_spread_max"])
    reason = ("measured_plane_meets_fixed_policy" if eligible else "measured_plane_coverage_below_fixed_minimum"
              if coverage is None or coverage < thresholds["pf_plane_coverage_min"] else "measured_plane_spread_missing_or_above_fixed_maximum")
    summary["plane"] = {"status": "complete", "measured": True, "coverage": coverage, "rel_spread": spread,
        "eligible": bool(eligible), "reason": reason, "camera_height_m": float(speed.camera_height_m), "n_steps": n - 1}


def load_metric_inputs(prepared, video_path, *, window):
    """Verify a complete eligible frozen package before any particle inference.

    Returns (Trajectory, PlaneSpeed, actual timestamps, K); no image extraction
    or graph is needed to reuse the independently measured metric odometry.
    """
    from .plane_speed import PlaneSpeed
    from .visual_odometry import Trajectory
    freeze = prepared["artifacts"]["freeze"]
    freeze_path = Path(freeze["path"]).resolve(strict=True)
    if file_sha256(freeze_path) != freeze["sha256"]:
        raise ValueError("native metric freeze changed")
    output = freeze_path.parent
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (frozen.get("scoring_has_not_run") is not True or frozen.get("profile_sha256") != PROFILE_SHA256
            or frozen.get("policy_config_sha256") != CONFIG_SHA256):
        raise ValueError("native metric package policy/profile is incompatible")
    for record in frozen["artifacts"].values():
        path = Path(record["path"]).resolve(strict=True)
        if not path.is_relative_to(output) or file_sha256(path) != record["sha256"]:
            raise ValueError("native metric frozen artifact changed or escaped its package")
    registration = json.loads((output / "metric_registration.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "metric_summary.json").read_text(encoding="utf-8"))
    if (summary.get("eligible") is not True or summary.get("status") != "complete"
            or registration["window"] != list(window)
            or registration["video"]["sha256"] != file_sha256(video_path)):
        raise ValueError("native metric package is ineligible or identifies different video/window")
    for name, expected in registration["source_sha256"].items():
        path = (output / "source_snapshot" / name).resolve(strict=True)
        if (not path.is_relative_to(output / "source_snapshot") or file_sha256(path) != expected
                or file_sha256(Path(__file__).resolve().parent / name) != expected):
            raise ValueError("native metric source identity changed")
    with np.load(output / "native_metric_inputs.npz", allow_pickle=False) as data:
        if (str(data["input_signature"]) != summary["input_signature"]
                or str(data["profile_sha256"]) != PROFILE_SHA256
                or _array_sha(data["timestamps"]) != summary["native_quality"]["timestamps_sha256"]):
            raise ValueError("native metric arrays lost their input/grid identity")
        trajectory = Trajectory(**{key: data[key].copy() for key in
            ("centers", "xz", "valid", "n_inliers", "rotations", "translations")})
        speed = PlaneSpeed(**{key: data["plane_" + key].copy() for key in ("lengths_m", "raw_m", "inliers")},
            coverage=summary["plane"]["coverage"], rel_spread=summary["plane"]["rel_spread"],
            camera_height_m=PROFILE["camera_height_m"])
        return trajectory, speed, data["timestamps"].copy(), data["K"].copy()
