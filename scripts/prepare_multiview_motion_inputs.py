"""Guarded Root-only native input replay; publishes no geometry or partial master."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from monocular_osm import multiview_motion_inputs as m


def available_commit():
    if os.name == "nt":
        import ctypes as c
        class Info(c.Structure):
            _fields_ = [("cb", c.c_ulong)] + [(key, c.c_size_t) for key in (
                "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
                "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
                (key, c.c_ulong) for key in ("HandleCount", "ProcessCount", "ThreadCount")]
        info = Info()
        info.cb = c.sizeof(info)
        m.require(c.WinDLL("psapi", use_last_error=True).GetPerformanceInfo(c.byref(info), info.cb),
                  "actual commit measurement unavailable")
        return int(info.CommitLimit-info.CommitTotal)*int(info.PageSize)
    fields = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if parts[0] in ("CommitLimit:", "Committed_AS:"):
            fields[parts[0][:-1]] = int(parts[1])*1024
    return fields["CommitLimit"]-fields["Committed_AS"]


class RuntimeGuard:
    def __init__(self):
        import psutil
        self.psutil = psutil
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.observations = 0
        if hasattr(self.process, "cpu_affinity"):
            cpus = self.process.cpu_affinity()
            m.require(len(cpus) >= 2, "CPU2 affinity unavailable")
            self.process.cpu_affinity(cpus[:2])

    def __call__(self):
        m.require(time.monotonic()-self.started <= 1800, "inner1800 deadline")
        rss = sum(p.memory_info().rss for p in [self.process, *self.process.children(recursive=True)])
        self.peak_rss = max(self.peak_rss, rss)
        self.observations += 1
        m.require(rss <= 3 << 30, "RSS3GiB cap")
        m.require(self.psutil.virtual_memory().available >= 8 << 30 and available_commit() >= 8 << 30,
                  "physical/commit8GiB headroom")


def original_clock(ledger):
    import numpy as np
    read = ledger.read
    basis = read["basis_refs"]
    m.require(set(basis) == {"native_vo", "native_clock", "native_facts", "native_root_acceptance", "native_replay_config"},
              "exact original lineage roles required")
    paths = {key: ledger.add(role) for key, role in basis.items()}
    clock = m.load_named_npz(paths["native_clock"], ("timestamps", "K"))
    signed = m.load_named_npz(paths["native_vo"], ("cache_signature", "vo_algo_version"))
    s = signed["cache_signature"]
    v = signed["vo_algo_version"]
    m.require(s.shape == () and s.dtype.kind == "U" and s.nbytes <= m.MAX_JSON
              and v.shape == () and v.dtype.kind in "iu" and int(v) == 3,
              "original scalar cache signature/version required")
    signature = m.strict_json(str(s.item()).encode())
    extraction = signature["extraction"]
    n = read["native_dense_count"]
    times, K = clock["timestamps"], clock["K"]
    m.require(times.dtype == np.dtype("float64") and times.shape == (n,) and np.isfinite(times).all()
              and np.all(times > 0) and np.all(np.diff(times) > 0), "exact positive native F64 clock required")
    raw_sha = hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()
    m.require(m.typed(times) == read["full_signed_native_clock_typed_sha256"]
              and raw_sha == read["full_signed_native_clock_raw_F64_sha256"]
              and raw_sha == extraction["timestamps_sha256"], "full signed native clock differs")
    video = read["video_binding"]
    m.require(signature.get("schema") == 2 and signature.get("vo_algo_version") == 3
              and type(extraction.get("n_frames")) is int and extraction["n_frames"] == n
              and extraction.get("stride") == read["original_video_ordinal_stride"]
              and signature.get("frames_sha256") == read["full_signed_native_frames_sha256"]
              and signature["video"]["path"] == os.path.normcase(str(m.safe(video["path"])))
              and signature["video"]["size"] == video["size"], "original video/extraction signature differs")
    h, w, _ = read["expected_native_shape"]
    f = (w/2.)/np.tan(np.deg2rad(70.)/2.)
    expected = np.array([[f, 0., w/2.], [0., f, h/2.], [0., 0., 1.]], dtype=np.float64)
    m.require(K.dtype == np.dtype("float64") and np.array_equal(K, expected)
              and m.typed(K) == read["native70_K_typed_sha256"], "original assumed70 native K differs")
    for row in read["windows"]:
        m.require(np.array_equal(times[row["native_read_rows"]], row["signed_decoder_times_s"]), "ledger selected clock differs")
    return times, K, signature


def execute(config_path, config_sha256, *, capture_factory=None, cv2_module=None, check=None):
    # These environment controls precede the production NumPy/OpenCV imports.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    check = RuntimeGuard() if check is None else check
    check()
    config, ledger = m.load_config(config_path, config_sha256, "prepare")
    m.require(Path(m.__file__).resolve() == ROOT/"monocular_osm/multiview_motion_inputs.py", "foreign protocol origin")
    output = m.safe(config["output_dir"])
    m.require(not output.exists(), "exclusive new prepare output required")
    read = ledger.read
    video = read["video_binding"]
    m.require(set(video) == {"path", "size", "sha256"} and type(video["size"]) is int and video["size"] > 0,
              "exact encoded video binding required")
    video_path = ledger.add({"path": video["path"], "sha256": video["sha256"]})
    m.require(video_path.stat().st_size == video["size"], "encoded video size differs")
    times, K, signature = original_clock(ledger)
    import numpy as np
    if cv2_module is None:
        import cv2 as cv2_module
    cv = cv2_module
    cv.setNumThreads(2)
    m.require(hasattr(cv, "CAP_PROP_N_THREADS"), "supported decoder CPU2 control required")
    factory = capture_factory or cv.VideoCapture
    check()
    output.mkdir(parents=True, exist_ok=False)
    cap = factory(str(video_path), cv.CAP_FFMPEG, [cv.CAP_PROP_N_THREADS, 2])
    dense_hash = hashlib.sha256()
    ring = deque(maxlen=7)
    records = []
    total_bytes = 0
    started = time.monotonic()
    first = read["first_native_video_ordinal"]
    last = read["last_native_video_ordinal"]
    stride = read["original_video_ordinal_stride"]
    ends = {row["native_read_rows"][-1]: row for row in read["windows"]}
    n = 0
    try:
        m.require(cap.isOpened() and cap.get(cv.CAP_PROP_N_THREADS) == 2, "bounded decoder did not open with CPU2")
        extraction = signature["extraction"]
        m.require(cap.get(cv.CAP_PROP_FPS) == extraction["fps"]
                  and cap.get(cv.CAP_PROP_FRAME_COUNT) == extraction["total_frames"], "original decoder metadata differs")
        if first:
            m.require(cap.set(cv.CAP_PROP_POS_FRAMES, first), "cannot seek exact registered first ordinal")
        for ordinal in range(first, last+1):
            check()
            ok, frame = cap.read()
            m.require(ok and cap.get(cv.CAP_PROP_POS_FRAMES) == float(ordinal+1), "native ordinal shifted/truncated")
            m.require(isinstance(frame, np.ndarray) and frame.dtype == np.dtype("uint8")
                      and list(frame.shape) == read["expected_native_shape"] and frame.flags.c_contiguous,
                      "full native BGR shape/dtype/layout differs")
            if (ordinal-first) % stride:
                continue
            ms = cap.get(cv.CAP_PROP_POS_MSEC)
            m.require(type(ms) in (float, int) and np.isfinite(ms) and ms > 0
                      and float(ms/1000) == float(times[n]), "positive actual decoder clock differs from cached F64")
            dense_hash.update(json.dumps([list(frame.shape), frame.dtype.str]).encode())
            dense_hash.update(memoryview(frame).cast("B"))
            ring.append(frame.copy())
            if n in ends:
                original = ends[n]
                m.require(len(ring) == 7, "complete original read window missing")
                arrays = {"frames_bgr": np.stack(tuple(ring)),
                          "video_ordinals": np.asarray(original["video_ordinals"], dtype=np.int64),
                          "timestamps": np.asarray(original["signed_decoder_times_s"], dtype=np.float64),
                          "K": K.copy(), "native_read_rows": np.asarray(original["native_read_rows"], dtype=np.int64)}
                headers = {key: m.typed_header(value) for key, value in arrays.items()}
                frame_hashes = [m.typed(value) for value in arrays["frames_bgr"]]
                name = output/f"window_{original['window_id']:02d}.npz"
                # Worst-case payload plus fixed NPY/ZIP overhead is reserved before writing.
                reserve = sum(value.nbytes for value in arrays.values()) + 65536
                m.require(reserve <= m.MAX_NPZ and total_bytes+reserve <= m.MAX_PREPARED,
                          "window/total prepared storage reserve exceeded")
                with name.open("xb") as f:
                    np.savez(f, **arrays)
                    f.flush()
                    os.fsync(f.fileno())
                size = name.stat().st_size
                m.require(size <= reserve, "actual NPZ exceeds reserved bytes")
                m.zip_preflight(name, m.WINDOW_MEMBERS)
                role = m.ref(name)
                ledger.add(role)
                records.append(dict(original, array_ref=role, bytes=size, typed_headers=headers,
                                    frame_typed_sha256=frame_hashes))
                total_bytes += size
                del arrays
            n += 1
    finally:
        cap.release()
        ring.clear()
    m.require(n == read["native_dense_count"] and len(records) == 32
              and dense_hash.hexdigest() == read["full_signed_native_frames_sha256"],
              "whole original dense pixel hash/count or all32 windows differs")
    manifest = {"schema": m.PREPARED_SCHEMA, "status": m.PREPARED_STATUS,
                "study_registration_ref": config["study_registration_ref"],
                "native_read_ledger_ref": config["native_read_ledger_ref"],
                "source_hashes": config["source_hashes"], "prepare_config_ref": {"path": str(m.safe(config_path)), "sha256": config_sha256},
                "video_binding": video, "windows": records, "window_count": 32, "total_npz_bytes": total_bytes,
                "full_dense_native_pixels_clock_verified": True, "no_geometry_computed": True,
                "full_native_proof": {"dense_count": n, "raw_ordinal_read_count": last-first+1,
                    "frames_sha256": dense_hash.hexdigest(), "frames_hash_prefix": "json.dumps([list(shape),dtype.str]) default separators",
                    "clock_typed_sha256": m.typed(times), "clock_raw_F64_sha256": hashlib.sha256(times.astype('<f8').tobytes()).hexdigest(),
                    "native70_K_typed_sha256": m.typed(K), "positive_CAP_PROP_POS_MSEC_exact_cached_F64": True,
                    "native_shape": read["expected_native_shape"], "decoder_threads": 2,
                    "independent_container_PTS_verified": False, "K_is_measured_calibration": False},
                "elapsed_decode_s": time.monotonic()-started, "geometry_computed": False,
                "runtime_versions": {"numpy": np.__version__, "opencv": getattr(cv, "__version__", "private-fake")},
                "resource_contract": {"cpu_threads": 2, "rss_bytes": 3 << 30, "headroom_bytes": 8 << 30,
                                      "inner_s": 1800, "outer_s": 1830, "external_guard_required": True},
                "neural_forward_run": False, "ground_truth_used": False, "policy_admission": False,
                "external_actor_closure_verified": False, "protected_hashes": dict(ledger.protected)}
    body = m.json_bytes(manifest)
    m.require(len(body) <= m.MAX_JSON and total_bytes+len(body) <= m.MAX_PREPARED, "complete prepared byte cap")
    check()
    m.output_membership(output, [f"window_{j:02d}.npz" for j in range(32)])
    ledger.verify()
    # No callbacks or fresh expected hashes after the final expected-byte pass.
    with (output/"manifest.json").open("xb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--config-sha256", required=True)
    args = p.parse_args(argv)
    execute(args.config, args.config_sha256)


if __name__ == "__main__":
    main()
