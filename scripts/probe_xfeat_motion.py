"""Frozen paired native ORB/XFeat motion proof, no localization or labels.

Prepare is model-free. Smoke/full execution require an accepted input SHA,
clean source and external8GiB physical/true-commit process-tree watchdog.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

if __name__ == "__main__":
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[key] = "2"
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", PROJ_NETWORK="OFF", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import psutil
from monocular_osm import paired_motion as core
from monocular_osm.visual_odometry import default_intrinsics
from scripts import replay_heading_sequence as utility, xfeat_backend as released

require, sha, load, write = utility.require, utility.sha, utility.load, utility.write
METHODS = ("orb", "xfeat")
INPUT_SHAPE = (720, 1280, 3)
FRAME_COUNT, NATIVE_COUNT, STRIDE = 2950, 8850, 3
PRIOR_SHA = "132ed9f30fbf6020a2fe7043dc2ddc152c7c715fbe59ee67fd9b1b8f3ad45371"
STATE_SHA = "f1911c6e5905e5ef62b0af843faeaa8e874df4d45e9e793f2f8a8fac109220f7"
PROFILE = {"version": "paired-native-ORB-XFeat-motion-v1", "posthoc_DEV": True,
    "core_profile_sha256": core.PROFILE_SHA256, "methods": list(METHODS), "window": [0., 295.],
    "native_shape": list(INPUT_SHAPE), "fps": 30., "stride": 3, "frames": 2950, "pairs_per_method": 2949,
    "K": "unchanged native default_intrinsics1280x720/HFOV70", "feature_request": 2500, "XFeat_actual_cap": 2500,
    "ORB_actual_resource_cap": 16384, "ORB_count_semantics": "nfeatures2500 is a soft target; retain every actual point, fail resource cap without trimming",
    "ORB": "ORB_create(nfeatures2500), remaining installed defaults; crosschecked Hamming",
    "XFeat": released.LOCAL_PROFILE, "expected_model_state_sha256": STATE_SHA,
    "expected_model_state_elements": 1551625,
    "matching": "official indexed XFeat.match(min_cossim=-1), save1-cosine from forward dot matrix; no ratio/cosine cutoff",
    "input_storage": "immutable full video SHA +all selected nativeBGR/RGB SHA +parse/actual neural-input SHA; independent auditor must re-decode",
    "superseded_proposal_sha256": PRIOR_SHA,
    "storage_amendment": "Do not duplicate8.16GB native RGB; lossless video+actual pixel/tensor hashes preserves re-decodable input identity",
    "feature_storage": "all points/descriptors/detection scores per frame, both methods; previous/current only resident",
    "pair_storage": "all indices/points/distances/displacements/selection/E/pre-recover mask COPY/postmask/reasons/Rt",
    "smoke": "fixed first4 kept native frames at ordinals0,3,6,9; runtime-only, no quality adoption",
    "full_requires_accepted_smoke": True, "first4_full_must_replay_smoke_feature_and_pair_hashes": True,
    "cpu_threads": 2, "interop_threads": 1, "opencv_threads": 2, "batch_size": 1,
    "minimum_free_physical_and_true_commit_GiB": 8, "tree_RSS_target_GiB": 3,
    "runtime_wall_budget_s": 7200, "external_watchdog_required": True,
    "selected_frontend": None, "confidence": None, "policy_adopted": False,
    "metric_scale": False, "failed_pair_is_stop": False, "network_or_downloads": False,
    "ground_truth_or_localization": False}
PROFILE_SHA256 = core.json_sha(PROFILE)


def source_identity(expected, previous=None):
    require(Path(__file__).resolve() == ROOT/"scripts/probe_xfeat_motion.py" and utility.ROOT == ROOT, "executed driver escaped pin")
    sources = utility.source_identity(expected)
    for name, module in list(sys.modules.items()):
        if name.startswith(("scripts.", "monocular_osm.")) and getattr(module, "__file__", None):
            path = Path(module.__file__).resolve(); require(path.is_relative_to(ROOT), "foreign loaded project source")
            sources[path.relative_to(ROOT).as_posix()] = sha(path)
    sources["scripts/probe_xfeat_motion.py"] = sha(__file__)
    require(previous is None or sources == previous, "executed source bytes changed")
    return sources


def copy(source, target, expected):
    source, target = utility.allowed(source).resolve(strict=True), utility.allowed(target)
    require(sha(source) == expected and not target.exists(), "copy source differs or destination exists")
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, target)
    require(sha(target) == expected, "copied bytes changed")


def specification(path):
    value = load(path)
    require(isinstance(value, dict) and set(value) == {"video", "native_control", "assets", "superseded_proposal"}, "exact input roles required")
    for role, key in (("video", "path"), ("native_control", "freeze"), ("superseded_proposal", "path")):
        row = value[role]
        require(isinstance(row, dict) and set(row) == {key, "sha256"} and isinstance(row[key], str)
            and utility.digest(row["sha256"]), "explicit role path/hash required")
        utility.allowed(row[key])
    require(isinstance(value["assets"], str), "inspected local asset root required")
    utility.allowed(value["assets"])
    return value


def verified_inputs(spec):
    native = utility.allowed(spec["native_control"]["freeze"]).resolve(strict=True)
    frozen = utility.freeze(native, spec["native_control"]["sha256"])
    require(frozen.get("status") == "complete" and frozen.get("ground_truth_opened") is False
        and {"measurement.json", "native_frames.json"} <= set(frozen["artifacts"]), "complete input-only native identity control required")
    summary = load(native.parent/"measurement.json")
    prior_input = native.parent.parent/"inputs_frozen.json"
    require(sha(prior_input) == summary["inputs_sha256"], "native producer input freeze differs")
    protected = load(prior_input)["protected_sha256"]
    require(isinstance(protected, dict) and 1 <= len(protected) <= 20000, "bounded native provenance ledger required")
    paths = [(utility.allowed(p).resolve(strict=True), d) for p, d in protected.items()]
    for path, digest in paths: require(utility.digest(digest) and sha(path) == digest, "native protected input/source changed")
    plan = load(native.parent.parent/"plan.json")
    require(protected.get(str((native.parent.parent/"plan.json").resolve())) == sha(native.parent.parent/"plan.json"),
            "native plan is not bound to accepted inputs")
    video = utility.allowed(spec["video"]["path"]).resolve(strict=True)
    require(sha(video) == spec["video"]["sha256"] == plan["video_sha256"], "same frozen source video required")
    frames = load(native.parent/"native_frames.json")
    require(len(frames) == NATIVE_COUNT and all(row["ordinal"] == i and utility.digest(row["pixels_sha256"])
        and isinstance(row["t_sec"], (int, float)) and np.isfinite(row["t_sec"])
        and abs(row["t_sec"]-i/30.) <= 1e-10 for i, row in enumerate(frames)), "exact native ordinal/actualPTS grid required")
    proposal = spec["superseded_proposal"]
    require(proposal["sha256"] == PRIOR_SHA and sha(proposal["path"]) == PRIOR_SHA, "original storage proposal must be preserved")
    assets = released.inspect_assets(spec["assets"])
    return {"frames": frames, "video": video, "assets": assets,
        "native_sha256": frozen["artifacts"]["native_frames.json"], "native_root": native.parent,
        "K": default_intrinsics(INPUT_SHAPE[1], INPUT_SHAPE[0], hfov_deg=70.)}


def prepare(spec_path, family, expected_commit):
    sources = source_identity(expected_commit); utility.check_headroom()
    spec_path = utility.allowed(spec_path).resolve(strict=True); declared = specification(spec_path)
    family = utility.allowed(family).resolve()
    roots = [ROOT, Path(declared["native_control"]["freeze"]).resolve().parent.parent,
        Path(declared["assets"]).resolve(), Path(declared["video"]["path"]).resolve().parent,
        Path(declared["superseded_proposal"]["path"]).resolve().parent]
    require(not family.exists() and not any(family.is_relative_to(p) or p.is_relative_to(family) for p in roots), "new isolated proof family required")
    bundle = verified_inputs(declared); family.mkdir(parents=True)
    try:
        copy(spec_path, family/"inputs/spec.json", sha(spec_path))
        copy(bundle["video"], family/"inputs/video.mp4", declared["video"]["sha256"])
        copy(bundle["native_root"]/"native_frames.json", family/"inputs/native_frames.json", bundle["native_sha256"])
        copy(declared["superseded_proposal"]["path"], family/"inputs/original_proposal.json", PRIOR_SHA)
        for name, digest in sources.items(): copy(ROOT/name, family/"inputs/source_snapshot"/name, digest)
        write(family/"plan.json", {"profile": PROFILE, "profile_sha256": PROFILE_SHA256, "source_commit": expected_commit,
            "source_sha256": sources, "declared": declared, "assets": bundle["assets"], "K": bundle["K"].tolist(),
            "native_frames_sha256": bundle["native_sha256"], "selected_frontend": None})
        verified_inputs(declared); source_identity(expected_commit, sources)
        write(family/"inputs_frozen.json", {"status": "prepared_only", "models_have_not_run": True,
            "ground_truth_or_localization": False, "artifacts": {p.relative_to(family).as_posix(): sha(p) for p in family.rglob("*") if p.is_file()}})
        return {"status": "prepared_only", "input_freeze_sha256": sha(family/"inputs_frozen.json"), "profile_sha256": PROFILE_SHA256}
    except Exception as error:
        write(family/"preparation_failed.json", {"status": "failed", "error": str(error)}); raise


def verify_prepared(family, expected_commit, expected_inputs):
    require(utility.digest(expected_inputs), "explicit accepted input SHA required")
    utility.freeze(family/"inputs_frozen.json", expected_inputs)
    plan = load(family/"plan.json")
    require(plan["source_commit"] == expected_commit and plan["profile"] == PROFILE and plan["profile_sha256"] == PROFILE_SHA256,
            "registered profile/source differs")
    source_identity(expected_commit, plan["source_sha256"])
    require(specification(family/"inputs/spec.json") == plan["declared"], "prepared input roles changed")
    bundle = verified_inputs(plan["declared"])
    require(bundle["assets"] == plan["assets"] and bundle["K"].tolist() == plan["K"]
        and bundle["native_sha256"] == plan["native_frames_sha256"]
        and sha(family/"inputs/video.mp4") == plan["declared"]["video"]["sha256"]
        and sha(family/"inputs/native_frames.json") == bundle["native_sha256"], "prepared assets/video/grid differs")
    return plan, bundle


def resource_gate(started):
    require(time.monotonic()-started <= PROFILE["runtime_wall_budget_s"], "registered wall budget exceeded")
    row = utility.check_headroom(); info = psutil.Process().memory_info()
    require(info.rss <= 3*1024**3, "owned resident memory exceeds3GiB target")
    return {**row, "rss_bytes": info.rss, "private_bytes": getattr(info, "private", info.vms)}


def features_orb(detector, bgr):
    keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), None)
    points = np.array([key.pt for key in keypoints], np.float32).reshape(-1, 2)
    descriptors = np.zeros((0, 32), np.uint8) if descriptors is None else np.ascontiguousarray(descriptors)
    return {"points": points, "descriptors": descriptors, "scores": np.asarray([key.response for key in keypoints], np.float32)}


class IndexedXFeat(released.ReleasedXFeat):
    """Released extraction/MNN with retained scores/IDs, no matcher changes."""
    def features(self, bgr):
        tensor = self.model.parse_input(released.rgb_view(bgr))
        parse_sha = core.array_sha(tensor.detach().cpu().numpy())
        expected, _, _ = self.model.preprocess_tensor(tensor)
        expected_sha = core.array_sha(expected.detach().cpu().numpy())
        seen = []
        def actual_input(module, args):
            seen.append(core.array_sha(args[0].detach().cpu().numpy()))
            self.runtime["feature_forward_calls"] += 1
        hook = self.model.net.register_forward_pre_hook(actual_input)
        try: value = self.model.detectAndCompute(tensor, top_k=2500, detection_threshold=.05)[0]
        finally: hook.remove()
        require(seen == [expected_sha], "actual XFeat input differs from official preprocessing")
        result = {name: np.ascontiguousarray(value[key].detach().cpu().numpy()).copy()
            for name, key in (("points", "keypoints"), ("descriptors", "descriptors"), ("scores", "scores"))}
        return result, {"parse_tensor_sha256": parse_sha, "actual_network_input_sha256": seen[0],
                        "parse_shape": list(tensor.shape), "network_shape": list(expected.shape)}

    def indexed_matches(self, first, second):
        self.runtime["pair_match_requests"] += 1
        if not len(first["descriptors"]) or not len(second["descriptors"]):
            return np.empty((0, 2), np.int32), np.empty(0, np.float32)
        a, b = (self.torch.from_numpy(np.ascontiguousarray(item["descriptors"])) for item in (first, second))
        ia, ib = self.model.match(a, b, min_cossim=-1)
        self.runtime["actual_mnn_calls"] += 1
        # Save distances in the same forward cosine algebra as official match.
        cosine = a@b.t()
        distance = (1.-cosine[ia, ib]).detach().cpu().numpy().copy()
        return np.column_stack([ia.cpu().numpy(), ib.cpu().numpy()]).astype(np.int32), distance


def match_orb(matcher, first, second):
    if not len(first["descriptors"]) or not len(second["descriptors"]): return np.empty((0, 2), np.int32), np.empty(0, np.float32)
    values = matcher.match(first["descriptors"], second["descriptors"])
    return np.asarray([[v.queryIdx, v.trainIdx] for v in values], np.int32).reshape(-1, 2), np.asarray([v.distance for v in values], np.float32)


def orb_parameters(detector):
    getters = {"max_features": "getMaxFeatures", "scale_factor": "getScaleFactor", "levels": "getNLevels",
        "edge_threshold": "getEdgeThreshold", "first_level": "getFirstLevel", "WTA_K": "getWTA_K",
        "score_type": "getScoreType", "patch_size": "getPatchSize", "fast_threshold": "getFastThreshold"}
    return {name: getattr(detector, method)() for name, method in getters.items()}


def check_features(features, method):
    require(set(features) == {"points", "descriptors", "scores"}, "exact feature roles required")
    p, d, s = (features[k] for k in ("points", "descriptors", "scores"))
    require(p.ndim == 2 and p.shape[1] == 2 and p.dtype == np.float32 and len(p) <= (16384 if method == "orb" else 2500)
        and s.shape == (len(p),) and s.dtype == np.float32 and np.isfinite(p).all() and np.isfinite(s).all(), "invalid feature coordinates/scores")
    require(d.shape == (len(p), 32 if method == "orb" else 64)
        and d.dtype == (np.uint8 if method == "orb" else np.float32) and np.isfinite(d).all(), "invalid descriptor package")
    require(not len(p) or (p.min() >= 0 and p[:, 0].max() < INPUT_SHAPE[1] and p[:, 1].max() < INPUT_SHAPE[0]), "features escaped native image")


def stream_native(family, frames, count):
    cap = cv2.VideoCapture(str(family/"inputs/video.mp4"))
    emitted = 0
    try:
        require(cap.isOpened(), "native video unavailable")
        for ordinal, expected in enumerate(frames):
            ok, pixels = cap.read()
            require(ok and pixels.shape == INPUT_SHAPE and pixels.dtype == np.uint8, "native pixel shape/truncation")
            require(int(round(cap.get(cv2.CAP_PROP_POS_FRAMES)))-1 == ordinal
                and abs(cap.get(cv2.CAP_PROP_POS_MSEC)/1000.-expected["t_sec"]) <= 1e-10, "actual ordinal/PTS differs")
            require(hashlib.sha256(pixels.tobytes()).hexdigest() == expected["pixels_sha256"], "native pixel bytes differ")
            if ordinal % STRIDE == 0:
                yield emitted, pixels, {"position": emitted, "ordinal": ordinal, "t_sec": expected["t_sec"],
                    "native_BGR_bytes_sha256": expected["pixels_sha256"],
                    "RGB_bytes_sha256": hashlib.sha256(released.rgb_view(pixels).tobytes()).hexdigest()}
                emitted += 1
                if count < FRAME_COUNT and emitted == count: break
        require(emitted == count, "incomplete exact kept-frame grid")
    finally: cap.release()


def quality_summary(valid, inliers, times, reasons):
    longest = total = 0.
    for good, dt in zip(valid, np.diff(times)):
        total = 0. if good else total+float(dt); longest = max(longest, total)
    return {"accepted_edges": int(valid.sum()), "edge_count": len(valid), "accepted_edge_fraction": float(valid.mean()),
        "frame_valid_fraction_including_identity": float((valid.sum()+1)/(len(valid)+1)),
        "longest_unknown_span_s": longest, "reasons": dict(Counter(reasons)),
        "accepted_inlier_median": float(np.median(inliers[valid])) if valid.any() else None,
        "unknown_is_not_stop": True, "unit_scale_is_not_metric": True, "quality_is_not_pose_truth": True}


def smoke_proof(family, expected, expected_inputs):
    require(utility.digest(expected), "explicit accepted smoke SHA required")
    frozen = utility.freeze(family/"smoke/frozen_measurement.json", expected)
    require(frozen.get("status") == "complete" and frozen.get("runtime_only_smoke") is True
        and frozen.get("input_freeze_sha256") == expected_inputs and frozen.get("profile_sha256") == PROFILE_SHA256
        and frozen.get("frames") == 4 and frozen.get("methods") == list(METHODS), "complete fixed4-frame smoke required")
    runtime = load(family/"smoke/runtime.json"); actual = runtime["model_runtime"]
    require(runtime["actual_model_state"] == {"sha256": STATE_SHA, "element_count": 1551625}
        and runtime["input_freeze_sha256"] == expected_inputs and actual["feature_forward_calls"] == 4
        and actual["pair_match_requests"] == 3 and actual["device"] == "cpu"
        and actual["cpu_threads"] == 2 and actual["interop_threads"] == 1
        and actual["lighterglue_constructed"] is False, "accepted smoke actual model/runtime differs")
    return load(family/"smoke/frame_evidence.json"), load(family/"smoke/pair_evidence.json")


def execute(family, expected_commit, input_freeze_sha256, *, smoke=False, accepted_smoke_sha256=None):
    family = utility.allowed(family).resolve(strict=True)
    plan, bundle = verify_prepared(family, expected_commit, input_freeze_sha256)
    replay = None if smoke else smoke_proof(family, accepted_smoke_sha256, input_freeze_sha256)
    target = family/("smoke" if smoke else "measurement"); target.mkdir(exist_ok=False)
    started = time.monotonic(); count = 4 if smoke else FRAME_COUNT
    frame_records, pair_records = [], []
    try:
        before = resource_gate(started); cv2.setNumThreads(2)
        model = IndexedXFeat(bundle["assets"]["root"])
        require(model.state == {"sha256": STATE_SHA, "element_count": 1551625}
            and model.assets == bundle["assets"] and model.runtime["cpu_threads"] == 2
            and model.runtime["interop_threads"] == 1, "actual released state/runtime differs")
        write(target/"model_loaded.json", {"assets": model.assets, "actual_state": model.state,
            "runtime_at_constructor": dict(model.runtime), "memory": resource_gate(started)})
        detector, matcher = cv2.ORB_create(nfeatures=2500), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        actual_orb = orb_parameters(detector)
        require(actual_orb["max_features"] == 2500, "ORB detector feature cap differs")
        write(target/"orb_parameters.json", actual_orb)
        relatives = {m: {"R": np.full((count-1, 3, 3), np.nan), "t": np.full((count-1, 3), np.nan),
            "valid": np.zeros(count-1, bool), "n": np.zeros(count-1, np.int64), "reasons": []} for m in METHODS}
        previous, times, ordinals = None, [], []
        for row, pixels, frame in stream_native(family, bundle["frames"], count):
            resource_gate(started); require(row == len(times), "nonsequential kept frames")
            current = {}; times.append(frame["t_sec"]); ordinals.append(frame["ordinal"])
            for method in METHODS:
                begin = time.monotonic()
                if method == "orb": features, preprocessing = features_orb(detector, pixels), {}
                else: features, preprocessing = model.features(pixels)
                check_features(features, method); current[method] = features
                hashes = {k: core.array_sha(v) for k, v in features.items()}
                path = target/method/"features"/f"frame_{row:04d}.npz"; path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **features)
                record = {**frame, "method": method, "feature_count": len(features["points"]),
                    "feature_sha256": hashes, "preprocessing": preprocessing, "elapsed_s": time.monotonic()-begin}
                frame_records.append(record)
                if row:
                    indices, distance = match_orb(matcher, previous[method], features) if method == "orb" else model.indexed_matches(previous[method], features)
                    result = core.estimate_pair(previous[method]["points"], features["points"], indices, distance, bundle["K"])
                    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
                    meta = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
                    pair_path = target/method/"pairs"/f"pair_{row-1:04d}.npz"; pair_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(pair_path, **arrays)
                    pair = {**meta, "method": method, "pair": row-1, "ordinals": ordinals[-2:], "times": times[-2:],
                        "arrays_sha256": {k: core.array_sha(v) for k, v in arrays.items()}}
                    pair_records.append(pair)
                    state = relatives[method]; state["reasons"].append(result["reason"])
                    if result["status"] == "complete":
                        state["R"][row-1], state["t"][row-1] = result["R"], result["t"]
                        state["valid"][row-1], state["n"][row-1] = True, result["n_inliers"]
            previous = current
            if row % 100 == 0: print(f"paired native motion frame {row+1}/{count}", flush=True)
        require(len(frame_records) == 2*count and len(pair_records) == 2*(count-1)
            and model.runtime["feature_forward_calls"] == count and model.runtime["pair_match_requests"] == count-1,
            "incomplete paired extraction/matching")
        if replay is not None:
            expected_frames, expected_pairs = replay
            require([{k: v for k, v in r.items() if k != "elapsed_s"} for r in frame_records[:8]]
                == [{k: v for k, v in r.items() if k != "elapsed_s"} for r in expected_frames]
                and pair_records[:6] == expected_pairs, "full first4 inputs/features/poses differ from accepted smoke")
        write(target/"frame_evidence.json", frame_records); write(target/"pair_evidence.json", pair_records)
        summaries = {}
        for method in METHODS:
            state = relatives[method]
            motion = core.chain_relative_poses(state["R"], state["t"], state["valid"], state["n"])
            np.savez(target/f"motion_{method}.npz", **motion, relative_R=state["R"], relative_t=state["t"],
                relative_valid=state["valid"], timestamps=np.asarray(times), frame_ordinals=np.asarray(ordinals), K=bundle["K"])
            summaries[method] = quality_summary(state["valid"], state["n"], np.asarray(times), state["reasons"])
        model.verify_unchanged()
        write(target/"quality.json", {"methods": summaries, "selected_frontend": None, "confidence": None,
            "runtime_only_smoke": smoke, "metric_or_localization_claim": False, "invalid_is_stop": False})
        write(target/"runtime.json", {"source_commit": expected_commit, "source_sha256": plan["source_sha256"],
            "profile_sha256": PROFILE_SHA256, "input_freeze_sha256": input_freeze_sha256,
            "actual_model_state": model.state, "model_runtime": model.runtime, "assets": model.assets,
            "elapsed_s": time.monotonic()-started, "memory_before": before, "memory_end": resource_gate(started),
            "opencv_threads": cv2.getNumThreads(), "numpy": np.__version__, "opencv": cv2.__version__,
            "frames": count, "selected_frontend": None, "ground_truth_or_localization": False})
        verify_prepared(family, expected_commit, input_freeze_sha256)
        if not smoke: smoke_proof(family, accepted_smoke_sha256, input_freeze_sha256)
        write(target/"frozen_measurement.json", {"status": "complete", "input_quality_only": True, "runtime_only_smoke": smoke,
            "source_commit": expected_commit, "profile_sha256": PROFILE_SHA256, "input_freeze_sha256": input_freeze_sha256,
            "accepted_smoke_sha256": accepted_smoke_sha256, "frames": count, "methods": list(METHODS),
            "selected_frontend": None, "ground_truth_or_localization": False,
            "artifacts": {p.relative_to(target).as_posix(): sha(p) for p in target.rglob("*") if p.is_file()}})
        return {"status": "complete", "frames": count, "freeze_sha256": sha(target/"frozen_measurement.json")}
    except Exception as error:
        write(target/"failure.json", {"status": "failed", "error": f"{type(error).__name__}: {error}",
            "completed_frame_records": len(frame_records), "completed_pair_records": len(pair_records), "ground_truth_or_localization": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("prepare"); setup.add_argument("--spec", type=Path, required=True)
    for name in ("smoke", "infer"):
        command = sub.add_parser(name); command.add_argument("--input-freeze-sha256", required=True)
        if name == "infer": command.add_argument("--accepted-smoke-sha256", required=True)
    for command in sub.choices.values():
        command.add_argument("--family", type=Path, required=True); command.add_argument("--expected-commit", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    with utility.offline():
        if command == "prepare": args["spec_path"] = args.pop("spec"); result = prepare(**args)
        else: result = execute(**args, smoke=command == "smoke")
    print(json.dumps(result)); return 0


if __name__ == "__main__": raise SystemExit(main())
