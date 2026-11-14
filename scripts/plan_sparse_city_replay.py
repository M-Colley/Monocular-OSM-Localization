"""Freeze an input-only replay plan over every completed archived city call.

Planning reads small reference-ID/coordinate members and hashes embeddings;
it never loads pixel members or runs retrieval, VO, a model, or evaluation.
The legacy embedding encoding identity remains explicitly unverified.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from monocular_osm.sparse_sequence import PROFILE, PROFILE_SHA256, _array_sha, _json_sha


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2): value.update(block)
    return value.hexdigest()


def require(condition, message):
    if not condition: raise ValueError(message)


def header(path, key):
    with zipfile.ZipFile(path) as archive:
        with archive.open(key + ".npy") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
            else: raise ValueError("unsupported reference NPY header")
        require(not dtype.hasobject, "reference metadata cannot require pickle")
        return {"shape": list(shape), "dtype": dtype.str, "zip_crc32": archive.getinfo(key + ".npy").CRC}


def call_records(result):
    """Only configuration/cache topology is read, never score/eval/GT fields."""
    config = result["provenance"]["config"]
    require(not any(config.get(key) for key in ("ground_truth_waypoints", "ground_truth_streets", "estimated_length_m",
        "openvo_trajectory_path", "vggt_long_trajectory_path", "ocr_video_path"))
        and (config.get("osm_around") is None or config.get("osm_around_source") == "video_title"),
        "original reference acquisition contains known external/GT/manual spatial priors")
    for token in result["provenance"].get("argv") or []:
        flag = str(token).split("=", 1)[0]
        require("ground_truth" not in str(token).replace("\\", "/").lower()
            and not (flag.startswith("--") and len(flag) > 2 and any(name.startswith(flag)
                for name in ("--osm-around", "--estimated-length-m", "--ground-truth", "--ground-truth-waypoints"))),
            "original acquisition command contains known label/manual inputs")
    source, data = config["vpr_source"], Path(config["data_dir"])
    require(source == "mapillary", "this archived metadata schema currently supports Mapillary only")
    cover = result["city_cover_retrieval"]
    coarse, fine = cover["coarse_regions"], cover["fine_regions"]
    require(coarse and len(coarse) == cover["n_cells"] <= 64
            and all(r["status"] == "ok" for r in coarse + fine), "all original city reference calls must have completed")
    calls = [{"kind": "ordinary_pass1", "cache": str(data / source)}]
    if result.get("vpr_coarse_to_fine") is not None:
        calls.append({"kind": "ordinary_pass2", "cache": str(data / (source + "_c2f"))})
    for phase, rows in (("coarse", coarse), ("fine", fine)):
        calls.extend({"kind": phase, **{key: r[key] for key in ("cache", "center", "radius_m", "ref_cap", "cell_index")}}
                     for r in rows)
    decoding = result.get("vpr_decoding") or {}
    stats = [decoding.get("pass1"), *([decoding.get("pass2")] if len(calls) == len(coarse) + len(fine) + 2 else []),
             *[row.get("track_decoding") for row in coarse + fine]]
    require(len(stats) == len(calls) and all(isinstance(row, dict) and row.get("n_queries") == 80
            and type(row.get("n_references")) is int and row["n_references"] > 0 for row in stats),
            "an original completed call lacks its actual80-query/reference-count record")
    for call, stats in zip(calls, stats):
        call["original_decoder"] = {key: stats.get(key) for key in ("decoder", "n_queries", "n_references")}
    for call in calls:
        relative = Path(call["cache"]).relative_to(data)
        require(".." not in relative.parts, "cache escaped the input data leaf")
        call["archive_relative"] = str(relative)
    require(len({c["archive_relative"] for c in calls}) == len(calls), "duplicate original cache call")
    whitelist = {key: config.get(key) for key in ("city", "data_dir", "video_path", "frame_stride", "vo_start_sec",
        "vo_end_sec", "max_frames", "frame_max_width", "vpr_source")}
    return calls, whitelist, cover.get("baseline", {}).get("backbone")


def inspect_pool(directory, call):
    metadata = sorted(directory.glob("mly_ref_meta*.json"))
    require(len(metadata) == 1, "original call has missing or ambiguous fetch metadata")
    meta = json.loads(metadata[0].read_text(encoding="utf8"))
    refs, signature = meta["refs"], meta["signature"]
    require(0 < len(refs) <= 100000, "original listing is empty or exceeds the replay bound")
    if "center" in call:
        expected = {"center": [round(float(x), 4) for x in call["center"]],
                    "radius_m": float(call["radius_m"]), "cap": int(call["ref_cap"])}
        require(signature == expected, "archived fetch signature differs from the original call")
    store = directory / "ref_img_store.npz"
    ih, xh = header(store, "ids"), header(store, "xy")
    require(len(ih["shape"]) == 1 and 0 < ih["shape"][0] <= 100000
            and xh["shape"] == [ih["shape"][0], 2], "reference identity members have invalid dimensions")
    with np.load(store, allow_pickle=False) as arrays:
        ids, xy = arrays["ids"].astype(str), np.asarray(arrays["xy"], float)
    require(len(set(ids.tolist())) == len(ids) and np.isfinite(xy).all(), "reference store IDs or positions are invalid")
    lookup = dict(zip(ids.tolist(), xy))
    served = [ref for ref in refs if str(ref["id"]) in lookup]
    served_ids = ["mapillary:" + str(ref["id"]) for ref in served]
    coords = np.asarray([lookup[str(ref["id"])] for ref in served], float)
    require(served and len(set(served_ids)) == len(served_ids)
            and np.array_equal(coords, [[ref["lat"], ref["lon"]] for ref in served])
            and np.all(np.abs(coords[:, 0]) <= 90) and np.all(np.abs(coords[:, 1]) <= 180),
            "servable IDs are not bound to the original ordered metadata positions")
    if "original_decoder" in call:
        require(len(served) == call["original_decoder"]["n_references"], "servable pool changed after the completed original call")
    fingerprint = hashlib.sha1(json.dumps([[r["id"], r["lat"], r["lon"]] for r in served]).encode()).hexdigest()
    embeddings = list(directory.glob("ref_emb_megaloc_" + fingerprint[:10] + ".npz"))
    require(len(embeddings) == 1, "matching original embedding fingerprint is missing")
    embedding = embeddings[0]
    eh = header(embedding, "emb")
    require(len(eh["shape"]) == 2 and eh["shape"][0] == len(served)
            and 0 < eh["shape"][1] <= 32768 and eh["dtype"] == "<f4", "cached embedding shape/dtype is invalid")
    with np.load(embedding, allow_pickle=False) as arrays:
        require(str(arrays["fingerprint"]) == fingerprint, "cached embedding identifies another ordered subset")
    return {**call, "archive": str(directory.resolve()), "metadata": {"path": str(metadata[0].resolve()), "sha256": sha(metadata[0])},
        "fetch_signature": signature, "listed_count": len(refs), "served_count": len(served),
        "ref_ids": served_ids, "ref_latlon": coords.tolist(), "ref_ids_sha256": _json_sha(served_ids),
        "ref_latlon_sha256": _array_sha(coords), "ordered_servable_fingerprint": fingerprint,
        "reference_store": {"path": str(store.resolve()), "ids_sha256": _json_sha(ids.tolist()),
            "xy_sha256": _array_sha(xy), "ids_header": ih, "xy_header": xh, "pixels_accessed": False},
        "embeddings": {"path": str(embedding.resolve()), "sha256": sha(embedding), "bytes": embedding.stat().st_size,
                       "header": eh, "encoding_identity": "unverified: legacy cache stores ordered reference fingerprint, not model revision/preprocessing"}}


def video_grid(video, config):
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        require(capture.isOpened(), "registered encoded video cannot be opened")
        fps, count = float(capture.get(cv2.CAP_PROP_FPS)), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally: capture.release()
    stride = config["frame_stride"]
    require(np.isfinite(fps) and fps > 0 and count > 0 and width > 0 and height > 0 and type(stride) is int and stride > 0
            and config["max_frames"] is None and config["frame_max_width"] is None, "full registered native grid is unavailable")
    first = int(config["vo_start_sec"] * fps)
    stop = min(count, int(config["vo_end_sec"] * fps) if config["vo_end_sec"] is not None else count)
    dense = max(0, (stop - first + stride - 1) // stride)
    require(dense >= 80, "registered video has fewer than the fixed80 planned query frames")
    sampled = np.linspace(0, dense - 1, 80).astype(int)
    return {"video": {"path": str(video.resolve()), "sha256": sha(video)}, "fps": fps,
        "container_frame_count": count, "native_shape_hw": [height, width],
        "registered_window": [config["vo_start_sec"], config["vo_end_sec"]],
        "stride": stride, "dense_count": dense, "sample_count": 80, "query_dense_indices": sampled.tolist(),
        "planned_frame_ordinals": (first + sampled * stride).tolist(),
        "timing_requirement": "capture actual decoded PTS/encoded-fps times; verify every ordinal before inference",
        "query_mode": "crops", "views_per_query": 3, "native_pixels": True, "reference_images_fetched": False}


def plan(result_path, archive, video, output, *, expected_calls=None):
    output, archive = Path(output), Path(archive).resolve(strict=True)
    require(not output.exists(), "refuse to overwrite an existing replay plan")
    result_path = Path(result_path).resolve(strict=True)
    calls, config, backbone = call_records(json.loads(result_path.read_text(encoding="utf8")))
    if expected_calls is not None: require(len(calls) == expected_calls, "original call count differs from the registered replay")
    actual = {str(p.parent.relative_to(archive)) for p in archive.rglob("ref_emb_megaloc*.npz")}
    require(actual == {c["archive_relative"] for c in calls}, "archive omits or adds an original reference call")
    pools = [inspect_pool(archive / call["archive_relative"], call) for call in calls]
    dimensions = {pool["embeddings"]["header"]["shape"][1] for pool in pools}
    require(len(dimensions) == 1, "reference descriptor dimensions differ between original calls")
    unique = {}
    for pool in pools:
        for identifier, coords in zip(pool["ref_ids"], pool["ref_latlon"]):
            require(identifier not in unique or unique[identifier] == coords, "duplicate reference ID has conflicting positions")
            unique.setdefault(identifier, coords)
    ordered_ids = sorted(unique)
    value = {"experimental": True, "validation_status": "posthoc_dev_unvalidated", "scoring_has_not_run": True,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_pin_required_before_models": True,
        "source": {str(path.relative_to(ROOT)): sha(path) for path in (Path(__file__).resolve(), ROOT / "monocular_osm/sparse_sequence.py")},
        "original_result": {"path": str(result_path), "sha256": sha(result_path), "fields_used": "configuration and completed call/cache topology only"},
        "query": video_grid(Path(video), config), "original_reported_backbone": backbone,
        "decoder_profile": json.loads(json.dumps(dict(PROFILE))), "decoder_profile_sha256": PROFILE_SHA256,
        "pools": pools, "call_count": len(pools), "total_served_occurrences": sum(pool["served_count"] for pool in pools),
        "union": {"count": len(ordered_ids), "ref_ids": ordered_ids, "ref_ids_sha256": _json_sha(ordered_ids),
            "ref_latlon_sha256": _array_sha(np.asarray([unique[key] for key in ordered_ids], float)),
            "order": "lexicographic public reference ID", "duplicate_vector_policy": "verify all occurrences agree within1e-6; otherwise refuse whole replay"},
        "execution_plan": {"status": "not_started", "cpu_threads": 4, "min_free_commit_gib": 8,
            "new_reference_fetches": False, "new_reference_forwards": False, "fresh_query_forwards": "80queries x3crops, one shared query descriptor cache",
            "memory": "stream one archived embedding pool at a time into deduplicated union; never load raw pixel members",
            "freeze_before_labels": "all32pool identities, fresh query pixels/descriptors/times, complete raw matrix, both fixed predictions and source/runtime hashes",
            "selected_variant": None},
        "limitations": ["Legacy reference encoding identity is unverified; filename and content fingerprint do not prove the model revision/preprocessing.",
            "All completed calls are reused uniformly without old similarity/route ranking; original fine-cell density was visually selected by the earlier hierarchy.",
            "Fresh crop queries, pooled references and sparse decoding all differ from the original regional warp answer; this is not a decoder-only A/B.",
            "Sparse top64 candidate pruning may remove the globally best full-reference path; neither output is calibrated or adopted by assisted-policy-v1."]}
    output.mkdir(parents=True)
    path = output / "replay_plan.json"
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")
    artifacts = {path.name: sha(path)}
    for name, expected in value["source"].items():
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        require(sha(target) == expected, "planner source changed while freezing its plan")
        artifacts[str(target.relative_to(output))] = expected
    frozen = {"scoring_has_not_run": True, "models_have_not_run": True, "plan_sha256": sha(path), "artifacts": artifacts}
    (output / "plan_frozen.json").write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf8")
    return value, frozen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-calls", type=int)
    args = parser.parse_args(argv)
    value, frozen = plan(args.result, args.archive, args.video, args.output, expected_calls=args.expected_calls)
    print(json.dumps({"status": "planned_only", "call_count": value["call_count"], "union_count": value["union"]["count"],
                      "plan_sha256": frozen["plan_sha256"], "models_have_not_run": True}))


if __name__ == "__main__": main()
