"""Model-free adapter from a complete frozen gallery to fixed top-eight pixels.

The full reference matrix/membership is retained. Only the already declared
local-matcher candidate pixels are materialized, without GPS/score-file input.
The source gallery's inference directory is never read or modified.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from scripts import pilot_image_gallery as gallery
from scripts.orb_rerank import stable_topk

require = gallery.require


class CandidatePixels:
    """Read-only candidate rows indexed in the original full gallery space."""
    def __init__(self, pixels, columns, reference_count):
        self.pixels = pixels
        self.columns = {int(column): row for row, column in enumerate(columns)}
        self.reference_count = int(reference_count)

    def __len__(self):
        return self.reference_count

    def __getitem__(self, index):
        require(not isinstance(index, (bool, np.bool_)) and isinstance(index, (int, np.integer)),
                "global integer reference index required")
        require(int(index) in self.columns, "reference is outside the fixed candidate-pixel union")
        return self.pixels[self.columns[int(index)]]


def stage_candidate_pixels(family, plan, columns, output, *, check=gallery.sparse.check_headroom):
    """Read at most one image at a time from each source ZIP member.

    No full raw union or raw-pool temporary is allocated. Source-file hashes
    bind the archives; per-ID pixel hashes bind every copied image row.
    """
    family, output = Path(family).resolve(strict=True), Path(output)
    cols = np.asarray(columns)
    require(cols.ndim == 1 and np.issubdtype(cols.dtype, np.integer)
            and 0 < len(cols) <= 640 and np.array_equal(cols, np.unique(cols)),
            "distinct sorted bounded top-eight union required")
    rows = plan["canonical_rows"]
    require(len(rows) <= 10000 and all(0 <= int(c) < len(rows) for c in cols), "candidate columns outside gallery")
    require(not output.exists(), "candidate pixels cannot overwrite an existing artifact")
    selected = [rows[int(c)] for c in cols]
    require(all(row["column"] == int(column) for row, column in zip(selected, cols)), "canonical column order changed")
    positions = {int(column): index for index, column in enumerate(cols)}
    seen, copied, memory = set(), [], []
    target = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint8, shape=(len(cols), 512, 512, 3))
    try:
        for source in plan["raw_sources"]:
            chosen = sorted((row for row in selected if row["source"] == source["source"]), key=lambda row: row["raw_row"])
            if not chosen:
                continue
            memory.append(check())
            archive = gallery.allowed_path(family / gallery.sparse.relative(source["copy"])).resolve(strict=True)
            require(archive.is_relative_to(family) and gallery.sha(archive) == source["sha256"], "frozen candidate archive changed")
            gallery.raw_header(archive, source["raw_shape"])
            with np.load(archive, allow_pickle=False) as saved:
                ids, xy = saved["ids"], saved["xy"]
            with zipfile.ZipFile(archive) as zipped, zipped.open("raw.npy") as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, _ = reader(stream)
                offset, row_bytes = stream.tell(), 512 * 512 * 3
                for row in chosen:
                    index, column = row["raw_row"], row["column"]
                    require(type(index) is int and 0 <= index < shape[0] and column not in seen,
                            "invalid or repeated canonical pixel row")
                    require("mapillary:" + str(ids[index]) == row["id"] and xy[index].tolist() == row["latlon"],
                            "candidate pixels differ from canonical public ID/position")
                    stream.seek(offset + index * row_bytes)
                    body = stream.read(row_bytes)
                    require(len(body) == row_bytes and hashlib.sha256(body).hexdigest() == row["pixels_sha256"],
                            "candidate pixel bytes differ from frozen per-ID ledger")
                    target[positions[column]] = np.frombuffer(body, np.uint8).reshape(512, 512, 3)
                    seen.add(column)
                    copied.append({"column": column, "id": row["id"], "pixels_sha256": row["pixels_sha256"],
                                   "archive": str(archive), "raw_row": index})
        require(seen == set(cols.tolist()), "one or more fixed candidates lack canonical pixels")
        target.flush()
        return {"candidate_count": len(cols), "reference_count": len(rows), "columns": cols.tolist(),
                "copied": copied, "shape": list(target.shape), "storage_color": "BGR",
                "minimum_headroom": gallery.minimum_memory(memory), "candidate_file_sha256": gallery.sha(output)}
    finally:
        gallery.close_map(target)


@contextmanager
def prepared_gallery_inputs(family, staging, *, expected_input_freeze_sha256):
    family, staging = Path(family).resolve(strict=True), Path(staging).resolve()
    require(not staging.exists() and not staging.is_relative_to(family)
            and not staging.is_relative_to(gallery.ROOT), "new isolated gallery staging root required")
    frozen_path = family / "inputs_frozen.json"
    require(gallery.sha(frozen_path) == expected_input_freeze_sha256, "declared gallery input freeze changed")
    gallery.verify_freeze(frozen_path)
    plan = gallery.load(family / "plan.json")
    require(plan["profile"] == gallery.PROFILE and plan["profile_sha256"] == gallery.PROFILE_SHA256,
            "unsupported prepared gallery profile")
    _, ledger, _ = gallery.frozen_inventory(family / "inputs/inventory/inventory_frozen.json")
    original_plan = gallery.load(family / "inputs/replay_plan.json")
    canonical, _ = gallery.canonical_rows(original_plan, ledger)
    require(canonical == plan["canonical_rows"], "prepared canonical ledger changed")
    control = gallery.control_inputs(family, plan)
    scores = control["raw_cosines"]
    candidates = stable_topk(scores - scores.mean(axis=0, keepdims=True), 8)
    columns = np.unique(candidates)
    staging.mkdir(parents=True, exist_ok=False)
    frames, raw, previous_temp = None, None, tempfile.tempdir
    candidate_path = staging / "candidate_raw.npy"
    try:
        pixel_record = stage_candidate_pixels(family, plan, columns, candidate_path)
        raw = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
        require(not raw.flags.writeable, "candidate pixel map must be read-only")
        native_temp = staging / "native_tmp"
        native_temp.mkdir()
        tempfile.tempdir = str(native_temp)
        frames = gallery.native_queries(family, plan, control["query_times"])
        pixel_hash = hashlib.sha256()
        for frame in frames.frames:
            frame = np.ascontiguousarray(frame)
            pixel_hash.update(str((frame.shape, frame.dtype.str)).encode())
            pixel_hash.update(memoryview(frame).cast("B"))
        query = {"pixels_sha256": pixel_hash.hexdigest(), "native_frame_shape": list(frames.frames[0].shape),
                 "fps": float(frames.fps), "frame_ordinals": plan["query"]["planned_frame_ordinals"],
                 "timestamps": control["query_times"].tolist()}
        evidence = {"similarities": scores, "ref_ids": control["ref_ids"],
                    "ref_latlon": control["ref_latlon"], "query_times": control["query_times"]}
        registration = {"source_family": str(family), "input_freeze_sha256": expected_input_freeze_sha256,
            "plan_sha256": gallery.sha(family / "plan.json"), "profile_sha256": plan["profile_sha256"],
            "ref_ids_sha256": gallery._json_sha(control["ref_ids"].tolist()),
            "ref_latlon_sha256": gallery._array_sha(control["ref_latlon"]),
            "raw_cosines_sha256": gallery._array_sha(scores), "query": query,
            "per_query_native_pixel_sha256": plan["per_query_native_pixel_sha256"], "candidate_pixels": pixel_record,
            "ground_truth_opened": False, "model_constructed": False, "new_images": False}
        def verify_inputs():
            require(gallery.sha(frozen_path) == expected_input_freeze_sha256, "original input freeze changed")
            gallery.verify_freeze(frozen_path)
            require(gallery.sha(candidate_path) == pixel_record["candidate_file_sha256"], "candidate pixels changed")
        yield {"registration": registration, "evidence": evidence, "query": query,
               "raw": CandidatePixels(raw, columns, len(control["ref_ids"])),
               "stream": enumerate(frames.frames), "verify_inputs": verify_inputs}
        verify_inputs()
    finally:
        if frames is not None:
            frames.close()
        if raw is not None:
            gallery.close_map(raw)
        candidate_path.unlink(missing_ok=True)
        tempfile.tempdir = previous_temp
