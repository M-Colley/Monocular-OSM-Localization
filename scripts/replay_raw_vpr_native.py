"""Root-executed pixel/clock replay only; no models, labels or route inference.

Run under the Root-owned CPU2/3GiB/8GiB-headroom/fixed-deadline guard. The
result is input evidence, never an actor-closure or historical-model receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from monocular_osm import raw_vpr_native as r


def execute(config_path, config_sha256, *, native_grid=None, check=lambda: None):
    ledger = r.Ledger(check)
    config_ref = {"path": str(Path(config_path).resolve()), "sha256": config_sha256}
    config = ledger.json(config_ref)
    r.need(config.get("schema") == "raw-vpr-native-replay-config-v1", "wrong replay configuration")
    sources = config["source_hashes"]
    r.need(set(r.SOURCE_PATHS) <= set(sources), "complete explicit code pins required")
    for name, expected in sources.items():
        path = (ROOT / name).resolve(strict=True)
        r.need(path.is_relative_to(ROOT), "source pin escaped repository")
        ledger.protect({"path": str(path), "sha256": expected})
    inventory = ledger.json(config["inventory"])
    bindings = ledger.json(config["own_request_bindings"])
    r.need(r.ref(bindings["inventory_ref"]) == r.ref(config["inventory"]), "Root ownership bindings belong to another inventory")
    execution_root = config.get("request_execution_root")
    if execution_root is not None:
        r.need(Path(execution_root).is_absolute(), "historical execution root must be explicit absolute path")
        origin = ledger.json(config["request_execution_origin_ref"])
        commit = origin.get("source_commit")
        r.need(r.key(origin["source"]) == r.key(execution_root) and origin.get("source_clean") is True
               and isinstance(commit, str) and len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
               "historical execution origin registration differs")
    plan = r.make_plan(inventory, config["calls"], bindings, ledger.json, ledger.json,
                       request_execution_root=execution_root)
    video = config["video"]
    video_path = ledger.protect({"path": video["path"], "sha256": video["sha256"]})
    r.need(type(video["size"]) is int and video_path.stat().st_size == video["size"], "encoded video size differs")
    vo = ledger.members(config["cues_vo"], ("timestamps", "K"))
    signed = ledger.members(config["signed_vo_cache"], ("cache_signature", "vo_algo_version"))
    signature_array = signed["cache_signature"]
    r.need(signature_array.shape == () and signature_array.dtype.kind == "U"
           and int(signed["vo_algo_version"]) == 3, "signed native VO version/signature schema")
    signature = json.loads(str(signature_array.item()))
    started = time.monotonic()
    grid_function = native_grid if native_grid is not None else r.load_native_grid(
        ledger, config["native_auditor_ref"], config["prepared_dependency_ref"])
    result = r.replay(plan, video, vo, signature,
                      lambda record: ledger.members(record, ("query_indices", "query_times")),
                      grid_function, max_lag_s=config["max_composed_lag_s"], check=check)
    result.update(config_ref=config_ref, metadata_inventory=config["inventory"],
                  own_request_bindings=config["own_request_bindings"], cues_vo=config["cues_vo"],
                  request_execution_origin_ref=config.get("request_execution_origin_ref"),
                  signed_vo_cache=config["signed_vo_cache"], source_hashes=sources,
                  elapsed_s=time.monotonic() - started, external_actor_closure_verified=False,
                  protected_hashes=dict(ledger.protected))
    output = Path(config["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # No output inventory/hash callbacks follow the final expected-input pass.
    ledger.unchanged()
    with (output / "native_replay.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(argv)
    execute(args.config, args.config_sha256)


if __name__ == "__main__":
    main()
