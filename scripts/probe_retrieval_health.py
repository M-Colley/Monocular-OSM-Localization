"""Input-only retrieval health from whitelisted saved VPR observations.

No GT file or evaluation value is opened. Unknown JSON values are skipped
without decoding them. Clip IDs identify records only; no routing policy is
selected or fitted. Output contains aggregate features, never coordinates.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
FIELDS = {"vpr_prior": {"track": True}, "n_frames": True, "n_valid_poses": True,
          "provenance": {"config": {"vo_start_sec": True, "vo_end_sec": True}}}
DECODER = json.JSONDecoder()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _white(text, index):
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _skip(text, index):
    """Skip one JSON value structurally; do not deserialize disallowed fields."""
    index = _white(text, index)
    quoted, escaped, depth = False, False, 0
    start = index
    for index in range(index, len(text)):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                if depth == 0:
                    return index + 1
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
        elif char in "]}":
            if depth == 0:
                return index
            depth -= 1
            if depth == 0:
                return index + 1
        elif depth == 0 and (char == "," or char.isspace()) and index > start:
            return index
    return len(text)


def _project(text, index, fields):
    index = _white(text, index)
    if index >= len(text) or text[index] != "{":
        return {}, _skip(text, index)
    selected = {}
    index = _white(text, index + 1)
    while index < len(text) and text[index] != "}":
        key, index = DECODER.raw_decode(text, index)
        index = _white(text, index)
        if text[index] != ":":
            raise ValueError("invalid object separator")
        index = _white(text, index + 1)
        if key not in fields:
            index = _skip(text, index)
        elif fields[key] is True:
            selected[key], index = DECODER.raw_decode(text, index)
        else:
            selected[key], index = _project(text, index, fields[key])
        index = _white(text, index)
        if text[index] == "}":
            break
        if text[index] != ",":
            raise ValueError("invalid member separator")
        index = _white(text, index + 1)
    return selected, index + 1


def project_inputs(raw):
    return _project(raw.decode("utf-8-sig"), 0, FIELDS)[0]


def _percentile(values, fraction):
    ordered = sorted(values)
    at = (len(ordered) - 1) * fraction
    lower, upper = math.floor(at), math.ceil(at)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (at - lower)


def _motion(points, times):
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    distances = [float(geod.inv(a[1], a[0], b[1], b[0])[2]) for a, b in zip(points, points[1:])]
    speed = [distance / (b - a) for distance, a, b in zip(distances, times, times[1:])]
    return {"arc_m": math.fsum(distances), "speed_mps": {
        "median": statistics.median(speed) if speed else None,
        "p95": _percentile(speed, .95) if speed else None,
        "max": max(speed) if speed else None}}


def derive_features(projected):
    """No labels, clip IDs, thresholds or selected-answer fields enter here."""
    cfg = (projected.get("provenance") or {}).get("config") or {}
    window = [value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
              else None for value in (cfg.get("vo_start_sec"), cfg.get("vo_end_sec"))]
    n_frames = projected.get("n_frames")
    n_valid = projected.get("n_valid_poses")
    n_frames = n_frames if isinstance(n_frames, int) and not isinstance(n_frames, bool) else None
    n_valid = n_valid if isinstance(n_valid, int) and not isinstance(n_valid, bool) else None
    fraction = (n_valid / n_frames if isinstance(n_frames, int) and n_frames > 0
                and isinstance(n_valid, int) and 0 <= n_valid <= n_frames else None)
    result = {"n_frames": n_frames, "n_valid_poses": n_valid, "valid_pose_fraction": fraction,
              "input_window_s": window, "status": "unavailable", "reason": "vpr_prior.track_missing",
              "track_count": 0}
    track = (projected.get("vpr_prior") or {}).get("track")
    if not isinstance(track, list) or not track:
        return result
    result["track_count"] = len(track)
    if any(not isinstance(row, list) or len(row) != 4 or
           any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in row)
           or abs(row[1]) > 90 or abs(row[2]) > 180 or not -1.0001 <= row[3] <= 1.0001 for row in track):
        result.update(status="invalid", reason="invalid_whitelisted_track_rows")
        return result
    times = [float(row[0]) for row in track]
    if any(b <= a for a, b in zip(times, times[1:])):
        result.update(status="invalid", reason="track_timestamps_not_strictly_increasing")
        return result
    points = [(row[1], row[2]) for row in track]
    similarities = [row[3] for row in track]
    smoothed = [(statistics.median([p[0] for p in points[max(0, i - 2):i + 3]]),
                 statistics.median([p[1] for p in points[max(0, i - 2):i + 3]])) for i in range(len(points))]
    raw, median5 = _motion(points, times), _motion(smoothed, times)
    duration = times[-1] - times[0]
    window_duration = None
    if (all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in window)
            and window[1] > window[0]):
        window_duration = window[1] - window[0]
    denominator_duration = window_duration if window_duration is not None else duration
    result.update(status="ok", reason=None, track_duration_s=duration, input_window_duration_s=window_duration,
                  denominator_duration_s=denominator_duration,
                  denominator_duration_source="input_window" if window_duration is not None else "track_span",
                  similarity_median=statistics.median(similarities), similarity_std=statistics.pstdev(similarities),
                  unique_coordinate_count=len(set(points)), dominant_coordinate_fraction=max(Counter(points).values()) / len(points),
                  raw=raw, running_median5=median5,
                  selected_track_length_over_5_5_duration=(median5["arc_m"] / denominator_duration / 5.5
                                                          if denominator_duration > 0 else None))
    return result


def record(source, clip, group):
    row = {"clip": clip, "group": group, "source": str(source.resolve())}
    if not source.is_file():
        return {**row, "source_sha256": None, "features": {"status": "unavailable", "reason": "result_file_missing"}}
    raw = source.read_bytes()
    row["source_sha256"] = sha_bytes(raw)
    try:
        selected = project_inputs(raw)
        row["whitelisted_projection_sha256"] = sha_bytes(json.dumps(selected, sort_keys=True,
            separators=(",", ":"), allow_nan=True).encode())
        row["features"] = derive_features(selected)
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        row["features"] = {"status": "invalid", "reason": f"whitelisted_input_parse: {type(exc).__name__}"}
    return row


def _display(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def run(args):
    if (args.out / "features.json").exists():
        raise SystemExit("feature snapshot already exists; use another output directory")
    manifest_path = args.root / "benchmarks/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible = [c["id"] for c in manifest["clips"] if c["split"] == "dev" and c["available_locally"]]
    baseline = args.baseline_clips if args.baseline_clips is not None else [
        cid for cid in eligible if (args.baseline_root / cid / "rep0/result.json").is_file()]
    if set(baseline) - set(eligible):
        raise ValueError("baseline IDs must belong to the available registered dev set")
    rows = [record(args.v6_root / cid / "rep0/result.json", cid, "frozen_v6") for cid in eligible]
    rows += [record(args.baseline_root / cid / "rep0/result.json", cid, "new_baseline") for cid in baseline]
    policy = {"input_projection": FIELDS, "ground_truth_opened": False, "evaluation_fields_decoded": False,
              "clip_ids_used_as_features": False, "route_selection_policy": None,
              "coordinate_equality": "exact saved values", "cosine_std": "population ddof0",
              "median5": "coordinatewise truncated centered window of five, matching vpr_track_only",
              "motion": "WGS84 geodesic arc; consecutive-query timestamps; linear p95",
              "selected_track_definition": "running median5 VPR route; result.answer is never read",
              "duration": "effective input window if bounded, otherwise observed track span"}
    result = {"exploratory": True, "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "script_sha256": sha_bytes(Path(__file__).read_bytes()), "manifest_sha256": sha_bytes(manifest_path.read_bytes()),
              "policy": policy, "n_registered_available_dev": len(eligible), "n_new_baseline_results": len(baseline),
              "records": rows}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "features.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Input-only retrieval health", "", "No GT or evaluation values were used. No routing rule was selected.", ""]
    for group in ("frozen_v6", "new_baseline"):
        lines += [f"## {group}", "", "| Clip | n | valid poses | median/std cosine | unique / dominant | raw / median5 arc m | raw speed median/p95/max m/s | median5 p95 m/s | length/(5.5 duration) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in rows:
            if row["group"] != group:
                continue
            f = row["features"]
            if f["status"] != "ok":
                lines.append(f"| {row['clip']} | {f.get('track_count', 0)} | {_display(f.get('valid_pose_fraction'))} | {f['reason']} | — | — | — | — | — |")
                continue
            speed = f["raw"]["speed_mps"]
            lines.append(f"| {row['clip']} | {f['track_count']} | {_display(f['valid_pose_fraction'])} | {_display(f['similarity_median'],3)} / {_display(f['similarity_std'],3)} | {f['unique_coordinate_count']} / {_display(f['dominant_coordinate_fraction'])} | {_display(f['raw']['arc_m'],0)} / {_display(f['running_median5']['arc_m'],0)} | "
                         f"{_display(speed['median'])} / {_display(speed['p95'])} / {_display(speed['max'])} | {_display(f['running_median5']['speed_mps']['p95'])} | {_display(f['selected_track_length_over_5_5_duration'])} |")
        lines.append("")
    lines += ["Missing tracks remain unavailable; no answer, fit, score or evaluation field substitutes for them.",
              "Source and whitelisted-input fingerprints are saved in features.json. Coordinates are never emitted."]
    (args.out / "features.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--v6-root", type=Path, default=ROOT / ".claude/worktrees/bench/output/benchmark_runs/v6")
    parser.add_argument("--baseline-root", type=Path, default=Path("E:/monocular-osm-runs/baseline-2b95f12/output/benchmark_runs/reliable_cold_dev"))
    parser.add_argument("--baseline-clips", nargs="+")
    parser.add_argument("--out", type=Path, default=ROOT / "output/retrieval_health_probe")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run(parser.parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
