"""Selectively extract comma2k19 segments from a Chunk_N.zip.

comma2k19 route directories embed a ``|`` (e.g. ``<id>|<datetime>``) which
is an illegal filename character on Windows, and a chunk is ~9 GB while we
only need a few segments. This extracts just the chosen segments'
``video.hevc`` + ``global_pose/`` (the parts the adapter reads), rewriting
``|`` -> ``_`` so the path is valid.

    python scripts/extract_comma_segments.py data/comma/Chunk_1.zip \
        "b0c9d2329ad1606b|2018-08-17--14-55-39" --segments 1-4 \
        --out-dir data/comma/extracted
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

_KEEP = ("video.hevc", "global_pose/")


def _segment_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("zip_path", type=Path)
    p.add_argument("route", help="route dir name inside the chunk (with '|')")
    p.add_argument("--segments", required=True, help="e.g. '1-4' or '1,2,3'")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    segs = set(_segment_list(args.segments))
    safe_route = args.route.replace("|", "_")
    z = zipfile.ZipFile(args.zip_path)
    prefix = f"Chunk_1/{args.route}/" if "Chunk_1" in z.namelist()[0] else None
    # Derive the chunk's top folder from the archive itself (Chunk_1, etc.).
    top = z.namelist()[0].split("/", 1)[0]
    prefix = f"{top}/{args.route}/"

    n_files = 0
    for name in z.namelist():
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        rest = name[len(prefix):]              # e.g. "3/global_pose/frame_times"
        seg_str, _, tail = rest.partition("/")
        if not seg_str.isdigit() or int(seg_str) not in segs:
            continue
        if not any(tail.startswith(k.rstrip("/")) for k in _KEEP):
            continue
        dest = args.out_dir / safe_route / seg_str / tail
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(name) as src, open(dest, "wb") as out:
            out.write(src.read())
        n_files += 1
    print(f"extracted {n_files} files -> {args.out_dir / safe_route}")
    print(f"route dir: {args.out_dir / safe_route}")


if __name__ == "__main__":
    main()
