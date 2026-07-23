"""Prefetch open-data LoD2 CityGML tiles for a GT clip or a point.

Downloads (and parse-caches) the tiles the --use-tile3d channel will
need, so a long pipeline run never blocks on tile downloads. Give it a
ground-truth JSON (disc covering all waypoints) or an explicit point:

    python scripts/fetch_lod2.py --gt ground_truth/berlin_lBlKR2ek0w4.json
    python scripts/fetch_lod2.py --lat 48.4059 --lon 9.9837 --radius 1500

Providers/licenses: src/citygml_lod2.py (Berlin dl-de/zero-2.0;
Baden-Wuerttemberg dl-de/by-2.0, attribution "LGL, www.lgl-bw.de").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.citygml_lod2 import fetch_lod2_mesh, provider_for_latlon  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", type=Path,
                    help="ground_truth/*.json — fetch a disc covering all "
                         "of its waypoints (+ margin)")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--radius", type=float, default=1000.0,
                    help="disc radius in metres (point mode; GT mode "
                         "derives it from the waypoints + this margin)")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "berlin", "bw", "nrw", "bavaria", "osm"])
    ap.add_argument("--max-tiles", type=int, default=120)
    opts = ap.parse_args(argv)

    if opts.gt:
        wps = json.loads(opts.gt.read_text(encoding="utf-8"))["waypoints"]
        lats = np.array([w["lat"] for w in wps])
        lons = np.array([w["lon"] for w in wps])
        lat, lon = float(lats.mean()), float(lons.mean())
        # small-angle metric estimate of the covering radius
        dy = (lats - lat) * 111_320.0
        dx = (lons - lon) * 111_320.0 * np.cos(np.radians(lat))
        radius = float(np.hypot(dx, dy).max()) + 300.0
    elif opts.lat is not None and opts.lon is not None:
        lat, lon, radius = opts.lat, opts.lon, opts.radius
    else:
        ap.error("give --gt FILE or --lat/--lon")
        return

    prov = opts.source if opts.source != "auto" else provider_for_latlon(lat, lon)
    # 'auto' outside CityGML coverage, or an explicit 'osm', prefetches the
    # worldwide OSM LoD1 extrusion instead of an official LoD2 tile set.
    use_osm = opts.source == "osm" or (opts.source == "auto" and prov is None)
    print(f"disc: {lat:.5f},{lon:.5f} r={radius:.0f} m  "
          f"provider={'osm' if use_osm else prov}")

    # dst CRS only affects the cached mesh key; use the UTM zone the
    # pipeline's osmnx projection will pick for this longitude.
    zone = int((lon + 180.0) // 6.0) + 1
    dst = f"EPSG:{32600 + zone}"
    if use_osm:
        from src.osm_buildings3d import fetch_osm_building_mesh
        mesh = fetch_osm_building_mesh(lat, lon, radius, dst_crs=dst)
    else:
        mesh = fetch_lod2_mesh(lat, lon, radius, dst_crs=dst,
                               provider=opts.source, max_tiles=opts.max_tiles)
    if mesh is not None:
        print(f"ready: {mesh.n_buildings} buildings, "
              f"{len(mesh.triangles)} triangles (mesh cached, CRS {dst})")


if __name__ == "__main__":
    main()
