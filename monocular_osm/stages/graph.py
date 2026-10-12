"""Stage 4: the OSM road graph."""

from __future__ import annotations

from pathlib import Path

from ..osm_data import fetch_city_graph, RoadGraph
from .context import RunContext


def _fetch_road_graph(
    city: str, cache_path: Path,
    around: tuple[float, float, float] | None = None,
) -> RoadGraph:
    """Fetch the OSM graph with a user-actionable error on geocode failure.

    osmnx raises a grab-bag of exceptions (InsufficientResponseError,
    ValueError, network errors) whose messages don't tell a CLI user
    what to fix. Re-raise as a ValueError that names the city and the
    expected format, keeping the original as __cause__.
    """
    from ..net import is_transient

    try:
        return fetch_city_graph(city, cache_path=cache_path, around=around)
    except Exception as e:
        if is_transient(e):
            # Retries are already exhausted inside fetch_city_graph; the city
            # name is probably fine, so don't send the user to check spelling.
            raise ValueError(
                f"Could not reach Overpass to fetch the OSM road graph for "
                f"{city!r} (network error after retries; try again later). "
                f"Original error: {e}"
            ) from e
        raise ValueError(
            f"Could not fetch an OSM road graph for {city!r}. Check the "
            f"spelling and use the form 'City, Country' (e.g. 'Ulm, "
            f"Germany'). Original error: {e}"
        ) from e


def load_road_graph(ctx: RunContext) -> None:
    """OSM driving graph for the city (or the VPR/seed disc) -> ``ctx.road``."""
    cfg = ctx.cfg
    osm_around = ctx.osm_around

    # 4. OSM road graph for the city.
    if osm_around is not None:
        la, lo, rad = osm_around
        print(f"[4/5] Fetching OSM driving graph around ({la:.4f}, {lo:.4f}) "
              f"r={rad:.0f} m for {cfg.city!r}")
        ctx.cache_path = cfg.data_dir / (
            f"{cfg.city.replace(',', '').replace(' ', '_')}"
            f"_around_{la:.4f}_{lo:.4f}_{int(rad)}.graphml")
    else:
        print(f"[4/5] Fetching OSM driving graph for {cfg.city!r}")
        ctx.cache_path = cfg.data_dir / f"{cfg.city.replace(',', '').replace(' ', '_')}.graphml"
    ctx.road = _fetch_road_graph(cfg.city, ctx.cache_path, around=osm_around)
    print(f"      -> {ctx.road.graph.number_of_nodes()} nodes, "
          f"{ctx.road.graph.number_of_edges()} edges, CRS={ctx.road.crs}")
