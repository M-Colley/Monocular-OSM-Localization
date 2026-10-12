"""Stage 4b/4c: absolute heading from the sun and OCR scene-text anchors."""

from __future__ import annotations

import numpy as np

from .context import _record_channel, RunContext


def estimate_sun_heading(ctx: RunContext) -> None:
    """Optional absolute heading from the sun -> ``ctx.result_sun_heading``.

    Activates only when the clip carries a usable capture time (container
    metadata or a burned-in clock); otherwise a graceful no-op.
    """
    cfg = ctx.cfg
    channel_status = ctx.channel_status
    video_path = ctx.video_path
    osm_around = ctx.osm_around

    # 4b. Optional: absolute heading from the sun. Activates only when the clip
    # carries a usable capture time (container metadata or a burned-in clock);
    # otherwise a graceful no-op. When available it pins the matcher's free
    # rotation DOF (monocular_osm/sun_heading.py). Computed + reported here.
    if cfg.use_sun_heading:
        try:
            from ..sun_heading import estimate_heading
            cen = osm_around[:2] if osm_around else None
            if cen is None:
                import osmnx as ox
                cen = ox.geocode(cfg.city)
            sh_times = list(np.linspace(cfg.vo_start_sec,
                                        cfg.vo_end_sec or (cfg.vo_start_sec + 420.0), 30))
            sh = estimate_heading(str(video_path), cen, sh_times,
                                  ocr_engine=cfg.ocr_engine)
            ctx.result_sun_heading = sh
            if sh and sh.get("available"):
                print(f"[4b] Sun heading: {sh['median_heading']:.0f} deg via {sh['source']} "
                      f"({sh['n_used']} frames, conf {sh['confidence']:.2f})")
            else:
                print(f"[4b] Sun heading: not available "
                      f"({(sh or {}).get('reason', 'module missing')})")
        except Exception as e:
            print(f"[4b] Sun heading failed ({e})")
            _record_channel(channel_status, 'sun', "failed", error=e)
            ctx.result_sun_heading = {"available": False, "reason": str(e)}
    else:
        ctx.result_sun_heading = None
