"""The GPS-overlay clip fleet — dashcam uploads that carry their own ground truth.

Every clip here burns a live lat/lon into the frame, so
:mod:`monocular_osm.gps_overlay` can OCR a GPS track off it and
``scripts/build_overlay_fleet.py`` turns that into a ``ground_truth/*.json``
with no manual labelling at all. Adding a clip is one entry in :data:`CLIPS`.

The two fields that actually need a human eye are ``region`` (which band of
the frame the stamp sits in) and ``window`` (the seconds to analyse). Both
are cheap to check: ``scripts/build_overlay_fleet.py --probe <id>`` writes a
single annotated frame.

Formats present in this fleet, and why that matters — each one broke the
parser in a different way, and together they are the regression corpus in
``tests/test_gps_overlay.py::test_parse_latlon_real_ocr_output``:

    VIOFO A229 Pro   "000MPH  N:41.8933 W:87.6216"   4 dp  (~11 m)
    WolfBox i07      "0 mPh  N:33.773609 W:84.371269" 6 dp (~0.1 m)
    ROVE R2-4K       "28MPH  N33.49497 W111.91312"   5 dp  (~1 m)
    DOD LS460W       "N42° 20' 18.07\"  W83° 3' 23.70\""  0.01" (~0.3 m)
    YouTube Capture  "42 22'59\"N  83 10'35\"W"       1"    (~31 m)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlayClip:
    """One overlay dashcam clip and how to read its stamp."""

    video_id: str
    #: Where the burned-in coordinates sit: "bottom", "top" or "full".
    region: str
    #: (start_sec, end_sec) analysed — downloaded, OCR'd and covered by GT.
    window: tuple[float, float]
    #: What the overlay looks like, for the reader. Not used by the code.
    overlay: str
    #: Human note: route, camera, why the clip is interesting.
    note: str
    #: Only when reverse geocoding picks a name the OSM graph fetch dislikes.
    city_override: str | None = None

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


# A 10-minute window is the default: it spans the ~400-500 s monocular-VO
# sweet spot with margin, and keeps eight clips at a few hundred MB rather
# than several GB (these uploads run 10-38 minutes).
_W = (0.0, 600.0)

CLIPS: tuple[OverlayClip, ...] = (
    OverlayClip(
        video_id="vCSEG6KaFng", region="bottom", window=_W,
        overlay="000MPH  N:41.8933 W:87.6216   (VIOFO A229 Pro, 4K)",
        note="Chicago Loop / River North, daylight. Dense high-rise canyon — "
             "the hardest case for sky/skyline channels, the easiest for OCR.",
    ),
    OverlayClip(
        video_id="LmIHvLMLqFk", region="bottom", window=_W,
        overlay="000MPH  N:41.8856 W:87.6250   (VIOFO, 1080p)",
        note="Chicago at NIGHT, wet road. Same city and camera as vCSEG6KaFng "
             "one week earlier — a day/night A/B on near-identical geometry.",
    ),
    OverlayClip(
        video_id="wJsEQTCAg1c", region="bottom", window=_W,
        overlay="28MPH  N33.49497 W111.91312   (ROVE R2-4K)",
        note="Phoenix / Tempe / Scottsdale, Arizona. Wide arterial grid — the "
             "US-suburb failure mode that Boreas probes in Canada.",
    ),
    OverlayClip(
        video_id="ZhGb8q1kliY", region="bottom", window=_W,
        overlay="N42d 20' 18.07\"  W83d 3' 23.70\"   (DOD LS460W)",
        note="Detroit, downtown to northwest, 2016. Hemisphere-PREFIX DMS; the "
             "degree sign OCRs as an 8.",
    ),
    OverlayClip(
        video_id="1nF_7l07i-E", region="bottom", window=_W,
        overlay="N42d 19' 53.68\"  W83d 3' 11.80\"   (DOD LS460W)",
        note="Detroit, driving INTO downtown, 2016. Same camera as ZhGb8q1kliY, "
             "opposite direction — a heading-disambiguation pair.",
    ),
    OverlayClip(
        video_id="g5lnpYCk1Ec", region="top", window=_W,
        overlay="42 22'59\"N  83 10'35\"W   (YouTube Capture, TOP band)",
        note="Detroit to Roseville, Michigan. The only TOP-band overlay, and the "
             "only 1-arcsecond one (~31 m quantisation) — treat its GT as coarse.",
    ),
    OverlayClip(
        video_id="y66ZkRpUh4k", region="bottom", window=_W,
        overlay="0 mPh  N:33.773609 W:84.371269   (WolfBox i07)",
        note="Atlanta at dusk, surface streets. From the 'ASMR Driving Videos' "
             "playlist (PL6v9miqhGL1OjKF_jSdVGlipisb48FVNQ), most of which is "
             "unrelated short-form video — only this and kxEDNj5L_yQ are dashcam.",
    ),
    OverlayClip(
        video_id="kxEDNj5L_yQ", region="bottom", window=_W,
        overlay="65 mPh  N:33.853474 W:84.431641   (WolfBox i07)",
        note="Atlanta — Vinings / Midtown / Buckhead, partly freeway. Freeway "
             "stretches are a deliberate hard case: no turns for the shape matcher.",
    ),
)

BY_ID = {c.video_id: c for c in CLIPS}
