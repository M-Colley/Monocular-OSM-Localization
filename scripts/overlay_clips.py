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
    #: Benchmark split. Every clip that existed when the method was tuned is
    #: "dev". A clip added AFTER the method is frozen is "test": add it with
    #: split="test" BEFORE running anything on it, and never tune on its result.
    split: str = "dev"

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
    OverlayClip(
        video_id="Wrn4_uCxRCQ", region="bottom", window=_W,
        overlay="2MPH  N54d 58' 55.29\"  W1d 36' 24.77\"   (Nextbase NBDVR402G)",
        note="Newcastle upon Tyne, UK — Tyne Bridge, Central Motorway, Jesmond, "
             "Coast Road. The first non-US clip in this fleet and the first with a "
             "SINGLE-DIGIT degree longitude (W1d), which no US clip exercises — "
             "every other stamp here sits at 83-111 degrees west. Also a fifth "
             "camera family (Nextbase), and its clock reads 14/01/2000, a reminder "
             "that the stamp's DATE is not to be trusted even where its position is.",
    ),
    # ---- HELD-OUT TEST CLIPS (added 2026-10-24, split="test") ----------------
    # Chosen from YouTube by a web search for burned-in lat/lon stamps (read off
    # YouTube's own frame thumbnails), in cities not used before, BEFORE any
    # pipeline run on them and while the 200 m-goal method was still being
    # developed on the clips above. Never tune on their results. city_override
    # is the city each TITLE names (the benchmark's fairness rule), not a
    # reverse geocode of the ground truth.
    # (Den Haag nsy7qRN_oaw was selected too, but YouTube reported it
    # unavailable when its window was downloaded on 2026-10-24; dropped.)
    OverlayClip(
        video_id="nsONSyk9wtc", region="bottom", window=_W, split="test",
        city_override="NYC, New York, USA",
        overlay="N40.70906 W73.99691   (ROVE R2-4K)",
        note="Midtown Manhattan, FDR Drive, Manhattan and Brooklyn Bridges. "
             "Dense high-rise canyon; the same uploader as the Scottsdale clip.",
    ),
    OverlayClip(
        video_id="bF5IfC2ZHjQ", region="bottom", window=_W, split="test",
        city_override="Ottawa, Canada",
        overlay="N:45.35308 W:75.78294 94km/h   (bottom left)",
        note="Ottawa: Draper Av to Queen St, partly on the Queensway highway.",
    ),
    OverlayClip(
        video_id="5b-tgjtwb2U", region="bottom", window=(0.0, 420.0), split="test",
        city_override="Manchester, UK",
        overlay="Nextbase DMS",
        note="Old Trafford to Manchester city centre (7:01 upload).",
    ),
    OverlayClip(
        video_id="zt8rHvHKBUU", region="bottom", window=_W, split="test",
        city_override="Bern, Switzerland",
        overlay="N 46 57'20.1\" E 7 26'51.3\"   (MiVue 798D, bottom right)",
        note="Bern city centre round trip, October 2019. First east-longitude "
             "DMS stamp from a MiVue camera.",
    ),
    OverlayClip(
        video_id="4LvkBo69jZo", region="bottom", window=_W, split="test",
        city_override="Strasbourg, France",
        overlay="N 48 34'25.5\" E 7 45'19.1\"   (MiVue 588)",
        note="Strasbourg city tour, December 2017.",
    ),
    OverlayClip(
        video_id="iJ2JvNk4Rmk", region="bottom", window=_W, split="test",
        city_override="Frankfurt am Main, Germany",
        overlay="MiVue 798D DMS",
        note="Frankfurt am Main city centre, December 2019.",
    ),
    OverlayClip(
        video_id="SjnUeNH-Kno", region="top", window=_W, split="test",
        city_override="Dublin, Ireland",
        overlay="N53.35xxxx W6.32xxxx   (DOD GS600, top left)",
        note="Dublin, September 2013: old, soft footage; Mapillary imagery is "
             "years newer, an appearance-gap case like KITTI.",
    ),
    OverlayClip(
        video_id="-1xwoetuoOg", region="bottom", window=_W, split="test",
        city_override="Sydney, Australia",
        overlay="S 33 54'43.7\" E 151 00'37.2\"   (Navman)",
        note="Sydney western suburbs: the only southern/eastern-hemisphere "
             "clip, and a route far from the city centre.",
    ),
)

BY_ID = {c.video_id: c for c in CLIPS}
