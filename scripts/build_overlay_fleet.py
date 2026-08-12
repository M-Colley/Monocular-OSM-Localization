"""Turn every GPS-overlay clip in scripts/overlay_clips.py into ground truth.

    python scripts/build_overlay_fleet.py                 # all clips
    python scripts/build_overlay_fleet.py --only vCSEG6KaFng LmIHvLMLqFk
    python scripts/build_overlay_fleet.py --probe g5lnpYCk1Ec   # one frame, no OCR

For each clip this downloads only its analysis window, OCRs the burned-in
coordinate stamp (:mod:`monocular_osm.gps_overlay`), reverse-geocodes the
midpoint into a ``--city`` string, and writes ``ground_truth/<slug>.json`` in
the schema ``main.py --ground-truth-waypoints`` already reads. Nothing is
labelled by hand.

It finishes by writing ``ground_truth/overlay_fleet.json`` — the index
``scripts/run_all_gt.py --overlay`` reads, carrying each clip's slug, video
path and the ``--osm-around`` disc derived from its extracted track. That disc
is a GT leak, so those runs are the *gated* number; ``run_all_gt.py --blind``
drops it.

Extracted tracks are cached next to each video, so re-running after a code
change costs downloads of nothing and OCR of nothing — pass ``--no-cache`` when
the parser itself changed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from monocular_osm.city_inference import guess_city_from_title, slugify_submission  # noqa: E402
from monocular_osm.download import (  # noqa: E402
    DownloadError,
    cached_video_metadata,
    download_video,
)
from monocular_osm.gps_overlay import (  # noqa: E402
    GpsFix,
    city_for_track,
    extract_gps_track,
    longest_continuous_run,
    osm_around_for_track,
    track_to_ground_truth,
)
from overlay_clips import BY_ID, CLIPS, OverlayClip  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GEOCODE_CACHE = ROOT / "data" / "reverse_geocode_cache.json"


def _window_stem(clip: OverlayClip) -> str:
    a, b = clip.window
    return f"input_{int(a)}-{int(b)}"


def _repo_rel(p: Path) -> str:
    """Repo-relative POSIX path, tolerant of --data-dir living elsewhere.

    Paths recorded in the index are consumed by scripts/run_all_gt.py
    relative to the repo root, so resolve before comparing: a relative
    --data-dir yields a relative path that Path.relative_to(ROOT) rejects
    outright.
    """
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()   # outside the repo: keep it absolute


def _already_downloaded(clip: OverlayClip, data_dir: Path) -> tuple[Path, str] | None:
    """An existing ``data/<slug>/`` for this clip, or ``None``.

    The slug embeds the video id, so a clip whose window is already on
    disk is findable without asking YouTube anything. That matters: a
    fleet build makes one metadata request per clip, and YouTube answers
    a burst of them with "Sign in to confirm you're not a bot" — which
    would otherwise strand eight fully-downloaded videos.
    """
    stem = _window_stem(clip)
    # Slugify the id the same way the directory name was built — ids carry
    # '_' and '-' ("kxEDNj5L_yQ", "1nF_7l07i-E") and both become '-'.
    id_slug = slugify_submission(clip.video_id, fallback_seed=clip.url)
    for d in sorted(Path(data_dir).glob(f"{id_slug}-*")):
        for suffix in (".mp4", ".mkv", ".webm"):
            if (d / f"{stem}{suffix}").exists():
                return d / f"{stem}{suffix}", d.name
    return None


def _acquire(clip: OverlayClip, data_dir: Path, max_height: int) -> tuple[Path, str, str]:
    """Download the clip's window. Returns ``(video, slug, title)``."""
    data_dir = Path(data_dir)
    try:
        meta = cached_video_metadata(clip.url, data_dir)
    except DownloadError:
        # No metadata and no cache. If the bytes are already here, keep
        # going — an OCR pass needs the video, not the title.
        local = _already_downloaded(clip, data_dir)
        if local is None:
            raise
        video, slug = local
        print(f"  offline metadata unavailable; using the downloaded {slug}")
        return video, slug, clip.video_id

    slug = slugify_submission(meta.video_id, meta.title, fallback_seed=clip.url)
    out_dir = data_dir / slug
    video = download_video(
        clip.url, out_dir, filename_stem=_window_stem(clip),
        max_height=max_height, section=clip.window,
    )
    return video, slug, meta.title or clip.video_id


def _probe(clip: OverlayClip, data_dir: Path, max_height: int, out_dir: Path) -> None:
    """Write one annotated frame so the region/window can be eyeballed."""
    import cv2

    from monocular_osm.gps_overlay import _crop_region

    video, slug, title = _acquire(clip, data_dir, max_height)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 30))   # 30 s in, past any intro
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"could not decode {video}")
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{clip.video_id}_frame.jpg"), frame)
    cv2.imwrite(str(out_dir / f"{clip.video_id}_band.jpg"),
                _crop_region(frame, clip.region))
    print(f"{clip.video_id} ({title})\n  slug={slug}\n  region={clip.region}\n"
          f"  -> {out_dir / (clip.video_id + '_frame.jpg')}\n"
          f"  -> {out_dir / (clip.video_id + '_band.jpg')}")


def _title_disagreement(city: str, title: str) -> str:
    """Warn when the OCR'd track lands somewhere the title never mentions.

    The dangerous failure of auto-extracted ground truth is not a track
    that looks broken — the auditor catches those — but one that looks
    like a perfectly ordinary drive in the wrong place. A parser bug on
    the Detroit clip shifted its whole track 37 km south while keeping
    every leg plausible; the *only* visible symptom was the reverse
    geocode reporting a township nobody had mentioned. So compare the
    two independent statements of where this clip is: the coordinates,
    and what the uploader called it.

    Advisory only. Titles routinely name a metro rather than the suburb
    actually driven ("Driving in Phoenix..." for a Scottsdale street),
    so this must not gate the build.
    """
    place = city.split(",")[0].strip().casefold()
    haystack = title.casefold()
    if place and place in haystack:
        return ""
    guess = guess_city_from_title(title)
    if guess and guess.split(",")[0].strip().casefold() == place:
        return ""
    # Accept the distinctive first word too: Nominatim returns "Newcastle upon
    # Tyne" where the uploader wrote "Newcastle Driving", and a warning that
    # cries wolf on a correct clip is worse than no warning at all. Still
    # catches the case this exists for — "Brownstown Charter Township" shares
    # no leading word with "Driving from Detroit ... to Roseville".
    head = place.split()[0] if place else ""
    if len(head) >= 4 and head in haystack:
        return ""
    return (f"   <-- CHECK: the title never says {city.split(',')[0]!r} "
            f"({title[:60]!r}); verify the stamp is being read correctly")


def _track_span_m(fixes: list[GpsFix]) -> float:
    from monocular_osm.gps_overlay import _haversine_m
    return sum(_haversine_m(a, b) for a, b in zip(fixes, fixes[1:]))


def _build(clip: OverlayClip, args) -> dict | None:
    video, slug, title = _acquire(clip, Path(args.data_dir), args.max_height)
    print(f"  video   {video.name} ({video.stat().st_size / 1e6:.0f} MB)")

    t0 = time.time()
    track = extract_gps_track(
        video, sample_interval_sec=args.interval, region=clip.region,
        use_gpu=not args.cpu,
        cache_path=None if args.no_cache else video.parent / "gps_overlay_track.json",
    )
    n_samples = int((clip.window[1] - clip.window[0]) / args.interval) + 1
    dt = time.time() - t0
    print(f"  OCR     {len(track)}/{n_samples} frames yielded a fix "
          f"({len(track) / max(1, n_samples):.0%}) in {dt:.0f}s"
          f"{' [cached]' if dt < 2.0 else ''}")
    # Uploads are often edited. Keep only the longest cut-free stretch: a
    # jump between two separate drives is not something the ground truth OR
    # the visual odometry can absorb, so the analysis window has to shrink
    # to a real, continuous drive.
    full = track
    track = longest_continuous_run(track, max_speed_kmh=args.max_kmh)
    if track and len(track) < len(full):
        print(f"  cut     upload is edited — dropped {len(full) - len(track)} fixes "
              f"outside the longest continuous run; window narrows to "
              f"{track[0].t_sec:.0f}-{track[-1].t_sec:.0f}s")

    if len(track) < args.min_fixes:
        print(f"  SKIP    only {len(track)} fixes (< --min-fixes {args.min_fixes}); "
              f"check region={clip.region!r} with --probe {clip.video_id}")
        return None

    lats = [f.lat for f in track]
    lons = [f.lon for f in track]
    span_km = _track_span_m(track) / 1000.0
    print(f"  bbox    lat[{min(lats):.5f},{max(lats):.5f}] "
          f"lon[{min(lons):.5f},{max(lons):.5f}]  path {span_km:.2f} km")

    city = clip.city_override or city_for_track(track, cache_path=GEOCODE_CACHE)
    if city is None:
        print("  SKIP    reverse geocoding failed and no city_override set")
        return None
    print(f"  city    {city}{_title_disagreement(city, title)}")

    gt = track_to_ground_truth(
        track, video_id=clip.video_id, video_url=clip.url, city=city,
        n_waypoints=args.waypoints,
        description=(
            f"{clip.note} Overlay: {clip.overlay}. Auto-extracted from the "
            f"burned-in GPS stamp by scripts/build_overlay_fleet.py: "
            f"{len(track)} OCR fixes over {track[0].t_sec:.0f}-{track[-1].t_sec:.0f} s "
            f"({span_km:.2f} km of route) subsampled to {args.waypoints} waypoints. "
            f"NOT hand-verified — the stamp's own precision bounds it."
        ),
    )
    out = ROOT / args.out_dir / f"{args.prefix}{clip.video_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gt, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote   {_repo_rel(out)} ({len(gt['waypoints'])} waypoints)")

    clat, clon, radius = osm_around_for_track(track)
    return {
        "clip": clip, "slug": slug, "title": title, "city": city,
        "gt_path": _repo_rel(out),
        "video_path": _repo_rel(video),
        "osm_around": f"{clat:.6f},{clon:.6f},{radius:.0f}",
        "n_fixes": len(track), "span_km": span_km,
        "vo_segment": f"{int(track[0].t_sec)}:{int(round(track[-1].t_sec))}",
        "n_dropped": len(full) - len(track),
    }


def _write_index(built: list[dict], path: Path) -> None:
    """Record what was built so ``scripts/run_all_gt.py`` can pick it up.

    The sweep reads this file rather than a hand-maintained copy of the
    slugs, video paths and GT-derived discs — those are all outputs of
    THIS script, and duplicating them by hand is exactly the kind of
    drift that silently evaluates a clip against the wrong GT.

    Entries MERGE with whatever is already indexed: rebuilt clips are
    replaced, untouched ones survive. Without that, adding a single clip
    with ``--only`` would silently drop every other clip from the fleet —
    the sweep would then quietly evaluate one clip and report it as the
    whole fleet.
    """
    existing: dict[str, dict] = {}
    if path.exists():
        try:
            for entry in json.loads(path.read_text(encoding="utf-8")).get("clips", []):
                existing[entry["video_id"]] = entry
        except (OSError, ValueError, KeyError):
            existing = {}

    for b in built:
        existing[b["clip"].video_id] = {
            "video_id": b["clip"].video_id,
            "name": f"{b['city'].split(',')[0]} ({b['clip'].video_id})",
            "slug": b["slug"],
            "title": b["title"],
            "city": b["city"],
            "video": b["video_path"],
            "ground_truth": b["gt_path"],
            "osm_around": b["osm_around"],
            "vo_segment": b["vo_segment"],
            "n_dropped_to_cuts": b["n_dropped"],
            "n_fixes": b["n_fixes"],
            "span_km": round(b["span_km"], 3),
            "note": b["clip"].note,
        }

    # Keep the manifest's order so the index is stable across rebuilds.
    order = {c.video_id: i for i, c in enumerate(CLIPS)}
    payload = {
        "note": "Written by scripts/build_overlay_fleet.py. Ground truth here is "
                "OCR'd from each clip's burned-in GPS stamp, NOT hand-verified — "
                "keep it separate from the hand-labelled/INS core fleet.",
        "clips": sorted(existing.values(),
                        key=lambda e: order.get(e["video_id"], 10 ** 6)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    kept = len(payload["clips"]) - len(built)
    print(f"\nwrote {_repo_rel(path)} ({len(built)} rebuilt"
          + (f", {kept} kept" if kept else "")
          + ") — scripts/run_all_gt.py --overlay reads this")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", metavar="VIDEO_ID",
                   help="build just these clips (default: all)")
    p.add_argument("--probe", metavar="VIDEO_ID",
                   help="dump one frame + the overlay band for this clip and exit "
                        "(use to verify `region` before a full OCR pass)")
    p.add_argument("--interval", type=float, default=5.0,
                   help="seconds between OCR'd frames (default 5)")
    p.add_argument("--waypoints", type=int, default=20,
                   help="waypoints written per clip (default 20)")
    p.add_argument("--min-fixes", type=int, default=12,
                   help="refuse to write GT below this many OCR fixes (default 12)")
    p.add_argument("--max-height", type=int, default=1080,
                   help="download height cap (default 1080; the stamp is "
                        "illegible below ~720)")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="ground_truth")
    p.add_argument("--prefix", default="overlay_",
                   help="ground_truth filename prefix (default 'overlay_')")
    p.add_argument("--max-kmh", type=float, default=160.0,
                   help="above this implied speed a leg is an edit cut, not "
                        "driving; the track is split there and only the longest "
                        "continuous run is kept (default 160)")
    p.add_argument("--cpu", action="store_true", help="force CPU OCR")
    p.add_argument("--no-cache", action="store_true",
                   help="re-OCR even when a cached track exists")
    args = p.parse_args(argv)

    if args.probe:
        clip = BY_ID.get(args.probe)
        if clip is None:
            raise SystemExit(f"unknown clip {args.probe}; known: {', '.join(BY_ID)}")
        _probe(clip, Path(args.data_dir), args.max_height, ROOT / "output" / "overlay_probe")
        return

    selected = [BY_ID[v] for v in args.only] if args.only else list(CLIPS)
    if args.only:
        unknown = [v for v in args.only if v not in BY_ID]
        if unknown:
            raise SystemExit(f"unknown clip(s): {', '.join(unknown)}")

    built: list[dict] = []
    failed: list[tuple[str, str]] = []
    for clip in selected:
        print(f"\n{'=' * 78}\n{clip.video_id}  [{clip.region} band]  {clip.overlay}\n"
              f"{'=' * 78}")
        try:
            row = _build(clip, args)
        except DownloadError as e:
            print(f"  FAIL    download: {e}")
            failed.append((clip.video_id, f"download: {e}"))
            continue
        except Exception as e:  # keep the sweep going; report at the end
            print(f"  FAIL    {type(e).__name__}: {e}")
            failed.append((clip.video_id, f"{type(e).__name__}: {e}"))
            continue
        if row is None:
            failed.append((clip.video_id, "too few fixes / no city"))
        else:
            built.append(row)

    print(f"\n\n{'=' * 78}\nSUMMARY  {len(built)}/{len(selected)} clips -> ground truth\n"
          f"{'=' * 78}")
    print(f"{'clip':14s} {'fixes':>6s} {'km':>6s}  city")
    for b in built:
        print(f"{b['clip'].video_id:14s} {b['n_fixes']:>6d} {b['span_km']:>6.2f}  {b['city']}")
    for vid, why in failed:
        print(f"{vid:14s} {'FAIL':>6s} {'-':>6s}  {why}")
    if built:
        _write_index(built, ROOT / args.out_dir / "overlay_fleet.json")


if __name__ == "__main__":
    main()
