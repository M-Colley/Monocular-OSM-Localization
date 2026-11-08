# How the test3 clips were chosen

The third held-out split (`split="test3"`, 2026-11-07) tests the metro-ring
search under rule `test3_2026_11` (`benchmarks/preregistered_rules.json`),
registered before any test3 clip was downloaded beyond the screen's frames.
Selection reads titles, durations and coordinate stamps only; nothing the
localization pipeline produces.

* `screen5.py` / `screen5.json`: 87 cities used by no earlier split or screen,
  three fixed queries each (`{city} dashcam GPS`, `... VIOFO`, `... drive`),
  the test2 pass rule (title names the city, >= 7 min, not timelapse or
  rear-facing; a stamp in one of YouTube's three frame thumbnails, then >= 2
  of 3 full-resolution frames at 25/50/75 % with a stamp within 40 km of the
  city). 987 videos screened, 20 passing in 11 cities. The thumbnail OCR ran
  sequentially for the first three cities and in six worker processes after
  that (speed only). YouTube answered with bot checks three times; the
  screen paused 30 min each time and nothing was recorded as "no result".
* `recheck5.py` / `recheck5.json`: 18 candidates had a thumbnail stamp but 0
  of 3 frames read. Re-reading them by direct stream seek failed the same
  way for all 18: OpenCV's stream read timed out, so they had been
  unverified rather than rejected.
* `recheck5b.py` / `recheck5b.json`: the same three instants, downloaded as
  one-second sections and read with the same OCR and rule. Five pass:
  Aarhus `Gw-mFsRIHEA`, Bratislava `5koGsNdaVvE`, Katowice `MMKAehhi-tA` and
  `5ewPAw_gIeI`, Zagreb `7n4XQUBEH5Y`.

**Selection.** In the screen's city order, at most two passing clips per city
in search order, the first ten whose ground truth builds
(`scripts/build_overlay_fleet.py` on the 0-600 s download, plausible under
`scripts/check_overlay_gt.py`, >= 500 m of route in the analysis window). A
clip that fails is replaced by the next passing clip of the same city, then of
the next city.

| Order | City | Clip | Ground truth |
|---|---|---|---|
| 1 | Düsseldorf | `8BUtb7TKUFA` | 1.8 km, plausible |
| 2 | Düsseldorf | `HfK2LuXQtfA` | 2.2 km, plausible |
| 3 | Eindhoven | `6GciDYJ1FUc` | 4.1 km, plausible |
| 4 | Bristol | `G6bUX4GvSbE` | 1.5 km (100 s cut-free), plausible |
| - | Belfast | `dsQZSk-dWGo` | failed: edited, 10 fixes in the longest cut-free run |
| - | Belfast | `BrM-LdwP2vA` | failed: edited, 8 fixes in the longest cut-free run |
| 5 | Cork | `vp_D8mxgEkA` | 9.4 km, plausible |
| - | Aarhus | `Gw-mFsRIHEA` | failed: edited, 8 fixes in the longest cut-free run (Aarhus has no second passing clip) |
| 6 | Riga | `K_54XHyzRn8` | 4.1 km (418 s cut-free), plausible after review (see below) |
| 7 | Bratislava | `5koGsNdaVvE` | 4.7 km, plausible |
| 8 | Katowice | `MMKAehhi-tA` | 8.4 km, plausible; replaces a Belfast clip |
| 9 | Katowice | `5ewPAw_gIeI` | 9.6 km, plausible; replaces a Belfast clip |
| 10 | Zagreb | `7n4XQUBEH5Y` | 5.3 km, plausible; replaces the Aarhus clip |

Selection complete on 2026-11-08, before any test3 run. Passing clips left
unused: Bucharest `tK7bHJXAJX8` and `yd5g7jsWiEM`, Sofia `jkkl1QgsPSo`,
Kansas City `oPbnvEIRm3A` and `ZLmQWroXthY`, New Orleans `hrbPCnxiQeQ`,
Albuquerque `jz6o-_Y6r2E`.

YouTube refused the Aarhus, Riga and Bratislava downloads with bot checks
on 2026-11-07; they were retried unchanged after 30 min waits (no cookies,
no change of source) and Aarhus downloaded on the fourth attempt.

**Riga review.** `check_overlay_gt.py` flags waypoint 4 (t = 114 s, a 420 m
detour for 55 m of progress). The cached raw fixes show a turn-around, not a
misread: latitude 56.97424 (104 s), 56.97407 (109 s), 56.97433 (114 s),
56.97493 (119 s), then a steady northbound run at 15-19 m/s, with longitude
moving smoothly throughout. The ground truth is kept unedited.

**Disclosure, not exclusion.** The first Katowice clip ('Podzamcze -
Katowice') is driven at the Podzamcze end in its first ten minutes, 40-47 km
from the Katowice geocode and so outside even the ring's 20 km outer
radius. The rule selects on the screen and on whether the ground truth
builds, not on where the route lies, so it stays in.
