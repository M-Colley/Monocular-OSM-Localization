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
| 6 | Aarhus | `Gw-mFsRIHEA` | pending (download refused by a bot check) |
| 7 | Riga | `K_54XHyzRn8` | pending (download refused by a bot check) |
| 8 | Bratislava | `5koGsNdaVvE` | pending (download refused by a bot check) |
| 9 | Katowice | `MMKAehhi-tA` | pending (replaces a Belfast clip) |
| 10 | Katowice | `5ewPAw_gIeI` | pending (replaces a Belfast clip) |

Next in line if a pending build fails: Zagreb `7n4XQUBEH5Y`, Bucharest
`tK7bHJXAJX8` and `yd5g7jsWiEM`, Sofia `jkkl1QgsPSo`, Kansas City
`oPbnvEIRm3A` and `ZLmQWroXthY`.
