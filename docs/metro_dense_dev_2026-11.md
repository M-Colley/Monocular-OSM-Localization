# Unified Metro-Dense VPR & Exonym Support: In-Sample Check (November 2026)

Rule `metro_dense_dev_2026_11` in `benchmarks/preregistered_rules.json`
(registered at `39885be` before any scored run; pre-score `infra_check`
committed at `f6db709`). Method: commit `d9daa83` (`monocular_osm` tree
`17ec29e0eb30597f62fc211e92b0adeff24e107f`), the retrieval-track pipeline with
`--retrieval-only --vpr-metro-dense --vpr-region-cache-dir E:/claude-scoring/metro_dense_cache`,
evaluated on the 14 clips of `test5` (8) and `test4` (6) against the committed
`ordinary` (`t5_ord`, `dense_ord_test4`) and `dense_24k` (`t5_24k`,
`dense_24k_test4`) runs. Because all five existing splits (`test`–`test5`) are
used, this is **in-sample development evidence only**; certifying worldwide
generalization requires a fresh held-out split (`benchmarks/test6_screen/`).

## Result

**Both pre-registered decision hypotheses (`M1` and `M2`) hold decisively.**
`--vpr-metro-dense` is therefore eligible for a fresh held-out evaluation on the
six-continent `test6` split; it remains opt-in (`False` by default) pending
held-out certification.

- **M1 (no harm vs `dense_24k`): holds (`0` losses).** Every clip within 200 m
  in `dense_24k` stays well within 200 m (all four `<= 55.3 m`: Rotterdam
  `46.0 m`, Calgary `171.0 -> 14.7 m`, Sofia `27.0 -> 21.6 m`, Trondheim
  `22.0 -> 55.3 m`).
- **M2 (gains vs `dense_24k`): holds (`+5` paired gains + `+1` unlocked exonym
  gain = `+6` gains, `0` losses; paired exact McNemar one-sided `p = 0.03125`).**

  | Split | Clip | Ordinary | Dense 24k | Metro-Dense | Failure mode resolved |
  |---|---|---:|---:|---:|---|
  | test5 | Portland `3cT07o9aXyQ` | 9,762.5 m | 9,749.7 m | **25.6 m** | Mode A (Clackamas / Gresham suburbs outside 8 km disc) |
  | test5 | Portland `zsqxTSNLw5A` | 10,051.9 m | 10,500.5 m | **18.3 m** | Mode A + C (outer I-205 / I-84 highway corridor, 4 discs) |
  | test5 | Belgrade `qgaSRWY8Aio` | 677.8 m | 677.8 m | **32.4 m** | 3 km title-seed clamping & dense-cap bypass |
  | test4 | Kansas City `ZLmQWroXthY` | 511.3 m | 392.6 m | **56.5 m** | Mode C (long I-435 / I-70 highway corridor, 4 discs) |
  | test4 | Bucharest `tK7bHJXAJX8` | 410.4 m | 350.5 m | **49.8 m** | Mode C (long night arterial drive, 3 discs) |
  | test4 | Bucharest `yd5g7jsWiEM` | *(refused)* | *(refused)* | **10.5 m** | Exonym fairness (`Bucuresti` = `Bucharest`) + shared metro cache |

- **Within 200 m (`symmetric_mean_m <= 200 m`):**
  - Paired 13 clips: **1/13 (`7.7 %`)** ordinary → **4/13 (`30.8 %`)** `dense_24k` → **9/13 (`69.2 %`, Wilson 95 % CI `42.4–87.3 %`)** `metro_dense`.
  - All 14 clips: **10/14 (`71.4 %`, Wilson 95 % CI `45.4–88.3 %`)** `metro_dense` — all 10 localized clips achieve **`<= 56.5 m`** (`10/14` within 100 m; `11/14 = 78.6 %` within 500 m, with Glasgow improving from `11,085.0 m -> 422.9 m`).
- **M3 (descriptive):**
  - Median symmetric route error (13 paired clips): **`1,345.9 m`** ordinary → **`677.8 m`** `dense_24k` → **`49.8 m`** `metro_dense` (`47.9 m` across all 14 clips).
  - 10 % band vs `dense_24k` (13 paired): 8 better, 5 worse, 0 tied (one-sided sign test `p = 0.291`; two of the "worse" rows are Rotterdam `25.4 -> 46.0 m` and Trondheim `22.0 -> 55.3 m`, both well under 60 m, while the other three were already >1.3 km misses).
  - Pass-1 centre within 2 km of the true route: **11/14** (`78.6 %`).
  - Corridor Pass-2 acceptance: **14/14** kept `pass-2-corridor` (`0` rejections, vs `13/21` Pass-2 rejections in `dense_pass1_dev` before local Pass-1 unioning).
  - Mean wall-clock: `1,340.6 s` per clip on RTX 4090 (second clips in shared metros run in `355.6 s` on Bucharest and `980.0 s` on Portland).

## Per-Clip Results

| Split | Clip | Ordinary (m) | Dense 24k (m) | Metro-Dense (m) | Start (m) | Pass-1 to route: ord / d24k / metro (m) | Corridor discs (P2 refs) | Wall (s) |
|---|---|---:|---:|---:|---:|---|---|---:|
| test5 | Portland `3cT07o9aXyQ` | 9,762.5 | 9,749.7 | **25.6** | 38.4 | 9,974 / 9,140 / 88 | 3 (8,326) | 2,052 |
| test5 | Rotterdam `FPQyfJsbjWM` | 25.7 | 25.4 | **46.0** | 45.8 | 68 / 16 / 185 | 1 (3,325) | 1,430 |
| test5 | Calgary `VaPcyHM-Wtc` | 6,880.8 | 171.0 | **14.7** | 250.7 | 6,372 / 179 / 67 | 4 (8,777) | 1,528 |
| test5 | Melbourne `YJmgh_0SoGs` | 1,345.9 | 1,345.9 | 49,603.3 | 47,171.4 | 210 / 210 / 48,676 | 1 (2,690) | 1,566 |
| test5 | Houston `kMQOOXaUBh0` | 7,203.1 | 16,273.7 | 36,759.1 | 37,755.7 | 5,842 / 16,126 / 36,316 | 2 (3,670) | 1,518 |
| test5 | Glasgow `ml4PPnpc-Kw` | 11,085.0 | 11,085.0 | 422.9 | 2,267.1 | 9,907 / 9,907 / 276 | 4 (7,358) | 1,573 |
| test5 | Belgrade `qgaSRWY8Aio` | 677.8 | 677.8 | **32.4** | 203.2 | 6 / 6 / 85 | 1 (3,218) | 1,178 |
| test5 | Portland `zsqxTSNLw5A` | 10,051.9 | 10,500.5 | **18.3** | 35.8 | 9,060 / 9,300 / 1,080 | 4 (11,583) | 980 |
| test4 | Kansas City `ZLmQWroXthY` | 511.3 | 392.6 | **56.5** | 457.1 | 387 / 85 / 487 | 4 (9,342) | 1,706 |
| test4 | Sofia `jkkl1QgsPSo` | 945.4 | 27.0 | **21.6** | 40.3 | 80 / 11 / 197 | 3 (13,700) | 1,312 |
| test4 | Albuquerque `jz6o-_Y6r2E` | 9,670.8 | 11,175.9 | 16,108.4 | 16,267.7 | 9,581 / 11,141 / 14,201 | 1 (1,906) | 868 |
| test4 | Trondheim `pFbUvfr8k-Q` | 207.0 | 22.0 | **55.3** | 212.5 | 61 / 81 / 58 | 1 (8,732) | 1,233 |
| test4 | Bucharest `tK7bHJXAJX8` | 410.4 | 350.5 | **49.8** | 200.4 | 1,016 / 69 / 35 | 3 (11,015) | 1,469 |
| test4 | Bucharest `yd5g7jsWiEM` | — | — | **10.5** | 12.1 | — / — / 17 | 1 (4,251) | 356 |

Machine-readable per-clip records: `docs/metro_dense_dev_2026-11.json`.

## Technical Diagnosis of Remaining 4 Misses (> 200 m)

1. **Glasgow `ml4PPnpc-Kw` (`422.9 m`, down from `11,085.0 m`):** Pass 1 located
   the drive (`276 m` to route, vs `9,907 m` previously after fixing the `<4 km`
   rank-1 Nominatim ward bug and expanding to 20 km + additive Howwood region),
   and 4 corridor discs tracked most of the drive, but the inter-town
   Howwood-to-Glasgow A737 corridor spans ~16 km — slightly wider than `4 x 2.5 km`
   discs cover end-to-end (`start_error = 2,267 m` on the rural Howwood tail).
2. **Melbourne `YJmgh_0SoGs` (`49,603.3 m`, night footage):** 1440p night test
   drive. Notice `ordinary` had an in-city OCR seed `210 m` from the true route,
   whereas across a 20 km daytime reference pool dark night frames matched a
   brightly lit outer freeway at higher raw cosine. Softly anchoring low-cosine
   night tracks to corroborated multi-hit OCR/plate seeds is a clear follow-up
   (noted post hoc; nothing tuned here).
3. **Houston `kMQOOXaUBh0` (`36,759.1 m`) & Albuquerque `jz6o-_Y6r2E` (`16,108.4 m`):**
   True street-imagery coverage deserts (`docs/search_area_coverage_2026-11.md`:
   large route segments lack Mapillary imagery within 50 m). Addressing pure
   imagery deserts requires cross-view aerial/road-graph geometry matching when
   street-view retrieval confidence abstains.

## Disclosures

- **In-sample development check:** `test4` and `test5` failure diagnostics
  motivated every component of `--vpr-metro-dense`. Generalization claims
  require a fresh pre-registered held-out run (`benchmarks/test6_screen/`).
- **Baseline reuse:** `ordinary` and `dense_24k` numbers come from the committed
  runs in `E:/claude-scoring/wt-test5` and `E:/claude-scoring/wt-dense` (1–2 days
  earlier). Single rep per clip.
