# Metro-Dense v2: Complete 38-Clip In-Sample Evaluation Across `test1`–`test5` (November 2026)

Rule `metro_dense_v2_all38_dev_2026_11` in `benchmarks/preregistered_rules.json`
(registered at `aec5008` before any v2 run; pre-score `infra_check` committed at
`7f6a4dc`). Method: commit `31a6f57` (`monocular_osm` tree
`75f8e90cdc63fdc328069264a72fcd02e5a2ace5`), retrieval-track pipeline with
`--retrieval-only --vpr-metro-dense --vpr-region-cache-dir E:/claude-scoring/metro_dense_cache`,
evaluated across **all 38 clips** of the five existing test splits (`test` [7],
`test2` [7], `test3` [10], `test4` [6], `test5` [8]) against the committed
`ordinary` and `dense_24k` (`--vpr-pass1-cap 24000`) runs. All five splits are
used splits, so this is **in-sample development evidence**; blind generalization
is evaluated separately under `test6_worldwide_2026_11`.

## Headline Outcome

**Both pre-registered hypotheses (`V1` and `V2`) hold with strong statistical
significance (`p = 0.00024` vs `dense_24k`, `p = 0.000004` vs `ordinary`, zero
losses across all 38 clips).**

- **Within 200 m (`symmetric_mean_m <= 200 m`):**
  - **Paired 37 clips:** `12/37 (32.4 %)` ordinary → `18/37 (48.6 %)` `dense_24k` → **`30/37 (81.1 %, Wilson 95 % CI 65.8–90.5 %)`** `metro_dense_v2`.
  - **All 38 clips (incl. exonym-unlocked Bucharest `yd5g7jsWiEM`):** **`31/38 (81.6 %, Wilson 95 % CI 66.6–90.8 %)`** — **all 31 localized clips achieve `<= 99.1 m`** (`31/38` within 100 m; **`34/38 = 89.5 %`** within 500 m; **`36/38 = 94.7 %`** within 900 m).
- **Median symmetric route error:**
  - Paired 37 clips: **`511.3 m`** ordinary → **`260.3 m`** `dense_24k` → **`41.7 m`** `metro_dense_v2` (**`41.2 m`** across all 38 clips).
- **Hypothesis Checks:**
  - **`V1` (no harm on `test1`–`test3` vs `dense_24k`):** **Holds (`0` losses across `test1`–`test3`, and `0` losses across all 38 clips).**
  - **`V2` (all-split gains):** **Holds (`+12` paired gains + `+1` unlocked gain = `+13` gains over `dense_24k`, McNemar one-sided `p = 0.000244`; `+18` paired gains + `+1` unlocked gain = `+19` gains over `ordinary`, McNemar one-sided `p = 0.000004`).**
  - **`V3` (10 % band & per-split breakdown):** 25 better, 7 worse, 5 tied vs `dense_24k` (one-sided sign test `p = 0.00105`).

### Per-Split Summary

| Split | Clips | Ordinary `<= 200 m` | Dense 24k `<= 200 m` | Metro-Dense v2 `<= 200 m` | Metro-Dense v2 `<= 500 m` | Metro-Dense v2 Median (m) |
|---|---:|---:|---:|---:|---:|---:|
| `test` | 7 | 5/7 (71 %) | 5/7 (71 %) | **7/7 (100 %)** | **7/7 (100 %)** | **19.3** |
| `test2` | 7 | 3/7 (43 %) | 4/7 (57 %) | **5/7 (71 %)** | **7/7 (100 %)** | **76.5** |
| `test3` | 10 | 3/10 (30 %) | 5/10 (50 %) | **10/10 (100 %)** | **10/10 (100 %)** | **24.1** |
| `test4` | 6 | 0/5 (0 %)* | 2/5 (40 %)* | **5/6 (83 %)** | **5/6 (83 %)** | **45.2** |
| `test5` | 8 | 1/8 (12 %) | 2/8 (25 %) | **4/8 (50 %)** | **5/8 (62 %)** | **235.6** |
| **Total** | **38** | **12/37 (32.4 %)** | **18/37 (48.6 %)** | **31/38 (81.6 %)** | **34/38 (89.5 %)** | **41.2** |

*\*`overlay_yd5g7jsWiEM` (`Bucharest`, `"Buzau - Bucuresti"`) was refused by the pre-exonym fairness guard in earlier baselines and localized to `10.5 m` in Metro-Dense.*

---

## Per-Clip Results Across All 38 Clips

| Split | Clip | Ordinary (m) | Dense 24k (m) | Metro-Dense v2 (m) | vs Dense 24k | Kept Track |
|---|---|---:|---:|---:|---|---|
| `test` | Sydney `-1xwoetuoOg` | 183.6 | 143.9 | **56.6** | better | `pass-2-corridor` |
| `test` | Strasbourg `4LvkBo69jZo` | 43.4 | 10.3 | **10.9** | tie | `pass-2-corridor` |
| `test` | Manchester `5b-tgjtwb2U` | 54.9 | 54.9 | **11.7** | better | `pass-2-corridor` |
| `test` | Ottawa `bF5IfC2ZHjQ` | 1,418.8 | 350.0 | **41.7** | **GAIN** | `pass-2-corridor` |
| `test` | Frankfurt `iJ2JvNk4Rmk` | 325.2 | 325.2 | **19.3** | **GAIN** | `pass-2-corridor` |
| `test` | NYC `nsONSyk9wtc` | 51.4 | 51.4 | **53.7** | tie | `pass-2-corridor` |
| `test` | Bern `zt8rHvHKBUU` | 168.2 | 16.1 | **16.1** | tie | `pass-2-corridor` |
| `test2` | Montreal `0jJbdTcDseI` | 3,457.6 | 12,893.7 | 392.4 | better (`33x`) | `pass-2-corridor` |
| `test2` | Budapest `1bGf8OD9Fqo` | 157.3 | 24.3 | **99.1** | worse (still `<100 m`) | `pass-2-corridor` |
| `test2` | Amsterdam `2sS6glBBm8g` | 177.5 | 175.1 | **76.5** | better | `pass-2-corridor` |
| `test2` | Warsaw `GnXsgpVPvww` | 89.5 | 89.5 | **30.8** | better | `pass-2-corridor` |
| `test2` | Paris `HpBfjRA60ws` | 24,363.8 | 11,703.3 | 370.8 | better (`32x`) | `pass-2-corridor` |
| `test2` | Munich `XO6H3mGMKYg` | 8,130.3 | 8,130.3 | **12.7** | **GAIN (`640x`)** | `pass-2-corridor` |
| `test2` | Warsaw `lCm2a3kOEmE` | 334.7 | 15.0 | **16.1** | tie | `pass-2-corridor` |
| `test3` | Katowice `5ewPAw_gIeI` | 2,686.9 | 89.1 | **21.2** | better | `pass-2-corridor` |
| `test3` | Bratislava `5koGsNdaVvE` | 24.2 | 18.2 | **16.3** | better | `pass-2-corridor` |
| `test3` | Eindhoven `6GciDYJ1FUc` | 21.0 | 21.0 | **27.1** | worse (still `<30 m`) | `pass-2-corridor` |
| `test3` | Zagreb `7n4XQUBEH5Y` | 271.8 | 12.0 | **16.8** | worse (still `<20 m`) | `pass-2-corridor` |
| `test3` | Düsseldorf `8BUtb7TKUFA` | 260.3 | 260.3 | **75.9** | **GAIN** | `pass-2-corridor` |
| `test3` | Bristol `G6bUX4GvSbE` | 6,322.4 | 8,683.6 | **43.1** | **GAIN (`201x`)** | `pass-2-corridor` |
| `test3` | Düsseldorf `HfK2LuXQtfA` | 4,126.8 | 4,126.8 | **11.1** | **GAIN (`372x`)** | `pass-2-corridor` |
| `test3` | Riga `K_54XHyzRn8` | 31.0 | 15.2 | **15.3** | tie | `pass-2-corridor` |
| `test3` | Katowice `MMKAehhi-tA` | 37,762.5 | 37,762.5 | **70.1** | **GAIN (`539x`)** | `pass-2-corridor` |
| `test3` | Cork `vp_D8mxgEkA` | 4,679.0 | 418.5 | **63.2** | **GAIN** | `pass-2-corridor` |
| `test4` | Kansas City `ZLmQWroXthY` | 511.3 | 392.6 | **56.3** | **GAIN** | `pass-2-corridor` |
| `test4` | Sofia `jkkl1QgsPSo` | 945.4 | 27.0 | **21.6** | better | `pass-2-corridor` |
| `test4` | Albuquerque `jz6o-_Y6r2E` | 9,670.8 | 11,175.9 | 16,108.4 | worse (desert) | `pass-2-corridor` |
| `test4` | Trondheim `pFbUvfr8k-Q` | 207.0 | 22.0 | **40.7** | worse (still `<45 m`) | `pass-2-corridor` |
| `test4` | Bucharest `tK7bHJXAJX8` | 410.4 | 350.5 | **49.8** | **GAIN** | `pass-2-corridor` |
| `test4` | Bucharest `yd5g7jsWiEM` | — | — | **10.5** | **UNLOCKED GAIN** | `pass-2-corridor` |
| `test5` | Portland `3cT07o9aXyQ` | 9,762.5 | 9,749.7 | 856.5 | better (`11.4x`) | `pass-2-corridor` |
| `test5` | Rotterdam `FPQyfJsbjWM` | 25.7 | 25.4 | **48.4** | worse (still `<50 m`) | `pass-2-corridor` |
| `test5` | Calgary `VaPcyHM-Wtc` | 6,880.8 | 171.0 | **14.8** | better (`11.6x`) | `pass-2-corridor` |
| `test5` | Melbourne `YJmgh_0SoGs` | 1,345.9 | 1,345.9 | 828.0 | better (`60x` vs v1) | `seed-corridor` |
| `test5` | Houston `kMQOOXaUBh0` | 7,203.1 | 16,273.7 | 36,759.1 | worse (desert) | `pass-2-corridor` |
| `test5` | Glasgow `ml4PPnpc-Kw` | 11,085.0 | 11,085.0 | 422.9 | better (`26x`) | `pass-2-corridor` |
| `test5` | Belgrade `qgaSRWY8Aio` | 677.8 | 677.8 | **28.6** | **GAIN** | `pass-2-corridor` |
| `test5` | Portland `zsqxTSNLw5A` | 10,051.9 | 10,500.5 | **18.4** | **GAIN** | `pass-2-corridor` |

Machine-readable data: `docs/metro_dense_v2_all38_2026-11.json`.

## Diagnostic Notes on Metro-Dense v2 Additions

1. **Corroborated Night-Drive Seed Recovery (`Melbourne`):** Triggered **only** on
   `overlay_YJmgh_0SoGs` (`Melbourne` 1440p night test) across all 38 clips,
   pulling error down from `49,603.3 m` (v1) and `1,345.9 m` (ordinary) to
   **`828.0 m`** (right inside the Berwick residential area, limited only by dark
   suburban street-level cosine precision within the 3.5 km seed neighbourhood).
2. **Low-Cosine Open-Provider Augmentation (`< 0.150`):** Helped or left
   unchanged 13 of the 14 clips where it fired, but on `Portland 3cT07o9aXyQ`
   (`p1_sim = 0.1202`), adding KartaView/Panoramax shifted part of the suburban
   Clackamas/Gresham track from `25.6 m` (v1 Mapillary-only) to `856.5 m` (still
   `11.4x` better than `9,750 m` in `dense_24k`). This indicates open-provider
   augmentation is best reserved for regions where Mapillary is genuinely sparse
   (`< 3,000` metro refs) rather than dense-Mapillary low-cosine drives.
3. **Only 2 Far Misses Left Out of 38 Clips (`Houston` & `Albuquerque`):**
   **36 of 38 clips (`94.7 %`) now localize within `< 860 m`**, **34 of 38
   (`89.5 %`) within `< 425 m`**, and **31 of 38 (`81.6 %`) within `< 100 m`**.
   The sole remaining multi-kilometre misses (`Houston` and `Albuquerque`) are
   verified street-imagery coverage deserts.
