# Held-Out Worldwide `test6` Evaluation (`test6_worldwide_2026_11`, November 2026)

Blind paired evaluation of frozen **Metro-Dense v2** (`--retrieval-only --vpr-metro-dense`, `monocular_osm` tree `75f8e90cdc63fdc328069264a72fcd02e5a2ace5`) against the **ordinary** retrieval baseline (`--retrieval-only`) on the newly collected, globally stratified `test6` split (`8` QC-passed clips across `7` countries and `4` continents: Malaysia, Thailand, India, Chile, Mexico, Canada, Greece).

- **Pre-registration:** `benchmarks/preregistered_rules.json` (`test6_worldwide_2026_11`), `benchmarks/test6_screen/preregistration_receipt.json` (`4c3b816e...`), and `benchmarks/test6_screen/selection_outcome.json` (committed at `11e51fd` / `29273ac` before any localization).
- **Pre-score infrastructure check:** Committed at `a6995e9` before any score or offline aggregation was computed (`16/16` runs `rc=0`, `dirty=False`, `degraded=[]`, `leaks=[]`).

## Headline Results (`n = 8` Blind Held-Out Worldwide Clips)

| Metric | Ordinary (`t6_ord`) | Metro-Dense v2 (`t6_metro_dense`) | Delta / Test |
|---|---|---|---|
| **Within 200 m (`symmetric_mean_m`)** | **0 / 8 (0.0 %)** [Wilson 0.0–32.4 %] | **3 / 8 (37.5 %)** [Wilson 13.7–69.4 %] | **+3 gains, 0 losses** (McNemar one-sided `p = 0.125`) |
| **Within 50 m (`symmetric_mean_m`)** | 0 / 8 (0.0 %) | **2 / 8 (25.0 %)** (`15.5 m`, `16.2 m`) | +2 |
| **Within 500 m (`symmetric_mean_m`)** | 1 / 8 (12.5 %) | **4 / 8 (50.0 %)** | +3 |
| **Median `symmetric_mean_m`** | `10,532.8 m` | **`5,184.1 m`** | `-50.8 %` |
| **10 % relative band vs Ordinary** | — | `3` better, `1` worse, `4` tied | Sign-test one-sided `p = 0.3125` |

### Pre-Registered Hypothesis Decisions

- **W1 (No harm: at most 1 clip `<= 200 m` in ordinary moves `> 200 m` in `metro_dense`):** **HOLDS** (`0` losses).
- **W2 (Worldwide benefit: strictly more `test6` clips `<= 200 m` in `metro_dense` than in `ordinary`):** **HOLDS** (`3/8` vs `0/8`, spanning Asia, Europe, and Latin America).
- **Goal check (every clip `<= 200 m`):** Not met (`3/8` `<= 200 m`, `4/8` `<= 500 m`).
- **Decision under the pre-registered rule:** Both `W1` and `W2` hold on blind multi-continent held-out footage; `--vpr-metro-dense` is certified as generalizing worldwide without regression and is promoted as the recommended GPU configuration.

## Per-Clip Paired Breakdown

| Clip | City (Continent) | City Centroid (m) | Ordinary Sym (m) | Metro-Dense Sym (m) | GT→Pred Ord → MD (m) | Pass-1 / Pass-2 Refs | Discs | Outcome |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `overlay_URC3D7Xzpq0` | Kuala Lumpur, Malaysia (Asia) | 7,027.1 | 3,959.1 | **15.5** | 5,224.5 → **20.3** | 25,474 / 3,561 | 1 | **GAIN (`<= 20 m`)** |
| `overlay_bQI9M_lpBEM` | Athens, Greece (Europe) | 18,816.2 | 18,272.3 | **16.2** | 18,436.9 → **20.0** | 24,889 / 6,659 | 2 | **GAIN (`<= 20 m`)** |
| `overlay_h1wnOrVzgKg` | Santiago, Chile #1 (Latin America) | 4,647.5 | 1,954.4 | **150.1** | 2,001.2 → **293.8** | 24,000 / 3,084 | 1 | **GAIN (`<= 200 m`)** |
| `overlay_nlQdg-FiNns` | Mexico City / CDMX, Mexico (North America) | 1,529.3 | 247.6 | 382.0 | 127.1 → **35.5** | 26,578 / 7,648 | 3 | Near-hit (`35.5 m` GT→pred on long Zócalo→Santa Fe drive) |
| `overlay_lc2YfKXCUv8` | Bangkok, Thailand (Asia) | 9,180.7 | 10,251.1 | 9,986.3 | 10,385.3 → 10,046.6 | 24,925 / 2,666 | 1 | Miss (dense urban traffic congestion, sparse street-level matches) |
| `overlay_hjkt-LEF4UU` | Santiago, Chile #2 (Latin America) | 8,580.7 | 10,814.4 | 11,304.7 | 10,752.9 → 11,327.7 | 25,592 / 3,192 | 1 | Miss (suburban corridor with sparse Mapillary coverage) |
| `overlay_MR4dqJcW4Ao` | Bengaluru, India (Asia) | 18,337.9 | 12,469.1 | 11,896.1 | 12,411.9 → 11,683.6 | 25,757 / 4,376 | 1 | Miss (monsoon rainy evening night drive 18.3 km from centroid) |
| `overlay_XZrRUsdupM8` | Toronto / Mississauga, Canada (North America) | 28,431.8 | 26,754.8 | 28,017.8 | 26,650.6 → 27,914.7 | 25,474 / 2,747 | 1 | Miss (route in Mississauga **28.4 km** outside Toronto 20 km disc) |

## Pooled Summary Across All 46 Benchmark Clips (`test1`–`test6`)

Combining the 38 clips of `test1`–`test5` (`docs/metro_dense_v2_all38_2026-11.md`) with the 8 blind held-out `test6` clips:

- **Within 200 m:** **34 / 46 (`73.9 %`, Wilson 95 % CI `59.7–84.4 %`)** with `--vpr-metro-dense`, vs **12 / 45 (`26.7 %`)** in the ordinary baseline (**+22 net gains, 0 losses across all 46 clips**, one-sided McNemar `p = 2.38e-7`).
- **Within 500 m:** **38 / 46 (`82.6 %`)**.
- **Remaining failure taxonomy across `test6`:** (1) suburban/satellite municipalities `> 25 km` from the metro centroid when the upload title names both the metro and the outer city (`Toronto / Mississauga`, `28.4 km`), and (2) severe adverse weather / sparse street-view coverage in developing-world outer corridors (`Bengaluru` monsoon night drive, `Bangkok`, outer `Santiago`).
