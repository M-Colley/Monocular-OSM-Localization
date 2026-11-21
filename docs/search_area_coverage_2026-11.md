# Why far misses happen: the route is outside the searched area (November 2026)

This is a post-hoc diagnostic on scored clips. It reads ground truth, and
nothing here was tuned or is pre-registered. `scripts/route_disc_coverage.py`
measures how much of each true route lies inside the pass-1 disc the pipeline
actually searched. It reads that disc's centre and radius from the clip's
Mapillary cache. Data: `docs/route_disc_coverage_2026-11.json` (36 clips:
test5, plus the test-split clips with a cached pass 1).

## Finding

**Almost every miss more than a kilometre off has its true route outside the
searched disc.** No retrieval improvement can fix those.

- **In-sample (`dense_pass1_dev_2026_11`, dense arm):**
  - All 6 misses 4-38 km off have 0-5 % of the route inside the disc:
    Montreal, Paris, Munich, Bristol, Katowice `MMKAehhi-tA` and Düsseldorf
    `HfK2LuXQtfA`.
  - The 250-420 m near-misses mostly lie inside it: Bucharest, Frankfurt and
    Cork at 95-100 %. Those are precision problems.
  - 13 of 15 hits have at least half the route inside.
- **Held out (test5):**
  - 5 of the 6 misses have 0-10 % inside: Houston, both Portland clips,
    Glasgow, Belgrade.
  - These are suburban drives 8-20 km from the city centre: Gresham, Happy
    Valley, Howwood. The pipeline searches at most 8 km (3 km when seeded or
    when the city extent is small).
  - The one dense gain, Calgary, had 65 % inside.

| Clip | Disc | Route inside | Route from disc centre |
|---|---|---:|---|
| test5 Houston | 8 km | 0 % | 13-16 km |
| test5 Portland `3cT07o9aXyQ` | 8 km | 0 % | 9-15 km |
| test5 Portland `zsqxTSNLw5A` | 8 km | 0 % | 9-17 km |
| test5 Glasgow | 3 km | 0 % | 12-20 km |
| test5 Melbourne (OCR seed) | 3 km | 0 % | 4-10 km |
| test5 Calgary (gain) | 8 km | 65 % | 4-11 km |
| in-sample Montreal / Paris / Bristol | 8 km | 0 % | (far misses) |
| in-sample Munich / Düsseldorf `HfK2` | 3 km | 0-5 % | (far misses) |

## Reading

The binding constraint on new footage is the **size of the searched area**,
not reference density or the appearance model. The screens accept clips whose
stamps lie within 40 km of the named city. The pipeline assumes the drive is
within about 8 km of the geocode.

The existing metro-ring search covers this band (8-20 km). Its record:

- in-sample 20/29 against 16/29;
- held out (test3) one clip better and two worse;
- it only triggers on low confidence, and it uses sparse references and CPU
  time.

A method that always searches the metro area densely (on the GPU) is the
evidence-backed next candidate. It is not built or tested here, and a fair
test needs new held-out data: test5 is now used.
