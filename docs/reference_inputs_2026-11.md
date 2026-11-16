# Pass-1 reference inputs: two diagnostics (November 2026)

Two leads came from reading a cross-view correspondence paper draft. It cuts
360-degree panoramas into perspective views before matching, and measures
Mapillary capture spacing at about 3.5 m. Both leads were checked against
this project's own first-pass (pass-1) references before building anything.
Both are diagnostics on already-used clips. Nothing was tuned, and neither
lead is pre-registered as a method.

## 1. How much of the route has a reference nearby

`scripts/ref_route_coverage.py` measures how much of each ground-truth route
has a served pass-1 reference within 50 m and 150 m. It compares the served
hash subsample (1,200 of the listed images, chosen by lowest sha1) with a
grid-stratified subsample of the **same listing and cap** (one image per 250 m
or 400 m cell). It reads ground truth, so it is diagnostic only. Data:
`docs/pass1_route_coverage_2026-11.json`.

| Clip | Listed images | Served (hash) within 50 m | Grid 250 m within 50 m | Any listed image within 50 m |
|---|---:|---:|---:|---:|
| Strasbourg `4LvkBo69jZo` (ok) | 194,039 | 62 % | 24 % | 100 % |
| Manchester `5b-tgjtwb2U` (ok) | 58,697 | 61 % | 25 % | 100 % |
| Eindhoven `6GciDYJ1FUc` (ok) | 804,948 | 62 % | 18 % | 88 % |
| Budapest `1bGf8OD9Fqo` (mode C) | 3,675,606 | 31 % | 24 % | 100 % |
| Bern `zt8rHvHKBUU` (mode B) | 299,009 | 20 % | 18 % | 100 % |
| Düsseldorf `8BUtb7TKUFA` (260 m) | 301,084 | 19 % | 14 % | 64 % |
| Ottawa `bF5IfC2ZHjQ` (mode A) | 1,193,243 | 14 % | 17 % | 89 % |
| Cork `vp_D8mxgEkA` (miss) | 90,898 | 6 % | 30 % | 80 % |

- **The listing covers the routes; the served subsample does not.** Pass 1
  keeps 1,200 of 59 thousand to 3.7 million listed images (7.5 million in
  Riga, 7.1 million in Detroit).
- **Coverage tracks the outcome.** The clips that localize have 61-62 % of
  their route within 50 m of a served reference. The misses have 6-31 %.
- **Even spatial sampling is not a free fix.** It helps Cork (6 to 30 %) and
  Ottawa a little, but hurts every clip whose route runs through densely
  captured areas. Hash sampling follows capture density, and capture density
  is itself a prior on where people drive.
- **A denser pass 1 was already tested.** Raising the cap to about 3,000 on
  four test3 clips moved no clip across 200 m
  (`docs/heldout_test3_2026-11.md`, "A denser first pass").

## 2. How many references are panoramas

Our MegaLoc descriptors see a 360-degree panorama squeezed into a 512 x 512
square. `scripts/pass1_camera_types.py` asks the Mapillary Graph API for the
`camera_type` of each served pass-1 reference. This is GT-free input metadata.
Data: `docs/pass1_camera_types_2026-11.json` (13 clips with a cached pass 1 and
a taxonomy entry).

- **Panoramas dominate only in Detroit:** 87 % of references for both Detroit
  clips (`g5lnpYCk1Ec` misses, `ZhGb8q1kliY` succeeds).
- **Everywhere else they are 1-17 %.** That includes the misses in Ottawa
  (10 %), Bern (1 %), Atlanta (3 %) and `wJsEQTCAg1c` (3 %).
- **Fisheye references are common in some cities:** 39 % in Berlin, 25 % in
  Ottawa and 20 % in Budapest. They have their own projection mismatch.

Panorama-aware global descriptors could therefore matter for Detroit-like
galleries at most. They do not explain the misses in general.

## Reading

Neither lead is strong enough to pre-register as a method. The sharper finding
is the coverage gap: misses have few served references near their route. Any
remedy has to raise coverage where the route actually is, without the prior
that density sampling gives for free, and that is not yet a method. Testing
anything new for generalization needs a fresh held-out split, because all four
existing test splits are used.
