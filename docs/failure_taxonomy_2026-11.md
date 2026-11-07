# Where the retrieval track fails: search coverage, November 2026

A diagnostic over every scored clip of the frozen retrieval-track method, the
default answer (`--answer track`): dev (run v6, commit `d87fb20`, whose answer
is the track; 9/15 within 200 m), test1 (`ccd0bbe`, 5/7) and test2
(`ccd0bbe`, 2/7). For each clip it measures how much of the ground-truth
route lay inside the two discs the method actually searched:

- the pass-1 disc: the title-derived prior when the run found place names
  in the video title or description, else 8 km around the city's geocode;
- the pass-2 (coarse-to-fine) disc, 2 km around the pass-1 centre.

It then sorts each miss into one mode:

| Mode | Meaning |
|---|---|
| A | route mostly outside the pass-1 disc (< 50 % of points inside) |
| B | route inside, but pass 1 picked the wrong area (< 25 % inside pass 2) |
| C | right area, but the 2 km pass-2 disc covers only part of the route (25-75 %) |

Ground truth is used here only, after every run was scored; test1 and test2
were already reported (`docs/heldout_test_2026-10.md`,
`docs/heldout_test2_2026-11.md`), and nothing in this note changes a method
or a past result. Reproduce with `scripts/failure_taxonomy.py`; per-clip
rows are in `docs/failure_taxonomy_2026-11.json`.

| Split | Clip | Route (km) | Pass-1 disc (km) | Route in pass-1 disc | Route in pass-2 disc | Confidence | Answer (m) | Mode |
|---|---|---|---|---|---|---|---|---|
| dev | Berlin (lBlKR2ek0w4), 0-420 s | 3.3 | 6 (title) | 1.00 | 0.75 | 0.189 | 135 | ok |
| dev | KITTI 2011_09_26 drive 0009 | 0.3 | 8 | 1.00 | 0.00 | 0.117 | 3,735 | B |
| dev | KITTI 2011_09_30 drive 0033 | 1.5 | 8 | 1.00 | 0.00 | 0.105 | 7,004 | B |
| dev | London (T4wTL3LpLqU) | 1.3 | 6 (title) | 1.00 | 1.00 | 0.193 | 62 | ok |
| dev | Detroit (1nF_7l07i-E) | 2.9 | 8 | 1.00 | 0.95 | 0.163 | 77 | ok |
| dev | Detroit (g5lnpYCk1Ec) | 10.4 | 8 | 0.20 | 0.25 | 0.155 | 1,573 | A |
| dev | Atlanta (kxEDNj5L_yQ) | 12.4 | 6 (title) | 0.65 | 0.00 | 0.148 | 6,024 | B |
| dev | Chicago (LmIHvLMLqFk) | 1.5 | 8 | 1.00 | 1.00 | 0.162 | 133 | ok |
| dev | Chicago (vCSEG6KaFng) | 1.4 | 8 | 1.00 | 1.00 | 0.186 | 169 | ok |
| dev | Phoenix (wJsEQTCAg1c) | 0.9 | 8 | 0.00 | 0.00 | 0.132 | 18,644 | A |
| dev | Newcastle upon Tyne (Wrn4_uCxRCQ) | 4.0 | 6 (title) | 1.00 | 0.95 | 0.159 | 35 | ok |
| dev | Atlanta (y66ZkRpUh4k) | 1.1 | 8 | 1.00 | 0.00 | 0.143 | 6,590 | B |
| dev | Detroit (ZhGb8q1kliY) | 1.5 | 8 | 1.00 | 1.00 | 0.167 | 55 | ok |
| dev | Ulm #2 (LC9Sa--u5KE) | 2.1 | 8 | 1.00 | 1.00 | 0.201 | 46 | ok |
| dev | Ulm 4K (ULl8s4qydrk), 0-300 s | 1.2 | 2 (title) | 1.00 | - | 0.268 | 15 | ok |
| test1 | Sydney (-1xwoetuoOg) | 3.3 | 8 | 1.00 | 1.00 | 0.151 | 192 | ok |
| test1 | Strasbourg (4LvkBo69jZo) | 5.2 | 8 | 1.00 | 0.70 | 0.232 | 34 | ok |
| test1 | Manchester (5b-tgjtwb2U) | 3.6 | 2 (title) | 0.85 | 0.85 | 0.182 | 56 | ok |
| test1 | Ottawa (bF5IfC2ZHjQ) | 8.5 | 8 | 0.30 | 0.10 | 0.131 | 1,380 | A |
| test1 | Frankfurt am Main (iJ2JvNk4Rmk) | 3.4 | 8 | 1.00 | 1.00 | 0.187 | 84 | ok |
| test1 | NYC (nsONSyk9wtc) | 4.5 | 4 (title) | 0.20 | 0.90 | 0.202 | 53 | ok |
| test1 | Bern (zt8rHvHKBUU) | 4.9 | 8 | 1.00 | 0.00 | 0.136 | 3,437 | B |
| test2 | Montreal (0jJbdTcDseI) | 4.9 | 8 | 0.00 | 0.00 | 0.152 | 3,004 | A |
| test2 | Budapest (1bGf8OD9Fqo) | 5.5 | 8 | 1.00 | 0.50 | 0.179 | 518 | C |
| test2 | Amsterdam (2sS6glBBm8g) | 2.5 | 8 | 1.00 | 1.00 | 0.213 | 172 | ok |
| test2 | Warsaw (GnXsgpVPvww) | 4.0 | 8 | 1.00 | 0.85 | 0.193 | 90 | ok |
| test2 | Paris (HpBfjRA60ws) | 14.8 | 8 | 0.00 | 0.00 | 0.127 | 24,364 | A |
| test2 | Warsaw (lCm2a3kOEmE) | 6.7 | 8 | 0.95 | 0.55 | 0.147 | 335 | C |
| test2 | Munich (XO6H3mGMKYg) | 2.2 | 6 (title) | 0.00 | 0.00 | 0.124 | 8,130 | A |

"Confidence" is the track's median decoded-reference cosine, the signal of
the pre-registered selective rule (τ = 0.162).

## What it shows

1. **When the pass-2 disc covers the route, the track is accurate.** Every
   clip with at least 75 % of its route inside the pass-2 disc is within
   200 m (14 of 14, across all three splits). Between 25 and 75 % it is 1 of
   4 (Strasbourg 34 m; Budapest 518 m, Warsaw `lCm2a3kOEmE` 335 m, Detroit
   `g5lnpYCk1Ec` 1,573 m); below 25 %, 0 of 10. (Ulm 4K's 2 km title disc
   needed no pass 2; 15 m.)
2. **Held-out misses are coverage misses.** Of the seven held-out failures,
   four are mode A (Ottawa, Montreal, Paris, Munich: 0-30 % of the route in
   the searched disc) and two are mode C (Budapest, Warsaw `lCm2a3kOEmE`: the
   right area, but 5.5-6.7 km routes against a 2 km disc). One is mode B
   (Bern).
3. **Dev misses are mostly appearance misses.** Of the six dev failures four
   are mode B with the route inside the disc (both KITTI drives, Atlanta
   `kxEDNj5L_yQ`, Atlanta `y66ZkRpUh4k` at night) and two are mode A
   (Detroit `g5lnpYCk1Ec`, Phoenix). The assisted policy was developed on
   these clips and mostly adds evidence for mode B (OCR, shop signs,
   sign-site verification, municipality grids). That fits its held-out
   result: on test2 it returned the track's answer on six of seven clips
   (`docs/heldout_test2_2026-11.md`).
4. **Long routes outgrow the fixed geometry.** The pass-2 radius is
   `max(2,000 m, 0.6 x length prior)`, and the length prior is the clip
   duration at 5.5 m/s (3,272 m for 595 s). The routes here run 0.3-14.8 km,
   and the highway clips are 2.4-4.5 times the prior (Warsaw `lCm2a3kOEmE`
   to Paris).
5. **The confidence signal nearly separates these cases.** Every one of the
   16 successes has confidence 0.151 or more; every one of the 13 misses has
   0.155 or less, except Budapest (0.179). The two groups overlap only in
   that narrow band (Sydney 0.151 succeeds; Montreal 0.152 and Detroit
   `g5lnpYCk1Ec` 0.155 miss). A weak track says that something is wrong; it
   does not say which mode.

## Detroit `g5lnpYCk1Ec` in detail (dev)

The pass-1 disc holds only the last 20 % of the 10.4 km drive. Pass 1 still
finds that part: its 2 km pass-2 disc lies on the freeway section, within
300 m of the car at 479 s. But the first 330 s of surface streets lie 12-15 km
from the city centre, outside both discs. Coverage is not the only problem.
Inside the pass-2 disc, with references 15-190 m from the true position at
380-569 s, the decoded positions are still 330 m to 4.5 km off (at 531 s the
best single match is 15 m away and the decode is 3.2 km away). Repetitive
freeway imagery leaves the sequence decoder without a stable signal there.
The exit numbers read by OCR (186B, 188) mark the right freeway section
(`docs/assisted_fleet_2026-11.md`), but an exit-centred dense search still
scored 809-909 m, consistent with this.

## Two observations for later work

- On Munich `XO6H3mGMKYg` (test2), the title prior resolved "Oberhaching,
  Camera, Bavaria" and searched a 6 km disc on central Munich, while the
  drive runs toward Germering, 12 km west. "Camera" comes from the
  description's equipment line. Treating such words as place names is a
  resolver bug. It was found on a held-out clip after scoring, so a fix
  cannot be credited there.
- Mode A is six of thirteen failures across the splits, and the earlier
  responses to it have not generalised. The seed ring (v7) was not adopted.
  The municipality grid misses on Detroit and exceeds the CPU budget on two
  test2 clips. A held-out claim for any new answer to modes A or C needs a
  third held-out split, since test1 and test2 have both been used.
