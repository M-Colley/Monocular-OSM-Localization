# Held-out test, November 2026 (`test2`)

Rule `test2_2026_11` (`benchmarks/preregistered_rules.json`), registered on
2026-11-06 before any test2 run: the assisted policy v3 (commit `a8add7a`,
`--assisted-only --assisted-profile=v3`, 14,400 s CPU budget from the
`assisted_v3_test1_replay` budget rule) against the frozen retrieval track
(commit `ccd0bbe`), one cold rep per clip, CPU retrieval, identical inputs
(the ground-truth builder's own 0-600 s download of each video,
`--skip-download`). The seven clips come from six cities that no development
step used, chosen by the fixed title/duration/stamp screen in
`benchmarks/test2_screen/`. Two further candidates (`jur2mWIr9IY`, Montreal,
and `y2om3vbdcKc`) failed the ground-truth builder and are not in the split.

Nothing was changed between seeing a test2 result and reporting all of them.

**Result in one line:** both methods put 2 of 7 clips within 200 m
(Warsaw 90 m, Amsterdam 172 m); the assisted policy returned the track's
answer on six clips and improved one (Paris, 24 km to 832 m). H1, H2 and H3
hold, but with no gain in the share within 200 m.

## Assisted v3 (complete)

Scored by the reviewed comparison adapter (`5bdae79`) with
`--manifest E:/claude-scoring/test2/manifest_test2.json`, ground truth
applied offline. Symmetric mean route error (m):

| Clip | City (as titled) | Path through the policy | **Assisted answer** | City centre |
|---|---|---|---|---|
| `GnXsgpVPvww` | Warsaw | strong baseline kept (median cosine 0.193) | **90** | 4,773 |
| `2sS6glBBm8g` | Amsterdam | strong baseline kept (0.213) | **172** | 571 |
| `lCm2a3kOEmE` | Warsaw | weak baseline; cues; expanded failed (budget) -> baseline | 335 | 5,286 |
| `1bGf8OD9Fqo` | Budapest | strong baseline kept (0.179) | 518 | 794 |
| `HpBfjRA60ws` | Paris | weak baseline; cues (VO 0.89, no sign branch); expanded selected | 832 | 15,259 |
| `0jJbdTcDseI` | Montreal | weak baseline; cues; expanded complete, not admitted -> baseline | 3,004 | 11,050 |
| `XO6H3mGMKYg` | Munich | weak baseline; cues; expanded failed (budget) -> baseline | 8,130 | 12,061 |
| **Within 200 m** | | | **2 / 7** (29 %, Wilson 95 % 8-64 %) | 0 / 7 |
| **Median** | | | **518** | 5,286 |

On dev the same commit and budget reach 13/15 (87 %, Wilson 62-96 %). The
held-out share is far lower and the intervals barely overlap: the assisted
policy's stage thresholds and fallbacks were tuned on the dev clips, and on
new footage the policy mostly falls back to the plain city-wide retrieval,
which then decides the outcome. It still beats the city-centre guess on
every clip (7 wins, 0 losses; one-sided sign test p = 0.008, H3 for the
assisted method).

Audit: the strict artifact audit (`check_dev_goal.py` with the test2 clips
as its target) first flagged Montreal: "descriptor receipt source is not the
actual frozen retrieval implementation". The receipts carry the hash of
`a8add7a`'s own `kartaview_vpr.py` (`9a191d6f...`, checked against the file);
the audit looked the declared hash up under a POSIX key while Windows runs
record `monocular_osm\kartaview_vpr.py`, and the one dev clip that reached the
check had no receipts, so it had never compared anything. With the lookup
normalised (`posix_sources`, now in `scripts/assisted_goal_audit.py` with a
regression test) all seven runs pass the audit; the only remaining reason on
five clips is the 200 m outcome itself.

Two runs were degraded by infrastructure without failing: on Montreal
`0jJbdTcDseI` and Warsaw `lCm2a3kOEmE` the cues stage's retail-inventory
request to Overpass returned 504/502 twice, 2-3 s apart, so their OCR cue is
`failed` while the run went on to an answer. The pre-registered rerun rule
covers runs that fail on infrastructure; the harness saw these as complete,
so they were scored as they are. The request now retries with the road-graph
backoff (commit 1eedf8e), which applies only to later runs. Separately, on
Munich and Warsaw `lCm2a3kOEmE` the metadata-union step was refused because
one to six sampled Mapillary images had no URL ("complete capped served tile
inventory required"; Detroit `g5lnpYCk1Ec` on dev has the same), which left
those expansions to the slower municipality grid that ran out of budget.

Two operator errors during the run, both before any score was looked at
(a misread wall clock made a finished run look stalled): the completed
Warsaw `lCm2a3kOEmE` result directory was moved aside and its per-run
retrieval caches under `data-assisted` deleted; the result directory was
restored unchanged (it passes the audit above, which binds every frozen
artifact by hash), the caches are gone. The Munich clip was stopped mid-run;
its directory is kept under `E:/claude-scoring/test2/interrupted/` and the
clip was rerun from restored inputs at the same commit (wall 14,278 s of the
14,400 s budget).

## Track (`ccd0bbe`) and the registered hypotheses

The frozen track method ran afterwards in the clean `ccd0bbe` worktree
(`track_test2`, `--cold-vpr`, its own data copy
`E:/claude-scoring/test2/data-track`, every clip `rc=0`, 1,355-1,780 s each,
no infrastructure rerun) and was scored by `ccd0bbe`'s own aggregator, as the
rule requires, with the same manifest, ground truth and geocode snapshot.
Symmetric mean route error of each method's answer (m):

| Clip | City | Track (`ccd0bbe`) | Assisted v3 (`a8add7a`) | City centre | Assisted vs track |
|---|---|---|---|---|---|
| `GnXsgpVPvww` | Warsaw | **90** | **90** | 4,773 | same answer |
| `2sS6glBBm8g` | Amsterdam | **172** | **172** | 571 | same answer |
| `lCm2a3kOEmE` | Warsaw | 335 | 335 | 5,286 | same answer |
| `1bGf8OD9Fqo` | Budapest | 518 | 518 | 794 | same answer |
| `HpBfjRA60ws` | Paris | 24,364 | 832 | 15,259 | **assisted wins** |
| `0jJbdTcDseI` | Montreal | 3,004 | 3,004 | 11,050 | same answer |
| `XO6H3mGMKYg` | Munich | 8,130 | 8,130 | 12,061 | same answer |
| **Within 200 m** | | **2 / 7** (Wilson 8-64 %) | **2 / 7** (8-64 %) | 0 / 7 | |
| **Median** | | 518 | 518 | 5,286 | |

- **H1** (assisted within 200 m on at least as many clips as the track):
  holds, 2 = 2. The assisted policy added no clip within 200 m.
- **H2** (assisted beats the track on more clips than it loses): holds,
  1 win, 0 losses, 6 ties; one-sided sign test p = 0.5. The six ties are not
  near-ties: the answers score identically to 0.1 m (the routes of Montreal
  and Amsterdam were compared point by point and are identical), because the
  policy either kept its baseline or fell back to it, and that baseline
  reproduced the track method exactly.
- **H3** (each method beats the city centre on more clips than it loses):
  holds for both. Track 6 wins, 1 loss (Paris), p = 0.063; assisted 7 wins,
  0 losses, p = 0.008.

The one difference is Paris, and it comes from the policy's expansion stage,
not from a better baseline. The assisted run's own baseline child also
failed there (16,149 m, rescored offline with `monocular_osm.metrics.score_route`
on the same ground truth; the track's 24,364 m reproduces with the same
call), it was weak (median cosine 0.141, under the 0.162 gate), and the
expanded search it then admitted scored 832 m. The two baselines differ on
this one clip only (track median cosine 0.127, about 17 km apart); the track
ran a day later with a cold cache, 11 of the 106 commits between `ccd0bbe`
and `a8add7a` touch the retrieval modules, and the cold cache was not kept, so whether
Mapillary changed or the code did is not known. Both baselines are worse than
the city-centre guess, so either way the win belongs to the expansion.

Reading: on footage from new cities the retrieval track, not the policy, sets
the outcome. The track reaches 2/7 here against 5/7 on test1 and 9/15 on dev;
the assisted policy's 13/15 on dev did not carry over (2/7), and its added
stages changed one answer in seven, from a 24 km failure to an 832 m miss.
Seven clips detect only large effects; this one is a null result for the
200 m criterion.
