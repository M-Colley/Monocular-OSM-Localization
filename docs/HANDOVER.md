# Handover: reliable localization work (state on 21 November 2026)

For the next agent. It covers where the project stands, how evaluation works
here, what has been tried, and what to do next. Read this first, then
`docs/code_knowledge.md` for the code itself.

## 1. Goal and where it stands

**Goal:** every clip within 200 m (symmetric mean route error between the
answer and the ground-truth route) from a monocular dashcam video plus a
city name. **It is not yet 100 %, and `--vpr-metro-dense` is now the certified
recommended GPU configuration (`34/46 = 73.9 % <= 200 m` across `test1`–`test6`,
0 losses vs ordinary).**

| Set | Method | Within 200 m | Record |
|---|---|---|---|
| dev (15 clips, used for tuning) | assisted v4 (`409a959`) | 14/15 | `docs/assisted_fleet_2026-11.md` |
| held out test1-test4 (30 clips) | frozen historical methods, pooled | 13/30 | `docs/heldout_test*_2026-1*.md` |
| test4 (6) | assisted v4 | 1/6 | `docs/heldout_test4_2026-11.md` |
| test5 (8) | plain pipeline, ordinary vs dense pass 1 | 1/8 vs 2/8 | `docs/heldout_test5_2026-11.md` |
| test1-test5 all 38 (in-sample) | `--vpr-metro-dense` v2 (`31a6f57`, GPU) | **31/38 (81.6 %)** (`30/37` paired vs `18/37` dense_24k & `12/37` ordinary, 0 losses, median 41.2 m) | `docs/metro_dense_v2_all38_2026-11.md` |
| **test6 worldwide (8, blind held-out)** | **ordinary vs `--vpr-metro-dense` v2 (`29273ac`, tree `75f8e90c`)** | **0/8 (0 %) vs 3/8 (37.5 %)** (`4/8 <= 500 m`, 0 losses; KL 15.5 m, Athens 16.2 m, Santiago 150.1 m) | `docs/heldout_test6_2026-11.md` |
| **Pooled test1-test6 (46 clips)** | **ordinary vs `--vpr-metro-dense` v2** | **12/45 (26.7 %) vs 34/46 (73.9 %)** (+22 net gains, 0 losses, `p = 2.38e-7`) | `docs/heldout_test6_2026-11.md` |

**Splits `test` through `test6` have all been evaluated.** `--vpr-metro-dense`
passed both pre-registered blind worldwide hypotheses (`W1` no harm, `W2`
worldwide benefit `3/8 > 0/8` across Asia, Europe, and Latin America). Remaining
misses on `test6` are satellite municipalities `> 25 km` from the metro centroid
(`Toronto / Mississauga` `28.4 km`) and sparse street-view / monsoon-night
corridors (`Bengaluru`, `Bangkok`, outer `Santiago`).

## 2. Repository and branch

- **Branch:** `codex/reliable-localization`, open as PR
  https://github.com/M-Colley/Monocular-OSM-Localization/pull/3. Last pushed
  commit `79517c1`; the working tree was clean at handover.
- **Pipeline:** `monocular_osm/` (stages in `monocular_osm/stages/`,
  retrieval in `kartaview_vpr.py`, CLI in `cli.py`, config in `pipeline.py`).
- **Running and scoring:**
  - Benchmark runner: `scripts/benchmark.py` (`run`, `aggregate --offline`).
  - Manifest: `benchmarks/manifest.json`, derived from
    `benchmarks/build_manifest.py`. Never hand-edit it. Check it with
    `python benchmarks/build_manifest.py --check`.
  - Pre-registrations and recorded outcomes:
    `benchmarks/preregistered_rules.json`.
- **Changelog:** `CHANGELOG.md` is current. `README.md` stops at test4; the
  dense-pass and test5 results are not in it yet.

## 3. How evaluation works here (keep to it)

1. **Pre-register before running.** Add a rule to
   `benchmarks/preregistered_rules.json` with: method commit, clips,
   flags, hypotheses, decision rule and disclosures. Commit and push it
   **before** any scored run.
2. **Never tune on a test split.** Diagnostics on used splits are allowed if
   labelled post hoc.
3. **Run from a clean detached worktree at the registered commit**, so edits
   in the main checkout cannot leak into a run. Examples:
   `E:/claude-scoring/wt-dense` (`b66b9ee`), `E:/claude-scoring/wt-test5`
   (`3f8f8f1`).
   - The runner passes each clip's registered `video.local_path` (relative
     to the checkout) as `--video-path`. So stage the worktree's untracked
     `data/` with junctions to the main checkout's folders (Python
     `_winapi.CreateJunction`). Use hard links where builder downloads were
     deleted (dev overlay clips → `E:/claude-scoring/v4fleet/data/<slug>/input.mp4`).
     Then verify every path resolves.
4. **Infrastructure check first.** Before reading any score, check:
   - exit codes, results present, `answer.method`;
   - for dense runs, pass-1 `n_references` and `decoder == viterbi`.

   Commit that check as `infra_check` in the rule.
5. **Score, then record.** Run `benchmark.py aggregate --offline` per run
   name, then the paired comparison:
   - `scripts/compare_paired_arms.py RUNS ORD_PREFIX NEW_PREFIX SPLIT...`
     for two arms named `<prefix>_<split>`;
   - `scripts/metro_ring_paired.py shadow/compare --record c2f_extension|metro_ring_retrieval`
     for one-run records;
   - `scripts/score_heldout_assisted.py` for assisted runs.

   Wilson intervals come from statsmodels. Write `outcome_recorded`, a
   `docs/` report and a CHANGELOG entry. Report every clip.

## 4. What has been tried (outcomes)

- **Assisted policy v1-v4:** dev 14/15. Held out it mostly keeps the
  baseline track: test4 1/6.
- **`--vpr-metro-ring`** (8-20 km cells, triggers when confidence < 0.162):
  in-sample 20/29 vs 16/29, held out (test3) +1/-2. It is opt-in.
- **`--vpr-c2f-extend`** (follows a drive past the 2 km pass-2 disc):
  failed its dev check (one loss, one gain; `docs/c2f_extension_dev_2026-11.md`).
  It is opt-in.
- **`--vpr-pass1-cap N`** (dense city-wide pass 1, `4a671a9`/`b66b9ee`):
  - Above 6,000 references, descriptors are streamed with no pixel store,
    and the exact Viterbi decode runs in torch on CUDA.
  - Dev check: 11/29 → 16/29, no loss (`docs/dense_pass1_dev_2026-11.md`).
  - test5: 1/8 → 2/8 (one gain, Calgary). It is opt-in and the default is
    unchanged.
- **Leads from a cross-view paper draft** (`docs/reference_inputs_2026-11.md`):
  - Panoramas are 87 % of pass-1 references only in Detroit.
  - Grid-stratified reference sampling is not a free win.
  - A cap of about 3,000 references moved nothing on test3.
- **Many retrieval and geometry models** (ImAge, QAA, SelaVPR++, MutualVPR,
  XFeat/Pair-VPR re-ranking, OrienterNet/OSMLoc, MapAnything, DVLT, particle
  filters): see `docs/recent_methods_2026-10.md`. None crossed the target.

## 5. Main diagnosis: the route is outside the searched area

From `docs/search_area_coverage_2026-11.md` (`scripts/route_disc_coverage.py`):

- **Far misses:** every in-sample miss over 4 km, and 5 of 6 test5 misses,
  have 0-10 % of the true route inside the pass-1 disc the pipeline
  searched.
- **The mismatch:** the pipeline searches at most 8 km around the city
  geocode, and only 3 km when seeded from a title, OCR or a plate, or when
  the city extent is small. The data screens accept clips within 40 km of the
  named city, so suburban drives (Portland's Gresham, Glasgow's Howwood,
  Houston suburbs) are unreachable.
- **Near misses (250-420 m)** lie inside the disc. They are track-precision
  problems.

## 6. Recommended next step (not started; the user has not decided)

A **dense, always-on metro-scale search** (about 20 km) on the GPU,
pre-registered as an in-sample check on the used test-split clips: 38 clips,
37 runnable, because the runner's fairness guard refuses `yd5g7jsWiEM`. If it
passes, it needs a fresh held-out split (test6).

Design notes:
- **Reuse `_metro_ring_vpr` in `stages/prior.py`.** Drop its confidence gate,
  or test it both ways, and give the cells dense caps.
- **Viterbi memory.** The decode is O(n²). The torch path (`_viterbi_decode_torch`)
  needs about 3 × n² × 4 bytes of GPU memory: 24,000 references fit in
  24 GB, about 60,000 would not. Beyond that, decode within cells or restrict
  each frame to its top-K references.
- **Cost.** A 20 km disc is 6.25 times the area of the 8 km one. Expect
  several GB of thumbnails per clip.
- **Ties by construction.** Seeded clips (title, OCR or plate priors with a
  3 km disc) are ties under the current cap. Decide whether the metro search
  also applies to them.

**Ask the user before** downloading new videos, using the GPU (their
llama-server normally holds it), or starting a new split.

## 7. Making a new held-out split (lessons from test5)

- **Selection:** follow `benchmarks/test5_screen/`.
  - Register the selection and method together before any search.
  - Use a fixed city pool, three queries per city, the unchanged V3 title
    eligibility, a stamp screen (thumbnail, then frames at 25/50/75 %,
    at least 2 within 40 km), and the unchanged ground-truth builder and QC
    (`build_one.py`).
  - Run `drive.py` until the target is reached or the pool runs out.
- **Yield is low:** 40 cities and 120 queries gave 8 clips in about 20 hours.
  Most uploads have no coordinate stamp. Budget for it, or use a larger pool.
- **YouTube access** (see `operational_amendment_0{1..5}.json`):
  - Read frames from HTTPS streams.
  - Give yt-dlp a JavaScript runtime (`--js-runtimes node` plus `pip install
    "yt-dlp[default]"` for yt-dlp-ejs).
  - Reopen refused streams.
  - Cap persistently unreadable candidates at 3 attempts.
  - Bot checks pause for 30 minutes. Never use browser cookies.
- **Fairness guard:** `scripts/benchmark.py city_fairness_problem` refuses
  clips whose title names the city as an exonym (test4's `yd5g7jsWiEM`
  title says "Bucuresti"). test5 avoided this by registering the title's own
  spelling. The guard could learn exonyms; that is a runner change.
- **Adding the split:**
  - add the clips to `scripts/overlay_clips.py`;
  - add the split description, cities and basis to
    `benchmarks/build_manifest.py`;
  - add the `--split` choice in `scripts/benchmark.py`;
  - add the allowed splits in `tests/test_benchmark_manifest.py`;
  - snapshot centroids with `python benchmarks/build_manifest.py --geocode`.

## 8. Environment gotchas

- **`MLY_TOKEN`** (Mapillary) is only in the Windows user registry.
  `benchmark.child_env()` reads it. **Never print or log its value.**
- **GPU:** an RTX 4090, usually held by the user's llama-server. Do not stop
  or touch it, or any other project's processes. Ask first. CPU-only runs use
  `CUDA_VISIBLE_DEVICES=-1` and `--min-free-vram-gb 0`.
- **Overpass:** one of `overpass-api.de`'s two IPs is black-holed from this
  network. `net.prefer_reachable_host` handles it. Outages still happen; with
  `--answer track` the geometry stage failing does not change the answer.
- **Disk:** C: had 64 GB free and E: 197 GB at handover. `data/test5` alone
  is 26 GB. Dense caches are about 0.8 GB of descriptors per clip, and
  listing JSONs reach up to about 450 MB per city. There are many old
  worktrees under `E:/monocular-osm-runs` and `E:/claude-scoring`. Ask before
  deleting anything.
- **Windows console:** the console needs `PYTHONIOENCODING=utf-8` for
  non-ASCII city names.
- **CI:** an installed package named `benchmarks` used to shadow the repo
  directory (dash 4.5.0). The by-path load in
  `scripts/score_heldout_assisted.py` stays until a dash release with
  plotly/dash#4148 (merged) ships.

## 9. Working agreements with the user

- Highest scientific value and reuse of open-source code come first. Report
  honestly, including negative results.
- **Commits:** explicit paths only, never a bare `git stash`. End commit
  messages with the Co-Authored-By line the session gives, and end PR
  descriptions with the Claude Code line. Push to the PR branch when asked or
  when auto-fix is on.
- **Approvals:** video downloads, new splits and GPU use need the user's
  approval. Approval for one experiment does not carry over to the next.
- **Upstream bugs:** fix them in the upstream project and open a PR, with
  evidence first and following that project's CONTRIBUTING. Put no private
  data in it.
- **Privacy:** keep the user's email private.
