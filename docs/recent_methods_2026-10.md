# Recent methods: research and controlled tests

Checked 28 October 2026. The complete cold development baseline remains
**9/15 within 200 m symmetric mean route error**. The experiments below do
not establish a new complete fleet result. All use development clips; the
held-out clips have not been used for method or parameter selection.

## What was searched

The official [NeurIPS 2025](https://neurips.cc/Downloads/2025) and
[2026](https://neurips.cc/Downloads/2026) programs were searched for place
recognition, visual/geographic localization, OpenStreetMap, cross-view
localization, odometry and related terms. This produced 18 and 28 matching
titles respectively, including false positives. Fifteen relevant poster
abstracts and their primary links received closer inspection. This is a
bounded relevance search, not a claim to have read every conference paper.

The accompanying GitHub review checked source, released checkpoints,
licenses, input assumptions and CPU compatibility. Detailed program scans,
retrieval timestamps, hashes and repository snapshots are under
`output/new_methods_20261028`. The eleven-method survey there also covers
CVPR/ICCV 2025–26 and older released controls.

| Method | Finding for this project |
|---|---|
| [ImAge, NeurIPS 2025](https://github.com/Lu-Feng/ImAge) | Released code and GSV weights. Actually encoded the full night gallery and native query views on CPU; results below. |
| [TRAIL, September 2026 preprint](https://arxiv.org/abs/2609.07373) | Particularly relevant to a query trajectory and unordered reference gallery. Uses learned emission and motion functions; an official runnable checkpoint was not verified. The hand-built spatial experiment below is not a reproduction. |
| [SelaVPR++](https://github.com/Lu-Feng/SelaVPRplusplus) | T-PAMI 2025 online publication, not CVPR. Released code and checkpoints; CPU path inspected, not yet timed or evaluated locally. |
| [EfficientVPR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.html) | Paper verified. Inspected repository snapshot had README and gitignore, without an implementation or checkpoint. |
| [A2GC, CVPR 2026](https://github.com/CV4RA/A2GC) | Evaluation/model implementation exists; trained checkpoint and license were not verified in this search. |
| [HOPS, ICCV 2025](https://github.com/CMalone-Jupiter/HOPS) | Descriptor fusion code exists, but needs corresponding places across reference conditions. Arbitrary nearby photos in an unordered city gallery do not supply that correspondence. |
| [Sequence-Matching Receptiveness, IROS 2025](https://github.com/QVPR/vpr-smr-predictor) | Assumes ordered comparable traverses and a trained predictor; unsuitable as a direct replacement for unordered-gallery decoding. |
| [To Match or Not to Match, CVPR 2025 IMW](https://github.com/FarInHeight/To-Match-or-Not-to-Match) | Motivated a paired geometry test. Its warning that reranking can hurt modern VPR makes an unchanged retrieval control essential. |
| [XFeat, CVPR 2024 control](https://github.com/verlab/accelerated_features) | Released compact local features and CPU inference; prepared as a controlled replacement for ORB extraction/matching. |

Other screened NeurIPS methods require inputs this project does not currently
have, such as LiDAR, a maintained posed scene map, floor plans or georeferenced
satellite tiles. A relevant paper title alone does not establish a usable
pretrained component. No paper's reported benchmark improvement is treated as
a measured improvement on this project.

## Released encoder and exact sequence decoding

The ImAge test uses the official repository at
`5d5e2d8cad11fe7ba699fdbd1f6b0a3c69dade18` and the released GSV checkpoint,
SHA256 `ffafd68a5dda19bf45ed211339f754cd3967448ea2fbabe998a5141778d927f8`.
The loaded numerical model-state identity is
`75ec289e5b1cd437b5ece5e269eeaa0eaec8db6f183a3c79518320e974578c75`.
The official RGB/tensor normalization/322-square resize order is retained.
Stored reference pixels are BGR and are explicitly converted to RGB, as are
all query crops. Both models use the same 1,498 reference IDs/images/geotags
and 80 native frames with three views each. Gallery selection is inherited
from the previously frozen, native-OCR-derived public reference request.

ImAge encoded every reference and query view afresh on CPU: 516.64 s, with
2.25 GB conservative resident-memory peak across the owned processes. MegaLoc
uses its frozen original similarity matrix as the paired control. Each model
has its own descriptors; vectors are never reused between encoders.

The initial top-64-per-query sequence test failed for all four variants.
An input-only diagnostic showed that pruning excluded the known full-gallery
MegaLoc path at **28/80 frames**, with its lowest-ranked state at rank 1,141.
Thus that sparse result could not fairly isolate encoder quality.

An exact full-reference dynamic program subsequently retained every reference
at every query with the same fixed objective: 30 m plus 40 m/s times each
actual query interval before an excess-distance penalty, followed by a
five-sample running median in metric coordinates. Raw and reference-mean
emissions were both declared before scoring. No thresholds were fitted to
the resulting errors.

| Night encoder/emission | Top-64 sequence error (m) | Exact full sequence error (m) | Exact start/end error (m) |
|---|---:|---:|---:|
| MegaLoc, raw | 1,996.02 | 208.85 | 298.79 / 756.38 |
| MegaLoc, reference mean | 1,727.29 | **124.12** | 298.79 / 756.38 |
| ImAge GSV, raw | 1,563.70 | **156.95** | 79.99 / 840.36 |
| ImAge GSV, reference mean | 2,176.48 | **167.03** | 79.99 / 422.17 |

All spatial scores use the same 20 registered GT waypoints. The exact MegaLoc
reference-mean path recovers all 80 reference IDs of the earlier successful
control and its 124.12 m headline. ImAge passes the 200 m spatial threshold in
both full-gallery variants and improves the start position, but does not beat
the best MegaLoc spatial score. Time-aligned means remain 453–562 m, using the
19 overlapping waypoint times without extrapolation. These are curve matches
with substantial endpoint and temporal errors, not precise localization at
every timestamp.

The exact test also decoded the complete 9,735-reference Detroit (`g5`) gallery:
**4,779.68 m raw** and **4,839.55 m reference mean**, versus the ordinary cold
baseline's 1,384.62 m. Removing pruning improves the declared numerical
objective but does not guarantee geographic correctness.

All six exact predictions were frozen together before new scoring. A separate
scalar-current-state implementation reproduced every selected reference,
route and objective exactly, including all 7,486,847,775 Detroit transitions
per variant. No dense reference-by-reference array was allocated. The actual
blocked replay took 124.42 s with a 104.6 MB conservative resident peak; the
independent audit took 49.63 s.

Reproducible implementation: `monocular_osm/full_sequence.py` and
`scripts/replay_full_sequence.py`, clean source
`ae65d2d0ff68723b2c140c88e607221443d5a5bf`, profile
`8ec24a47a29bc734f7872047c61bca4dd18677d9d4df935599d64df921b695c6`.
The combined freeze is
`a23dc75b3ab8fe307827668c567d736625d709647fd324b4766e00cf913e648f`;
the independent audit is
`47e53e40ee76d3b86e78199d508c3d57463c32408f7ef26364d4b51dfae2740d`.
Local artifacts are `E:/monocular-osm-full-sequence-proof/run1`,
`run1_input_audit` and `run1_score`. The score JSON SHA256 is
`61c96ea01f14bfedd55672bb86f30570a59e1282d3eacb61242e1b9c665c8882`.

## Additional negative controls

These tests preserve their failures. None changes the selected answer or the
fixed assisted policy.

| Test | Actual result |
|---|---|
| ORB geometry on the night clip | Fixed top eight centered references, three crops, 1,920 image pairs. No pair reached the preregistered eight reciprocal matches; every output fell back to centered retrieval. Both scored 602.75 m. Inference took 82.68 s. |
| Detroit native-motion plus public exit targets | Fresh 4,441-pose native VO, all literal exit-number candidates from public OSM, all-null assignment retained. Cue-only/raw/reference-mean fits scored 1,466.61 / 1,600.35 / 1,558.23 m. The sign text does not identify the camera position or prove that an exit was taken. |
| Detroit full-support spatial posterior | All 9,735 references, density-normalized spatial appearance, finite-boundary motion and uniform recovery. Raw/reference-mean peak routes scored 7,233.69 / 4,610.70 m. Independent kernels reproduced every posterior exactly; marginal peaks nevertheless jumped between distant modes. |

The spatial experiment is an original, training-free diagnostic inspired by
probabilistic localization. It does not implement TRAIL's learned potentials.
It demonstrates why independently choosing each marginal mode does not ensure
one coherent trajectory.

All experiments bind source, pixels, frame ordinals/timestamps, reference IDs,
model and preprocessing identities. Prediction freezes and independent input
checks precede separate label access. Negative variants are retained alongside
positive ones; no per-clip oracle combines them. A complete normal assisted
15-clip run and held-out validation are still required before any fleet claim.
