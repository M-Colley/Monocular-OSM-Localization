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
| [ImAge, NeurIPS 2025](https://github.com/Lu-Feng/ImAge) | Released code and GSV weights. Actually encoded the full night and Detroit galleries and native query views on CPU; results below. |
| [TRAIL, ECCV 2026](https://arxiv.org/abs/2609.07373) | Particularly relevant to a query trajectory and unordered reference gallery. Uses learned emission and motion functions; an official runnable checkpoint was not verified. The hand-built spatial experiment below is not a reproduction. |
| [SelaVPR++](https://github.com/Lu-Feng/SelaVPRplusplus) | T-PAMI 2025 online publication, not CVPR. Official base release verified; full 9,735-reference comparison completed. All eight new routes fail 200 m; results below. |
| [EfficientVPR, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_EfficientVPR_Toward_Efficient_Visual_Place_Recognition_via_Scene-Aware_Prompt_Tuning_CVPR_2026_paper.html) | Paper verified. Inspected repository snapshot had README and gitignore, without an implementation or checkpoint. |
| [A2GC, CVPR 2026](https://github.com/CV4RA/A2GC) | Evaluation/model implementation exists; trained checkpoint and license were not verified in this search. |
| [HOPS, ICCV 2025](https://github.com/CMalone-Jupiter/HOPS) | Descriptor fusion code exists, but needs corresponding places across reference conditions. Arbitrary nearby photos in an unordered city gallery do not supply that correspondence. |
| [Sequence-Matching Receptiveness, IROS 2025](https://github.com/QVPR/vpr-smr-predictor) | Assumes ordered comparable traverses and a trained predictor; unsuitable as a direct replacement for unordered-gallery decoding. |
| [To Match or Not to Match, CVPR 2025 IMW](https://github.com/FarInHeight/To-Match-or-Not-to-Match) | Motivated a paired geometry test. Its warning that reranking can hurt modern VPR makes an unchanged retrieval control essential. |
| [XFeat, CVPR 2024 control](https://github.com/verlab/accelerated_features) | Released compact local features and CPU inference; actually tested as a controlled replacement for ORB extraction/matching. |
| [OSMLoc, Information Fusion 2026](https://github.com/WHU-USI3DV/OSMLoc) | Released image-to-OSM code and checkpoints; already represented in this repository's historical experiments. Those use oracle local priors and odometry, so they do not establish city-wide monocular localization. |
| [RHO, CVPR 2026](https://github.com/InSAI-Lab/RHO) | Official source exists, but its two-branch model uses a holistic panorama. The current videos provide one forward-facing camera; a ready checkpoint was not verified from the inspected README/releases. |
| [SeqLoc, August 2026 preprint](https://arxiv.org/html/2608.07835v1) | Online entropy-weighted pose-volume fusion and map-shaped recovery are relevant. Its main experiments use reference-pose-derived ideal odometry and a 32 m initial crop error; appendix tests add simulated drift. That is a different problem from finding a route several kilometres away using this project's actual VO. Official runnable release not verified. |
| [BEVMapMatch, IROS 2026](https://github.com/ssuralcmu/BEVMapMatch) | Released model/download instructions; its sensor-derived BEV front end is based on UniTR/nuScenes and a Linux stack. Converting one dashcam video into the required BEV inputs needs additional work; no local accuracy result claimed. |
| [Hierarchical OSM relocalization, June 2026 revision](https://arxiv.org/abs/2603.01613v2) | DINO tokens plus coarse FFT correlation and uncertainty-controlled refinement. The current title is “Uncertainty-Aware Hierarchical Re-Localization in OpenStreetMap via Semantic Alignment”; an official runnable release was not verified. |

[Dash2Sim](https://www.cs.cmu.edu/~dash2sim/) is another relevant 2026 direction:
joint reconstruction of video and co-located geotagged street-view images can
transfer metric scale and georeferencing. Its author page still marks code and
data as forthcoming. It is a possible downstream use of verified image matches,
not a measured replacement for the present retrieval stage.

A supplemental primary-source check also examined
[DC-VLAQ](https://dc-vlaq.github.io/): DINOv2/CLIP residual fusion and query-residual
aggregation are relevant to appearance changes. The project page's Code text
has no repository link and no checkpoint was verified. No local inference or
NeurIPS acceptance claim is made. The supplemental record is
`output/new_methods_20261028/supplemental_primary_recheck.json`.

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
| XFeat geometry on the same night inputs | Same candidate/view/geometry/ranking profile, replacing only ORB/matching with pinned official sparse XFeat and reciprocal cosine matching. Found positive geometric support in 1,871/1,920 pairs and changed 57/80 selections, but worsened symmetric error from 602.75 to **721.90 m**. Inference took 79.99 s; native process-tree peak was 702 MB resident / 577 MB private. |
| Detroit native-motion plus public exit targets | Fresh 4,441-pose native VO, all literal exit-number candidates from public OSM, all-null assignment retained. Cue-only/raw/reference-mean fits scored 1,466.61 / 1,600.35 / 1,558.23 m. The sign text does not identify the camera position or prove that an exit was taken. |
| Detroit full-support spatial posterior | All 9,735 references, density-normalized spatial appearance, finite-boundary motion and uniform recovery. Raw/reference-mean peak routes scored 7,233.69 / 4,610.70 m. Independent kernels reproduced every posterior exactly; marginal peaks nevertheless jumped between distant modes. |

The spatial experiment is an original, training-free diagnostic inspired by
probabilistic localization. It does not implement TRAIL's learned potentials.
It demonstrates why independently choosing each marginal mode does not ensure
one coherent trajectory.

The XFeat source is pinned at
`e92685f57f8318b18725c5c8c0bd28c7fe188d9a`; its 6,247,949-byte weight file has
computed SHA256
`0f5187fd7bedd26c7fe6acc9685444493a165a35ecc087b33c2db3627f3ea10b`,
verified against the official Git blob. This is not an author-provided SHA256
manifest. The adapter uses CPU sparse features, without LighterGlue or
semi-dense refinement. All 558 feature forwards and 1,920 matching requests
completed. The frame-independent reranker produced a 194.85 km curve, versus
124.17 km for its already discontinuous control and 1.12 km for GT. This test
does not establish how a separately designed temporal geometry model would
perform.

The XFeat freeze is
`8e1ee524c33a39b799a3d64bd912d25027ad81c72b1abaeb67f28153f466d174`;
the independent pre-scoring audit is
`107eaa4e6b2d834d8e5895046dec17f16e74250894b2c5834e3cc8305e97dd49`.
That audit checks source, loaded numerical state, input identities, exhaustive
pair coverage, score arithmetic, rankings and routes. Only correspondence
hashes were retained, so the auditor could not independently replay fundamental
matrices, masks or hull coverage. The limitation is preserved in the separate
score ledger. Both paired scores use all 20 identical registered waypoints;
all protected source, input, prediction and label hashes stayed unchanged.
Local outputs are `E:/monocular-osm-xfeat-rerank-proof/run1` and
`run1_comparison`, with score JSON SHA256
`119763e9eceee0dae22cfd62b0c58dab90c19caf22d25335aad1a4ef827dcfcc`.

All experiments bind source, pixels, frame ordinals/timestamps, reference IDs,
model and preprocessing identities. Prediction freezes and independent input
checks precede separate label access. Negative variants are retained alongside
positive ones; no per-clip oracle combines them. A complete normal assisted
15-clip run and held-out validation are still required before any fleet claim.

The full tracked regression suite at `567bf54` passed **2,938 tests**, with
five skips and four dependency deprecation warnings, in 405.95 s. This includes
the exact decoder, spatial posterior, ImAge/XFeat/ORB adapters, metadata
collector, full-gallery runner and paired panorama experiment.

## Public camera metadata and the next controlled tests

A separate input-only acquisition requested the same seven public metadata
fields for **every one of the 9,735 Detroit reference IDs**, without selecting
by location, similarity or GT. All 195 fixed batches returned HTTP 200, with
2,469,214 decoded bytes and no retries. An independent 14,331-check audit
reproduced the normalized records from the saved response bodies and verified
request budgets, IDs, coordinates, source and freezes.

The uniform result contains **7,785 panoramas (79.97%)**: 5,085 `spherical`
and 2,700 `equirectangular`; 1,947 images are `perspective`, and three have
unknown camera type. The official MapillaryJS converter treats both panorama
labels as spherical. Computed heading/rotation is available for 9,703 IDs,
but a panorama's camera orientation is not the vehicle's travel direction.
Only 2,921 references belong to repeated capture sequences, with at most six
members each; this sampled gallery is not a collection of complete traverses.

Hash-verified sample images confirm complete panoramas stretched into the
cache's 512-square raster. Comparing them with ordinary dashcam crops using a
pinhole fundamental-matrix model has a projection mismatch. The experimental
`monocular_osm/panorama_views.py` converts a verified complete equirectangular
reference into four fixed 90-degree perspective views, preserving the original
ID and channels. Unknown/fisheye types retain their unmodified pixels. The
transform makes no world-heading or exact central-camera claim for stitched
panoramas. Its 36 tests and an independent 3D-ray calculation passed; that is
geometry validation, not a localization result. A separate paired XFeat test
uses a freshly reproduced unmodified control and retained correspondences so
that geometry can be audited directly. This borrows a preprocessing principle
from [RHO's public utility](https://github.com/InSAI-Lab/RHO/blob/ee6883f6fc7780c4ded3aa06b62d79d84949df8d/maploc/data/mapillary/utils.py);
it does not reproduce RHO's network or its three-view 120-degree configuration.

The Detroit metadata freeze is
`21639dac26d61a4794210603f7810dcb38c391d313ff09e5af8e25ce5e30600d`,
under `E:/monocular-osm-reference-metadata/g5_run1`. The collector's clean
source is `6f2cfc8b87b3296a6f690ad9b88703c33c50ae6b`, with 66 focused
metadata/geometry tests passing. It does not produce a location or route.

The same acquisition on all 1,498 night references found only **17 panoramas**,
with 1,465 perspective images and 16 fisheye/brown images. In the fixed top-eight
local-matcher lists, 12/640 night occurrences are panoramas, versus 278/640 for
Detroit. The format hypothesis therefore applies much more strongly to
Detroit. The night gallery cannot be explained by panoramic distortion alone.

The paired panorama inference completed in 402.05 s on CPU2, with a conservative
process-tree peak of 1.29 GB resident memory. Both unmodified/rectified routes
for both galleries were frozen together:
`98d02f28f7c8df3733c3b76d2334cf687ef9363540c61ac6ddf689b5ccf8a0f6`.
An independent 170,398-check audit reproduced all 7,320 OpenCV fits and masks,
hull scores, rankings and all four routes exactly; 1,977 protected files stayed
unchanged. Its SHA256 is
`05611027916cf49a3488756a5f084089a3c3df5e44d12a74fe5a9ee047a95e8a`.
The original 1,920 night pair metrics and all 80 route IDs exactly reproduce
the previous XFeat control. The 51.4 MB of saved correspondences remove that
older study's hash-only replay limitation. Separate official scoring gives:

| Gallery | Original XFeat error (m) | Panorama-aware XFeat error (m) | Original / panorama-aware route length (km) |
|---|---:|---:|---:|
| Night | 721.90 | 720.38 | 194.85 / 190.95 |
| Detroit | 1,716.25 | 2,034.92 | 593.84 / 670.28 |

All four fail the 200 m threshold. Rectification does not supply a useful
local-reranking improvement in this fixed test, and neither variant includes a
temporal decoder. The large route lengths expose repeated jumps between remote
matches. This result does not test panorama-aware **global** descriptors or a
joint temporal geometry model: the top-eight candidate list still comes from
the original whole-image embeddings. No extra candidate search or parameter
choice follows from these scores. The official score JSON is
`E:/monocular-osm-panorama-xfeat-proof/run1_score/official_scores.json`, SHA256
`d8792f9f91501a6625315e609db47e91ce7e9b2a65036dc7c2bde2268f1fdd16`.

The unchanged whole-image ImAge experiment also started on the full Detroit
gallery from clean `d18e629bba6b702c5dda3796affc19ee50ce8f78`. All 32 raw
archives and 9,735 canonical IDs have verified pixels, without duplicate
conflicts; every native query pixel, source ordinal and actual timestamp
matches the original 80-frame control. The runner retains all four outputs
before scoring, streams one raw pool at a time and checks both physical and
actual commit headroom before each descriptor batch. It passed 135 focused
gallery/adapter/decoder tests, with one Windows symlink capability skip.
It completed in 2,623.92 s of inference (about 44 minutes), with a conservative
process-tree peak of 2.275 GB resident memory. The actual model-state identity
matches the night ImAge test. All four predictions are frozen under
`E:/monocular-osm-image-gallery-proof/run1/inference`, SHA256
`acff5f18daabb9774755884adf78ce1b09b233a3e212c5fc6317534be89d334b`.
Independent audit verified all 9,735 pixel identities, the fresh native query
grid, 177 official checkpoint tensors and the loaded model-state hash. CPU4
descriptor products reproduce the saved similarity matrix bit-for-bit. A
separate scalar-current-state recurrence reproduces all four complete DP paths,
routes and objectives exactly. Source/input/prediction hashes are unchanged.
The accepted audit SHA256 is
`d1e8b6434686f362ed3d751f8c5b367e5ea8c36f5c993613d60ec83cf831e4e4`.

| Detroit encoder | Raw error (m) | Reference-mean error (m) |
|---|---:|---:|
| Frozen MegaLoc control | 4,779.68 | 4,839.55 |
| Fresh ImAge GSV | 1,475.68 | 1,341.31 |

Both original controls reproduce the earlier exact full-gallery result. ImAge
improves this matched decoder/gallery comparison, but both variants remain far
above 200 m. Its start error is 5,368.51 m; raw/reference-mean end errors are
1,860.68/5,285.53 m and time-aligned means are 3,666.61/3,491.77 m. The ordinary
cold baseline's 1,384.62 m is a different gallery/pipeline configuration, not
the paired control in this table. Legacy MegaLoc gallery-encoding provenance
remains an explicit limitation on causal encoder-only claims.

Every spatial result uses all 20 identical registered waypoints. Timed
diagnostics use the 19 covered timestamps, without extrapolation. The G5 file's
rounded header says 4–598 s, but its authenticated points span 5.0–598.9 s,
inside the registered 4–599 s extraction window; no points were edited or
discarded to make header strings agree. The official score JSON is
`E:/monocular-osm-image-gallery-proof/run1_score/official_scores.json`, SHA256
`b0659ef519e00dcb03fb6cce34870268fb4cfe3a8766caed6bad812c273cde95`.

Complete eight-row tables, timed diagnostics, input/score freeze hashes and
auditor/scorer recovery records are under
`output/new_methods_20261028/gallery_and_panorama_results.{json,md}`. All
inference and scoring jobs for these completed comparisons are terminal. No default or fixed-policy selection
rule changed. The established full-fleet result remains 9/15.

## Heading and temporal evidence: completed diagnostic

The next experiment adds public optical camera headings to the two frozen
Detroit descriptor sets. It retains all 9,735 references and three native
query views. Only perspective-photo headings are used; panoramas and missing
headings remain neutral. Local query yaw comes from consecutive actual
world-to-camera rotations, without treating VO translation units as metres.
Quality checks retain 65 of the 79 query intervals; missing intervals contribute
no measured turn. Crop offsets use the actual pixel centers and native camera
intrinsics.

Each encoder produces six paths: raw/reference-mean appearance crossed with
heading removed, reference heading, and reference heading plus optional local
yaw. One fixed 16-bin joint position/heading model searches every physical
transition edge, with no appearance-based pruning. This is a hand-built DEV
hypothesis, not a reproduction of TRAIL or another released learned model.

The source is clean `9f7e4a4`, runner profile
`1271d70b27215068ec8d38bd2f1174e85561d82ec26754699cd1352b01f60429`.
The accepted input freeze is
`5528668f329701b4fe868804b044ea77b3f4d99393b404a5d2b4ea93cc82acfb`
under `E:/monocular-osm-heading-proof/run2`. All twelve paths finished before
the independent input/output audit and separate GT scoring.

| Appearance | Heading evidence | MegaLoc error (m) | ImAge error (m) |
|---|---|---:|---:|
| Raw | Removed | 208.53 | 1,464.90 |
| Raw | Public camera heading | 613.60 | 1,530.08 |
| Raw | Camera heading + local yaw | 244.52 | 1,528.78 |
| Reference mean | Removed | 233.83 | 1,266.26 |
| Reference mean | Public camera heading | 210.73 | 1,300.61 |
| Reference mean | Camera heading + local yaw | 210.73 | 1,300.61 |

None meets 200 m. The large MegaLoc improvement over its old 4,779.68 m
full-gallery control comes from the revised temporal model with headings
removed; this is not evidence that public camera headings improve that case.
Even the 208.53 m route has 1,368.37/1,860.68 m start/end errors, a 17,040 m
path versus 10,368 m GT, and 778.00 m time-aligned mean error. The spatial
coverage metric does not establish accurate timing or endpoints.

Execution took 1,280.77 s on CPU2, with 0.83 GiB peak sampled process-tree RSS.
The master freeze is `180c4a005bd308d334574614dc2829e2c84a526ecf2fdb4d782e0fac2ae9ea33`.
Independent audit `2cce72bc01bd132584039b52b2236383cc017547dcc9633ff22ed36a67422e0b`
verified all 12 paths, 79 physical-graph steps, descriptor algebra and objective
sums; 1,900 protected files remained unchanged. It did not independently rerun
the full joint-state optimum. An initial auditor field-name error and its exact
failed report/source are preserved; producer artifacts did not change.
Official score SHA256 is `c09f871477f6fdfb67ee6cd5402158dd78575f6c12fb9406604b7876ce29bf29`.

A separate posthoc DEV follow-up applied the already-established five-point
metric running median uniformly to all twelve paths, preserving timestamps
and endpoints' ordinary truncated windows. Every scalar coordinate median was
checked before scoring. All twelve still fail: the lowest smoothed error is
210.35 m; the raw MegaLoc heading-removed result becomes 222.18 m. The new
freeze is `84e6ddfba3d23a4a7f20ad6fcef0af44c3f2f9ebe4442bd20fd2a67f96709599`,
audit `87ae8480b54c32ab60b375f79c8c052e906ac3e8e8a660980f718e896dc1f7b6`, and score
`91506b3b972b95eda6701f355e6a0e281f6814afcf398aedf46ba2ba10250d04`.
No method was selected for the policy or pooled into a fleet success claim.

## SelaVPR++ runtime readiness

The official standard base checkpoint is 482,674,957 bytes, SHA256
`b048490dbd1c27dee67fce6faaec7bec267d19a044c85877af94b8588e596a62`, matching
GitHub's published release digest. Source is fixed at `56bd921cbd3d53e9c5f91d0aafff147f95fb362a`.
The CPU adapter retains the official computation and removes only an unused
FAISS import. The actual strict CPU state load succeeds: 97,909,629 state
elements, 2,048-dimensional normalized descriptors, RGB/ImageNet normalization
then 322-pixel tensor resize, ordinary Torch attention fallback.

Clean `01fdfb0` encoded 12 native query crops and 32 reference photos in a
runtime-only smoke. All 44 descriptors and bitwise canonical preprocessing
checks passed. Constructor time was 4.89 s and mean forward time 0.459 s on
CPU2/inter-op1; external peak RSS/private memory was 0.77/1.19 GiB. No route or
accuracy score was computed. Runtime freeze is
`17b4fed96d33a03e02672f970a1c59af2f29fddd466f9058fdac4148ac7f3092`.
The separate full evaluation below retains previous MegaLoc/ImAge controls,
with all outputs fixed before new scoring.

## SelaVPR++ full gallery: completed comparison

Clean source `2bbd344` encoded all 9,735 canonical reference photos and 240
native query crops on CPU2/inter-op1, batch one. This tests the official
standard base global descriptor, without its optional hashing or learned
reranker. All eight new routes and sixteen byte-identical historical controls
were frozen together before new GT scoring. The eight configurations are
declared DEV comparisons; none is a newly selected policy.

| Decoder / appearance | Heading evidence | MegaLoc error (m) | ImAge error (m) | SelaVPR++ error (m) |
|---|---|---:|---:|---:|
| Original exact / raw | None | 4,779.68 | 1,475.68 | 4,753.36 |
| Original exact / reference mean | None | 4,839.55 | 1,341.31 | 1,260.70 |
| Revised temporal / raw | Removed | 208.53 | 1,464.90 | 1,187.03 |
| Revised temporal / raw | Public camera heading | 613.60 | 1,530.08 | 1,145.26 |
| Revised temporal / raw | Camera heading + local yaw | 244.52 | 1,528.78 | 1,146.44 |
| Revised temporal / reference mean | Removed | 233.83 | 1,266.26 | 860.31 |
| Revised temporal / reference mean | Public camera heading | 210.73 | 1,300.61 | 1,188.65 |
| Revised temporal / reference mean | Camera heading + local yaw | 210.73 | 1,300.61 | 1,165.42 |

All eight Sela routes fail 200 m. Its lowest spatial error is 860.31 m, with
2,685.86 m time-aligned mean error. It does not improve on MegaLoc with the
revised temporal decoder in this gallery. The original exact decoder includes
its established median-five postprocessing; the revised decoder does not.
Comparisons within each row keep the decoder fixed. Legacy MegaLoc reference
encoding identity remains UNVERIFIED, limiting encoder-only causal claims.

Execution took 5,133.56 s, with 1.82 GiB peak sampled process-tree RSS and
1.28 GiB private memory. Independent audit checked all 9,975 canonical tensor
hashes, fresh native frames/ordinals/timestamps, exact 44-descriptor smoke
replay, full descriptor algebra and both simple DP paths. It checked all six
heading objectives and physical graphs, without claiming a second full joint
DP optimum proof. All 2,585 protected files stayed unchanged. An initial
auditor-only safe-global list-order failure is preserved; the corrected audit
sorts that allowlist alone, retaining duplicate/addition/omission rejection.
Producer inputs and outputs did not change.

Artifacts are under `E:/monocular-osm-selavpr-gallery-proof`:
master `db72758e66f3e0280bc7256a87deb6619e854a42850f10578245006a8964ce58`,
audit `2827e77d3319f4d0debe7e8fde0620ba26f3575f9770b7922f138c46e05055d7`,
score `3987f2caaf5bd2d92f3d40a2d8e73badb28ff2868eea831a8c9bc5bdcbf3b0b2`.
All 24 scores use the same 20 authenticated waypoints; timed diagnostics use
19 overlapping timestamps without extrapolation. All jobs are terminal.

## Timed text cues: completed negative test

Clean `f589e88` tests capture-time POI evidence on the complete 2,998-reference
night union. It retains both literal branches and null, with raw and reference
mean appearance: eight outputs frozen before scoring. Shared-capture heading,
normalized POI associations, physical transition normalization, a fixed outlier
mixture and explicit uncertainty prevent treating every sign match as a precise
camera position. Two brands in the first capture share a mall directory pylon;
their text centers are uncertain location proxies, not two storefront rays.
No GPS overlay, GT position or manual branch choice enters the decoder.

| Appearance | Null (m) | First POI branch (m) | Second POI branch (m) | Joint model (m) |
|---|---:|---:|---:|---:|
| Raw | 711.00 | 711.00 | 711.00 | 711.00 |
| Reference mean | 681.34 | 681.34 | 681.34 | 681.34 |

Both joint models select null. Every cue path is identical to its null control;
the soft factors did not change the selected trajectory. Time-aligned means
are 1,280.68 m raw and 1,364.53 m reference mean. All eight fail 200 m. This
standalone diagnostic does not repair the normal v2 wrong-region result.

The CPU2 replay took 528.42 s without new neural inference. Independent audit
matched all four global normalizers, six component objectives and both joint
choices; 1,228 protected files stayed unchanged. Actual full DP optimality was
not independently replayed; a separate tiny exhaustive test covers the recurrence.
Artifacts are under `E:/monocular-osm-timed-poi-proof`: master
`537497653a99c4d64c8a2d39cee2bea6910758a5c3feb42ec46c4b1ef0313ffb`, audit
`23b3568cca84bdd6e6692ad0ed136dcd6c152a7cb81accf787c0b90e09771abe`, score
`8c71fc4330c40bbdfe667ba207c2b3d3be775c76fbbab08d8d5a9920950c0a69`.

## Project improvements supported by these results

- Prioritize sequence consistency and calibrated ambiguity over another encoder
  swap. Complete evidence acquisition does not establish a trustworthy location:
  the normal v2 union was complete and still selected a region 11.09 km away.
- Add standard time-aligned error, timestamp coverage, endpoint errors and route
  length ratios alongside the existing goal metric. A spatial coverage pass can
  conceal incorrect timing and endpoints.
- Finish the verified served-subset cache fast path. The numeric request-key
  bug is fixed in `f3dfa71`; known missing photos can still force loading a
  large raw store despite a valid descriptor subset.
- Extract versioned artifact/resource helpers from chained experiment-runner
  imports, and separate acquisition, model, decoding and audit state behind a
  typed retrieval session. Preserve old artifact readers and independent auditors.

The full suite at clean `f589e88` passes **3,386 tests**, with six skips and four
warnings in 393.62 s. All 416 tracked source/test/doc files stayed unchanged
during the run. JUnit is `output/tests-current-20261028.xml`, SHA256
`dee611191b9537452a7cc6964f4cb31fe84be4623ad198c92aaf761d33f04ce6`.
No default policy changed; the complete fixed-profile result remains **9/15**.

## Road-flow coverage and quality: paired input experiment

The normal night run did not measure road-plane speed after its weak ORB VO.
An independent full-window input probe therefore tested the existing road-flow
estimator without VO directions or stop flags. It decoded all 8,850 native
frames in the registered 0–295 s window, tracking 1,474 intervals at stride six.
The fixed profile uses 960×540 tracking, generic 70-degree FOV and 1.4 m camera
height, the existing 10–20 m depth interval and unchanged confidence gates.
It computes neither a geographic route nor an accuracy score.

The legacy seed band starts at row 297. At the calibration search's lower
horizon boundary, every tracked starting point projects to 2.87–9.83 m:
**none reaches the required 10–20 m interval**. All 1,474 travel estimates are
unknown, rather than measured zero movement. Independent cached algebra and
ray-plane checks confirm the coverage inconsistency. The original assertion
that the seed band covered all relevant pitches was too broad.

Source `a25445e` adds an explicit optional tracker band and the pure
`road_seed_top_row` projection helper. It projects the far road point at the
existing horizon search bound, floors the result and unions it with the legacy
band. This derives row **247** for the declared geometry. The default band,
depth interval, calibration and quality thresholds remain unchanged.

| Input measurement | Legacy band | Geometry-derived band |
|---|---:|---:|
| Seed top row | 297 | 247 |
| Tracked correspondences | 549,034 | 563,857 |
| Finite forward estimates / 1,474 intervals | 0 | 1,450 |
| Horizon sharpness | 0 | 0 |
| Confidence | 0 | 0 |
| Eligible for metric localization | No | No |

Both measurements use exactly the same 8,850 native frame records and pixel
stream. Widening the band repairs support but does not repair the night flow:
1,078 raw steps are negative, and clipping creates 1,140 zeros that cannot be
interpreted as stops. Fast-step relative spread is 0.866, above the unchanged
0.55 rejection threshold. No metric localization follows from these inputs.

The paired run finished in 101.53 s on CPU2. Its master is
`6a197bae3e730b53fe2b63c45fff51b4c112791b928e59388f485008d2035013`;
independent cached audit `caadc250f3f2265541d0d52727efa099fbafc637cfff36768ba066417354129f`
passes 161 checks, including independent projection, motion/spread arithmetic
and gate rejection. That audit verifies cached native records and video hash
binding; it does not claim an independent video re-decode or feature retracking.
The legacy probe/audit and its initial missing-helper import failure are
preserved separately under `E:/monocular-osm-road-flow-proof`.

## Rank fusion: completed negative comparison

Clean `d1db35e` computes float64 average-tie percentile ranks for each raw
MegaLoc, ImAge and SelaVPR++ query row, then their fixed equal-weight mean.
All 9,735 references and 80 actual query times are retained. Single-encoder
rank controls separate fusion from the rank transformation. Reference-mean
correction occurs after ranking/fusion. This is an explicitly posthoc DEV
hypothesis after the earlier encoder outcomes; ranks are not calibrated
localization probabilities.

All sixteen new routes and twenty-four historical cosine controls finished
before the forty-output master, independent audit and separate scoring:

| Appearance | Exact simple raw (m) | Exact simple reference mean (m) | Revised temporal raw (m) | Revised temporal reference mean (m) |
|---|---:|---:|---:|---:|
| MegaLoc percentile | 1,189.14 | 2,449.96 | 1,379.21 | 2,054.56 |
| ImAge percentile | 1,266.69 | 1,614.19 | 1,516.69 | 1,513.75 |
| SelaVPR++ percentile | 4,808.89 | 1,202.30 | 1,592.77 | 1,204.39 |
| Equal-three mean | 1,597.77 | 4,621.97 | 1,335.47 | 4,125.66 |

All sixteen fail 200 m. The ensemble does not improve on the original
MegaLoc revised temporal control's 208.53 m. Encoder disagreement did not
establish complementary correct matches. Every original control's official
score reproduces exactly. No rank or fusion configuration is adopted.

Execution took 2,038.25 s on CPU2, with 0.28 GiB peak sampled process-tree RSS.
Independent audit reproduced all four rank/fusion matrices bit-for-bit and
all eight simple DP paths, objectives and scalar median-five routes exactly.
All eight revised temporal objectives and physical graphs also matched; no
second global joint-heading optimum proof is claimed. All 2,887 protected
files remained unchanged. Scores preserve unrounded metrics as well as the
ordinary display format, using the same 20 waypoints and 19 covered times.

Artifacts under `E:/monocular-osm-rank-fusion-proof`: master
`ea034eeed2e8ca9f88cb8b644ea6ba7c74b50e7da94a30ad5c659c1f785b3bc1`, audit
`8cc38fcb47912be6ef34a1561841a45f8138104cf50bf821bf2aef459538ee8f`, score
`5feeff81a833230939b296467024e03b5c1364e664c24ef3d869203253f2eed7`.

## Normal Phoenix assisted-v1: completed failure

The ordinary public CLI at clean `d1db35e` completed in 1,506.7 s with fresh
query/OCR/automatic VO and no forced child or region. All six public metadata
regions completed: 8,700 references, cache hits, zero lost tiles or missing
URLs, identical actual query/model/decoder identity. Historical reference
encoding remains UNVERIFIED.

The metadata candidate's raw median 0.1863955 exceeds baseline 0.1323504 and
the fixed admission requirements, but 29/80 repeated coordinates give 0.3625
dominance, above the fixed 0.25 limit. The fallback municipality plan needs
at least 65 cells, exceeding its 64-cell bound. Expansion is therefore partial,
and the unchanged policy selects baseline. This is not a missing-cache failure
or an incomplete metadata-region acquisition. The earlier 25.57 m component
result does not establish a deployable policy success.

Independent inference audit passed before scoring. The selected answer fails:
**18,644.08 m** symmetric mean, with 18,624.53/18,920.49 m start/end errors.
All 1,091 consumed protected files stayed unchanged through ordinary offline
aggregation. The strict goal checker reports this error and fourteen missing
clips; a single normal proof is not a complete fleet evaluation.

Parent master is `4556b893f09a1d5921ef265816a363ab932a157d580b9e62eee19efae3377f49`,
audit `30c3dd8acbdbcba7e044fd22c81a6125d3486a4a7e4f56e26de0536d15537f8c`, benchmark
summary `68e9bbf9335f6e03709857ff486b6d3e18bea0b0516ec61a6b0c69d14a1e5c3d`.
Input-only route-support analysis is under
`output/policy_geometry_review_20261028`; extent or repeated occupancy alone
cannot prove correct place or a validated stop. No quality limit changed.

## Consecutive learned features: runtime smoke passed

Clean `87282b2` adds a matched, input-only ORB/XFeat motion experiment. Both
frontends use the same 2,950 native frames at stride three, native intrinsics,
parallax/static filters, stable top-300 moving matches and essential-matrix
gates, with RNG zero for each pair. All rejection means unknown motion,
not a verified stop. Translations have unit scale, not metres.

The fixed first-four-frame runtime smoke completed in 10.14 s on CPU2/inter-op1
with 0.43 GiB peak sampled RSS. Independent audit checked the native pixels,
actual preprocessing tensors, model state, regenerated ORB features, reciprocal
matches and pose bookkeeping. All six pairs fail the parallax gate; no actual
essential fit occurs in this smoke. Synthetic tests cover fit/mask branches.
This does not establish frontend quality. Saved XFeat features are bound to
model/source/input proof, without a second neural feature-generation run.

Smoke master `63bca94f04403589f08daff5ad07d4c9f1880d00346ed7e4c941410f3612d9fd`
and independent audit `6b28e0e43c243287ce07f4cb3046025baab2fcd9de749258ffdd10fe08fbf9c6`
are under `E:/monocular-osm-xfeat-motion-proof`. The full paired sequence has
completed and passed independent input audit; it supplies no geographic
localization accuracy result.
The four new files pass 104 focused tests including existing compatibility
cases; those overlap and are not added to the full-suite count below.

The full suite at clean `a25445e` passes **3,490 tests**, with six skips and four
warnings in 453.21 s. All 436 tracked non-label/result file hashes stayed
unchanged, and label/result boundaries were checked by metadata only. Artifacts
are `output/tests-rank-flow-20261028.{xml,log,source.json,review.json}`. This is
regression evidence, not a completed fifteen-clip accuracy result.

## Full consecutive-feature comparison

The same clean `87282b2` paired run completed all 2,950 native frames in
854.77 s on CPU2/inter-op1, with 0.49 GiB peak sampled process-tree RSS.
Both frontends receive the same pixels, intrinsics, pair filters and pose
gates. This measures accepted geometric estimates, not pose truth:

| Frontend | Accepted pairs / 2,949 | Accepted fraction | Longest unknown span | Median accepted inliers |
|---|---:|---:|---:|---:|
| Fresh ORB | 1,431 | 48.52% | 91.2 s | 98 |
| Released XFeat | 1,478 | 50.12% | 78.7 s | 111 |

The modest coverage gain leaves long unknown intervals. Neither frontend is
selected. Rejected motion is not a validated stop, and accepted translations
have unit scale rather than metres. The existing ground-flow checks still
reject metric travel estimation; no threshold was relaxed.

Independent audit redecoded all 8,850 native frames and 2,950 preprocessing
tensor proofs, regenerated all ORB features and replayed all 5,898 pair
checks, including 3,079 essential-matrix fits. Both complete camera chains
and planar projections match exactly, as do the first four accepted smoke
frames. The audit made no neural forward or model constructor call. Saved
XFeat features remain bound to the released source/state/input proof, without
a second neural feature-generation run. All 12,081 protected files remained
unchanged through 24,282 checks in 421.83 s.

Full master under `E:/monocular-osm-xfeat-motion-proof/run1` is
`b084a7d0ecc660d0ddaf2d9d23f6a7be946bf9b4888d48546fa330b5872d6e1a`;
independent report under sibling `run1_full_input_audit` is
`8ccab46983b594b703d29d02cf518385ed96344e417a01330b25cd08e7b2b98b`.

## Road geometry input repair and further released candidate

The whole-Michigan physical-road extraction at `3170b27` conservatively
failed on three GDAL warnings about unclosed polygon rings. The failed
outputs and partial databases are retained, with freeze
`8c8957d4d0bfdac83a89491e6fb353552df918dd96c6f1a534621b74a5693dde`.
A synthetic case reproduced the same warning from an unrelated forest
multipolygon alongside a valid service road. Clean `3868b81` declares
driver-level line interest before reading. It retains warning refusal and
the independent selected source-way ID/tag/node-reference-count completeness
guard; a selected road with a missing middle node still fails. Independent
reviews and 102 focused/compatibility tests pass. A new immutable run2 is
extracting the same public source. No projection has run. Coordinates/order
remain GDAL-authority, with no legal-routing or topology claim.

The official [FoL repository](https://github.com/chenshunpeng/FoL) now exposes
ViT-B and ViT-L checkpoints. This is the AAAI 2025 FoL release; availability
does not establish a FoL++ checkpoint or an accuracy gain here. The base
checkpoint was downloaded from fixed Hugging Face revision
`27f7dbfbfd0adbdf4190dab8debd03edcc99496e`: 1,080,932,591 bytes, verified against
publisher LFS SHA256 `0f0aae410e05699cb56d26f62ae75470bc3ba780882fb788bf87365ed222c8b0`.
Seven required source/license/preprocessing files match official Git blobs
at `09cd23bd5a86c4d2d6ae4d117299bbd59255bd1f`. FakeTensor metadata inspection
with CPU, mmap, weights-only loading and a narrow NumPy metadata allowlist
finds 192 state tensors, 90,056,257 elements and 360,225,028 state bytes.
No model constructor or neural forward has run. Runtime/CPU feasibility and
strict complete-state compatibility must pass before any gallery trial.
