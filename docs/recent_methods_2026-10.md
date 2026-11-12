# Recent methods: research and controlled tests

Current status (12 November 2026): the completed assisted v4 development
fleet reached **14/15**, with Detroit `g5lnpYCk1Ec` still failing. See
[code knowledge](code_knowledge.md) and the [fleet record](assisted_fleet_2026-11.md).
The October 30 live-run statements below are historical, not current process
claims. The former held-out cohorts have since been scored and used to
inform development. Six new TEST4 videos are registered and frozen separately;
their localization has not run and their method binding remains pending.

The 11 November sequence study also completed two protected reference-snapshot
controls, retaining all 42 original calls and weighting each provider image
once across its feature alternatives. Both fixed full-window controls were
negative: baseline **7,152.842366478733 m**, expanded **6,338.264241249098 m**,
against the retained Detroit Answer's **1,387.2377579105248 m**. The baseline
reproduced the earlier strict-bank trial exactly. Fresh map preparation,
complete field reconstruction, native-clock checks and closed predictions
preceded scoring. No policy or fleet gain follows. Full source/proof details
are in the fleet record linked above; older running-status entries below
retain their original historical context.

The separately registered adaptive-resampling study at `54a85e4` also completed
four negative controls: 1,619.89/6,040.93 m with original heading and
3,603.44/20,274.61 m with normalized heading. Fewer resampling events did not
repair full-route localization. No new Answer policy or fleet gain is claimed.

The subsequent fixed500k study at `fcc3203` also failed all four controls:
2,289.26/13,752.97 m with original heading and 1,136.61/3,529.62 m with
normalized heading. Computing only the final pure summary made the budget
feasible while retaining full genealogy. It did not establish accurate
localization or a new selected-Answer fleet result.

A subsequent native32-window geometric motion pilot at `a902b8f` verified its
full original pixel/clock stream, then refused all32 independent windows under
fixed triangulation, parallax and held-out gates. Synthetic-image feasibility
did not establish useful actual-video motion; no new pose or route is adopted.

The fixed road-context epsilon-mixture study at `67be7cb` subsequently completed
all four controls, with exact transition replay and all predictions closed
before scoring. Errors were 1,638.2766488560471 / 19,760.57970939646 m with
legacy heading and 18,845.335325828848 / 20,289.24917215141 m with normalized
heading, in baseline/expanded order. All failed 200 m. One paired Run9 control
improved and three worsened; no new route, parameter adjustment or policy was
adopted. Validated fleet accuracy remains **14/15**. The completion record is
`E:/monocular-osm-joint-vpr-proof/run19/study_completion.json`, SHA256
`af1f1e036e3fd0428648b0442a8958a42dd75b27242eed6461be26b1ee394cad`.

At the 30 October 2026 checkpoint, the complete cold development baseline was
**9/15 within 200 m symmetric mean route error**. These experiments did not
establish a new complete fleet result. They used development clips; the
held-out clips had not then been used for method or parameter selection.

At that checkpoint a serial all-fifteen run of the fixed opt-in v3 policy was running
at clean source `f72fdaf`. Its complete prepared-input audit passed before
inference. Every terminal attempt, including a missing answer or failure,
remained in the denominator; no new fleet accuracy was available then.
Historical execution counts and handles are recorded in `output/goal_progress.json`.
Completed executions are not accuracy passes.
KITTI0009 reached the two-hour deadline without a parent answer, so this
registered run cannot establish 15/15. An isolated source fix addresses both
deadline finalization and a metadata-union duplicate-reference identity error;
the original run continued unchanged. Neither fix had an accuracy result then.

## What was searched

A further bounded primary-source scan on 11 November identified
[Déjà View/DVLT](https://research.nvidia.com/labs/dvl/projects/dvlt/), listed
as a NeurIPS 2026 Oral. Its [official repository](https://github.com/nv-tlabs/dvlt)
at `f59bb18259ab446a7585c4f47731a77cfdefa426` provides inference code and a
publisher checkpoint link. The 117M-parameter model estimates cameras and
dense geometry jointly from multiple views. This suggests a distinct motion
frontend to examine after the unsuccessful sparse fixed-K window pilot;
that transfer is a hypothesis, not an accuracy result. Linux/CUDA is the
publisher-tested environment. Windows/CPU2/3 GiB feasibility, checkpoint
identity and native-video geometric support were initially unverified.
On 12 November, two prospectively fixed synthetic profiles ran using the
official complete checkpoint and unchanged source in a separate pinned CPU
environment. Two 224-square views took 2.719 s forward / 10.484 s guarded
total and peaked at 1,267,519,488 bytes RSS; two 504-square views took
17.422 / 24.922 s and peaked at 1,383,497,728 bytes. Root independently
verified all 368 raw-to-loaded state tensors, generated/preprocessed inputs,
finite outputs and actor closure. A separately registered seven-view504/K12
test then took 83.812 s forward / 91.500 s guarded total, peaking at
2,882,199,552 bytes sampled aggregate RSS. All nine arrays (90,684,300 bytes)
were retained, finite and independently checked with the same full weights.
No actual video or GT was used. These exact synthetic capacities are now
measured; native-video geometric/metric validity remains unverified. Internal consistency and
Sim(3)-aligned benchmark results would not independently certify metric
speed or covariance on the project video.

The separate masked native pilot at `3823286` then completed all32 model
windows and all224 detector frames in 1978.656 seconds. Model state, retained
outputs, pure geometry recomputation and process closure passed. Primary
full-extracted-CHECK pose-only and model-depth/pose results were both **0/32**.
The persistent-CHECK secondary comparison accepted two pose-only windows and
no model-depth/pose window; both positives fail the primary coverage rule.
Peak numerical actor RSS was 2,145,640,448 bytes. This is native runtime
feasibility with a negative primary geometry result, not a route or fleet
improvement. See [code knowledge](code_knowledge.md#masked-native-dvlt-diagnostic)
for the complete gate and closed artifact identities.

A bounded 12 November source check added two candidates to these notes. It
does not establish exhaustive NeurIPS coverage or another ready CPU localizer.
Neither candidate has run on project inputs:

| Candidate | Verified source finding | Next feasibility gate |
|---|---|---|
| [Pi3X / Pi3](https://github.com/yyfz/Pi3), ICLR 2026 paper with a December 2025 Pi3X update | Reference-free geometry; upstream has a PyTorch RoPE fallback and an F32 attention path. The inspected `9fa3ddb` model is substantially heavier than DVLT. | Pin complete weights/config/state and public camera readout, then measure a separate generated-input CPU profile. Learned scale remains uncalibrated. |
| [Pair-VPR](https://github.com/csiro-robotics/Pair-VPR), [RA-L 2025](https://arxiv.org/abs/2410.06614v2) | Learned pair classification reranks global retrieval. At `fd764f9`, the CPU evaluator leaves `device` undefined and the no-xformers cross-attention fallback has an argument mismatch. | Review a narrow CPU compatibility repair, strict complete-state loading and both reranking orders before a generated-input timing test. Global-only encoding would not reproduce the method. |

Pair-VPR's published speed control uses 322-pixel ViT-B inputs and top100
bidirectional reranking. Its dense F32 features would occupy about 45.4 GiB
for 30,000 images; 80 queries imply 16,000 pair directions at that TopK.
CPU16/four-hour capacity is unmeasured. The released research license and
model/dependency identities must be retained. Reranking cannot recover a
place absent from the retrieved candidate set.

The known [MapAnything](https://github.com/facebookresearch/map-anything)
method also has an explicit public CPU path with AMP disabled; this is newly
verified feasibility information, not a newly discovered approach. Its full
runtime/dependency/weight gate remains separate. The bounded source receipt,
exact revisions and resource caveats are in
`output/dvlt_native_pilot_design_20261112/next_methods_algorithm.json`, SHA256
`c69177531e2fc59e273892cf7e43904ab6a75438df5c5a39c8586612562a03d5`.

An additional focused primary-source check on 10 November found the
following candidates. This supplements the earlier bounded conference scan;
it is not a claim to have read every newly published paper. No model from
this supplementary check has been run on the project.

| Candidate | Verified release/input finding | Implication |
|---|---|---|
| [PGGT-Loc](https://arxiv.org/abs/2610.41833), posted October 30, NeurIPS 2026 | The paper uses an image plus planar map primitives. Its author-linked `3dv-casia/PGGT-Loc` main README returned HTTP 404 through GitHub on November 10. | A runnable release was not verified. OSM road centerlines alone do not supply its planar primitive map; this is an input mismatch, inferred from its stated inputs. |
| [AutoCompass](https://nianticspatial.github.io/autocompass/), ECCV 2026 | The authors describe learning image-to-map pose distributions from noisy position labels and optional relative poses. The inspected project page did not expose a verified code/checkpoint download. | Useful motivation for retaining distributions and uncertain motion; it is not an available replacement demonstrated here. |
| [TOL/TOLoc](https://arxiv.org/abs/2604.01644), April 2026 | The [official repository](https://github.com/WHU-USI3DV/TOL) contains released code and checkpoint download links. Its [text generator](https://github.com/WHU-USI3DV/TOL/blob/main/data_prepare/generate_text_from_raster.py) produces cardinal-direction object descriptions from OSM rasters. | Its text input differs substantially from observed shop/exit OCR. Generating a query description from the unknown true pose/map tile would leak location; a future trial needs an independent video-to-description front end and verified weight/runtime provenance. |

A further 11 November primary-source check found two SMC inference papers:
[Psi-Sampler, NeurIPS 2025](https://arxiv.org/abs/2506.01320) studies posterior-aware
particle initialization for score-model reward alignment;
[Diffusion Path Samplers via SMC](https://arxiv.org/abs/2601.21951), revised
8 November and marked accepted at NeurIPS 2026, provides
[official code](https://github.com/akyildiz-group/dpsmc) for diffusion-based
sampling of unnormalized targets. These are generative sampling papers, not
released OSM localizers. The inferred project experiment is a bounded backward
appearance guide for initial/turn/ancestor proposals with exact importance
corrections, preserving each target observation once. It needs no new model,
and the four registered controls have now completed at `05b06c5`. Their
baseline/expanded errors were 4,910.15/6,041.08 m with the original heading
target and 3,605.82/1,058.16 m with normalized heading. All fail 200 m. The
1,058.16 m route also has endpoint errors above five kilometres and covers
only part of the drive. Later common-null visual-target and heading-variance
controls also failed; none was adopted. The shared mode-readout bug was fixed
and tested, but its correction reproduced the same failed routes. The bounded
research receipt is `output/reference_snapshot_next_methods_20261111/research.json`.

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
| [MutualVPR, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/04388b6780bcf1a39ad0a91422f09a85-Abstract-Conference.html) | Official source and author-linked weights verified at commit `bd373c6`. Strict loading and 44 CPU forwards pass an independent runtime audit; a full-gallery Night/Detroit comparison is running, with no accuracy result yet. |
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

The 30 October follow-up also inspected the September preprint
[AdaptVPR](https://arxiv.org/abs/2609.04369) and its
[official release](https://github.com/chenshunpeng/AdaptVPR). It constructs
training positives through appearance editing, geometry/diversity checks and
limited regeneration. The release supplies generation code and metadata;
generator implementations/weights and original images require separate setup.

A further 30 October sweep rechecked the NeurIPS 2026 program and official
releases. It found no new matching title relative to the earlier search;
this remains a bounded relevance search. The most practical newly identified
released candidate is [QAA, ICRA 2026](https://github.com/arplaboratory/QAA):
its official demo supports CPU, and the author-linked
[1024-dimensional safetensors checkpoint](https://huggingface.co/xjh19972/QAA-1024)
is public. Its 366,461,972-byte payload now matches the publisher's SHA256
`1bb5f47ccc9f00abffc337c00e3611c9836b8bf24f1ed2f1b92d424bd6eff977`.
Independent safetensors SDK metadata agrees with all 200 saved float32 tensor
entries, including the unused dust-bin scalar. Released QAA1024 then completed
strict loading of all 200 tensors and all 44 CPU forwards in **28.422 s**, using
CPU2/inter-op1, batch one. Construction took 4.171 s; mean query/reference
forward times were 0.4309167/0.4615 s. The internal sampled peak RSS was
1,073,102,848 bytes. The complete checkpoint/loaded-state digest is
`0e7dba5fc25f44fccee023dbe8d7bee180bcd13f771912946a0dbb1d7e5a083d`.

Independent model-free audit
`d3387d80050b6454a192e1a82bdd30b6df3aafa5cd3181144a648fb27346b4b1`,
using verifier
`56806b8aaad8262d0de211037e1284e846d395a77907d240e1f64e9270259ea0`,
passed before root rehashed all 510 protected files and confirmed every owned
process birth had exited. It checked the input, preprocessing, full-state and
runtime receipts without replaying neural output values. This establishes
runtime compatibility, **not localization accuracy**.

Model-free preparation of the paired Night/Detroit full galleries also
completed in 26.359 s: 30 frozen members, including ten positive input copies
totalling 760,654,338 bytes. The prepared master is
`b6d390b0a29197446cd6db64a17a9bf5fe7da49e08f84639a52c6d7fcb57709b`.
The independent full-input check passed in 105.156 s; root rehashed its 338
protected files. Report is
`c16ff8e5edabfcef5d8a5339df6bcfd838ff83ba61751a0925a9592cf4bd7318`.
The serial Night then Detroit comparison completed on CPU2/inter-op1, batch
one. All sixteen declared QAA/cached-MegaLoc outputs froze before the complete
independent audit and separate label access. The audit passed in 517.735 s;
root rehashed all 395 protected files and confirmed the owned processes exited.
All eight learned-QAA routes fail the 200 m target:

| Decoder / normalization | Night error (m) | Detroit G5 error (m) |
| --- | ---: | ---: |
| Exact simple / raw | 1,891.69 | 4,824.37 |
| Exact simple / reference mean | 1,808.26 | 4,808.64 |
| Heading erased / raw | 365.41 | 4,879.42 |
| Heading erased / reference mean | 398.76 | 4,891.05 |

Threshold decisions use the unrounded values. The unchanged cached-MegaLoc raw
controls reproduce 173.095194 m for Night exact-simple and 208.529452 m for
Detroit heading-erased. Their historical encoding identity remains unverified.
No QAA variant is adopted, and this comparison establishes no fleet gain.
The paired master is
`c69c2b75a6fb690e558572c3a822f8295d8996df7d0cea301a8176d23bbd95b3`;
complete audit
`10c9ac68cc3c501b07de61940e602810401c0e9ca23d59bf3ecd94afed25d5d4`;
all-sixteen score
`d06ca22fe75f3e727a93d61be3949b6e9caa68f89878f1a389504bd77e88b9b8`;
root score acceptance
`e1550bbafab8670bb8d231e958c8f761685cef2d128e5e284a177674262a83e9`.
The family is `E:/monocular-osm-qaa-gallery-proof/run1`, with scores in sibling
`run1_paired16_score`. MutualVPR has separately finished Night and is processing
Detroit; its paired accuracy result is pending.

D2-VPR has author-linked weights,
but the anonymous checkpoint payload was not verified. UniPR-3D's released
checkpoints exceed the present 3 GiB process budget individually. The inspected
[TF-VPR source](https://github.com/ddfs430/TF-VPR) has a syntax error and its
gallery-token initialization exceeds the budget for the Night gallery.

The current [CROSS release](https://github.com/jiaming-ai/CROSS) includes
monocular, stereo and RGB-D modes. Its author describes CPU support as tests
only; mapping needs a GPU and a previously constructed posed memory. It is
therefore not a direct city-wide replacement for this pipeline. Exact source
pins, primary links, checkpoint metadata and static checks are recorded in
`output/new_methods_20261030/final_primary_sweep_20261030.json`.
It is a training-data approach, not a verified pretrained retrieval replacement
for this CPU experiment. No local inference or accuracy result is claimed.
The inspected README is saved at `output/new_methods_20261030/AdaptVPR_README.md`.

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
reviews and 102 focused/compatibility tests pass. Immutable run2 completed in
294.97 s. An independent compiled-Protobuf audit checked all 5,107,511 source
ways, all 1,415,696 selected physical ways and 12,135,432 vertices, plus every
WKB/index row and the complete content digest. All 22 protected files remained
unchanged. Master is `896bc4c3bd3f44720c9c22577f1f70b90a81fe59a4008635490a25fa69261d91`;
audit is `7998a8c258f4b6c5b00a3eccf87bfc1160839003446b81d9cd269ae37a10d6da`.
No projection has run. Coordinates/order remain GDAL-authority, with no
legal-routing or topology claim.

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

## Full-support metadata candidate: Phoenix replay passes

Clean `70c1d20` adds a standalone, unadopted replay of every reference from
all six complete metadata pools. The 80-by-8,700 raw matrix uses the exact
original native query/model identity, with no new neural forward or API call.
The baseline's two original audit pools are retained independently; every
selected baseline row maps to its actual fine-pass pool, without requiring
baseline references to belong to the metadata union. Historical reference
encoding remains UNVERIFIED.

Both full-sequence raw and reference-mean outputs were fixed before new GT
access. Reference-mean is the sole new policy candidate. It passes the exact
existing gates: raw selected-reference median 0.1659418 versus required 0.162,
and unsmoothed coordinate dominance 0.175 versus maximum 0.25. The raw
diagnostic still has 0.3625 dominance and cannot rescue a declined primary.
Already-admitted legacy metadata and earlier baseline/POI/PF decisions retain
precedence. No appearance, dominance or physical-motion threshold changed.

Independent audit reconstructed the lexical union, both complete optimal
paths/objectives, scalar metric median-five routes and original raw-column
proofs. Both objective differences are zero. It also checked both original
baseline pools, all 18 retained source control artifacts and exact Decimal
admission arithmetic. All 1,353 protected files stayed unchanged through
3,148 checks. Inference took 133.84 s on CPU2 with 0.13 GiB peak sampled RSS;
independent audit took 163.38 s.

Separate scoring retained both new outputs and all eight original controls:

| Frozen output | Symmetric mean (m) | Mean at covered GT times (m) |
|---|---:|---:|
| **Fixed reference-mean primary** | **27.87** | **76.44** |
| Raw diagnostic, not selectable | 25.65 | 73.06 |
| Original selected baseline | 18,644.08 | 18,901.82 |
| Original metadata selected raw references, not normal answer | 30.26 | 78.87 |

All six original regional reference tracks were scored too: 18,817.59,
6,087.77, 30.26, 24,477.94, 20,274.55 and 38,656.53 m in original order.
The original selected baseline's symmetric score reproduces exactly. All
scores use the registered 400:595 window and the same 20 waypoints; timed
interpolation uses covered timestamps without extrapolation. A geographically
accurate legacy reference track can still be declined by a fixed input gate;
the new experiment preserves that refusal and supplies a distinct candidate.

Artifacts under `E:/monocular-osm-metadata-union-proof`: all-output master
`d5925d1fa77f5b6f2d3144e0213e17c928cdafeb81124b6d151b3300425e9a3f`, independent audit
`6c01acce3651629f047396a6ef9520e98c0871a1f753e7292e705436b5c60d69`, all-ten score
`858e98dca49c0419dd6a0816a3135c8f5ed18599928a87b086a3b5d073b824c6`.
This is a successful posthoc DEV replay. It is integrated into the ordinary CLI
as explicit `--assisted-profile v3` at `f72fdaf`. A fresh normal-CLI run now
independently verifies the same 27.87 m selected-answer result, as detailed
below. It is not adopted as a default or a complete fifteen-clip result.

V3 preserves v1's earlier-stage precedence and exact original metadata gate.
Only a complete legacy decline can request the new full-support decode before
municipality fallback. All actual metadata calls are retained, including an
ordinary coarse/fine refinement. Both new decoder outputs and the original
controls are frozen before admission; only reference mean is selectable.
The saved child answer preserves the exact metric-median route and timestamps.
Independent review caught and closed two audit gaps: an admitted union could
otherwise be relabeled as legacy, and an embedded candidate could hide a changed
displayed answer. Regression cases now reject both coherent altered transcripts.

The complete regression suite at clean `70c1d20` passes **3,668 tests**, with
six skips and four warnings in 427.08 s. All 448 tracked non-label/result
files remain byte-identical, with all 30 excluded label/result boundaries
unchanged by metadata checks. See
`output/tests-metadata-map-motion-20261028.{xml,log,source.json,review.json}`.
The last complete fixed-policy selected-answer fleet remains **9/15**.

The newer complete suite at clean `f72fdaf` passes **3,860 tests**, with six
skips and four warnings in 514.73 s. All 459 tracked non-label/result files
stayed byte-identical, and all 30 excluded label/result metadata boundaries
were unchanged. CPU thread counts were OpenCV2/Torch2/inter-op1, with external
test networking disabled. See
`output/tests-assisted-v3-20261028.{xml,log,source.json,review.json}`. The clean
normal Phoenix v3 preparation contains all six public pools/8,700 reference
rows, original video and pinned model/runtime assets; no old query descriptors,
OCR, VO or routes are copied. This fresh normal run completed in 1,688.33 s.
The complete-output audit passed before scoring: all reached children, original
requests, native video clocks/pixels, OCR/VO input lineage and strict v3
union/admission checks passed, with 1,351 protected files unchanged. It did not
rerun neural forwards or claim a second full-DP optimum. Ordinary unchanged
benchmark aggregation then scored `result.answer` at **27.87 m**, versus the
old 18,644.08 m baseline. There were no provenance leaks or aggregate warnings.
Root rehashed all 1,357 scoring inputs after completion. This proves the normal
Phoenix path; it does not add Phoenix to the old fleet as a new pooled result.

Phoenix parent freeze:
`673c554c3125dcbe8ba920ddd6414bcc1a5b5767da5005126243f4f27743bf11`;
independent audit:
`18aef39b03f34957059f258bf3c041111b078c4ab5401142cce21fa3c6d8f56a`;
score receipt:
`7666083a726d53625f48720fff4312fd50e855c4f262705cb26853c33d78caf4`.
The family is `E:/monocular-osm-assisted-phoenix-v3-proof/run1`; separate scores
are in its sibling `run1_score`. The latest complete fixed-policy fleet remains
9/15; fresh all-fifteen v3 evaluation is still required.

## Physical-road projection: all twelve variants fail the target

The fixed projection ablation completed all twelve original heading routes
uniformly, with twelve byte-identical controls. Every one of the 960 rows and
all timestamps/end rows were retained. The full index uses all 1,415,696
audited physical ways and 12,135,432 vertices, with one AEQD frame centred on
the mean of all 9,735 public reference coordinates. All actual route points
are inside the public extract polygon; the outside/hole unchanged rule remains
explicit and tested. No distance cap, smoothing, trimming, endpoint pinning,
alignment or radius search was introduced.

Projection moves points by mean 4.2401 m, median 4.1677 m and maximum 13.6774 m.
It took 54.69 s on CPU2 with 0.85 GiB peak sampled process-tree RSS.
An independent implementation rebuilt the complete metric map, proved global
nearest ways using rectangle candidates and all candidate distances, and
computed closest points through line project/interpolate. All 107 distinct
source points and all 960 output rows, numeric-ID ties, inverse coordinates,
displacements and byte-identical controls passed; 1,992 protected files stayed
unchanged. This audit uses no producer nearest-point implementation.

All twelve new projected routes still exceed 200 m. The lowest symmetric mean
is **208.3916317 m**, versus **208.5294520 m** for its original MegaLoc
raw/heading-erased control, an improvement of only 0.1378203 m. Across all
twelve pairs, changes range from -0.6851 to +0.8073 m. Every original numeric
metric reproduces exactly. No variant is adopted; this small lateral
projection does not solve the remaining route-selection error. It does not
test connected or directionally legal map matching.

Under `E:/monocular-osm-road-projection-proof`: master
`d50f71c08b89790393c682e031564647332074db7a8752ec723140247dbd1352`, independent audit
`7807b5266f1ee3d187887295a66dd42e213fe07331bda881d5d3f1e29ce5bb0c`, all-24 score
`10e14e71b45558afbc1ea8ef73015da98a0e79ca5eea909bcde1915c65e56507`.
The first scorer assertion compared metric dictionaries with older annotation
fields; it failed with no numerical differences. Its source/failure receipt
is preserved. The corrected scorer checks every numerical metric and retains
the older annotations separately; no prediction or geometry changed.

A separate connected-road-distance experiment completed on the full
9,735-reference Detroit gallery. It replaces only the sequence decoder's
transition distance and radius legality with shortest undirected physical-road
distance, bounded below by the original Euclidean distance. Shared OSM node
IDs define connections; geometric crossings alone do not. All references and
80 queries remain. Two fresh Euclidean controls and two connected outputs were
frozen before scoring. The map coordinates retain their GDAL-authority
limitation, and the graph does not model legal driving directions or access.
Inference took 2,341.34 s with a 2.45 GiB sampled process-tree RSS peak. A
separate 456.52 s audit reproduced both unchanged D1 controls, checked all four
saved objectives and median filters, and independently recomputed 95 selected
graph-pair shortest distances with zero difference. It does not independently
rebuild the graph or prove a second connected-heading MAP optimum.

All eight declared scores were retained:

| Distance | Raw, original references (m) | Reference mean, original references (m) | Raw, median5 (m) | Reference mean, median5 (m) |
|---|---:|---:|---:|---:|
| Euclidean control | 208.53 | 233.83 | 222.18 | 237.53 |
| Connected physical roads | 493.43 | 221.58 | 546.03 | 229.04 |

**Every output fails 200 m.** The connected reference-mean variant improves its
corresponding control, but neither connected primary route beats the unchanged
raw control. No variant or default is adopted. The median5 curves are secondary
diagnostics, and these development results establish no fleet gain. All 152
consumed files were rehashed unchanged after scoring. Under
`E:/monocular-osm-connected-road-distance-proof`: master
`11f7a5094ccf9eee57b6f152c1868b81110cb5dbbf3171b3865bbaa31c664958`,
independent audit
`acb1cc3fee57d49e9c3fa407093d788782799e2d6ec84474761e1184cc426306`,
all-eight score
`1f5e1cde1d2c73bf576e45d3798228f4c868de5f82101176ee8a422886036220`.

## FoL-base: strict complete-state CPU smoke and audit pass

The first strict load failed before any forward because the released checkpoint
contains `aggregator.dust_bin`, absent from the GitHub inference class. That
failed run and all 195 source/checkpoint files remain frozen. The authors'
[published demo implementation](https://huggingface.co/spaces/shunpeng/fol-visual-place-recognition/blob/de6483cd4165412c01786b4fffc14ed8992c0415/fol/aggregator.py)
documents retaining this unused training scalar for strict checkpoint loading.

A declared compatibility subclass calls the unchanged Git constructor and
adds only that author-published parameter assignment. It inherits the original
forward and region selection methods unchanged. Symbolic base-768 schema checks
and 61 focused tests pass. No checkpoint key is ignored; all 192 tensors and
90,056,257 elements load strictly, including the actual scalar value. Actual
model state matches the full checkpoint digest
`870ad36903cf658a1ee6897139a81f678df87b921fe6054b659bce826e3fbaaa`.

The new runtime smoke completed four native frames at ordinals 0,3,6,9 in
11.17 s on CPU2/inter-op1, with 0.82 GiB peak sampled RSS. All seven output
components are saved. Independent audit redecoded the first ten native frames,
reconstructed all four normalized/resized input tensors, checked every saved
output shape/dtype/typed hash and global norm, and independently hashed the
complete weights-only CPU checkpoint and retained scalar. It made no neural
forward or model constructor call; it does not establish neural output-value
correctness or geographic localization quality.

Under `E:/monocular-osm-fol-runtime-proof`: failed-run freeze
`96a7851b248b7ee02874c14140076d013cd3f71efc9f014398997bec5adcb229`, successful run2
master `8a718914a015e879ddd52d9240c93a912e2c97fe973b236bdc268e0d5bac4129`, audit
`77c817c5585d903ba74f80b4b7bb26b04e88a74ca0367a91ba90582db4df6f3f`.
The frozen local DINO source revision remains UNVERIFIED, and 322 is an explicit
method choice rather than the official parser's 504 default. No FoL++ or
gallery accuracy claim is made. The complete existing Night gallery test
finished separately on CPU2: all 2,998 references and 240 native query crops,
with four FoL global-stage routes and four new cached-matrix MegaLoc controls
fixed before scoring. It retains all seven forward-output schemas and hashes,
but only the global descriptor values; this is not the local reranking stage.
Input freeze `5cc8744f3ca98ba684a40f822b674f3aa222322d1b0f94c6de45bc4998092a26`
and profile `d1e72cb128b6b0ef7b11dd2b1a7b7bc136f87cb6a92f7e6be66b3de10106a95a`
are fixed. No selected variant or default adoption is claimed.
All 3,238 forwards and eight routes completed in 1,783.83 s; peak sampled RSS
was 1.30 GiB and private commit 1.78 GiB. The full192 state matched the strict
checkpoint. Root rehashed all 18 output artifacts, 15 prepared artifacts and
eight executed sources before accepting master
`24d4366d3317e7e660d246dcdb0d22555a630b81b03ad4dba676702d7066d13a`.
The independent output audit passed all reference/native crop and canonical
tensor checks, complete192 checkpoint/state binding, all 719,520 per-view
cosines bit-for-bit, and the max-of-three reduction. Independent float64 dot
products differ by at most `5.47e-8`. It reconstructed all four exact-simple
optimal paths and checked all four erased-heading path objectives and physical
edge counts; it does not claim a second heading-global optimum or neural
output-value replay. All 379 protected files stayed unchanged. Audit
`343863f166b54dc2d6f10a721628bd36a075719ef284425eb501de5017814be9`
passed before separate all-eight scoring:

| Fixed Night output | FoL global stage (m) | New cached MegaLoc control (m) |
|---|---:|---:|
| Exact simple, raw | 10,958.01 | 173.10 |
| Exact simple, reference mean | 1,862.74 | 11,090.49 |
| Heading erased, raw | 428.98 | 711.00 |
| Heading erased, reference mean | 556.12 | 681.34 |

**All four FoL routes fail 200 m; FoL is not adopted.** All eight routes were
scored, with no selected variant. Score freeze:
`50167df9a22bc0a966b6b50a6d9a47b6121e12669d193cec557417994c1ce799`.
The legacy MegaLoc reference encoding remains UNVERIFIED, so this compares
the declared bundled methods rather than isolating an encoder-only effect.
The second registered comparison on G5's complete 9,735-reference gallery
completed all 9,975 forwards in 5,500 s. Peak process-tree RSS was 1.64 GiB.
Its independent complete-output audit passed before scoring: all 13,939 raw
occurrences, both query index systems, all 80 native frames/240 crops and all
2,336,400 per-view cosines were checked. Float32 replay was bitwise exact;
float64 dot products differed by at most `8.00e-8`. All four exact-simple paths
were reconstructed independently, while all four erased-heading saved path
objectives and physical edges passed. It does not claim a second heading-DP
optimum or neural output-value replay. All 331 protected files stayed unchanged.
Separate scoring retained all eight fixed outputs:

| Fixed G5 output | FoL global stage (m) | New cached MegaLoc control (m) |
|---|---:|---:|
| Exact simple, raw | 1,526.70 | 4,779.68 |
| Exact simple, reference mean | 1,321.66 | 4,839.55 |
| Heading erased, raw | 1,516.32 | 208.53 |
| Heading erased, reference mean | 1,094.24 | 233.83 |

**All four G5 FoL routes fail 200 m**, as did all four Night routes. No variant
or default is adopted. These remain bundled global-stage comparisons at 322;
they do not rule out FoL's full local reranking method. Earlier failed G5
preparations and their exact schemas remain preserved. The complete master is
`87e7f322f9baea9510b36e3c5dae7c3fd68e7a9e2944c4e3b1902b04292c1916`,
actual audit `b0a2068f74540b72acfb81a11e743b0b164bfe2bc510cf0bd5b968abcda0d24a`,
all-eight score `577c18e624e7b7e94902b5e71699c93f04bd3d2fc5151c5ac532ba7ef46946da`.

The further NeurIPS source check located
[MutualVPR's official repository](https://github.com/Gucci233/MutualVPR),
including an author-linked checkpoint. Five inspected source files match their
Git blobs at `bd373c6e734556f33d9b4cbc5396862592e624a7`; the source-only receipt is
`output/mutualvpr_readiness_20261028/source_receipt.json`. A subsequent static
readiness check verified 18 source blobs and anonymously downloaded the
384,797,922-byte author-linked checkpoint. Restricted weights-only FakeTensor
inspection found 646 tensors matching every expected architecture key, shape
and dtype after only the uniform `module.` prefix is removed. This includes
all twelve convolutional adapters and BatchNorm buffers. The local checkpoint
SHA is `e8ca2129256bee963d7a8f0e9db7ef0763a16b520bb06811511ea1992ab2f1f0`;
no publisher digest was available. Root and checkpoint licensing remain
unverified; the backbone is Apache 2.0. The example
sets GPU visibility at import and uses an empty checkpoint path with
`strict=False`; a separate bounded strict-load wrapper was prepared and passed
32 synthetic tests. Its 44-image input freeze contains 32 references selected
by fixed ID spacing and the first four registered native queries with three
crops each. The first actual runtime attempt stopped before model construction
or any forward because the isolated environment lacks the author's declared
`timm==1.0.15` dependency. The failed run and original executor/input/deployment
bytes remain preserved. The exact official `timm==1.0.15` wheel was then verified
against its PyPI digest and all 287 RECORD members, and extracted into an
isolated target without changing the global environment. A source guard fix
admits a legitimate public ImageNet resource only through its exact pinned
wheel-relative path and digest; generic label/alias refusals remain in place.
Original prepared attempts remain immutable.

The new 44-image CPU2/inter-op1 run completed in **202.30 s**, using the official
504-pixel preprocessing, strict 646-tensor loading and unchanged dense fallback.
Peak sampled process-tree RSS was 1.08 GiB. Mean forward time was 3.805 s for
the twelve query crops and 3.769 s for the 32 references. Independent audit
checked all staged inputs, native reconstruction, 44 canonical PIL tensors,
all descriptor hashes/norms and the complete weights-only state digest against
the frozen runtime attestation. All 1,207 protected files remained unchanged.
It performed no model constructor or forward replay and does not independently
prove neural output values. The later full-gallery results are reported below.

The runtime family is
`E:/monocular-osm-mutualvpr-runtime-proof/run2_guardfix1`, master
`2d11a7cfe3182f9ebdcf2ec7ed0146aa03ded2856d2ef318e66c082aea9192d6`,
independent audit
`59ec8164de2cbc3600ef31400ede6d7fc2f23bff83627048c2982b136056e129`.
The measured cost suggests roughly 3.4 and 10.5 CPU hours for the Night and G5
neural work alone, before decoding and audit. The full-gallery comparison ran
serially, Night then G5, with a uniform 18-hour internal limit per
case. Before inference, a separate equivalence audit bound all 2,998 and 9,735
reference rows and the two sets of 80 native queries/240 RGB crops to previously
audited inputs, and confirmed the same released model/dependencies as the
44-image runtime. All 991 protected files were rehashed unchanged. The actual
Night run has produced positive forward-input receipts with the accepted
complete model-state digest; that is runtime evidence, not localization accuracy.
All sixteen declared new method/control outputs must be frozen and audited
before new MutualVPR scoring. The input freeze is
`9f90a069371b4c15bbe36a33aaf9c8e5ea5ea24b0ecc9e99b4114e96144080eb`,
input-equivalence report
`b9c35b866fbb2a4816646a08dd82fe6c2a7f08c8504a15152a4810bc7fc16450`.

## Local reference averaging: no failed clip repaired

A model-free posthoc DEV experiment used every reference to average coordinates
around each existing cached-MegaLoc route. Temperature was fixed at 0.02;
Gaussian distance scales were 25, 50, 100 and 200 m. Both raw and reference-mean
emissions and both existing decoders were retained. There was no new smoothing,
neural inference, road snapping or per-clip selection. Legacy reference encoding
remains unverified.

All 32 refinements and eight byte-identical controls froze before new label
access. Independent arithmetic agreed within 2.132e-14 degrees; root rehashed
336 audit inputs and 344 scoring inputs. Preparation, inference, audit and
scoring took 2.031, 3.031, 2.031 and 8.125 seconds.

**All sixteen Detroit refinements fail 200 m.** Raw heading-erased refinement
worsens its 208.529452 m control at every scale. Four Night refinements pass, but
their control already passes; its best change is 173.095194 to 170.194868 m.
No failed clip was repaired, no variant is adopted, and the complete fleet
result remains 9/15.

| Decoder / emission | Scale (m) | Night error (m) | Detroit G5 error (m) |
| --- | ---: | ---: | ---: |
| exact simple / raw | original | 173.10 | 4,779.68 |
| exact simple / raw | 25 | 170.42 | 4,768.22 |
| exact simple / raw | 50 | 170.19 | 4,806.55 |
| exact simple / raw | 100 | 175.67 | 4,810.89 |
| exact simple / raw | 200 | 179.71 | 4,809.58 |
| exact simple / reference mean | original | 11,090.49 | 4,839.55 |
| exact simple / reference mean | 25 | 11,082.26 | 4,832.35 |
| exact simple / reference mean | 50 | 11,081.05 | 4,853.39 |
| exact simple / reference mean | 100 | 11,081.64 | 4,851.28 |
| exact simple / reference mean | 200 | 11,104.83 | 4,862.81 |
| heading erased / raw | original | 711.00 | 208.53 |
| heading erased / raw | 25 | 708.93 | 208.85 |
| heading erased / raw | 50 | 705.66 | 209.45 |
| heading erased / raw | 100 | 705.48 | 209.58 |
| heading erased / raw | 200 | 706.36 | 210.66 |
| heading erased / reference mean | original | 681.34 | 233.83 |
| heading erased / reference mean | 25 | 681.16 | 233.65 |
| heading erased / reference mean | 50 | 681.64 | 233.91 |
| heading erased / reference mean | 100 | 679.64 | 231.33 |
| heading erased / reference mean | 200 | 672.47 | 230.06 |

Every declared output is shown; threshold decisions use unrounded values.
Family: `E:/monocular-osm-emission-refinement-proof/run1`; scores: sibling
`run1_score`. Master:
`41ac90ebb35fd094b7741739c0f61398acc7c839633bbc3ce6621255adaefd36`;
audit: `38f6df44d0716314ef6c07574430ea5084e17ab801fa37f90d9766e89a4533ad`;
scores: `5336f70f11e1d816dc164de3f728928956f9e2827ca9c0c3d440534d102db373`;
root acceptance:
`28aa3fa6632af83fc42fd4a0e6b2cbd4989c12cf49c2c964fdc71b1e0c2b1981`.

## Runtime and metadata identity repairs: regression verified

The shared deadline and source-occurrence fixes passed **3,982 tests, with six
skips and four warnings**, at isolated commit `bc0cc4b`. All 463 protected
non-label files and 30 label/result stat-only boundaries stayed unchanged.
Root and an independent reviewer confirmed all 33 observed process births had
exited. CPU/OpenCV/Torch threads were 2/2/2, with one Torch inter-op thread;
external network access and GPU use were disabled during the regression.

The fixes are integrated at `a8add7a`. Its code blobs match the tested commit;
the remaining differences are these research notes and the explicit `psutil`
dependency, already installed in the tested environment. The main CLI budget
smoke check also passes. No admission threshold or source-identity tolerance
changed. This is operational and identity correctness evidence; a fresh normal
all-fifteen accuracy evaluation is still required. Full regression artifacts
are under the isolated worktree's `output/tests-assisted-runtime-union-20261030.*`.

## Antipodal heading hypothesis: rejected before decoding

A proposed bearing-mixture change affects only exactly opposite heading pairs.
All four existing Night/Detroit erased controls contain **zero such pairs among
316 selected transitions**. Since the change can only subtract penalties from
other paths, it cannot alter these selected MAP paths. The experiment stopped
without a new decode, neural forward or ground-truth coordinate access.
Root independently checked all bin differences and rehashed the 14 consumed
files. Count proof: `924ae6aa70a1f51634ceed0b0b1127e6bdeee9294baee4f4a1a1d59164c58da3`;
root acceptance: `b3fec860151949b9f363978a36414435c4b708696ec2f7efffeac8cc53b19d15`.

## Fresh fifteen-clip evaluation and comparison clock fix

The fixed `a8add7a` pipeline is running all fifteen clips with the same v3
policy, 7,200-second pipeline budget, 7,800-second outer guard and no retries.
Before inference, preparation copied 488 files, a separate input audit passed,
and root rehashed all 973 protected inputs. The accepted input receipt is
`abf12cdb6f40070da81a7d5c15fa51c3747b0ad75c12dd84f952df1624176478`.
No new overall accuracy result is available; the last complete result is 9/15.

Review found that configuration fingerprints included the absolute monotonic
deadline. Identical policies naturally start at different clock values. Commit
`32c2079` excludes only `runtime_deadline_monotonic_s` from method comparisons,
including comparisons between repeats. Raw configuration snapshots, relative
budgets, both reserves and runtime receipt checks remain unchanged. Independent
compatibility checks passed 285 tests with one skip; root reran the nine new
regressions successfully. The live inference checkouts and outputs were not
changed. Their eventual evaluation uses the separately pinned and reviewed
comparison adapter for both benchmark module dependencies, with its source
recorded explicitly.

## MutualVPR full-gallery comparison: no failed clip repaired

The serial full-gallery run completed both cases, with 2,998 Night and 9,735
Detroit references and 240 query views per case. All sixteen declared routes
froze before new label access. The independent audit passed in 634.5 seconds;
root then rehashed all 1,219 protected files before authorizing separate scoring.
Scoring took 7.125 seconds, and root rehashed all 98 consumed scoring files.
All eight cached-MegaLoc controls reproduce their previous unrounded spatial
and timed metrics exactly. Legacy MegaLoc reference encoding remains unverified.

**All eight new MutualVPR routes fail 200 m.** Neither difficult clip is
repaired, and the model is not adopted. This two-case DEV experiment does not
change the last complete fixed-policy result of 9/15. Every declared route is
shown below; threshold decisions use unrounded errors.

| Model | Decoder / emission | Night error (m) | Detroit G5 error (m) |
| --- | --- | ---: | ---: |
| MutualVPR | exact simple / raw | 2,239.25 | 1,403.33 |
| MutualVPR | exact simple / reference mean | 11,665.43 | 4,570.04 |
| MutualVPR | heading erased / raw | 393.81 | 527.74 |
| MutualVPR | heading erased / reference mean | 11,626.83 | 2,785.22 |
| cached MegaLoc | exact simple / raw | 173.10 | 4,779.68 |
| cached MegaLoc | exact simple / reference mean | 11,090.49 | 4,839.55 |
| cached MegaLoc | heading erased / raw | 711.00 | 208.53 |
| cached MegaLoc | heading erased / reference mean | 681.34 | 233.83 |

Family: `E:/monocular-osm-mutualvpr-gallery-proof/run2`; scores: sibling
`run2_paired16_score`. Paired master:
`e972cfe9f0a752561e73a407764b6d338c66887885b0384a2aa1086da2297078`;
independent audit:
`5a4790643f8243fd59178ee0ae8b9f8ae772a34d2e0cc8b1b1994d1851f76298`;
scores: `715fb62b700000325dfbebdb648b23719103521279c8bf0947c2c47e47a074ed`;
root score acceptance:
`22cea4d36cfee4333529ec241a0ec3a58631147c97d7de40241e631743dcc73e`.

## Metric depth: bounded CPU feasibility verified

Depth Anything V2 Metric VKITTI Small completed exactly four fixed synthetic
720x1280 images on CPU, with two Torch threads, one inter-op thread and batch
one. Official RGB/255, cubic lower-bound 518/multiple-14 and ImageNet
preprocessing produced the exact admitted tensors. The complete 239-tensor,
24,785,089-element float32 checkpoint loaded strictly. Root independently
recomputed its numerical state hash and confirmed it matched both the loaded
and final model state.

Initialization took 4.563 seconds. The four forward times were 2.172, 1.812,
2.468 and 2.391 seconds; the outer run took 15.266 seconds. Sampled peak process
RSS was 823,910,400 bytes, within the 3 GiB limit. All four saved native-sized
float32 depth maps are finite and bounded by 0-80 m; root rehashed all eight
runtime files, nine prepared files and their declared dependencies. Resource
guards and observed process termination passed.

This establishes CPU feasibility only. Synthetic depth values do not establish
metric scale, pose quality or localization accuracy. No native video, GPS,
ground-truth coordinates or routes entered this synthetic experiment. The
subsequent masked-native runtime and pair-consistency checks follow.

Family: `E:/monocular-osm-metric-depth-runtime-proof/synthetic_run1`; runtime:
`1443e04c476bd0f97385c16b26adb5b04455838dd7a60cf5802c2d1e3d095052`;
whole checkpoint state:
`ebb6e31f96aa52a747d836d6557da0dd67146a21afaf8ac9a524f22251da5181`;
root runtime acceptance:
`8d9988453fc1cb70f77fe4148a8b7e0f009f1c04c634742c1598bf0bc8751174`.

The same checkpoint then completed four masked native video frames: Night
ordinals 0 and 8,847, and Detroit ordinals 119 and 17,879. Exact original native
pixels were reconstructed from previously audited overlapping crops. A fixed
mask blacks the top and bottom quarters plus all returned text-detector boxes,
preserving the original 720x1280 grid. All four masked-only previews passed
visual review for visible GPS coordinate overlays; detector recall is not
certified. No recognizer or geographic-coordinate decoder ran.

The four native depth forwards took 2.078, 1.687, 1.719 and 1.844 seconds. Total
outer runtime was 16.25 seconds, with sampled peak RSS 836,104,192 bytes. Root
independently checked all four actual input tensors, finite native-sized depth
maps and unchanged 239-tensor state, and rehashed 196 declared files. This is
runtime evidence only; it does not validate metric scale, pose or localization.
The pair experiment below uses eight fixed adjacent-frame pairs per clip and
retains every failed or unobservable pair.

Native family: `E:/monocular-osm-metric-depth-runtime-proof/native_four_run1`;
runtime: `48158acc828591940c36638345ae2046ce52190c0181d5945c91e40115782f27`;
root acceptance:
`426e30068358c9c2dfe33da366ec9e3aa57b56567be832ba1223de49712152be`.

The fixed pair experiment completed all 32 native depth forwards and all 16
reports in 69.047 seconds, with sampled peak RSS 861,470,720 bytes. Complete
original sampling signatures were reproduced before masking. Independent
decodes verified all 32 selected native frames and timestamps; pixel checks
and masked-only visual review passed. Root checked all 32 actual model input
tensors, native float32 depth arrays and unchanged checkpoint state, and
rehashed 447 declared paths. Matches were frozen before depth inference.

| Pair diagnostic | Night | Detroit G5 |
| --- | ---: | ---: |
| Retained pairs | 8 | 8 |
| Measured / unknown | 4 / 4 | 8 / 0 |
| Supported triangulation points per measured pair | 181-289 | 141-273 |
| Nonpositive direct signed motion samples | 435 / 1,200 | 1,101 / 2,330 |
| Pairs with negative median direct signed motion | 1 / 4 | 3 / 8 |

Depth-to-triangulated-depth ratios often agree between the two frames, but
direct 3D displacement exposes conflicting motion signs. Neither agreement
nor a small rigidity residual certifies absolute scale: multiplying every
depth by a constant changes scale without breaking geometric consistency.
The generic 70-degree camera intrinsics also remain uncalibrated. No scale,
confidence, filter input or route was adopted, and no ground truth was used.
The last complete fixed-policy localization result remains **9/15**.

Pair family: `E:/monocular-osm-metric-depth-pair-proof/run1`; master:
`4720f479972318bbaae5eab3251531a64c0d29a71189cd778f6b1477d9958883`;
root runtime acceptance:
`2718324bbcd773e7688c7e6e85ca047348366f3cfe54ddcd1977511b9a4733d4`.

A second, model-free experiment fitted depth-assisted EPNP poses on the same
saved inputs. Original match-row IDs fixed the fit/check split before depth
validity filtering. Each direction used its own source depth, and check
matches never entered RANSAC. The fixed profile passed 62 synthetic checks;
four additional independent synthetic scenarios verified known motion,
isolation of check matches, partition stability and global depth-scale
ambiguity.

All 16 pairs were retained in a 4.062-second run: 13 had independently fitted
forward and reverse poses, while three empty-match pairs remained unknown.
Normalized translation-cycle errors ranged from 0.118 to 0.883 on Night and
0.044 to 1.836 on Detroit. The first five Detroit pairs were relatively
consistent (0.044-0.094); its final three were substantially less consistent
(0.376, 0.929, 1.836). Reprojection tails on check matches remained large.
These observations do not justify a general motion-quality or absolute-scale
claim. No pose, confidence, route or localization policy was changed.

PnP family: `E:/monocular-osm-depth-pnp-proof/run1`; master:
`f2d5414e75a62a1af0a72841e1d387f651aacc4ec250ee61b563c27b60bf8c2c`;
root acceptance after independently checking the saved projections and cycles:
`42f63965c23808dc9fed3e3199e79190a3ff12e9a31f5e45de39087be045a1a2`.

An essential-matrix control then used exactly the same frozen masked matches
and original fit/check partition. All 16 pairs were retained in 3.047 seconds:
13 bidirectional fits and three empty-match UNKNOWNs. Root independently
checked the native-pixel Sampson distances through the fundamental matrix,
unit-pose and cheirality arrays, and 562 declared paths.

The follow-up comparison evaluates all three pose families with the same
native-pixel Sampson residual on identical odd-ID matches. Entries below are
medians of per-pair medians, in pixels; they are not localization scores.

| Common pair cohort | Cached poses | Depth PnP | Essential control |
| --- | ---: | ---: | ---: |
| Night, four pairs with all three poses | 0.437 | 0.541 | 0.466 |
| Detroit G5, all eight pairs | 0.403 | 0.403 | 0.386 |

This provides no consistent depth-specific pose gain or metric-scale
validation. The cached estimator may already have used some check points;
masking alone is also not isolated from changed features, refitting and the
partition. Forward/reverse essential agreement uses the same 2D measurements
and is not independent pose truth. No motion or localization policy is adopted.

Essential family: `E:/monocular-osm-essential-control-proof/run1`; master:
`bad081e11687460703312edd8e9688010626af2b94d9f0032715dd0ccfc57b72`;
common-metric findings:
`7a35d822c3938c75ae1e7670193fabb255837b5c6b71f3af2f59c6812e0301f7`.

[UniDepthV2](https://arxiv.org/abs/2502.20110) is the next candidate because it
predicts camera parameters and metric 3D geometry from RGB alone. Its official
ViT-S checkpoint contains 34,184,323 float32 elements. Pinned source,
safetensors and isolated dependencies passed review. The unmodified released
model then completed four synthetic CPU forwards in a 29.516-second outer run,
with sampled peak RSS 1,116,958,720 bytes. All 352 state tensors matched the
checkpoint digest before and after inference; actual encoder inputs matched
the independently prepared tensors. All six raw outputs were retained.

This establishes CPU feasibility, not calibration, metric scale, pose quality
or localization accuracy. Preparation of the same 32 masked native frames
retains all 16 pairs, including unknown cached poses. The source convention
review also identified a required pixel-center and resize conversion for
future geometric tests; each target frame needs its own learned intrinsics.
Resampled XYZ need not project exactly onto its array coordinates, so this
conversion alone does not validate the predicted camera or geometry.

Synthetic family: `E:/monocular-osm-unidepth-runtime-proof/synthetic_run1`;
runtime: `7e2ea4dcf42c3317df547490d53a1a5ff4831c04c46122c80ea0c82035ce4ea1`;
root acceptance:
`dfc4862faa67fc2dd2128b123194be9a981e0751af56739044927ebb2d8cbefd`.

The same checkpoint subsequently completed all 32 masked native frames in
127.015 seconds, with sampled peak RSS 1,211,863,040 bytes. Root independently
checked all 192 raw arrays, their actual encoder-input hashes, camera-Z and
radius arithmetic, unit rays, complete pair retention and unchanged state
receipts, then rehashed 1,765 declared paths. This preserves all 16 fixed
pairs, including unknown cached poses. No pose fitting or accuracy evaluation
was part of this runtime check.

Native family: `E:/monocular-osm-unidepth-runtime-proof/native32_run1`;
runtime: `ede75b7f07e8b9ac62894e914598319db13b79f0ff50e04d6077c0318ab0c138`;
root acceptance:
`15c241e350ed8c1f466c4a5b8cb4f381e06d40c4b491940f99831788eeec005b`.

A subsequent model-free PnP trial sampled the raw source XYZ and used the
target frame's own converted camera matrix. It preserved the original BF
match IDs and fit/check split. All 32 self-projection diagnostics were frozen
before the first pose fit. The 11.187-second trial retained all 16 pairs:
13 had bidirectional fits, and three empty-match pairs remained unknown. Root
independently checked the samples, camera conversions, projections and cycles,
and rehashed 1,885 declared paths. Six additional synthetic scenarios checked
known motion with different cameras, split isolation and scale ambiguity.

On G5, normalized translation-cycle errors ranged from 0.0065 to 0.3337,
lower on several pairs than the earlier depth-assisted trial. However,
held-out reprojection tails remained large; two reverse directions also
contained nonfinite projected check points. Several fitted steps imply
14-17 nominal metres over approximately 0.134 seconds. The learned focal
lengths themselves vary between adjacent G5 frames by 8.46% and 9.09% at
the median across the eight fixed pairs. These are prediction diagnostics,
not measured vehicle speed or known calibration error.

This does not validate metric motion or justify policy admission. A lower
cycle error cannot establish absolute scale, and comparing learned cameras
against the earlier generic 70-degree camera does not isolate depth's effect.
All raw predictions remain unchanged; no pose, scale or route was adopted.

PnP family: `E:/monocular-osm-unidepth-pnp-proof/run1`;
master: `4a3281aef37b173d30038cc6b2a806b7f6c7c3280e061d410d466781950d417c`;
root acceptance:
`30ea86e35e38498d9a7729a8fb391787fca5c4553950434ddac44f9676174359`.

The final comparison used exactly the earlier common odd-ID matches, with
each new pose's source and target cameras in the fundamental matrix. Root
independently verified all 26 fitted directions through camera-ray pixel
gradients, all held-out and 128 self-projection statistics, unchanged legacy
cohorts and 97 consumed paths. Forward-direction medians of per-pair median
Sampson distances are below, in native pixels.

| Common pair cohort | Cached poses | Earlier depth PnP | Essential control | UniDepth PnP |
| --- | ---: | ---: | ---: | ---: |
| Night, four pairs | 0.437 | 0.541 | 0.466 | 0.588 |
| Detroit G5, eight pairs | 0.403 | 0.403 | 0.386 | 0.510 |

UniDepth provides no median improvement on these cohorts. All eight pairs per
clip remain in the study, including unknowns. G5 reverse projections retain
12 nonfinite check points out of 1,158, rather than excluding them from the
reported denominator. These are geometric diagnostics, not localization
scores or a depth-only causal comparison.

Findings: `E:/monocular-osm-unidepth-pnp-proof/run1_findings/findings.json`;
SHA: `1bd6721b8ea0e96187286089d56d6dd7c7de85fdd7050d71c6a9eab4b24df877`;
root acceptance:
`49ae4229df2ad7cc5213b18bbf90d0fc7536772313b2d69f4c08570f6aa6b865`.

A separate source review found that the strongest historical component
routes use different query views, reference scope or objectives from the
ordinary pipeline. Night's 124 m experiment used one OCR branch; the normal
two-branch union failed, so retrospectively choosing the successful branch
would be an oracle. G5's best 208.53 m control still exceeds 200 m. The new
experiment uses one fixed full-support sequence decoder on all 15 ordinary
runs' own matched inputs. The method, parameters and complete clip order were
registered before the first new parent or array read. Each terminal clip can
enter the comparison after the original strict/native audit, independent root
acceptance and immutable input copies. Both complete fifteen-clip runs and
their independent/root audits remain required before accuracy scoring.

Seven completed A8 clips have passed the original integrity checks; this is
not an accuracy result. The first matched input, Ulm4k, contains all 1,499
observed reference columns across 80 query frames. Root independently checked
every copied raw similarity and the sorted column mapping before launching
the fixed raw/erased decoder. It completed in 105.266 seconds; root checked all
80 selected reference IDs, coordinates, raw scores and timestamps against the
frozen input and confirmed process closure. No accuracy scoring has run.
The comparison uses no new neural inference.
The separate input/copy transport passed 40 synthetic peer tests after a
scoped-alias identity check was corrected; the numerical helper remains
unchanged. No accuracy score or production policy change follows from these
checks. Live counts and artifact receipts are in `output/goal_progress.json`.

The next six input attempts remained unavailable: their overlapping references
require descriptor evidence whose receipt payloads are absent from the
original closed audit's protected map. These attempts remain in the
fifteen-clip denominator. A further evidence audit must establish the exact
consumed descriptor rows and immutable payloads before those cases can enter
the comparison; the existing requirement has not been bypassed.

All 48 frozen receipt payloads for those six clips were subsequently recovered
and matched their recorded hashes. A separate byte audit copied the descriptor
archives before checking their consumed tensor hashes. Root independently
verified all 48 payloads and twelve tensor hashes; the original audits and
unavailable attempts remain unchanged.

The exact descriptor test then rejected Ulm2: three of 959 shared reference
vectors differ numerically, with maximum L2 distance 1.57008e-7. All shared
coordinates are exactly equal, and the largest raw-cosine difference is
2.98023e-8. This is an
observed numeric difference; it does not establish historical encoder identity.
A new experiment was registered over all fifteen clips before reading any new
accuracy labels. It permits descriptor L2 distance at most 1e-6 from the lexical
canonical row, computed in float64. Exact per-pool tensor hashes, receipts,
query/model/clock/coordinate checks and the existing 1e-6 raw-score envelope
remain required. Its algorithm profile is separate from the original data
authority; the earlier exact-equality experiment is preserved.

The first actual assembly under the new rule retained both Ulm2 pools as 2,040
unique references over 80 frames. Root independently checked every canonical
raw value, coordinate, ID and timestamp. A diagnostic replay through the
unchanged raw/erased decoder completed in 144.313 seconds. An independent check
verified all 80 physical rows and 79 transition terms; its recomputed objective
differs by 3.41e-13. No neural re-encoding or localization scoring occurred.
These are input and execution checks, not a complete all-fifteen accuracy result
or a production-policy adoption.

This comparison cannot borrow a favourable research gallery or choose a
decoder per clip. The last complete fixed-policy result remains **9/15**.
