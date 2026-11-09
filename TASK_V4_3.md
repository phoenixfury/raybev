# V4.2 audit and v4.3 retraining

V4.2 is not yet a lightweight accuracy substitute for ROSA. It restores sensible
class-conditioned heavy-vehicle dimensions, but still misses/misclassifies heavy
vehicles and overfits heading. Another unchanged continuation is not justified.

## Measured results

Run: `raybev_v4_2_r18_fg_48m_clidar_2x3080ti`. All three inspected checkpoints
(`best_map`, `best_heading`, `last`) have finite parameters and the expected
calibration/teacher contracts. Best mAP is epoch 9, best high-confidence car
heading is epoch 3, last is epoch 12. Train loss falls 3.55 -> 2.54 while
validation loss increases 7.88 -> 8.48. Epoch-12 mAP falls to 4.56%.

The following uses the same diagnostic center-distance AP evaluator, current
674-frame validation GT, CLIDAR -> BASELINK conversion, +/-48 m square and six
class mapping. Van has no validation GT and is excluded from mAP. ROSA was
reevaluated from its verified cached predictions; no ROSA inference was launched.
Native decoders still differ: ROSA NMS=.5, RayBEV=.25, both capped at 150 with
score floor .01. These are NOT official ROSA FDS/NDS benchmark numbers.

| Metric | V4.2 best epoch 9 | Cached ROSA |
|---|---:|---:|
| mAP | 4.720% | 16.921% |
| Car AP | 23.408% | 50.763% |
| Pedestrian AP | 0.194% | 12.035% |
| Cyclist AP | 0% | 17.107% |
| Truck AP | 0% | 0.281% |
| Bus AP | 0% | 4.419% |
| Car AOE, matches at floor .01 | 55.49 deg | 32.52 deg |
| Car AOE, score >= .3 | 26.70 deg | 22.76 deg |
| Car recall, score >= .3, 2 m | 19.02% | 50.97% |

V4.2 improves over v4.1 best mAP 4.36%, but remains below historical v3.1
MobileNet best 5.03%. ROSA itself is weak on trucks here: it is not a reliable
rare-class oracle. Low bus AOE in the RayBEV log is selection bias: only 10/206
buses match at 2 m; trucks have only 2/134 matches. Size priors are not a cure for
foreground recall or classification. AP also applies minimum precision/recall
gates: zero AP is not evidence of a numerical zero-output failure.

Direct unaugmented oracle-center probes (60 VAL / 48 TRAIN frames, stratified
across all four bags plus rare-class frames) isolate errors before peak/NMS:

| Probe | TRAIN | VAL |
|---|---:|---:|
| Correct truck class | 10/14 | 0/11 |
| Correct bus class | 9/11 | 2/17 |
| Car heading AOE | 13.69 deg | 51.68 deg |
| Car heading flip rate | 4.96% | 25.31% |

On VAL the mean bus length/GT ratio is 1.15 when decoded with the GT class,
but only 0.53 with the predicted class. Mean objectness at true truck/bus centers
is only .031/.026. This is both a proposal and a visual classification problem,
not just a score threshold problem. Priors prevent car-sized buses ONLY if the
class is correct. On all validation labels, median truck L/W/H is 6.36/2.43/2.67 m
versus TRAIN 5.68/2.19/2.18 m; validation priors must not be fitted to hide this.
Heavy vehicles are not simply farther in VAL: truck/bus median ranges are
22.8/24.5 m in VAL versus 24.3/29.3 m in TRAIN.

Machine-readable evidence: `analysis/v4_2_epoch9/audit.json`. It includes the
checkpoint SHA, configuration, logged original metrics, exact sampled tokens,
mean oracle-center diagnostics, ROSA reevaluation and heading breakdowns.
`rosa_provenance/manifest.json` validates prediction/checkpoint/annotation hashes
and equality of source GT. Reported original RayBEV ASE uses the old metric;
do not compare it directly with corrected ROSA ASE.

## Confirmed math fixes

1. **Stride-8 feature lattice.** Both backbones' /8 main paths have origin 0:
   input feature centers are pixels 0,8,16,... . The old LUT and ray cues used
   resize-bin centers 3.5,11.5,... . Correct mapping is:

   ```text
   u_net = (u_raw + .5) * resize_scale - .5 + pad_left
   u_feature = (u_net - feature_pixel_origin) / 8
   grid_x = 2 * (u_feature + .5) / feature_width - 1
   ```

   New origin is 0; historical missing fields restore origin 3.5. In the front
   camera, 3.5 resized pixels corresponds approximately to .85 deg near the
   optical axis (~.71 m transverse at 48 m). Correcting this does not modify K,
   D, extrinsics or the OpenCV lens model. Rebuild ray cues and LUT together.

2. **Facing-vector supervision.** Old `softplus(-2*dot(f,GT_unit))` constrains
   only one projection. For GT yaw 0, f=[-100,4] has loss .000335 but decodes
   to -170 deg with a predicted axis error of only 10 deg. The perpendicular
   component is unconstrained. New loss is raw two-component SmoothL1 toward
   `[sin(yaw),cos(yaw)]`, consistently for GT, ROSA and track references. It
   penalizes both components, has a bounded optimum, and retains a gradient at
   exact 180-degree reversal. The doubled-angle axis and output shapes remain.
   This closes a real nullspace, not all visual ambiguity: the probe finds wrong
   decoded car heading despite a positive GT-facing projection in 4.51% of VAL
   cars, versus 25.31% total flips. Most remaining flips need better evidence and
   generalization. Facing confidence remains uncalibrated and is NOT multiplied
   into object scores. New loss magnitudes are not comparable with v4.2 losses.

3. **Aligned-size IoU union.** Correct union is V_pred+V_GT-intersection, not
   the enclosing box volume. E.g. dimensions [4,2,2] vs [2,4,2] give intersection
   8, union 24, IoU 1/3, ASE 2/3; old code gave ASE 3/4. New metrics record
   `center_distance_v2_aligned_size_union`. AP and heading definitions do not
   change, and historical logs/checkpoints are not rewritten.

4. **Unequal microbatches.** Accumulation now weights each loss by the number
   of frames in that microbatch / actual frames in its accumulation window.
   Previously a short last microbatch had excess weight at batch_size>1. The
   default batch_size=1 is unchanged. Single-GPU epoch losses now also use frame
   means rather than means of unequal batch means.

The comparison utility now also reports foreground softmax probabilities rather
than sigmoid(class logits), and applies classwise candidate limits to final scores
instead of masking logits and accidentally renormalizing softmax. This corrects
diagnostic ablations, not the v4.2 trainer's primary decoder or logged AP.

Checked and retained: OpenCV KB4 forward projection against cv2 on front rays;
inverse ray/feature reprojection; resize half-pixel and letterbox convention;
CLIDAR center and heading transformation once into BASELINK; BEV cell-center,
axis sign, offset and size encode/decode contracts; canonical local CenterNet
radius; rotated polygon NMS; focal/CE/regression denominators; independent class
and regression weights; objectness*softmax(class)*sqrt(quality) scoring; heading
wrap/axis/front-back conventions; fixed BEV BN train/eval equality; exact DDP
validation gathering and additive class statistics. Upright boxes still discard
roll/pitch after the rig transform, as an explicit modeling approximation. The
quality target is a localization-ranking heuristic, not calibrated 3D IoU.
Analytical tests do not establish real-rig calibration accuracy or dataset quality.

## Lightweight ROSA ideas retained / missing

ROSA's local implementation scores multi-camera observations conditioned on the
current object query, aggregates individual keypoints, and updates its spatial
observations during refinement. RayBEV has geometry-conditioned pyramid gates
and learned per-pixel/height confidence, but NO object-query attention or temporal
inference. These gates are lightweight attention-like weighting, not ROSA's
object-conditioned observation selection.

V4.3 addresses an information bottleneck: the old transport averaged all four
heights into one 32-channel feature map. Now it keeps four camera-fused height
maps before the BEV 1x1 convolution. With weight w_hc=prior_hc*sigmoid(logit_hc):

```text
D = max(sum_h,c w_hc, 1e-4)
V_h = sum_c(w_hc * sampled_feature_hc) / D
BEV = concat(V_0,V_1,V_2,V_3, coverage[4], mean_height, XY/range[3])
```

The denominator is COMMON to all heights: weak/unseen planes are not amplified.
The sum of V_h exactly equals the old pooled visual map. Recovery repeats the
old 32 input-channel convolution weights once per height WITHOUT dividing by 4,
then copies the eight geometric-channel weights. This preserves the first BEV
preactivation when the lattice is held constant, while allowing height-specific
weights to learn. The deliberate lattice correction changes full-model predictions.
No size/head/backbone weights are reset when recovering current v4.2.

This is a modest ROSA-inspired evidence-retention change, not a claim that a
dense ray-plane detector now equals ROSA. If heading/classification still fail,
the next experiment should isolate object-conditioned multi-point appearance
evidence or aligned TRAIN-only feature/class distillation, not add more arbitrary
rare-class weights. The current teacher cache contains only matched yaw/score:
it transfers neither ROSA features nor size/class distributions, and rejects
large teacher-vs-GT heading disagreements. It cannot transfer the full detector.
ROSA's native OCam feature coordinates must not be substituted for student
OpenCV coordinates during any future distillation. Also test additional heavy
vehicle appearances/tracks and per-bag generalization; repeated frames do not
create new heavy-vehicle examples.

## TDA4VH implications

| Four-camera frame inventory | V4.2 pooled | V4.3 separate heights |
|---|---:|---:|
| Parameters | 11,369,233 | 11,372,305 |
| Convolution GMAC | 77.356 | 77.809 |
| BEV input channels | 40 | 136 |
| BEV handoff at INT8 | 5.625 MiB | 19.125 MiB |

The image encoder is 74.310 GMAC/frame (~95.5% of v4.3 convolution work).
This ResNet model is not cheap merely because it has few weights. The reference
transport remains host/C7x work outside the learned ONNX graphs; neither MACs
nor tensor footprints include all transport traffic or represent peak memory.
The concatenated handoff needs DDR/tiled/streamed implementation, not an assumed
on-chip fit. Keep a pooled run as an explicit lower-memory ablation. The learned
graphs remain conventional convolutions/elementwise gates; no transformer,
LayerNorm, dynamic attention decoder or GridSample is added to them. Class
softmax, geometry, NMS and decoding remain outside the TIDL graphs.

Before promising throughput: compile both encoder/head on the installed TIDL
SDK, inspect actual offload/fallbacks, calibrate INT8 using representative TRAIN
frames across all cameras/classes, reevaluate VAL INT8 accuracy, and measure
four-camera latency plus C7x transport and NMS on the board. Reducing encoder
cost or revisiting MobileNet is a separate controlled experiment after correcting
geometry; swapping backbones alone has not solved these accuracy failures.

## Retrain on the server

From `projects/raybev_lite`, copy the changed source plus the existing TRAIN
teacher JSON and v4.2 checkpoint. The launcher's default ROOT is the previous
server dataset path; set ROOT if your dataset lives elsewhere.

```bash
OUT=work_dirs/raybev_v4_3_r18_geometry_48m_clidar_2x3080ti \
EPOCHS=12 LR=0.0001 \
bash scripts/launch_2x3080ti.sh \
  --init-from work_dirs/raybev_v4_2_r18_fg_48m_clidar_2x3080ti/best_map.pt \
  --height-fusion concat --warmup-steps 200
```

Use **weights-only initialization, not resume**: new optimizer/scheduler and
epoch numbering are required. Original checkpoints/exports are untouched.
All image backbone stages remain trainable at half the main learning rate.
Class difficulty weights remain (1,2,4,4,4,4); no additional multiplier is stacked.
Existing teacher JSON is reused offline; ROSA need not run on the server.

For the lower-memory control, use a different OUT ending `_pooled` and replace
`--height-fusion concat` with `--height-fusion pooled`. Lattice/facing/metric fixes
still apply. Do not mix their output directories or resume across configurations.

Inspect epochs 2-4 before another long continuation. `history.jsonl`, CSV and
checkpoints now include TRAIN/VAL `center_diagnostics`: class accuracy, objectness,
true-class fused scores, GT-class vs predicted-class size ratios, heading/axis
errors and flips. These are oracle-center probes, NOT detection recall. Demand
better validation truck/bus classification AND proposal recall, lower heading
flips at comparable recall, and improved per-bag AP. A falling training loss or
bus AOE measured on a few surviving predictions is insufficient.

Reproduce the current audit (no ROSA launch; omit `--rosa-analysis` on a server
without the native ROSA validation cache):

```bash
python scripts/audit_checkpoint.py \
  --checkpoint work_dirs/raybev_v4_2_r18_fg_48m_clidar_2x3080ti/best_map.pt \
  --root /path/to/raybev_4bags_native_800x640_v2 \
  --output analysis/new_v4_2_audit \
  --rosa-analysis ../ROSA/work_dirs/combined_4bags_analysis_20261107
```

Validation performed locally: focused regression/math tests and ONNX Runtime
parity; real two-rank CPU DDP gradients and center-stat reduction with an empty
rank; six full-resolution TRAIN steps from actual v4.2 epoch 9 with ROSA/track
losses, finite gradients and unchanged fixed BEV BN statistics. No v4.3 accuracy
training or TIDL board measurement has been performed. These checks establish
implementation consistency, not a future AP gain.
