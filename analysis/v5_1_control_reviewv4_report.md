# V5.1 YOLO26n control: validation comparison and bottlenecks

## Verdict

The control stabilizes the previous confidence collapse, but does **not** solve
heavy-class recognition or directed heading. It is not yet a successful light
ROSA. Do not extend this control unchanged expecting the missing classes to
appear. This run uses `facing_loss=unit_vector`, **not circular_bins**: it cannot
accept or reject the proposed circular-heading experiment.

Run: `raybev_v5_1_yolo26n_control_reviewv4_48m_clidar_2x3080ti`.
All files below are under that run's `rosa_comparison/` directory.

## Video and evaluation contract

- Video: `validation_raybev_vs_rosa.mp4`, H.264, 2560x900, all **674 VAL frames**,
  2 FPS, **337 seconds (5:37)**, 117,786,111 bytes. Frame count verified by ffprobe.
- RayBEV checkpoint: `best_map.pt`, epoch **5**. `best_heading.pt` also selects
  epoch 5; last is epoch 6. Model parameters are finite.
- ROSA: verified existing epoch-6 inference cache; no ROSA retraining/inference.
- Identical reviewed `heavy_review_v4/val.pkl` GT for both: Car 7345,
  Pedestrian 313, Cyclist 13, Truck 188, Bus 152, Van 0.
- OpenCV video overlays; GT, ROSA predictions and LiDAR transformed once from
  CLIDAR to BASELINK. Video displays both at score >=0.10, with explicit scene
  cuts. This is dataset playback, not an end-to-end runtime/FPS measurement.
- Cache metadata records 800x800 source images for ROSA (all 2696 camera records),
  versus RayBEV's native 800x640 inputs. These are matched frames/GT with each
  model's native pipeline, **not a controlled input-preprocessing/backbone-only
  ablation**. The square source resize does not imply extra original image detail.
- Native decoders retained: RayBEV NMS .25, ROSA cached NMS .5, both capped at
  150 predictions/frame for the primary comparison. Scores are not calibrated
  equally. Equal display thresholds are **not equal recall**.
- Same lightweight evaluator, not official ROSA FDS/NDS. AP uses predictions
  down to .01 and center matching at 0.5/1/2/4 m. Van is excluded from mean AP.
- Reviewed classes differ from original ROSA-cache annotations, but raw geometry
  is identical. Predictions are NOT retrospectively relabeled. Hashes, tokens,
  video SHA and provenance are in `manifest.json`.

## Measured results

| Metric | RayBEV control | ROSA |
| --- | ---: | ---: |
| Same-evaluator mAP | 5.81% | 16.54% |
| Car AP | 26.99% | 50.76% |
| Pedestrian AP | 2.06% | 12.03% |
| Cyclist AP | 0 | 17.11% |
| Truck AP | 0 | 0.20% |
| Bus AP | 0 | 2.58% |
| Car precision at matched 44% recall | 49.99% | 93.30% |
| Car directed AOE at matched 44% recall | 33.30 degrees | 19.85 degrees |
| Car >90-degree flip rate at matched 44% recall | 16.00% | 8.48% |

Matched-recall measurements use score-ranked Car predictions and 2 m matching,
without filtering by angle. Different models can match different GT subsets;
equal recall is more informative than equal score, not identical object coverage.
At 30% Car recall, AOE is 25.78 vs 16.09 degrees and precision 80.0 vs 96.9%.
ROSA is also imperfect, especially on this heavy-class GT.

Training logs select epoch 5 at 5.8147% mAP; fresh local AMP inference gives
5.8090%. Small numerical/NMS differences do not change the conclusion.
V5 best was 6.2444%; V5.1 is lower by **0.43 percentage points**. Relative to its
V5 epoch-3 initializer (5.4078%), it is a small gain, not a heavy-class solution.
Train loss improves 4.081 -> 3.749; VAL stays 6.652 -> 6.655 across six epochs.

## Bottleneck 1: rare-class learning, not just confidence thresholds

Best-epoch GT-center statistics (oracle probes, **not detection recall**):

| Quantity | Truck | Bus |
| --- | ---: | ---: |
| TRAIN class accuracy | 62.1% | 4.3% |
| VAL class accuracy | 6.9% | 17.1% |
| VAL mean objectness | .03565 | .03136 |
| VAL mean true-class probability | .12374 | .19314 |
| VAL mean fused true-class score | .00442 | .00487 |

Van TRAIN class accuracy is **0%**; Van has no VAL GT. At VAL GT centers,
166/188 trucks choose Car. Buses choose Car 72 times, Truck 49 times and Bus
only 26 times. At display score .1, correct-class Truck and Bus detection TPs
are both zero. Misclassification and weak objectness jointly remove objects
before they can contribute useful boxes/heading.

The current sampler is class-*frame-presence* balancing with a 5% threshold.
Truck appears in 562/3787 frames, Bus in 282 and Van in 300, so all receive
**repeat factor 1.0**. It expands 3787 frames to just 3807 (DDP uses 3806).
Expected Truck/Bus exposures are only 645 -> 652 / 337 -> 346, while Car is
38155 -> 38275. Fourfold class weights do not erase this imbalance: approximate
weighted Car exposure is still ~15x Truck and ~28x Bus. These are exposure
ratios, not measured gradient norms.

Unlike V5's late collapse, Bus true-class probability rises .172 -> .194 and
Truck stays near .126 -> .123 over this run. Freezing has stabilized the trend,
but Bus/Van are now clearly underlearned even on TRAIN. The corrected taxonomy
is being adapted from an old, Car-dominated head with only six low-LR epochs.
The isolated contribution of freezing vs label cleanup cannot be inferred from
this one arm.

Scoring ablations confirm this is not primarily the quality multiplier or top-k:

- Quality factor disabled, objectness retained: mAP 5.8009%.
- Score floor .001 instead of .01: mAP 5.8275%; Truck AP still zero,
  Bus AP only 0.0924%.
- Per-class top-k 50: mAP 5.5453%, not an improvement. This still respects the
  checkpoint's single-label decoder and is not an oracle proposal-recall test.

No score boosts were applied.

## Bottleneck 2: front/back ambiguity remains largely unresolved

Bus VAL GT-center directed AOE is **121.57 degrees**, but unsigned-axis AOE is
only **12.45 degrees**: **70.4%** exceed 90 degrees. This is primarily front/back
selection, not simply an inaccurate long-axis angle. Mean facing-vector norm is
.244, consistent with a weak/ambiguous unit-vector regressor. At a Bus oracle
center, the median directed error is 172.88 degrees.

This is not only uncertain-label outliers: true-bus track 17_09/984 contributes
79/152 VAL Bus boxes; its oracle flip rate is 64.6%. Track 17_09/953 contributes
25 with a 76% flip rate. Full per-track probes: `heavy_track_diagnostics.json`.
Do not confuse these oracle probes with detected-bus heading accuracy: at .1
the model has no correctly detected buses.

The teacher currently supervises **yaw only**. Its original <=30-degree GT-yaw
agreement gate excludes large GT/teacher disagreements; neighbor-label smoothing
also cannot repair a consistently reversed track. Neither provides appearance,
objectness or class distillation. Circular bins can test a better facing model,
but cannot resolve wrong labels or absent visual front/rear evidence by itself.

## Bottleneck 3: hard class-dependent size decoding amplifies class errors

The size head uses `class_relative` LWH and selects its prior using the predicted
class. At GT centers, mean decoded length / GT length is:

| Decode assumption | Truck | Bus |
| --- | ---: | ---: |
| Oracle correct class | .877 | 1.289 |
| Actual winning class | .663 | .644 |

Thus the visual impression of a small Bus/Truck box is partly the Car/Truck
classification error selecting the wrong dimensional prior. Even with the
correct class, Bus dimensions are not solved: the prior/residual overestimates
length. The final size projection was intentionally reset for changed priors;
this short fine-tune has not learned adequate instance-specific corrections.
Oracle ~0.1 m *offset* errors are not evidence of accurate proposal localization.

## Bottleneck 4: supervision quality and diversity are still limiting

The final TRAIN fork and teacher match the checkpoint hashes. Counts are
645 Truck / 337 Bus / 312 Van, but only 69 / 58 / 15 sequence-track IDs.
These are track-ID counts, not independently verified unique physical vehicles.
TRAIN class review was conservative, not certification of all 3D geometry/yaw.

VAL is concentrated: 128/152 Bus boxes are in bag 17_09; one track supplies 52%.
It still has zero annotated Van GT despite the approved Van taxonomy.
The existing contact sheets show **review candidates**, not safe automatic
track-wide corrections: 30_09/2630, 2664 and 2723 look van-like; 25_09/2953 and
30_09/2766, 2778 have car/background crops associated with bus-sized boxes.
Occlusion/association can mislead a single crop. These need multi-frame/LiDAR
review, not blanket relabeling from thumbnails. No VAL labels were changed here.
This can distort evaluation but does not explain the very poor TRAIN Bus/Van
recognition or missed visually real buses.

A motion/heading screening probe also finds >120-degree disagreement on 9.0%
of eligible TRAIN Car pairs and 13.3% of Bus pairs. This is **not a label-error
rate**: reversing, track association and pose errors can cause it too. It supplies
review candidates only; no yaw targets were rewritten.

## Bottleneck 5: BEV resolution is not image/depth information

RayBEV has stride-8 image features (100x80), fused to 32 channels, and four fixed
height planes. Adjacent 0.25 m BEV cells often sample very similar image features.
At VAL Bus centers, a 0.25 m forward displacement moves the projection by a median
of only .069 feature cells in the best visible camera; Truck is .124. These are
geometric differential probes at GT centers, not measured CNN feature changes.
A 384x384 output grid therefore does not create independent image evidence.

Source inspection shows ROSA's evaluated checkpoint uses VoVNet-99, 256-channel
features at strides 4/8/16/32, six object-conditioned refinement stages, dedicated
size/face refinement and ray/range auxiliary losses. Its temporal instance count
is **zero**; this comparison is not evidence of a temporal-model advantage.
RayBEV's fixed projection and single dense decoding do not reproduce those
object-conditioned corrections. That representational gap is a plausible
contributor, not isolated by this experiment. The result is not proof that
YOLO26n is the wrong backbone or that full attention is required.

Calibration/LiDAR/teacher/annotation hashes all pass. Frozen YOLO /8 and /16
weights and BN buffers are exactly unchanged from the initializer; all 72 /32
state tensors changed. No failed weight load or accidental full-backbone freeze
was found. Across all 2696 VAL camera records, recorded LiDAR-to-camera matrices
agree with the model's fixed rig matrices to max absolute difference 1.06e-7.
Matching hashes/math do not certify physical rig calibration accuracy.

## Recommended next work — not implemented as model changes here

1. Finish frame-level taxonomy/geometry/heading review and maintain an immutable
   reviewed evaluation baseline. Retain this baseline when reporting any new
   label revision; do not treat label-only metric changes as learning gains.
2. Replace ineffective heavy-frame balancing with explicit heavy-track/batch
   coverage and a controlled foreground-class balancing experiment. Repeating
   whole images also repeats their Cars; balance object/class supervision, not
   just frame counts. Keep dense objectness background normalization stable;
   do not repeat the old macro-positive/global-negative mismatch.
3. Run the pending circular-heading arm on the *same* current labels/recipe to
   isolate facing formulation. Judge it by matched-recall heading and heavy-class
   metrics, not the lowest high-threshold AOE. It is not a class-recognition fix.
4. Test size decoding independent of a hard winning-class prior, and/or stronger
   instance-specific size supervision. Keep classification and size failures
   separately measurable.
5. Add TRAIN-only appearance/objectness/class or sparse BEV distillation after
   validating teacher taxonomy. The current yaw-only cache cannot teach the
   missing recognition/depth representation. A projected image-space auxiliary
   can improve training without adding deployment inference layers; visibility
   and occlusion still require care.
6. Only then test selective /16 unfreezing and a compact object-conditioned/local
   refinement or depth-confidence module. Avoid changing backbone, sampler,
   heading, size and capacity together. TDA4VH favors foldable convs and host LUTs,
   not copying ROSA's full six-stage attention decoder.

TIDL compile/offload, INT8 accuracy, memory and TDA4VH latency remain unmeasured.
The existing 136x384x384 BEV handoff is ~19.1 MiB at INT8; a more detailed output
grid alone would increase deployment cost without fixing these demonstrated
classification and supervision failures.

## Artifacts and reproduction

`run_analysis.json`: six-epoch trends, checkpoint hashes/contracts/freeze checks.
`comparison.json`: common-GT metrics, ablations, confusion and oracle probes.
`matched_recall.json`: score-ranked equal-recall Car heading comparison.
`data_bottlenecks.json`: sampling, track counts and projection/motion probes.
`heavy_track_diagnostics.json`: per-track heavy-class oracle probes.
`input_geometry_checks.json`: all-frame input metadata and extrinsic consistency.
These generated files and the 112.3 MiB video are under ignored `work_dirs/`;
do not add them to Git to recreate the previous large-file push problem.

The comparison utility now accepts explicit reviewed TRAIN/VAL annotations and
records class edits separately from unchanged geometry. 21 focused calibration,
revision and comparison-contract tests pass. No network weights, training labels,
training recipe or production training run were modified/launched for this audit.

```bash
OMP_NUM_THREADS=2 PYTHONPATH=.:../ROSA/work_dirs/onnx_deps_py38 \
conda run --no-capture-output -n openmmlab python scripts/compare_validation_rosa.py \
  --checkpoint work_dirs/raybev_v5_1_yolo26n_control_reviewv4_48m_clidar_2x3080ti/best_map.pt \
  --root /home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640_v2 \
  --calibration configs/calibration_baselink.json \
  --rosa-analysis ../ROSA/work_dirs/combined_4bags_analysis_20261107 \
  --train-ann /home/mostafa/mmdetection3d/projects/raybev_lite/datasets/heavy_review_v4/train.pkl \
  --val-ann /home/mostafa/mmdetection3d/projects/raybev_lite/datasets/heavy_review_v4/val.pkl \
  --output work_dirs/raybev_v5_1_yolo26n_control_reviewv4_48m_clidar_2x3080ti/rosa_comparison \
  --stage all --device cuda --batch-size 2 --workers 0 --fps 2 \
  --score-threshold .1 --ray-title 'RayBEV v5.1 YOLO26n control'
```

The completed video is protected against overwrite; use a new output directory
for a full rerun, or `--stage analyze` to reuse verified predictions without video.
