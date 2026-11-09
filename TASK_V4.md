# Recipe 4: ResNet-18, continuous axis/facing, training-only ROSA supervision

This replaces the MobileNet/four-bin heading recipe, not its historical
checkpoint contracts. Native OpenCV cameras, CLIDAR input labels converted
once to BASELINK, +/-48 m range, 0.25 m cells, train-only size priors and the
stable global-count class losses remain unchanged. Truck/Bus retain 4x object
weights; dense background is not multiplied by difficulty weights.

## Network and heading contract

The default backbone is ImageNet ResNet-18. Its /8, /16 and /32 features have
128, 256 and 512 channels, projected to the same 32-channel geometry pyramid.
All backbone stages are trainable at half the new layers' LR. The BEV handoff
remains 40 channels; there is no inference teacher or temporal state. Matching
TorchVision ResNet-18 weights load strictly through `--backbone-weights`; by
default TorchVision loads `ResNet18_Weights.IMAGENET1K_V1`. MobileNet weights
are not compatible with this backbone. `--backbone mobilenet_v2` remains an
explicit ablation, using the new heading unless loading a historical config.

The head returns four tensors:

```
heatmap [B,6,H,W], regression [B,8,H,W], quality [B,1,H,W], facing [B,2,H,W]
```

The last two regression channels are **sin(2*yaw), cos(2*yaw)**. They represent
the continuous unoriented vehicle axis. Raw SmoothL1 plus the existing norm
penalty trains these values. This removes the old quadrant-residual fold at
45/135 degrees. `make_targets` still stores global GT sin/cos for backward
compatibility; the loss transforms them into doubled-angle targets internally.

The independent two-channel facing output is a continuous classifier vector
ordered `[sin-direction feature, cos-direction feature]`, **not two softmax
logits**. For GT heading theta, its binary front/rear training logit is
`2*dot(facing, [sin(theta),cos(theta)])`, with target front=1. The rear has the
negative logit. Even an exact reversal gets a nonzero corrective gradient.
This continuous formulation avoids introducing fixed front/back target seams.

Host decoding:

```
axis = 0.5 * atan2(regression[6], regression[7])
projection = dot(facing, [sin(axis), cos(axis)])
yaw = wrap(axis + (pi if projection < 0 else 0))
facing_confidence = sigmoid(2 * abs(projection))
```

Use `decode_heading` rather than interpreting facing as bins. Confidence is
uncalibrated and kept separate from object score. Object score stays
`sigmoid(class) * sqrt(sigmoid(quality))`; public detection arrays remain
`[N,9]` with global BASELINK yaw. Size encoding and NMS are unchanged.

## Use the bundled teacher on the server (no ROSA launch)

Copy `teacher/raybev_heading_teacher_train.json` along with the updated project.
The launcher now uses this project-local file by default, not a path inside the
dataset. ROSA model code, its checkpoint, and its native image dataset are not
needed on the server to consume this cache. Paths recorded inside the JSON are
provenance only; the loader never opens the ROSA checkpoint.

The cache stores both the original annotation-file hash and a portable GT
signature. New caches validate every TRAIN token and ordered BASELINK GT box
independently of pickle serialization, frame ordering and machine-specific
paths. Changed labels, calibration, label frame or split remain errors. Old
caches without this signature retain their strict annotation-file hash check.
See `teacher/README.md` for the generated artifact's provenance and coverage.

## Regenerate the ROSA training teacher locally

The available comparison caches contain **validation** frames and cannot be
used here. Teacher input must contain exactly the prepared TRAIN tokens, with
explicit gravity-center boxes and class names. The preparation step matches
same-class ROSA detections within 1 m, requires score >=0.1, and rejects heading
disagreement >30 degrees with CLIDAR GT. It never copies reversed teacher
directions or replaces ground-truth labels. A fully empty matched cache fails.

From the updated RayBEV project directory, supply an existing TRAIN prediction
JSON from the verified ROSA inference workflow:

```bash
ROOT=/home/mustafa/mmdetection3d/data/raybev_4bags_native_800x640_v2
python scripts/prepare_heading_teacher.py \
  --root "$ROOT" \
  --predictions /path/to/ROSA_TRAIN/predictions.json \
  --teacher-checkpoint ../ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth \
  --output teacher/raybev_heading_teacher_train.json
```

Or generate TRAIN predictions using the existing ROSA runner first. Use its
original native dataset/pipeline, not an unverified replacement image root:

```bash
python scripts/prepare_heading_teacher.py \
  --root "$ROOT" \
  --rosa-config configs/rosa_train_teacher_inference.py \
  --rosa-data-root /home/mustafa/mmdetection3d/data/combined_improved_detzero_4bags \
  --teacher-checkpoint ../ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth \
  --device 0 --output teacher/raybev_heading_teacher_train.json
```

Adapt these server paths if needed. The supplied native root must have the
same TRAIN tokens/labels; split mismatches fail rather than being silently
subsampled. ROSA inference can be lengthy and is not run by the training loop.
`--teacher-box-frame CLIDAR` is the default for native ROSA predictions;
BASELINK must be explicitly selected only for already converted predictions.
Teacher output is transformed once into BASELINK using the supplied LiDAR
calibration. Cache metadata binds annotation, predictions, checkpoint and both
calibration hashes, label frame and Z convention. Resume checks teacher content.

## Training-only auxiliary losses

`--heading-teacher-weight 0.2` adds sparse output/response distillation of the
teacher axis and facing at matched GT centers. Portable ROSA JSON does not
contain hidden features or raw heading logits: this is **not feature/logit KD**.
Detection confidence weights/gates teacher matches; it is not asserted to be
a calibrated yaw probability. Collision handling uses the exact GT regression
owner. Sparse terms use the global GT count, avoiding the failed macro balance.

`--track-heading-weight 0.1` adds a pose-aware reference from adjacent TRAIN
track labels. Neighbor timestamps must be within 1.5 s and world-heading
agreement within 20 degrees. References are circular-averaged in world space,
then rotated into the current BASELINK; pedestrians are excluded. This is
label-based track consistency, **not** student-to-student temporal feature
learning. No adjacent camera forward or temporal deployment buffer is added.
Validation labels/teacher predictions never build training references or losses.

## Launch a new run

```bash
ROOT="$ROOT" BATCH_SIZE=1 ACCUM_STEPS=4 bash scripts/launch_2x3080ti.sh
```

Output defaults to `work_dirs/raybev_v4_r18_axis_48m_clidar_2x3080ti`.
The launcher requires `teacher/raybev_heading_teacher_train.json` and defaults
to ResNet-18. Override
`HEADING_TEACHER` for a different verified cache. Batch 1 per GPU with four
accumulation steps is the conservative starting point: ResNet costs much more
compute than MobileNet. Do not assume the old batch 4 fits. Start fresh from
ImageNet, **without `--resume`**; backbone and heading meanings both changed.
Ordinary resume is allowed only within this identical new recipe and cache.
For an explicit no-ROSA ablation, call the combined Python trainer directly
without `--heading-teacher`; it prints teacher weight=0. NPZ training lacks
track/token metadata and rejects teacher caches rather than silently using them.

## Heading reporting and checkpoint selection

Every validation epoch writes `heading_diagnostics` into `history.jsonl`:

- Object score >=0.01 / 0.1 / 0.3: matches, directed/median/axis AOE, reversal
  rate, within-15-degree rate, precision and recall, separately by class.
- GT range 0-20 / 20-35 / 35-68 m after global matching, including GT counts
  and recall; separate results for each sequence and score threshold.
- No matched objects means AOE=null, not an invented one-radian heading.

Score-conditioned summaries also enter CSV/TensorBoard. The historical main
`mAOE` remains unchanged for comparison; the new diagnostics explicitly avoid
its no-match placeholders. Diagnostics are also saved inside each checkpoint.
Axis-regression and facing-classification losses are separately logged as
`heading_axis_loss` and `heading_direction_loss`; the old `heading_bin_loss`
key is retained as an alias of the facing loss for tooling compatibility.
`best_map.pt` still selects detection AP.
`best_heading.pt` independently selects lowest car AOE at object score >=0.3,
requiring >=50 matches and >=10% car recall to avoid a tiny easy subset. Its
selection rule is deliberately car-specific, not an all-class heading claim.
Both checkpoints must be evaluated for detection/heading/recall trade-offs.

## Deployment and validation limits

The learned graph contains ordinary ResNet convolutions/BN/ReLU/pooling and
the existing geometry-conditioned convolutional pyramid/BEV head. The LUT
transport and angle/facing decoding remain host/C7x work. ONNX metadata records
the new facing-vector and doubled-angle contract; teacher/track components
are not exported. Old MobileNet checkpoints default to their original backbone
and binned/global heading semantics and still load strictly.

ResNet-18 is an explicit accuracy experiment, not a verified TDA4VH speed
upgrade. Target SDK compilation, INT8 parity and actual board profiling remain
required. Training and validation must establish whether the new recipe
improves heading/generalization; loss/round-trip tests cannot establish that.

The full-size new model has **11,367,760 parameters** and **77.1577 convolution
GMAC per four-camera frame** (image encoder 74.3104, BEV/head 2.8473). This
compares with the previous MobileNet recipe's 1,988,690 parameters / 14.3977
GMAC. The roughly 5.4x convolution-work increase is not a measured latency
ratio. Teacher/track supervision adds no deployed graph components.

Full-resolution ImageNet-initialized optimizer steps on real Truck/Bus camera
samples passed with finite gradients for every parameter, including track
reference loss. Two-rank CPU DDP with alternating empty ranks passed. Focused
tests cover axis/facing seams/reversals, strict offline weights, historical
checkpoints, cache leakage/calibration/owner guards, a synthetic teacher-enabled
training epoch, confidence/range/sequence diagnostics, and ONNX parity/metadata.
Only synthetic teacher targets were used in those integration tests; actual
teacher-generation coverage/provenance is recorded separately in
`teacher/README.md`. Full retraining and held-out accuracy validation remain
outstanding; training-set teacher coverage is not a validation AP result.
