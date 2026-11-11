# V7: 225 queries, richer calibrated observations, eight combined bags

The new launcher uses architecture version **5**, four decoder stages, 96
channels, stride 4/8/16/32 features, and **225 total matching queries**.
161 slots retain TRAIN-fitted anchors; 64 slots receive image-scored calibrated
ground hypotheses. Denoising slots remain TRAIN-only. The backbone is the
verified official pretrained YOLO26n convolutional trunk.

## Dataset and launch

Run from `projects/raybev_lite`:

```bash
bash scripts/launch_v7_yolo_rosa_observe.sh
```

Defaults:

- Root: `../../data/Custom/combined_bags`.
- TRAIN/VAL: `annotations/fisheye3dod_infos_{train,val}.pkl` under that root.
- Exactly eight bags are required in both splits: 8,815 TRAIN / 908 VAL frames
  in the inspected local dataset. The existing chronological split is preserved.
- Native 1920×1536 images are resized on load to 800×640. Camera order remains
  front, rear, left, right, with explicit ROSA-to-ARC camera-name mapping.
- CLIDAR input labels become BASELINK gravity-centred boxes, ±48 m XY.
- Extents: `datasets/combined_8bags/train_yolo_extents_v1.json`.
- Output: `work_dirs/raybev_v7_yolo_rosa_observe_q225_c96_p2_8bags_48m_clidar`.
- 96 epochs, batch **4 per GPU**, accumulation **4**, two GPUs: effective batch
  **32**. Profile memory before raising the microbatch; `BATCH_SIZE=8
  ACCUM_STEPS=2` preserves the effective batch but changes BN statistics.

`ROOT`, `TRAIN_ANN`, `VAL_ANN`, `OUT`, `EXTENT_ANNOTATIONS`, `BATCH_SIZE`,
`ACCUM_STEPS`, and `EPOCHS` can be overridden. Paths passed to the trainer as
relative annotation paths are relative to the dataset root, so prefer absolute
`TRAIN_ANN`/`VAL_ANN` overrides.

**Start a new run, not a V6.1 resume.** Changed feature widths, queries and
supervision are incompatible with the old sparse checkpoint. The new network
starts from the verified YOLO weights; decoder, adapters and proposals start
fresh. Resume only this run's own checkpoint using the same recipe.

The four-bag yaw teacher is deliberately not a default. `HEADING_TEACHER` may
point to a newly generated teacher that passes the full eight-bag TRAIN hash
and calibration checks. Combined labels are read as supplied; this change does
not silently apply old four-bag track IDs/review rules to the new annotation
file. Old four-bag mAP is not a valid paired baseline for the new holdout.

## Native preprocessing and calibration

Training and validation use `resize_camera_image`: RGB, PIL bilinear resize,
integer letterbox if needed, then ImageNet normalization. The calibrated
native dimensions are checked, so an incorrectly cropped/stretched source
cannot silently use the same intrinsics. Already-prepared network-sized images
are accepted without a second resize.

The projector explicitly constructs and uses **K_network = A_resize K_native**:

```
sx = resized_width / original_width
sy = resized_height / original_height
fx' = sx fx                 fy' = sy fy
cx' = sx (cx + 0.5) - 0.5 + pad_left
cy' = sy (cy + 0.5) - 0.5 + pad_top
```

For 1920×1536 → 800×640, sx=sy=5/12 and padding is zero. Distortion coefficients
and extrinsics do not change. Projection/visibility still respect the native
camera boundary. No undistortion, OCam conversion or double scaling is used.
Export metadata records native K, network K, native dimensions and unchanged D.
Tests compare resized projections directly with OpenCV, including non-integer
resizes and padded aspect ratios. One sample per bag was checked against the
combined annotation extrinsics; maximum matrix discrepancy was 1.1e-7.

## Architecture and objectives

**Proposal coverage.** A 24×24 calibrated XY lattice samples /8 camera features
at ground_z+0.9 m. A small MLP predicts objectness and within-cell XY offsets;
the top 64 visible cells seed query slots. Selected slots retain the fitted
TRAIN size/yaw mixture rather than a predicted-class size lookup. Invisible
slots fall back to their anchors. Selected box centres use ground_z + half the
prior height, so different-sized hypotheses rest on the same ground. Dense objectness/offset losses train the
proposal branch despite discrete top-k selection. `GROUND_Z` is BASELINK metres
(default 0); a single ground plane is an assumption, not terrain estimation.

**Pyramid and observations.** Optional stride-4 features come from pretrained
YOLO layer 2, with a C96 adapter and top-down fusion. Existing learned support
points remain. The richer observation branch uses OpenCV polynomial derivatives
and the resized intrinsic matrix to compute local pixels/radian, anisotropy,
camera overlap, ray tangents and angular feature-level compatibility. These
condition attention keys and reliability. A compact multiplicative
appearance-by-spherical-ray residual conditions values before camera fusion.
This adapts the mechanisms inspected in ROSA's `fisheye_evidence.py` and
`fisheye_box_decoder.py`; it does not import ROSA's OCam projection or C256 head.
Jacobian arithmetic is FP32 even under AMP; singular-value metadata is detached
to avoid undefined gradients at isotropic rays, as in ROSA.

**Heading.** Raw doubled-axis regression is retained. A separate binary
hemisphere objective supervises front/rear along the detached predicted axis.
It uses stable log-sum-exp logits and BCE-with-logits for AMP safety. Each camera
also predicts facing evidence from the two longitudinal support points before
camera fusion. Cameras missing either point contribute zero. Decoding selects
the hemisphere with greater probability. The cosine heading prior defaults to
zero for the new experiment and is separately configurable. Facing confidence
never multiplies detection scores. Confidence is supervised, but good calibration
is an empirical result: Brier score/ECE must be inspected after retraining.

**Projected extents.** Predicted corners plus samples on all twelve box edges
are projected into valid cameras. Curved fisheye edges therefore have additional
support beyond their corners (the finite sampling is still an approximation).
Image-space SmoothL1 constrains only approved visible edges. The loss can train
length, height and unsigned yaw. Residuals are normalized by the measured 2D
width/height (minimum four pixels), not the full canvas, to retain far-object
gradients. A projected extent cannot distinguish a 180°
front/rear flip by itself.

## Independent camera-box supervision

The bundled eight-bag file contains **35,837** accepted weak camera observations
over **8,140** TRAIN frames: Car 33,658, Pedestrian 1,300, Truck 583, Bus 280,
Cyclist 16. No Van extent passed the gates. Validation is never used to prepare
these targets. Original 3D annotation files and classes remain unchanged.

Regenerate on a machine with source caches:

```bash
python scripts/prepare_sparse_extents.py \
  --root /absolute/path/to/combined_bags \
  --output datasets/combined_8bags/train_yolo_extents_new.json
```

The converter reads `combined_bags/summary.json`. Its four November sources
are under `tools/detzero_temporal_pipeline_complete/runs`; the four September
sources are under `data/Custom`. Use `--source-summary` or repeated
`--source sequence_id=/path/to/dataset` if those paths differ on the server.
The generated annotation JSON is portable and can be copied directly into
RayBEV on the server: training does not need the source caches or ROSA running.

These are **weak YOLO measurements, not human-verified amodal boxes**. Native
timestamp/frame identity is checked. Matches require confidence ≥0.6, IoU ≥0.4,
a ≥0.15 uniqueness margin, compatible broad classes, and one-to-one association.
Potential occlusion by nearer known/ignored 3D objects rejects the target;
truncated image/projection edges are masked individually. This is conservative
visibility estimation, not a guarantee against unlabelled occluders. Targets
receive weight 0.25×YOLO confidence and retain source cache/manifest hashes.
The annotation loader rejects mismatched TRAIN/calibration hashes, unproven
YOLO provenance and missing visibility masks. Reviewed 3D classes are never
overwritten with YOLO classes. The previous four-bag generated extent file was
superseded and removed; existing user annotations were not removed.

## Measurements and controlled comparisons

`history.jsonl` and the metric CSV include score/class-agnostic one-to-one raw
query recall at **0.5, 1, 2 and 4 metres**, for initial queries and every stage,
by class and 0–20 / 20–35 / ≥35 m range. Matching maximizes valid-pair count
before distance, so duplicate queries cannot inflate recall. This is XY proposal
coverage, not AP. The console prints initial/final coverage.

Standard post-NMS directed AOE, unsigned-axis AOE, flip rate and matched coverage
remain separate. Additional facing reliability reports Brier/ECE and those
heading errors on explicitly labelled pre-NMS class-correct matches at score
≥0.1 and XY<2 m. Loss components include proposal, binary facing and extent.
TRAIN-only extents, denoising and teacher terms are absent from VAL loss.

Use different `OUT` directories and keep the same dataset, seed, epochs and
effective batch for controlled training:

| Arm | Environment overrides before the launcher |
|---|---|
| Q225 C64 baseline | `CHANNELS=64 STRIDE4=0 OBSERVATION_CONDITIONING=0 PROPOSAL_QUERIES=0 EXTENT_WEIGHT=0` |
| Add ground proposals | same, with `PROPOSAL_QUERIES=64` |
| Test C96 | previous arm with `CHANNELS=96` |
| Add stride 4 | previous arm with `STRIDE4=1` |
| Add observation conditioning | previous arm with `OBSERVATION_CONDITIONING=1` |
| Add weak extents | previous arm with `EXTENT_WEIGHT=1` (full default) |
| Facing-prior test | full arm with `FACING_COSINE_PRIOR=1` |
| Multi-camera-facing test | full arm with `MULTI_CAMERA_FACING=0` |

Run `scripts/infer_yolo_rosa_sparse.py` on each resulting checkpoint with the
same eight-bag VAL annotation path. It records AP, raw coverage, heading and
network latency. No accuracy improvement is claimed before those runs train.

`scripts/benchmark_sparse_observation.py` measures paired native-shape synthetic
latency and optional backward memory. The local short AMP run on an RTX 3060
Laptop GPU measured ~31.8 ms for Q225/C64 and ~47.4 ms for the full architecture,
with 1,886,231 parameters. These are network-only timings, batch 1, random
weights/input, five measured iterations; small differences between intermediate
arms are noisy. See `analysis/v7_observation_native_profile.json`. They do not
include resize, IO or NMS and do not establish TDA4VH performance.

The real-data native AMP smoke uses all eight bags, pretrained YOLO, weak
extents and two disposable optimizer updates. Its report is
`analysis/v7_observation_8bags_final_smoke.json`; it is not a trained model.
Two-rank CPU DDP also passed with empty/ignored targets on one rank and three
synchronized updates. The focused suite passed 80 tests, with two optional
dependency/export skips. ONNX code now handles four pyramid levels, proposals and
facing confidence, but export/parity could not run in this environment because
`onnx` is unavailable. TIDL operator support, INT8 accuracy and ≥5 FPS on the
actual board remain unmeasured; top-k, dynamic sampling and Q225 attention must
be included in that deployment check.
