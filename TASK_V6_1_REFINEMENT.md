# V6.1: heading recovery, precise centers, and ROSA-style heavy-vehicle evidence

## Changes

YOLO26n, Q100, four stages, C64, native 800x640 four-camera inputs, OpenCV
fisheye projection and the +/-48 m region are retained. `architecture_version=4`
selects the changes below. Version 3 and old checkpoints keep their original
inference/loss semantics; the original V6 launcher is unchanged.

### Heading

The doubled-axis vector is supervised **before normalization**. A physical
90-degree axis error now has a corrective gradient rather than the stationary
point in V6's normalized-vector objective. Near-zero vectors fall back to the
previous valid axis before atan2, avoiding undefined gradients while the raw
regression loss remains active. Eight circular facing logits and explicit
front/rear evidence are retained. Tests cover both exact 90-degree axis recovery
and 180-degree facing recovery; they are distinct problems.

### Localization

Stages 1–2 retain spherical direction/log-range updates for coarse capture.
Stages 3–4 use range-independent XYZ residuals in metres:

| Stage | Maximum XY correction/axis | Maximum Z correction |
| --- | ---: | ---: |
| 3 | 2 m | 1 m |
| 4 | 1 m | 0.5 m |

The residual is `limit * tanh(raw)`, so near-zero corrections remain smooth and
continuous, not quantized. Center SmoothL1 beta changes from 1 m to **0.1 m**,
strengthening sub-meter errors. These bounds are correction limits, not promised
accuracy. There is no dense BEV lattice or cell resolution in this architecture.

### What was transferred from ROSA for Truck/Bus

The actual ROSA epoch-6 comparison config has **no special Truck/Bus detector
or class-selected dimensional prior**. Its generic mechanisms are transferred
at reduced width/capacity:

- **Seven fixed plus six learned support points**, just like ROSA's keypoint
  generator. Learned local coordinates are `(sigmoid(Linear(query)) - .5) * LWH`,
  rotated by the current yaw and reprojected at every stage. Large objects get
  proportionally larger sampling regions. Point counts remain fixed, not a
  runtime loop that grows with box size.
- **Separate nonlinear classification and size contexts**, conditioned on box
  state, instead of only one linear layer per task. C64 residual MLPs replace
  ROSA's much wider C256 trunks. Extents remain class-independent.
- **Additive log-size updates** without V6's +/-0.5 tanh cap, with safe absolute
  log-size bounds. Extent supervision uses L1 in log-LWH, like ROSA's box loss.
  A mistaken Car winner never selects a Car dimensional template.
- **Lightweight P5 -> P4 -> P3 semantic fusion**, using two residual 1x1+BN
  blocks rather than ROSA's wide four-level FPN. Zero-gamma initialization
  preserves the initial input function and permits safe weight migration.
- **Class-stratified denoising**: one representative per present class before
  filling the remaining slots. Box-relative XY noise is rotated along the
  object's axes, including for long buses/trucks.
- **ROSA's detached centerness target `exp(-XY distance)`**, without V6's extra
  log-size penalty. Size is already supervised separately; it no longer
  suppresses the quality target a second time.

Existing foreground-only class weighting, explicit rare-class/track sampling,
and regression for all known matched GT remain. We do not copy ROSA's class
weight vector by index: its class order differs. The existing reviewed RayBEV
weights are retained. No confidence boosts or GT-dependent inference are added.

Verified ROSA sources, relative to `projects/ROSA`:

- `work_dirs/combined_4bags_analysis_20261107/new_best6/resolved_inference_config.py`
- `rosa/fisheye_box_decoder.py`: `FisheyeBoxKeypointGenerator`
- `rosa/hybrid_box_decoder.py`: dedicated task trunks and log-size refinement
- `rosa/integrations/sparse4d_components.py`: centerness target
- `../mmdet3d_plugin/models/detection3d/target.py`: class-stratified DN

ROSA camera-model values are **not** copied. Our OpenCV K/D/extrinsics and
CLIDAR-to-BASELINK label conversion remain unchanged. No label edits were made.

## Launch on the server

Copy the updated `raybev/`, `scripts/` and this guide to the server project.
Keep the existing reviewed `datasets/heavy_review_v4/`, TRAIN heading teacher,
calibration and original YOLO weights. No new annotations or teacher inference
are needed. Use the same working Python/Ultralytics environment.

From the project directory, start fresh from official YOLO26n:

```bash
bash scripts/launch_v6_1_yolo_rosa_refine.sh
```

Alternatively, reuse the compatible parts of your V6 epoch-23 checkpoint:

```bash
INIT_FROM=work_dirs/raybev_v6_yolo_rosa_q100_l4_b16a1/best_map.pt \
bash scripts/launch_v6_1_yolo_rosa_refine.sh
```

This is an **audited weights-only warm start**, not resume. Source class order,
semantic annotations, teacher, calibration, frames and geometry must match.
The optimizer, scaler, schedule and best-checkpoint tracking start fresh. V3
axis/size output layers and the final two center outputs are intentionally reset
because their semantics changed. Existing fixed-point attention slots transfer;
six new slots and new modules retain their initialization. The source checkpoint
SHA, epoch and transfer report are recorded in every new checkpoint.

Both commands default to:

- Batch **16 per GPU**, accumulation **1**, two GPUs, AMP: global batch 32.
- **96 new epochs**, LR **0.0002**, backbone multiplier **0.1**, warmup 500
  optimizer updates, all image stages unfrozen, gradient clip 10.
- `work_dirs/raybev_v6_1_yolo_rosa_refine_48m_clidar_b16a1`.

At this dataset size, 96 epochs schedule approximately 11,424 updates, versus
2,856 in the completed B16/24-epoch run. This is a longer training experiment,
not a guarantee of improvement. Warm-start epochs are additional epochs; they
do not count the source's 24 epochs. Monitor the full VAL split and best models.

Override `ROOT`, `OUT`, `BATCH_SIZE`, `ACCUM_STEPS`, `EPOCHS`, `WORKERS`, `LR`,
`HEADING_TEACHER`, or `YOLO26_WEIGHTS` when appropriate. Choose a new `OUT` for a
different experiment. Do not bypass strict recipe checks to resume old V6.

Resume only the new run with its original batch/schedule/teacher settings:

```bash
bash scripts/launch_v6_1_yolo_rosa_refine.sh \
  --resume work_dirs/raybev_v6_1_yolo_rosa_refine_48m_clidar_b16a1/last.pt
```

## Metrics and evaluation

The trainer now prints a validation progress bar, all eight TRAIN/VAL loss
components, mAP, mATE, mASE, mAOE/radians/degrees, and per-class AP, TP/FP/FN,
geometry errors and match counts. JSON and CSV contain the same expanded fields,
heading by score/range/bag and measured-only geometry with no-match status.
An existing CSV with a different header is preserved; new-schema resume metrics
go to `epochs_metrics_v2.csv` rather than corrupting its old rows.

The inherited evaluator's aggregate errors still contain a 1.0 placeholder for
an active class with no matches. Those legacy values are explicitly labeled;
measured-only values exclude unmatched classes and include coverage. Never
judge Bus heading from three easy matches or an empty Van class. TRAIN teacher
and DN losses are absent on VAL, so total objectives are not identical.

Use the existing sparse inference/export commands with a new checkpoint. ONNX
metadata identifies 13 support points, raw axis outputs and fine metric stages.
Old seven-point V6 exports must not be reused after retraining.

## Verification and deployment cost

- Complete legacy + sparse tests: **345 passed**; 20 existing/deprecation or
  expected static-export trace warnings.
- Exact 90-degree axis and 180-degree facing recovery; zero-axis stability;
  synthetic near/far sub-meter fit; geometry/learned-point gradients; DN isolation;
  all-parameter finite backward on foreground, empty and ignored targets.
- Audited V3 -> V4 warm-start, fresh optimizer/schedule, strict data contract,
  V4 resume, expanded metrics, and safe old-CSV handling tested end to end.
- Two-rank CPU DDP: repeated updates with empty/ignored rank, all gradients
  finite and weights synchronized; the complete trainer also finished two
  synthetic DDP epochs with exact VAL gathering and all component/metric logs.
- Four real-image native-resolution GPU AMP smoke updates from official YOLO,
  and four after migrating the actual V6 epoch-23 checkpoint. GradScaler backed
  off initial scales and finite updates passed with clipping. Native encoder/head
  ONNX parity passed after the fresh updates. Proofs are under
  `analysis/v6_1_verification/`; smoke weights are not production weights.

Measured at four-camera native resolution: **1,440,372 parameters**, including
1,159,456 in the encoder; convolution cost **7.5281 GMAC** versus V6's 7.4871.
Linear-layer MACs are approximately 0.1787 G, excluding attention matrix
multiplications and geometric/elementwise work. The extra convolution cost is
about 0.55%, but point sampling grows from 33,600 to 62,400 nominal feature
vectors per four-stage frame. Encoder/pyramid handoff remains C64.

The sparse head remains a CPU/C7x reference, not certified TIDL offload. Board
SDK compatibility, INT8 accuracy, memory transfers and full four-camera >=5 FPS
must still be profiled. No new dataset accuracy or bus/truck AP gain is claimed
until this version is actually trained and evaluated. Production training was
not launched by this implementation.
