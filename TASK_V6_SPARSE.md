# V6: YOLO-ROSA-Lite Q100 / L4 / C64 / P7

## What was built

A separate, trainable sparse 3D detector—not a smaller dense BEV head and not
the original ROSA package with its six-stage checks removed. Existing dense
RayBEV models and ROSA source/checkpoints are preserved.

```
4 x RGB 800x640 images
  -> shared official YOLO26n conv+SPPF backbone
  -> native /8, /16, /32 feature maps, each projected to 64 channels
  -> 100 TRAIN-initialized candidate boxes + 64D query features
  -> four stages of query attention, calibrated image observations and updates
  -> six class scores, continuous 3D boxes and localization quality
```

- No dense BEV, fixed height planes, YOLO PAN/2D Detect or temporal inference.
- Three independent image scales (100x80, 50x40, 25x20), not a single collapsed
  stride-8 image feature map.
- Every stage projects **seven moving box points**: center, front/rear,
  left/right and top/bottom (at 45% of the current dimensions, just inside faces).
- Query attention: width 64 / four heads. Observation-key width 32. FFN width 128.
- Query/content/geometry-conditioned camera-by-scale attention, then point
  attention. All-invalid views yield zero weights, not NaNs/uniform invalid reads.
- Compact polynomial ray embedding; dynamic box-conditioned projection/sampling.
- Continuous center state updated in viewing direction and log range, and
  positive class-independent log-LWH updates. Classification cannot hard-select
  a Car dimensional prior for a Bus.
- Continuous doubled heading axis plus eight circular facing logits, with
  explicit front/rear observation features. Facing confidence does not boost
  object confidence. Circular CE has a nonzero soft-target entropy floor even
  at an excellent fit; judge heading with directed AOE/flip rate as well as loss.
- Static camera calibration is shared, but sampling coordinates are recomputed
  after each box update. Known calibration does not provide object depth/yaw.

Implemented in `raybev/sparse_{config,model,loss,training}.py`.
Public API: `from raybev import SparseConfig, YOLOROSALite`.

## Data and geometry contract

Source annotations remain CLIDAR, converted once by the existing dataset loader
to BASELINK gravity centers. BASELINK is x-forward, y-left, z-up. OpenCV fisheye
projection uses the four supplied distortion coefficients, not OCam parameters.
Projection/sampling and geometric state arithmetic stay FP32 under AMP, including
rays beyond 90 degrees where valid. Image inputs are ImageNet-normalized RGB;
the YOLO stages restore RGB [0,1] internally, exactly like the verified adapter.

Camera order is **ARC_F, ARC_R, ARC_SL, ARC_SR**. The evaluation region remains
X/Y centers in [-48,48). The sparse detector has continuous coordinates, so
there is no BEV-cell resolution parameter. Its 68 m center-range bound covers
the square's XY corners; it is not a 48 m radius crop.

Default annotations: `datasets/heavy_review_v4/{train,val}.pkl`.
Taxonomy: minibuses, ambulances, panel vans -> Van; cargo trucks -> Truck;
trams -> Bus. No labels/geometry/splits were edited while implementing V6.
Validation still has no annotated Van GT and retains the review limitations
documented in `analysis/v5_1_control_reviewv4_report.md`.

The current data has max 37 TRAIN / 35 VAL objects per frame. More GT than free
queries raises an error instead of silently truncating targets. One hundred
queries is a capacity limit, not guaranteed spatial coverage.

## Training behavior

- 100 anchors fit deterministically using TRAIN-only class-weighted spatial
  clustering and real representative box dimensions/headings. They are not
  the first 100 of ROSA's 300 anchors, and inference does not receive GT.
- One-to-one Hungarian matching at every stage; center, relative-size and focal
  class costs. All four stages receive class/center/size/axis/facing/quality loss.
- Foreground-only class weights `(1,2,4,4,4,4)` with a common fixed TRAIN-object
  normalization. No macro-positive/global-negative mixture and no score boosts.
- Sampler explicitly mixes 30% uniform TRAIN frames with 70% rare-class -> track
  -> frame draws. Cars stay supervised in those images; class exposure counts
  are written each epoch. This replaces the ineffective 5%-presence repeats.
  Sampling is seeded/sharded for DDP and preserves background-only frames.
- Ignored targets do not become positives. Unmatched query centers inside their
  conservative footprints are excluded from background classification; matched
  known GT within the same region stays supervised. No VAL ignores.
- Up to 24 noisy GT queries are appended **only in training**. They have known
  supervised correspondence; ordinary queries cannot attend to them. No extra
  queries or GT inputs exist in inference. Class/box/heading denoising is logged
  separately with weight .5.
- Reviewed TRAIN-only heading cache is validated and used on Hungarian-matched
  known GT. Yaw-cache supervision has no dense-cell-owner assumptions. No new
  ROSA inference is required. It is yaw-only, not full class/feature distillation.
- New network, new optimizer/schedule. Default: 24 epochs, 2 GPUs, batch 2/GPU,
  accumulation 2 (effective batch 8), AdamW head LR 2e-4, backbone LR multiplier
  .1, 500-step warmup, AMP, gradient clip 10, all image stages trainable.
- Validation uses the full exact non-padded split; internal mAP, per-class AP,
  per-bag metrics, and heading by score/range/bag are logged every epoch.
  These are the existing lightweight metrics, not official ROSA FDS/NDS.

TRAIN and VAL loss differ in teacher/denoising auxiliaries; compare components,
not just raw totals. Initial low confidence/zero AP is possible with a new head;
monitor matched-query classification and validation recall as learning proceeds.
An empty class's heading statistic is not evidence of correct heading.

## Server setup and launch

Copy updated project code, `datasets/heavy_review_v4/`,
`teacher/raybev_heading_teacher_train_reviewv4.json`, and verified official
`weights/yolo26n.pt` to the server's existing project. Preserve old runs.
Use the working YOLO environment (tested Ultralytics 8.4.182 and 8.4.208); do not
upgrade Torch/NumPy or downgrade a working Ultralytics install unnecessarily.
SciPy is required for TRAIN-only Hungarian matching; if missing, install the
additional `requirements-sparse.txt` in that environment. It was already present
locally, so no packages were installed for this implementation.

From the project directory:

```bash
bash scripts/launch_v6_yolo_rosa_sparse.sh
```

ROOT is auto-detected under the repository's `data/` or `data/Custom/` directory.
Override `ROOT`, `OUT`, `BATCH_SIZE`, `ACCUM_STEPS`, `WORKERS`, `EPOCHS`, `LR`,
`HEADING_TEACHER`, or `YOLO26_WEIGHTS` as needed. The annotation manifest is
verified before torchrun starts. Output:

```
work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti/
  history.jsonl
  epochs.csv
  last.pt
  best_map.pt
  best_heading.pt   # only after minimum heading-match/recall requirements pass
```

**Do not resume v5/v5.1 or ROSA weights into this architecture.** The new run
uses verified official YOLO backbone weights, with fresh adapters/decoder.
Resume only this same sparse run and recipe:

```bash
bash scripts/launch_v6_yolo_rosa_sparse.sh \
  --resume work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti/last.pt
```

Resume checks architecture, class order, camera/LiDAR hashes, known and ignored
geometry, track IDs, teacher, world size, optimizer/schedule recipe and label
conventions. Image paths can relocate without weakening semantic label checks.
Epoch-boundary seeds recreate augmentation, sampling and DN randomness.

## Standalone validation and deployment export

```bash
python scripts/infer_yolo_rosa_sparse.py \
  --checkpoint work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti/best_map.pt \
  --root /absolute/data/root --device cuda --amp \
  --output work_dirs/v6_validation.json

python scripts/export_yolo_rosa_sparse.py \
  --checkpoint work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti/best_map.pt \
  --output work_dirs/v6_export --verify
```

ONNX / ONNX Runtime are optional for export/parity. Output includes:

- `encoder_single_camera.onnx`: invoke four times with shared weights.
- `encoder_four_camera.onnx`: reference batched encoder.
- `sparse_head.onnx`: four-camera pyramids -> 100 boxes, six logits, quality,
  doubled axis, eight facing logits, and visibility support.
- `metadata.json`, and `parity.json` when `--verify` is used.

Camera calibration and anchors are baked into the exported head. Postprocessing
uses max foreground sigmoid * sqrt(quality), finite/support/range filtering,
then optional class-aware rotated NMS. No velocity/attributes are predicted.

**The sparse head is a portable CPU/C7x reference, not a certified fully-offloaded
TIDL graph.** It contains dynamic GridSample, MatMul, normalization and geometry.
The encoder is the first accelerator candidate; exact-SDK compilation and
per-tensor INT8 calibration/parity must be checked before board deployment.
Do not confuse this ONNX export with the old dense learned-subgraph/LUT export.

## Verification and cost

Measured at native four-camera resolution:

| Property | Value |
| --- | ---: |
| Whole-model parameters | 1,385,164 |
| Image encoder parameters | 1,151,008 |
| Sparse decoder/query parameters | 234,156 |
| Convolution GMACs / four-camera frame | 7.4871 |
| Nominal box-point camera projections | 11,200 |
| Nominal sampled feature vectors before visibility masking | 33,600 |
| Three-scale four-camera handoff at INT8 | 2.5635 MiB |

Convolution MACs exclude Linear/attention, projection, sampling, normalization,
elementwise work and NMS. Handoff size is not peak memory or board latency.
`analysis/yolo_rosa_sparse_profile.json` records these counts.

Checks cover calibrated projection/gradients, invalid-view stability, DN
isolation, all-parameter gradients on foreground/empty/ignored batches,
one-to-one matching, ignored-background gradients, independent dimensions,
TRAIN-only anchors/sampling, strict checkpoint/inference contracts, and ONNX
encoder/head parity. A small head-only fit deliberately reverses Truck/Bus
anchor headings by 180 degrees and checks that optimization can recover them;
this is a learnability test, not evidence of dataset generalization.

Additional checks performed:

- Two-rank sparse DDP backward with empty/ignored rank and denoising.
- Complete two-rank synthetic trainer: two epochs, exact VAL gathering and saved
  checkpoints (`analysis/yolo_rosa_sparse_ddp_training_smoke.json`).
- Four real-image native-resolution GPU AMP updates with pretrained YOLO,
  reviewed teacher, Truck/Bus/Van and ignored examples. Finite updates were
  reached after GradScaler backoffs, with gradient clipping. See
  `analysis/yolo_rosa_sparse_native_smoke.json`.
- Native encoder/head ONNX parity after four smoke updates:
  `analysis/yolo_rosa_sparse_native_export.json`. Exported smoke graphs are under
  ignored `work_dirs/sparse_v6_verification/native_onnx/`, not deployable trained
  weights. Initial random/smoke weights are NOT production checkpoints.

The full legacy+sparse test suite passed: **305 tests**. The 18 warnings are
existing GradScaler deprecations and expected static-shape ONNX trace warnings.
No production training was launched. No dataset accuracy gain or TDA4VH latency
is claimed. The target remains >=5 FPS for the full four-camera pipeline (<=200 ms),
including preprocessing, transfers, all four stages and postprocessing. Processor
identification, SDK import/offload, INT8 accuracy and on-board profiling remain
required; sparse query count alone cannot establish that target.
