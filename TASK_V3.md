# MobileNet task recipe 3.1: heading, heavy-vehicle size and classification

This is the current training/head contract. The backbone/BEV architecture
remains v2: MobileNetV2-1.0, OpenCV fisheye, CLIDAR input labels converted once
to BASELINK, 800x640 images, +/-48 m, 0.25 m cells. All retained MobileNet
stages remain trainable at half the new layers' LR. No ResNet or transformer
is introduced. The older [V2.md](V2.md) is the geometry/history reference.

## Correction after the zero-AP v3 run

The original v3 run reached four logged epochs with all class APs zero. Its
macro-averaged foreground loss used a different normalization from the dense
background loss. Stacking this with difficulty weights, object-frequency
repeats and a 0.5-weight class CE suppressed cars and promoted rare-class false
positives. A checkpoint replay found the correct class at only 14/115 sampled
car centers; disabling quality scoring or single-label decode did not recover
AP. This is a classification-training failure, not an empty validation set.

Recipe 3.1 restores **global object-count normalization** for focal, box,
heading and quality losses (`class_balanced_losses=False`), reduces auxiliary
class CE to **0.1**, and restores **frame-presence** repeats. Foreground and
regression difficulty weights remain `(1,2,4,4,4,4)`: Truck/Bus still receive 4x
the per-object gradient, without additional inverse-microbatch class counts.
The heading/size architecture, calibration, score rule and MobileNet are
unchanged; there is no added inference compute. Historical checkpoints retain
their saved loss flags and inference behavior when loaded, but cannot resume
into the corrected defaults. Start fresh from ImageNet in the new directory.

## What changed

- Heading: four coarse direction logits plus a shared local sin/cos residual.
  The nearest bin covers a +/-45 degree residual. Training uses bin cross
  entropy and **raw**, not pre-normalized, residual regression with a soft
  unit-norm penalty. A reversed bin therefore gets a corrective classification
  gradient even when the residual starts at (0,1). Global heading is decoded
  by adding the selected bin center and wrapping to [-pi,pi].
- Size: the three outputs learn `log(size / class_median_size)`. The medians
  are fitted from original, range-filtered **training labels only**, before
  balancing. Validation/test labels never fit the priors. The size head starts
  with zero residuals and learns variations; boxes are not forced to fixed sizes.
  Decode adds the **predicted class** log-prior. Incorrect classes can still
  yield incorrect dimensions, hence the coupled classification changes below.
- Classification: positive focal and dense background losses share the total
  positive-center denominator; regression/quality/heading use the total box
  count. Difficulty weights `(1,2,4,4,4,4)` apply only to foreground/object
  terms, not dense negatives. A 0.1-weight auxiliary foreground class CE
  explicitly discourages Car from winning at a Truck/Bus GT center. Ambiguous
  different-class centers sharing one grid cell are excluded from this CE.
- Sampling: repeats use the milder 5% frame-presence rule, not object frequency.
  Each frame is retained 1-3 times, with deterministic seeded rounding; empty
  frames are retained once. This is combined-dataset training only. NPZ training
  shares the fitted size priors and losses, but has no frame-repeat sampler.
- Decode: only one class wins per cell before fused-score local peak selection.
  This avoids decoding one shared box as both a car and a class-prior-sized bus.
  Object score stays `sigmoid(class) * sqrt(sigmoid(quality))`; no artificial
  rare-class multiplier is applied at inference.

Heading-bin confidence is available from `decode_heading` as the maximum bin
softmax probability. It is **separate from object confidence**, is not yet
calibrated, and is not a guarantee of an accurate residual angle. Existing
center/size quality does not magically become yaw quality. Annotation direction
errors and visually ambiguous front/back views still require data auditing.

## Train fresh

Do not `--resume` the old epoch-18/v2 or zero-AP v3 checkpoint: heading/dimension
semantics or the loss/sampling recipe differ. Defaults use ImageNet
MobileNet initialization and a separate output directory; old weights and
metrics are untouched. Resume is supported only within an identical new run.

From `projects/raybev_lite`, on the existing server:

```bash
bash scripts/launch_2x3080ti.sh
```

The existing server ROOT is preserved. On this workstation, override it:

```bash
ROOT=/home/mostafa/mmdetection3d/data/Custom/raybev_4bags_native_800x640_v2 \
  bash scripts/launch_2x3080ti.sh
```

Default output: `work_dirs/raybev_v3_1_tasks_48m_clidar_2x3080ti`.
The conservative batch defaults remain one image/GPU with four accumulation
steps. If the previous four-image/GPU run fits memory, use:

```bash
BATCH_SIZE=4 ACCUM_STEPS=1 bash scripts/launch_2x3080ti.sh
```

Effective batch size is eight in either case, but object-count normalization
and BatchNorm statistics are per microbatch/rank, so these recipes are not
identical and must not be switched on resume. Repeating more frames also
lengthens an epoch.

Both combined trainers print the fitted training-only priors and class counts.
The NPZ trainer fits from box arrays without loading camera images for fitting;
its `--data` directory must contain training samples only. Unseen training
classes retain explicit fallback priors and appear with count zero; no held-out
labels are used to fill them. Checkpoint config stores the priors, counts and
all task/decode settings. Fresh-run defaults use fallback ones only until this
fitting step; custom training code must call `fit_training_config` too.

The log now separates heading-bin, center-classification and size-regression
losses. Prediction/GT counts per class are now printed and added to CSV so
rare-class flooding/car disappearance is visible before mAP improves.
Validation JSON/CSV includes matched heading count, directed AOE,
axis AOE, front/back error rate (>90 degrees) and fraction within 15 degrees.
These matched-object diagnostics must be read alongside AP/recall: a model
that misses difficult objects can look artificially good on its few matches.
The center-distance mAP itself does not directly measure heading correctness.

## Runtime and compatibility

```python
heatmap_logits, regression, quality_logits, heading_logits = model(images)
# [B,6,384,384], [B,8,384,384], [B,1,384,384], [B,4,384,384]
```

Regression output is offset row/column, center Z, three class-relative log-size
residuals, and local sin/cos heading residual. For compatibility, `make_targets`
still stores global GT sin/cos; `detection_loss` converts it to the selected
bin's local target internally. All trainers request the exact regression-owner
class map with `return_class_ids=True` for colliding centers. Public decoded
boxes remain `[N,9]` with global BASELINK yaw and metric dimensions.

Use `decode_sizes` and `decode_heading` for custom probes instead of directly
exponentiating the size outputs or applying atan2 without the heading bin.
Export metadata records all these rules. Bin argmax/softmax, atan2, prior
lookup, scoring and NMS stay in host/C7x postprocessing, not the learned ONNX
graphs. Existing checkpoints loaded through `ModelConfig.from_checkpoint`
retain their old three outputs, absolute sizes, global yaw and score rules.

At full resolution, the new model has 1,988,690 parameters and 14.3977
convolution GMAC/four-camera frame versus 1,987,118 / 14.1854 previously.
The BEV handoff stays 40 channels. Four heading-logit channels add 0.5625 MiB
of detector output per frame at INT8. These inventories are not peak memory or
TDA4VH latency; matching SDK compilation, INT8 accuracy and board tests remain.

## Verification and limits

Focused tests cover reversed-heading gradients, angular wrap/bin boundaries,
Truck/Bus size round trips, global focal normalization, a dense-map regression
that reproduces the failed v3 car-bias gradient, checkpoint/resume compatibility,
and sparse classification training through decode/AP. The latter uses known
class features and oracle geometry; it is not a camera-network accuracy test.
Historical macro-loss/object-frequency behavior remains tested separately.
All 104 focused tests (including ONNX parity) and the two-rank CPU DDP
Truck/Bus/empty-rank smoke test passed after this correction. Full-size fresh
MobileNet optimizer steps on real Truck/Bus camera samples had finite gradients
for every parameter. On the same four-frame failed-checkpoint replay, the
aggregate car-bias gradient changed from +0.200 (downward bias update) to -0.453
(upward update). These in-memory checks wrote no checkpoints. Full held-out
validation after retraining is still required: they do not establish an AP
gain or guarantee convergence on the camera dataset.

For this prepared TRAIN split, fitted L/W/H medians are approximately
Truck `5.683 x 2.186 x 2.181 m` and Bus `11.294 x 2.904 x 3.438 m`.
The stable seeded sampler returns to 3,853 entries (805 Truck and 617 Bus
object exposures), versus the failed v3 sampler's 5,940 entries (1,907 Truck
and 1,585 Bus exposures). These are repeat exposures, not independent labels.
Do not compare raw losses directly with
the previous recipe; compare held-out AP, heading/size diagnostics and recall.
