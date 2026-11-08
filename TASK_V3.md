# MobileNet task recipe 3: heading, heavy-vehicle size and classification

This is the current training/head contract. The backbone/BEV architecture
remains v2: MobileNetV2-1.0, OpenCV fisheye, CLIDAR input labels converted once
to BASELINK, 800x640 images, +/-48 m, 0.25 m cells. All retained MobileNet
stages remain trainable at half the new layers' LR. No ResNet or transformer
is introduced. The older [V2.md](V2.md) is the geometry/history reference.

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
- Classification: positive focal and regression/quality/heading losses now
  average by the classes present in each microbatch, with difficulty weights
  `(1,2,4,4,4,4)` for car/pedestrian/cyclist/truck/van/bus. Adding many cars
  no longer dilutes one Bus center's foreground/box gradient. Dense background
  focal supervision is unchanged. A 0.5-weight auxiliary foreground class CE
  explicitly discourages Car from winning at a Truck/Bus GT center. Ambiguous
  different-class centers sharing one grid cell are excluded from this CE.
- Sampling: repeats use object frequency, not the old 5% frame-presence rule.
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

Do not `--resume` the old epoch-18/v2 checkpoint: heading and dimension outputs
have new meanings and the loss/sampling recipe changed. Defaults use ImageNet
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

Default output: `work_dirs/raybev_v3_tasks_48m_clidar_2x3080ti`.
The conservative batch defaults remain one image/GPU with four accumulation
steps. If the previous four-image/GPU run fits memory, use `BATCH_SIZE=4
ACCUM_STEPS=1`. Effective batch size is eight in either case, but class-loss
normalization is per microbatch/rank, so these recipes are not identical and
must not be switched on resume. Repeating more frames also lengthens an epoch.

Both combined trainers print the fitted training-only priors and class counts.
The NPZ trainer fits from box arrays without loading camera images for fitting;
its `--data` directory must contain training samples only. Unseen training
classes retain explicit fallback priors and appear with count zero; no held-out
labels are used to fill them. Checkpoint config stores the priors, counts and
all task/decode settings. Fresh-run defaults use fallback ones only until this
fitting step; custom training code must call `fit_training_config` too.

The log now separates heading-bin, center-classification and size-regression
losses. Validation JSON/CSV includes matched heading count, directed AOE,
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

99 focused tests cover reversed-heading gradients, angular wrap/bin boundaries, Truck/Bus
size round trips beyond their priors, class-balanced gradients under added cars,
foreground Car-versus-Bus competition, object-frequency repeats, single-label
decode, orientation diagnostics, old checkpoints and ONNX parity. A two-rank
CPU DDP backward with alternating empty ranks and real truck/bus-frame optimizer
steps passed. No full new-recipe training run has been performed yet: these
changes address measured failure mechanisms, not establish a validated AP gain.

For this prepared TRAIN split, fitted L/W/H medians are approximately
Truck `5.683 x 2.186 x 2.181 m` and Bus `11.294 x 2.904 x 3.438 m`.
The seeded sampler produces 5,940 entries instead of 3,853; repeated Truck
object exposure rises 805 -> 1,907, Bus 617 -> 1,585. These are repeat exposures,
not additional independent labels. Do not compare raw losses directly with
the previous recipe; compare held-out AP, heading/size diagnostics and recall.
