# Recipe 4.2: foreground classification and restored size priors

## What v4.1 actually achieved

Read all 12 validation epochs plus `last.pt`, `best_map.pt`, `best_heading.pt`.
The normalization fix worked, but the rare-class changes did not.

| Logged full-validation metric | v4 best | v4.1 best-map (epoch 9) |
|---|---:|---:|
| mAP | 2.43% | 4.36% |
| Car AP | 12.17% | 21.78% |
| Car axis error | 36.65 degrees | 17.51 degrees |
| Car recall, score >=0.3 | 0.83% | 18.58% |
| Truck / bus AP | 0 / 0 | 0 / 0 |

V3.1 MobileNet still has the better best mAP, 5.03%. V4.1 epoch 12 falls to
4.23%; training loss keeps dropping while validation loss rises. Do not just
extend its unchanged schedule. Best heading checkpoint is epoch 3: car AOE
18.50 degrees at score >=0.3, but only 13.22% recall. Best-map epoch 9 has
31.73-degree AOE at 18.58% recall. Both score-conditioned and all-match heading
metrics matter; bus AOE from 18 or 24 matched examples is not detection success.

Additional deterministic, stratified replay of `last.pt`: 60 VAL frames and
48 TRAIN frames, with extra rare-class frames. At GT centers, VAL bus median
predicted LWH is 4.86/1.95/1.77 m versus GT 10.56/2.87/3.57 m. Truck median
is 4.44/1.88/1.70 m versus GT 5.87/2.24/2.44 m. Pedestrian sizes also regress.
Shared size prediction has fallen back toward car dimensions: even TRAIN bus
median length is only 8.21 m versus GT 11.56 m. Its features are not entirely
dead; this is not fixed by a score threshold or another checkpoint conversion.

Foreground class accuracy in that replay is TRAIN bus 9/11 and truck 9/14,
but VAL bus 2/17 and truck 0/11. VAL median true-class probabilities are
0.00231 for buses and 0.00086 for trucks. TRAIN/VAL car heading AOE at GT centers
is 23.68/54.74 degrees. This is substantial generalization failure, not just
insufficient score threshold. Calibration and teacher hashes match; frozen
BEV BN counters remain unchanged at 38520 as intended. Tensors are finite.

## Targeted changes (not a validation-accuracy promise)

1. Retain fixed-statistics BEV BatchNorm, trainable affines, ResNet-18,
   +/-48 m, 0.25 m cells, 800x640 OpenCV cameras, CLIDAR input conversion,
   axis/facing heading, and the existing offline TRAIN-only ROSA/track losses.
2. Separate class-agnostic object-center detection from foreground semantics.
   One new one-channel convolutional objectness head gets Gaussian focal
   supervision for ALL object centers. Six existing class logits now get
   weighted CE ONLY at unambiguous GT centers (weight 1, smoothing 0.05).
   Background no longer trains rare semantic classes down across the grid.
   Existing heavy positive weights remain 4x, the fixed TRAIN-mean denominator
   remains, and no class-macro loss/object-frequency oversampling is added.
3. Disable the extra heavy-vs-car ranking penalty. Foreground softmax already
   provides class competition, with light vehicles as legitimate negatives.
4. Restore TRAIN median class size anchors. The v4.1 shared-anchor experiment
   removed useful dimension priors and regressed buses/pedestrians. Reset only
   the last size projection; a zero residual starts with the correct class's
   prior, not a car-sized bus. This deliberately restores class dependence:
   WRONG semantic classes can still give WRONG dimensions. The foreground
   classifier targets that remaining problem; we do not claim it is solved
   before measuring retraining.
5. Stronger TRAIN-only brightness, white balance, contrast and gamma variation
   addresses the observed appearance overfit without cropping/flipping pixels
   or changing geometry, labels, validation images, or the teacher cache.

The new objectness head adds only 1473 parameters at head width 32. It is
ordinary separable/pointwise convolution with fixed BN; class softmax stays
on the host, outside the learned ONNX graph. This is NOT a target-hardware
latency claim. New raw outputs are:

```
class_logits [B,6,H,W], regression [B,8,H,W], quality_logits [B,1,H,W],
facing_vector [B,2,H,W], objectness_logits [B,1,H,W]
```

Host score is `sigmoid(objectness) * softmax(class)[class] * sqrt(sigmoid(quality))`.
Use updated decoding/export metadata; do not reuse old four-output consumers.
Difficulty weights do not multiply inference scores. Public decoded boxes
remain `[x,y,z,l,w,h,yaw,class,score]`. Historical checkpoints keep their OLD
head, score, size, normalization and loss semantics through `from_checkpoint`.

## Retrain from v4.1's best detection checkpoint

Copy the updated `raybev/` and `scripts/` to the server; the existing teacher
JSON is unchanged and no ROSA inference is needed. From the project directory:

```bash
OUT=work_dirs/raybev_v4_2_r18_fg_48m_clidar_2x3080ti \
EPOCHS=12 LR=0.0001 \
bash scripts/launch_2x3080ti.sh \
  --init-from work_dirs/raybev_v4_1_r18_norm_heavy_48m_clidar_2x3080ti/best_map.pt \
  --warmup-steps 200
```

Set `ROOT` if your dataset is elsewhere. Preserve previous run directories.
Migration is strict: objectness features and its initial logits are copied
from the learned Car heatmap; remaining compatible weights/BEV statistics are
retained, size output projection is zero-reset, and optimizer/schedule/epoch/
best tracking start NEW. No silent partial load and no old optimizer resume.
New-run `--resume` is only for v4.2 with its ORIGINAL schedule and recipe.

Treat this as a controlled recovery experiment. Inspect epochs 2-4: objectness
and foreground CE separately, class AP/ASE/recall, and heading diagnostics by
score and sequence. Compare BOTH 0.1/0.3 operating points and ranked AP because
score semantics changed. A finite training loss or restored prior is not a
claim of improved held-out detection. If rare-class AP remains zero, audit
image/GT visibility and class-center confusion before extending training or
stacking more class weights. Do not launch another unchanged 24-epoch run.
