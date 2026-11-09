# Recipe 4.1: consistent BEV normalization and heavy vehicles

Keep ResNet-18, OpenCV fisheye calibration, CLIDAR labels converted once to
BASELINK, +/-48 m, 0.25 m cells and the axis/facing heading outputs.
These code changes are not a claim of improved validation AP or target throughput.

## Changes

- BEV BatchNorm always uses fixed running statistics, including in `train()`.
  Its affine weights/biases remain trainable. Image-encoder BN is unchanged.
  Fresh models start with identity statistics; recovery imports the checkpoint
  statistics and learns under the SAME normalization used for inference.
  This is not the failed post-hoc recalibration or `model.train()` at validation.
  Fixed BN folds into convolution; no GroupNorm/LayerNorm runtime operators.
- Object losses use a constant TRAIN-only mean object count per frame, times
  batch size. Accumulation no longer gives one sparse frame the same object
  weight as one crowded frame. Focal positives/background, box, quality,
  heading, teacher and track terms all use this normalization. No per-class
  macro averaging or stronger repeat sampler is introduced.
- The existing class heatmap gets an additional weight-0.2 hard-negative
  ranking term, margin 0.5 logit, only at unambiguous vehicle GT centers.
  Truck/bus compete with car/van and each other; car/van centers explicitly
  suppress false truck/bus logits. Normalize by the same object denominator,
  retain existing 4x heavy-positive weights, and never boost inference scores.
  Loss is logged as `train/val_heavy_classification_loss`.
- Shared size reference: fit ONE L/W/H median from range-filtered TRAIN labels.
  Predict `log(physical_size / shared_reference)` and decode without looking
  at predicted class. Misclassifying a bus cannot itself shrink its dimensions.
  `class_relative` remains supported strictly for historical checkpoints.

The learned output tensors and parameter count are unchanged. Ranking and
sampling are training-only; shared size decode is host postprocessing. Export
metadata explicitly records normalization and the new size convention. Old
exports/postprocessors that add a predicted-class prior must not be reused.

## Recover from v4 on the two-GPU server

From the RayBEV project directory, with the existing dataset and teacher JSON:

```bash
OUT=work_dirs/raybev_v4_1_r18_norm_heavy_48m_clidar_2x3080ti \
EPOCHS=12 LR=0.0001 \
bash scripts/launch_2x3080ti.sh \
  --init-from work_dirs/raybev_v4_r18_axis_48m_clidar_2x3080ti/last.pt \
  --warmup-steps 200
```

Set `ROOT` if the dataset path differs from the launcher's server default.
Keep the original v4 run untouched. This command creates a NEW optimizer,
schedule, epoch numbering and best-checkpoint tracking. It strictly checks
backbone, geometry, heading, classes, training counts and calibration; it does
not partially load mismatched architectures. It preserves the learned model
and fixed BEV statistics but ZERO-resets the last size projection: an old
class-relative residual cannot be migrated by adding one constant. The size
branch must relearn truck/bus dimensions, so inspect validation, not initial size.
Source hash, source epoch, config changes and reset keys are saved as `warm_start`.
The source checkpoint is never overwritten and no ROSA inference is launched.

The 12-epoch recovery schedule is a starting experiment, not an accuracy promise.
Check car recall at score 0.3, per-sequence heading/axis/flip rates, truck/bus AP
and size error over the first few epochs. Do not judge bus heading on a tiny
matched subset while AP stays zero. Keep v3.1 as the regression baseline.

To continue an interrupted NEW v4.1 run, use its `last.pt` with `--resume`, omit
`--init-from`, and retain the original total epochs/LR/warmup/batch recipe.
Old v4 `--resume` fails closed because its normalization and size semantics differ.
`--bev-normalization batchnorm` and `--heavy-classification-weight 0` are explicit
ablation controls, not recommended recovery defaults.
