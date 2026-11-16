# V8: heading, heavy-class discrimination, and YOLO26m

Prepared 2026-11-16. Training has not been launched. Changes are experimental;
passing gradient/contract checks establishes implementation correctness, not
improved held-out accuracy.

## Baseline and diagnosis

`work_dirs/raybev_combined_10/best_map.pt` is epoch 57, mAP 19.317%. Latest
epoch 96 has mAP 18.636%. The next run starts with a fresh optimizer/decoder
and COCO-pretrained image backbone. Do not resume a nano checkpoint as medium.

At epoch 96, score >=0.1, class-aware center distance <2m:

| Class | AP | Precision | Recall | Directed AOE at score >=0.1 |
|---|---:|---:|---:|---:|
| Car | 54.63% | 23.19% | 86.22% | 25.70 deg |
| Pedestrian | 19.24% | 12.59% | 74.56% | 79.95 deg |
| Truck | 1.79% | 5.11% | 29.66% | 22.94 deg |
| Van | 36.16% | 26.36% | 55.98% | 6.23 deg |
| Bus | 0% | 0% | 0% | unavailable |

Latest **final-stage** raw class/score-agnostic 2m query coverage is Truck
91.53% and Bus 73.91%. The gap to class-correct recall motivates classification
and ranking changes. Raw 0.5m Truck coverage is only 13.56%, so regression is
also a limitation. Earlier stage-1 coverage should not be called final-stage
recall. No raw-query coverage value is detector recall.

The old sigmoid focal objective already suppresses wrong-class logits, but has
no explicit matched-object softmax competition. The quality head was supervised
on matches only, although every detection score is multiplied by predicted
quality. This motivates a background-quality experiment; it is not proof that
quality alone caused the false positives.

## Implemented changes

All new objectives default to the old behavior when loading existing configs.
Every flag is stored in `cfg` and protected by resume equality checks.

1. `--backbone yolo26m` selects the medium architecture. Feature widths at
   /4,/8,/16,/32 are 256/512/512/512 rather than nano's 64/128/128/256. FPN
   adapters and metadata follow the selected variant. Retain official layers
   0..9 through SPPF, excluding C2PSA and the original YOLO heads. All retained
   pretrained tensors load strictly. RayBEV inference reconstructs the selected
   architecture offline from its checkpoint without the original COCO file.
2. `--matched-class-weight 1`: auxiliary softmax CE on Hungarian-matched
   foreground logits, including DN positives. This directly penalizes a Car
   logit outranking Truck/Bus. Class weights use the square root of existing
   positive weights; the sampler and ordinary focal weights are unchanged.
   Do not expect increased rare-class weights alone to solve false positives.
3. `--quality-background-weight .25`: zero-quality BCE on unmatched background
   queries. Ignore-region queries and queries within 2m of a known object are
   excluded to avoid contradicting ignored objects or accurate duplicates.
   Existing positive localization-quality targets are unchanged.
4. `--heading-hard-weight 2`: detached error-dependent weights between 1 and 3
   focus axis regression on axis errors and circular/bin-hemisphere supervision
   on incorrect front/back decisions. Gradients cannot game the weighting.
5. `--facing-binary-weight 1` increases explicit front/rear loss from 0.5.
   `--facing-label-smoothing .05` mildly smooths the circular and binary targets
   to reduce excessive certainty. This is not a substitute for correct labels.
6. `history.jsonl` now includes `heavy_confusion`: class-agnostic, score-greedy
   XY<2m matches for Truck/Van/Bus, missed GT, and predicted/GT LWH ratios for
   each class transition. This is diagnostic; the AP evaluator is unchanged.
   New `matched_class` and `quality_background` losses are logged separately.

The decoder width C96, Q225 queries, four stages, native image size, proposals,
sampling, extent targets and score formula stay consistent across new arms.
Medium is supported by the installed Ultralytics 8.4.182 implementation and its
[official YOLO26 configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26.yaml).

Local medium source: `../../checkpoints/yolo/yolo26m.pt`, SHA256
`401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`.
Its architecture and every retained feature tap were verified against that
local COCO checkpoint; provenance is recorded as trusted local weights.

## Launch

From `projects/raybev_lite`, in the environment with requirements installed:

```bash
conda activate openmmlab
EXPERIMENT=medium_fixes bash scripts/launch_v8_heading_heavy.sh
```

Defaults: two GPUs, four four-camera samples per GPU, 2 accumulation steps,
effective global batch 16, 64 epochs, LR 2e-4, backbone LR multiplier .1,
seed 3407. Output: `work_dirs/raybev_v8_medium_fixes`.
Batch sizes 8 and 6 were reported to exceed GPU memory; batch 4 is the requested next setting.
This reduces the effective batch from 32 to 16 relative to batch 8/accumulation 2.
BatchNorm also sees four samples per GPU instead of eight. Use the same
microbatch and effective global batch across all new controls. LR stays 2e-4.

```bash
# Show the exact command without training:
DRY_RUN=1 EXPERIMENT=medium_fixes bash scripts/launch_v8_heading_heavy.sh

# Change GPU count; accumulation automatically preserves global batch 16:
NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 EXPERIMENT=medium_fixes \
  bash scripts/launch_v8_heading_heavy.sh

# Resume this same experiment (keep its original settings):
EXPERIMENT=medium_fixes bash scripts/launch_v8_heading_heavy.sh \
  --resume work_dirs/raybev_v8_medium_fixes/last.pt
```

## Controlled comparison

Run each arm separately; the launcher never queues training implicitly:

| EXPERIMENT | Backbone | New objectives |
|---|---|---|
| nano_control | YOLO26n | disabled |
| nano_fixes | YOLO26n | enabled |
| medium_control | YOLO26m | disabled |
| medium_fixes | YOLO26m | enabled |

Medium-fixes versus medium-control tests the objective bundle. Nano-fixes
versus medium-fixes tests the backbone. For finer ablations, run a new `OUT`
with `MATCHED_CLASS_WEIGHT=0`, `QUALITY_BACKGROUND_WEIGHT=0`, or disable heading
changes with `HEADING_HARD_WEIGHT=0 FACING_BINARY_WEIGHT=.5 FACING_LABEL_SMOOTHING=0`.
Controls are fresh runs, not warmed from epoch 57. A different backbone cannot
strictly resume nano weights. Keep validation hashes, microbatch, global batch,
sampling, image size, and total schedule identical across comparisons.

At epochs 24/48/64 compare best mAP and class AP, Truck/Bus->Car confusion,
dimension ratios with counts, precision/recall at scores .1 and .3, final-stage
0.5/1/2m coverage, and class-specific heading AOE/flip rate at matched recall.
Preserve separate axis AOE and directed AOE: pedestrian axis error near 43 deg
shows its problem is broader than a 180-degree ambiguity. Avoid interpreting
aggregate AOE shifts caused by missing classes as heading improvement.
The existing `best_heading.pt` selector still uses Car AOE at score .3; it is
not an all-class heading checkpoint. Inspect epoch histories for heavy classes.

## Data review before interpreting improvements

Reproduce the read-only audit with a new output filename:

```bash
python scripts/audit_heading_heavy_data.py \
  --root ../../data/Custom/combined_bags_camera_ray_dedup_v2 \
  --output analysis/v8_heading_heavy_data_audit.json
```

Prepared audit: `analysis/v8_heading_heavy_data_audit.json`. In-range TRAIN
counts: Car 102012, Pedestrian 3567, Cyclist 122, Truck 2178, Van 1465, Bus 522.
Distinct TRAIN tracks: 5027/487/61/270/103/81 respectively. VAL has only
18 Truck tracks, 3 Bus tracks, and 1 Cyclist track. Bus/Truck VAL tracks are
TRAIN-unseen by `(sequence_id, gt_id)`, whereas 138/234 Van VAL boxes and
4842/8583 Car VAL boxes share a TRAIN-seen track. Repeating frames cannot add
independent heavy-object diversity.

No track changes class within either split. This does **not** validate a class
label that is consistently wrong for the entire track. Review Truck/Bus->Car
visual examples and the weak YOLO extent targets. The audit lists 18 adjacent
TRAIN pairs with world-frame yaw changes >160 degrees within 2 seconds
(13 Car, 5 Truck); verify those image/point-cloud pairs before editing labels.
World orientation removes ego-yaw changes, but a flagged pair can still be an
ID switch or actual maneuver. No labels were changed automatically. The audit
can detect temporal contradictions, not validate pedestrian body heading.

The existing split stays fixed for comparison. A larger held-out set of distinct
heavy tracks/bags and reviewed pedestrian heading annotations is needed before
claiming generalization. Medium provides more capacity, not new label evidence.

## Verification

- 86 focused tests passed, 1 existing test skipped: nano compatibility, medium
  feature equality against the local COCO model, class-competition gradients,
  ignored/duplicate quality masking, corrective gradients on reversed heading,
  nano/medium finite backward, strict checkpoint reload, and old trainer tests.
- A real four-camera 800x640 sample passed medium Q225/C96/L4 forward and all
  active objective calculations. Model: 10,248,439 parameters, finite boxes.
- All four launch variants passed dry-run argument checks and shell syntax.
- The dataset audit completed on the actual combined TRAIN/VAL annotations.
- The existing TRAIN extent cache passed its annotation/calibration contract.
- No full training, CUDA memory/AMP smoke, or accuracy improvement is claimed.
