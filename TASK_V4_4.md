# V4.3 assessment, reviewed heavy labels, and next controlled experiment

V4.3 improves cars and confident heading, but does not solve heavy vehicles.
Do not continue the epoch-12 checkpoint unchanged or interpret a low heading
error on a small selected subset as ROSA-level detection.

## Actual run assessment

Run: `raybev_v4_3_r18_geometry_48m_clidar_2x3080ti`. Twelve epochs; best mAP
checkpoint is epoch 6, best guarded car-heading checkpoint epoch 3. All three
checkpoints inspected have finite parameters, correct OpenCV/CLIDAR contracts,
zero-origin stride-8 geometry, separate heights, and unit-vector facing loss.
Recovery used v4.2 epoch 9 with only the intended three configuration changes.

| Logged result, original GT | V4.2 best epoch 9 | V4.3 best epoch 6 |
|---|---:|---:|
| Internal mAP | 4.720% | 5.311% |
| Car AP | 23.408% | 26.231% |
| Pedestrian AP | 0.194% | 0.324% |
| Truck / bus AP | 0 / 0 | 0 / 0 |
| Car AOE, floor .01 | 55.49 deg | 48.40 deg |
| Car AOE, score >= .3 | 26.70 deg | 17.09 deg |
| Car recall, score >= .3, 2 m | 19.02% | 19.37% |

The mAP gain is ~12.5% relative but only 0.59 percentage points. This now exceeds
the historical v3.1 best 5.03%, yet remains far below ROSA. Epoch 3 has the best
car AOE at score .3 (15.42 deg) with recall 18.01%, NOT the best general detector.
Training loss decreases 4.59 -> 2.46; validation reaches its minimum at epoch 2
(7.10) then increases to 7.66. Epoch-12 mAP is 5.25%. Backbone stages were ALL
trainable in this run. Actual microbatch was 4/GPU, accumulation 1; preserve
that when comparing experiments because image BatchNorm is still live.

Oracle-center diagnostics from every logged frame expose the remaining gap:

| Epoch-6 center diagnostic | TRAIN | VAL, original GT |
|---|---:|---:|
| Truck class accuracy | 93.91% | 0/134 |
| Bus class accuracy | 92.87% | 12/206 |
| Mean truck objectness | .392 | .039 |
| Mean bus objectness | .242 | .028 |
| Car AOE | 17.60 deg | 55.88 deg |

By epoch 12 TRAIN truck/bus accuracy reaches 98.39%/97.57%, while VAL remains
1/134 and 4/206. This is severe generalization failure, not a zero-output numeric
collapse. Bus/truck matched counts at epoch 6 are only 19/5 at floor .01 and 2 m;
neither class has a correct detection at score .3. Small offset errors (~.1 m)
at known centers do not imply that proposals, appearance classes or facing work.

## The tram and real heavy vehicles

All 32 validation truck/bus tracks were inventoried and one best projected crop
was inspected per track. Suspicious examples were checked across cameras/frames
and against available native-resolution images. A crop does not by itself prove
that a partially occluded or badly projected object is absent.

The visibly confirmed `17_09` tram tracks **910, 914 and 980 were already Bus**.
They remain Bus. `names_to_ids` now explicitly accepts `Tram`/`tram` -> Bus for
future inputs; no currently trained labels were secretly remapped by this alias.
Real examples include delivery trucks 17_09/897 and 990, trucks 30_09/2630 and
2744, and passenger buses 16_09/2122 and 17_09/953 and 984.

The user approved the following class-only corrections after visual inspection:

| Sequence / track | Original | Reviewed | Raw boxes | In +/-48 m |
|---|---|---|---:|---:|
| 17_09 / 932 | Bus | Truck: box-bodied delivery vehicle | 10 | 10 |
| 30_09 / 2722 | Bus | Truck: cargo/refuse vehicle | 43 | 39 |
| 30_09 / 2724 | Bus | Truck: green cargo/refuse vehicle | 5 | 5 |

Thus 58 raw validation annotations change consistently across entire tracks;
54 participate in the current metric, 26.2% of the previous in-range Bus labels.
Bus count becomes 206 -> 152, Truck 134 -> 188. No geometry, dimensions, heading,
track IDs, images, sample tokens, split assignment or original PKLs were changed.
TRAIN was copied byte-for-byte, so its priors and existing TRAIN-only teacher
cache remain valid. **Correcting VAL does not improve training supervision.**

Portable revision files: `datasets/heavy_review_v1/{train.pkl,val.pkl,rules.json,manifest.json}`.
The manifest records approval, every changed token/box, input/output hashes and
counts. Semantic source hashes allow equivalent GT after pickle/path relocation,
not changed classes, boxes, IDs or token sets. Original labels remain available.

Evidence: `analysis/v4_3_heavy_labels/contact_sheet_0.jpg`, `contact_sheet_1.jpg`,
`native_17_09_932.jpg`, `native_30_09_2722.jpg`, `native_30_09_2724.jpg` and
`multiview_*.jpg`. `tracks.json` provides exact image/token/track associations.
Source annotations match the suspicious labels (box differences <2e-6); this is
not a RayBEV class-ID swap. Some source records include `camera_track_hold` and
constant-velocity predictions. For example 17_09/1003 projects onto a facade in
several inspected frames. It is flagged for LiDAR/manual review, NOT deleted.
17_09/1008 also needs review; poor visibility alone is insufficient to remove GT.

## Full validation control: labels vs learned improvements

The actual epoch-6 checkpoint was rerun over all 674 frames on the local GPU,
using FP16 image/BEV inference and FP32 decoding. Each prediction is reused
unchanged for both GT versions. Native ROSA cached predictions were independently
validated and scored on both versions; no ROSA inference or prediction relabeling.

| Same predictions / evaluator | Original GT | Reviewed GT v1 |
|---|---:|---:|
| RayBEV internal mAP | 5.310% | 5.310% |
| RayBEV truck / bus AP | 0 / 0 | 0 / 0 |
| ROSA internal mAP | 16.921% | 16.537% |
| ROSA truck AP | .281% | .204% |
| ROSA bus AP | 4.419% | 2.576% |

The small RayBEV difference from the distributed training log is a rerun
precision/batch/kernel effect, not new training. These are diagnostic center-
distance metrics, NOT official ROSA FDS/NDS. Native NMS differs (.25 vs .5).
ROSA itself agrees with some wrong Bus annotations: correcting GT can lower its
score. Never present annotation-induced changes as a learned-model gain.

On reviewed VAL, RayBEV's oracle-center class accuracy is 0/188 trucks and
10/152 buses. The problem persists on actual passenger vehicles. Mean decoded
bus length/GT is .495 with predicted class versus 1.153 with GT class. The shared
class-relative residual head mostly follows its priors; classification and
generalization still matter, and long trucks are especially sensitive to the
TRAIN taxonomy. ROSA is not a clean rare-class teacher simply because its total
AP is higher. Existing yaw-only distillation does not teach class appearance.

Artifacts: `analysis/v4_3_review_control/evaluation.json` (both metrics/heading
and class diagnostics), `predictions.json` (checkpoint-bound original predictions),
and `rosa_provenance/manifest.json`.

## Next fixes and experiment

Confirmed changes made now:

- Explicit Tram -> Bus taxonomy guard; approved cargo labels fixed in a fork.
- Annotation hashes stored in new training histories/checkpoints. Resume rejects
  changed annotation contents or a changed backbone-freeze policy.
- Optional early-stage freezing correctly freezes BOTH gradients and BN buffers.
  Frozen parameters are excluded from AdamW. The policy is reapplied after each
  `model.train()`; setting `requires_grad=False` alone would be incomplete.

The next controlled NN experiment is `--freeze-image-stages early`: freeze
ResNet stem/layers 1-3, train layer 4 plus ray/FPN adapters and all BEV/task heads.
This preserves the source checkpoint's already-finetuned early features; it does
NOT restore ImageNet weights. The hypothesis is reduced feature overfitting,
not a confirmed diagnosis or promised accuracy fix. `none` is the matched
all-trainable control. Defaults outside this explicit experiment remain unchanged.

No extra attention head, transformer, memory bank, input resolution or inference
operation is added. TDA4VH inference compute/memory remains v4.3's ~77.81 GMAC
and 19.125 MiB INT8 BEV handoff. Device compilation, quantized accuracy and
latency still require actual measurements.

Read-only sensitivity probes on 29 VAL / 32 TRAIN frames found no normalization
switch that rescued truck classification. Zeroing BEV geometry did not rescue
it either; zeroing visual features destroyed proposals/heading. Per-frame BEV
batch statistics improved a few bus classifications but reduced sampled car
classification from 99.0% to 71.4%. Do not reintroduce live BEV BN as a fix.

Before spending another long run, review TRAIN annotations too. Candidate sheets
were generated in `analysis/v4_3_train_label_candidates` for 159 raw truck/bus
tracks (154 have eligible in-range crops). Only candidates, NOT fully reviewed
or automatically corrected labels. Initial sheets contain suspicious classes and
occlusions; validation corrections alone cannot resolve these. Future TRAIN
changes require refitting priors and regenerating the TRAIN teacher cache from
the stored raw TRAIN predictions, not bypassing the semantic-hash checks.

## Run the paired short experiment on the server

Copy changed source, `datasets/heavy_review_v1`, existing v4.3 epoch-6
`best_map.pt`, and the unchanged teacher JSON. Activate the normal training env.
The root images still come from the original dataset; no image duplication.

From `projects/raybev_lite` (server project may be named `raybev`):

```bash
# Main experimental arm, six epochs, early backbone stages frozen:
bash scripts/launch_v4_4_review_experiment.sh early

# Run separately as the matched all-trainable control:
bash scripts/launch_v4_4_review_experiment.sh none
```

Use ROOT=/your/dataset/path if different from the previous server path. Separate
outputs are `raybev_v4_4_r18_early_reviewv1_48m_clidar_2x3080ti` and
`raybev_v4_4_r18_none_reviewv1_48m_clidar_2x3080ti`. Both initialize from v4.3
best epoch 6 with a fresh optimizer/schedule, LR=.0001, batch=4/GPU, accumulation=1,
warmup=200, same augmentation/class weights and teacher losses. Do not run these
two 2-GPU jobs simultaneously on the same GPUs, and do NOT resume v4.3 into them.
The unfrozen arm controls for restarting the schedule; comparison with the old
v4.3 curve alone would confound that change with freezing.

At epochs 2-3 compare both arms on the SAME reviewed GT: actual truck/bus AP and
recall, oracle-center class/objectness, predicted-class size error, heading flips
at comparable recall, and per-bag results. No long continuation merely because
TRAIN loss falls. If freezing fails, the next representation experiment is
object-conditioned multi-point image evidence, not another class-loss multiplier.

Reproduce/verify review files:

```bash
python scripts/prepare_label_revision.py --root /your/dataset/path \
  --output datasets/heavy_review_v1 --verify

# A new TRAIN/VAL review pass; never overwrites existing audit folders:
python scripts/audit_heavy_labels.py --root /your/dataset/path \
  --split train --output analysis/train_review_new
```

Local verification: 156 focused tests passed; both backbone types' frozen stages
retain identical weights and BN buffers after backward/optimizer steps; real
two-rank CPU DDP with frozen stages and an empty rank passed; four actual full-
resolution steps from v4.3 epoch 6 with the existing teacher/tracks passed. No
full v4.4 accuracy training or board benchmark has been launched.
