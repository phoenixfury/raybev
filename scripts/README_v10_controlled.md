# V10: frozen manual-v5 data, fresh medium, one-factor experiments

Prepared, not trained. This is an experiment matrix, not a claim that any change
already improves accuracy. Do not combine all arms before their individual results.

## Annotation audit and remaining review

`analysis/v10_manual_review_v5/annotation_audit.json` independently checks every
source annotation against the final V5 rows and saved decisions, including exact
source-index removal, untouched object fields, metadata, all 5,201 YOLO records,
their pixel evidence, and final TRAIN fingerprint. This is exhaustive consistency
checking, **not new visual certification** of every user classification/cuboid.

All 4,494 decisions applied: 4,409 retained/reviewed, 85 removed (71 TRAIN, 14 VAL),
444 retained class changes. Frames/images remain intact. Five linked YOLO GT
classes changed; none of the removed annotations had linked weak extent records.

Two remaining mixed-class TRAIN tracks in bag `08_27_10_12_18`:

| Track ID | Retained classes | Action |
| --- | --- | --- |
| 5051 | Bus 33, Truck 2 | Check temporal identity/class; no automatic majority vote. |
| 5053 | Van 2, Truck 2 | Check identity switch versus inconsistent labels. |

There are no simultaneous duplicate instances of these IDs within a frame, but
that alone does not rule out temporal ID switches. Cross-class geometry checks
found **89 flagged cuboids**:

- 49 Cars with unusually large dimensions: 23 manually class-reviewed and 26
  other/original Cars.
- 35 Vans with heavy-vehicle-sized dimensions (including two `5053` frames above).
- Five newly Pedestrian-labeled boxes on `11_09_08_01_03/211`, all still
  **11.81 × 3.05 × 3.37 m**. Their projected region includes a pedestrian and a
  passing foreground van; the contact sheet does not certify the cuboid's physical
  association. The vehicle-sized geometry is not credible pedestrian supervision.

These flags are not automatic class corrections. Review temporal/LiDAR association
and geometry; do not infer the target from the foreground occluder, automatically
resize boxes, or change VAL labels to fit model predictions.
`analysis/v10_manual_review_v5/pedestrian_211_preview.jpg` and
`van_4946_preview.jpg` give four-camera temporal examples. The latter shows a
visually plausible delivery Van with retained 16 m cuboids: check depth/center
and temporal association too, not just the class or a guessed replacement size.

`followup_review.jsonl` addresses these 126 distinct objects in **final V5 indices**, correctly
accounting for all earlier removals. To inspect them first, from the project:

```bash
python scripts/review_heavy_annotations.py \
  --root ../../data/Custom/combined_bags_camera_ray_dedup_v2 \
  --train-ann datasets/combined_bags_camera_ray_manual_review_v5/fisheye3dod_infos_train.pkl \
  --val-ann datasets/combined_bags_camera_ray_manual_review_v5/fisheye3dod_infos_val.pkl \
  --extents datasets/combined_bags_camera_ray_manual_review_v5/train_yolo_extents_v1.json \
  --classes car truck van bus pedestrian cyclist \
  --review-ledger analysis/v10_manual_review_v5/followup_review.jsonl \
  --state work_dirs/manual_v5_followup/decisions.json \
  --output datasets/combined_bags_camera_ray_manual_review_v6
```

Use the `unknown` status filter to restrict the view to the flags. Original V5 and
its completed decisions remain intact. Review classification separately from
geometry. A class confirmation does not mark a box as credible endpoint GT.

If creating V6, audit/prepare it with `--release ...v6 --source ...v5 --output ...v6`;
set matching `REVIEW_DIR` and `AUDIT_DIR` when launching. Every comparison must
use the same frozen annotation release, including the control.

## Common control

- Fresh COCO YOLO26-medium backbone, random RayBEV heads, no V8/V9 warm start,
  no ROSA teacher. Seed 3407, 64 epochs, LR 0.0002, backbone LR multiplier 0.1.
- Batch 4 per GPU, accumulation 2, 2 GPUs: effective batch 16.
- Q225, C96, four stages, 64 proposals, stride-4 FPN, existing observation cues,
  verified TRAIN extents. Same track sampler/augmentation in every arm.
- Unit class weights for focal/regression/DN; no blanket rare-class multiplier,
  heavy logit boost, extra hard-heading weight, final-layer resampling patch,
  or quality-negative change. This is a **new baseline**, not a controlled direct
  comparison against the old V9 recipe or V4 annotation metrics.
- Correct per-cell proposal supervision common to all arms. Matching schedule
  stays unchanged except the dedicated `matching` arm.

## Arms and valid comparisons

| `V10_ARM` | Change | Compare against |
| --- | --- | --- |
| `control` | Common recipe only | Fresh baseline |
| `shapes` | Shape-balanced learned-anchor and proposal size hypotheses | `control` |
| `anchors` | Learned-anchor sizes only | `control` |
| `proposals` | Proposal size priors only | `control` |
| `foreground_ce` | Unweighted matched foreground CE | `control` |
| `foreground_balanced` | Capped object-level weighting of matched foreground CE | `foreground_ce` |
| `objectness` | Binary objectness + matched-only softmax semantics | `control` |
| `objectness_balanced` | Foreground object balancing in split objective | `objectness` |
| `context` | 18 additional object-aligned, multi-height/context points in every stage | `control` |
| `early` | Earlier-stage loss weight 2 versus final-stage 1 | `control` |
| `matching` | Class-cost warmup/ramp only | `control` |
| `endpoint` | Weight .25 on explicitly geometry-approved heavy TRAIN cuboids | `control` |

The `shapes` bank uses TRAIN geometry, not predicted/GT semantic class for size
selection: length bins <2.5, 2.5–5.5, 5.5–8.5, 8.5–14, >=14 m, with budgets
8/60/12/12/8%. Shared spatial fitting remains the existing TRAIN-only fit in every
arm; **we do not call the spatial fitting itself class-independent**. Keeping XY
fixed isolates size hypotheses and preserves the same raw XY Car coverage.
Z follows each new height/ground hypothesis; no semantic size lookup exists.

For V5: learned slots have geometry-bin counts `[13,97,19,19,13]` versus control
`[8,144,5,4,0]`; proposal size slots `[5,38,8,8,5]` versus `[4,53,4,3,0]`.
Static learned-anchor VAL Car XY coverage is unchanged: 5,603/8,637 at <2 m
(64.87%). This does not measure image-scored proposals or trained performance.

Foreground balancing uses capped square-root inverse TRAIN object counts, then
normalizes within matched objects. It never weights background by rare class.
The split-objective arm trains background only through objectness, with separate
positive/negative means; semantics uses matched foreground objects. Its decoder
uses softmax semantic probability × objectness × quality, not raised heavy logits.
Its unchanged focal-form assignment cost uses semantic × objectness probability,
not sigmoid applied to conditional-softmax logits. The explicit matching-cost
schedule is still changed only in the `matching` arm. Objectness and semantic
component losses are logged separately.

Endpoint is deliberately blocked: there are **zero explicitly geometry-approved
boxes**. `geometry_approvals_TEMPLATE.json` binds approved normalized TRAIN boxes
to the TRAIN fingerprint and calibration. Add only independently geometry-reviewed
heavy boxes with actual evidence; an empty template is rejected. No class-review
decision automatically becomes a geometry approval. Directed tram headings need
their own temporal/LiDAR/image review; do not globally flip Bus headings.

## Prepare and launch

Assets are already generated locally. Re-prepare on a server with the same files:

```bash
python scripts/prepare_v10_experiment.py --root /path/to/dataset
```

The launcher checks the frozen file hashes and refuses the remaining mixed-class
IDs and class/dimension flags unless resolved in a new release, or explicitly
acknowledged using `ALLOW_TRACK_CONFLICTS=1` and `ALLOW_GEOMETRY_FLAGS=1`.
Neither override corrects annotations or enables unapproved endpoint supervision.
Preview commands before starting:

```bash
ALLOW_TRACK_CONFLICTS=1 ALLOW_GEOMETRY_FLAGS=1 DRY_RUN=1 V10_ARM=control bash scripts/launch_v10_controlled.sh
ALLOW_TRACK_CONFLICTS=1 ALLOW_GEOMETRY_FLAGS=1 DRY_RUN=1 V10_ARM=shapes bash scripts/launch_v10_controlled.sh
```

After resolving/acknowledging the flags, remove `DRY_RUN=1` to run a chosen arm.
Run control then shapes first, with separate fresh output directories. Later
compare the foreground pair, then objectness pair, context, early supervision,
matching, and only then approved geometry. No `--resume`/`--init-from` is accepted
by this controlled launcher. Exact resumes can use the printed trainer command,
preserving its recipe/data/approval hashes, rather than starting another arm.

Override `ROOT`, `YOLO26_WEIGHTS`, `OUT`, `NPROC_PER_NODE` for server paths/hardware.
Use a fresh `OUT` for each seed/arm. GPU memory has not been validated here. Context
increases evidence sampling from 13 to 31 points: if batch 4 OOMs, use batch 2 /
accumulation 4 **for both context and its paired control**, not only context, to
hold microbatch/BatchNorm conditions fixed. Do not claim AMP/VRAM performance
from the small CPU tests.

## What to read after each epoch

`history.jsonl` now includes:

- `raw_query_coverage`: complete VAL raw XY coverage for initial queries and every
  decoder stage; all classes/ranges, one-to-one, score/class agnostic.
- `train_raw_query_coverage_sampled`: TRAIN raw coverage every 50 local batches;
  this is sampled augmented training exposure, not full-TRAIN population recall.
- `stage_supervision`: full TRAIN/VAL per-stage/per-class GT, matched, supported,
  correct argmax, <2 m geometry, semantic weights, and matched semantic NLL.
- `geometry_matched_stage_classification`: separate class/score-agnostic XY<2 m
  maximum-cardinality matching, avoiding training-assignment optimism. TRAIN is
  sampled; VAL complete. Includes heavy confusion and uncovered GT at every stage.
- `controlled_evaluation.fixed_recall`: per-class FP/frame at recall .25/.5/.75.
  Exact score ties are kept; unreachable recall is explicitly reported, never
  compared as if it were reached. Domain is post-NMS candidates decoded >=.01.
- `at_fixed_car_recall`: all-class FP decomposition at Car recall .5/.75. Compare
  only common reachable recall points, not a fixed score across different scores.
- `seen_unseen`: GT recall and matched localization/directed/unsigned heading
  error, globally matched before slicing. No spurious FP from dropping the other
  group. Includes distinct track counts; `heavy_per_track` provides individual
  heavy track results. IDs are evidence of overlap, not proof of physical-identity independence.
- `tram_heading`: independent tram-track registry slice; directed AOE, unsigned
  axis AOE, flip rate, TP and GT counts. These are errors relative to current yaw
  annotations, which class review has **not** independently certified.

Current VAL: all 114 Trucks and 23 Buses have unseen TRAIN IDs; Vans are 138 seen /
32 unseen. Report those separately. Choose changes using coverage, per-class
recall/FP and matched localization, not macro mAP alone. Freeze comparison epochs
and add a second seed before promoting a winning combination. No trained V10
result exists yet, so no claim that these changes solve the failures is warranted.
