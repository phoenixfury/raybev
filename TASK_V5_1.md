# V5 analysis and next experiment: reviewed heavy labels + circular facing

## Result: lighter backbone works, but heavy-class generalization fails

Run: `work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti`.
The checkpoints have finite parameters, the expected calibration hash and the
verified official YOLO26n pretrained-weight SHA. This is not a failed weight load.

| Checkpoint | Epoch | Internal mAP | TRAIN loss | VAL loss | Car heading AOE at score≥0.3 | Matched Car recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| best_heading | 3 | 5.41% | 4.55 | 6.57 | 12.3° | 12.3% |
| best_map | 10 | 6.24% | 2.94 | 7.32 | 21.8° | 26.5% |
| last | 24 | 5.79% | 2.19 | 8.03 | 26.3° | 30.0% |

The heading rows are **not equal-recall comparisons**: epoch 3 is not proved
better at the coverage of epoch 24. mAP averages the five present validation
classes; Van has no validation GT. It is the project's evaluator, not ROSA's
official metric or nuScenes NDS. Existing R18 reviewed-GT control was about
5.31% mAP: YOLO improves overall mAP and compute, but does not solve heavy classes.

At epoch 24, oracle-center TRAIN Truck/Bus classification accuracy is 99.5% /
99.0%; VAL is 0% / 3.3%. Truck true-class probability drops from 0.148 at epoch 3
to 0.0088 at epoch 24. Bus peaks around 0.358 at epoch 7 and falls to 0.066.
VAL objectness is only 0.024 / 0.018 for Truck/Bus at epoch 24.
These are **GT-center probes, not detection recall**. Regression offset error
at an oracle center is not evidence that proposals localize the object.

Truck AP stays zero throughout the run. Bus briefly gets nonzero AP, but both
heavy APs are zero in best_map/last. At epoch 24, score≥0.3 still yields one true
positive per heavy class, not literally zero boxes; recall/precision are too low
for the evaluator's AP gates. Lowering display thresholds or multiplying rare
class scores would hide the training problem and increase false positives.

The training/validation gap, collapsing unseen class probabilities, and rising
VAL loss indicate overfitting. Contradictory/uncertain heavy labels compound it.
Continuing epoch 24 is not the proposed next experiment.

Full epoch/center data and checkpoint hashes: `analysis/v5_analysis.json`.

## TRAIN-label review and a safe annotation fork

Inspected 159 heavy-track candidate records: 154 eligible contact-sheet crops
and five without an eligible crop. Checked 41 selected tracks across up to three
observations with projected centers and heading arrows, then 15 tracks in a Van
taxonomy follow-up (55 unique tracks across both passes). Evidence:
`analysis/v5_train_review_evidence/`; earlier complete contact sheets:
`analysis/v4_3_train_label_candidates/`; follow-up:
`analysis/v5_van_review_evidence/`.

New fork: **`datasets/heavy_review_v4`**. Original annotations, v5 checkpoints,
all images, tokens, track IDs, physical box coordinates and splits are preserved.

- **191 raw TRAIN Bus→Truck corrections across 23 tracks** (186 boxes in range).
  Includes box-bodied cargo vehicles and the SITRAK tracks 30_09/2298, 2303,
  2305, 2306 and 2308 that were explicitly labeled Bus.
- **257 raw TRAIN Truck→Van corrections across 12 tracks** (254 in range).
  Minibuses and ambulances are Van, as explicitly confirmed by the user; visually
  resolved panel vans are also Van. Covered cargo pickup 16_09/1345 stays Truck.
  Total TRAIN class corrections: **448 raw / 440 in range across 35 tracks**.
- **149 uncertain raw TRAIN boxes across 15 tracks** are preserved with
  `gt_ignore=True` (145 in range). Apparent association/geometry inconsistencies,
  occlusion and construction-vehicle subtype ambiguity make a blanket relabel
  unsafe. Example: 16_09/1193 appears on different Car/Van crops across frames.
  We do not invent a class or a metric box from those images.
- Ignored boxes create **neither foreground targets nor dense background
  penalties** within a conservative rotated-footprint envelope plus one cell.
  Other known positive centers in the envelope remain supervised. No frame is
  deleted. Ignored boxes cannot enter size priors, class counts, teacher matching
  or track references. Validation ignores are forbidden.
- True buses/trams remain Bus. **VAL labels exactly equal v5's reviewed GT**:
  Truck 188, Bus 152. No difficult validation targets were removed or relabeled
  for this experiment. The old three approved validation cargo corrections are
  retained from heavy_review_v1.

Trainable in-range counts: Car 38,155; Pedestrian 1,849; Cyclist 144; Truck 645;
Van 312; Bus 337. Size priors are recomputed from these TRAIN-only targets, **not
manually corrected geometry**. Geometry and direction quality still require
further 3D review; exact fitted priors are in `analysis/v5_analysis.json`.

This is **conservative curation, not certification of every annotation**.
Remaining contact-sheet-only/occluded candidates are not guaranteed correct.
No uncertain Car/Van association is silently relabeled. The approved rules,
counts, original/output hashes and individual edits are in the fork's manifest.
`heavy_review_v2` is the intermediate class-only fork; v3 adds ignores; both are
preserved. **V5.1 uses v4**, including the user-confirmed Van taxonomy.
Van still has no annotated validation GT: this controlled experiment preserves
VAL, but that absence does not certify its physical taxonomy. A separate VAL
review would require a new evaluation baseline, not silent scoring changes.

## Heading change

The current doubled-angle axis + two-vector facing regression is mathematically
valid, but a mean-facing vector can shrink toward zero when front/rear evidence
conflicts. In v5 there are many front/back errors, not just small axis errors.

Opt-in **`--facing-loss circular_bins`** retains continuous `sin(2*yaw), cos(2*yaw)`
for the axis and replaces the two facing outputs with **eight circular logits**.
Training uses adjacent-bin interpolated targets with CE, continuous through both
bin boundaries and the -π/π seam. Host decoding selects the strongest facing mode
to choose the axis's front/rear sign, preserving continuous angular precision.
It does not average opposite modes to a zero facing vector. Hemisphere posterior
confidence is reported separately and **never boosts object scores**.

The same loss is used consistently for GT, verified ROSA yaw targets and track
references. The default and historical checkpoints retain their original
two-vector behavior. This is a hypothesis to test, **not a measured heading gain**;
it cannot manufacture a visible front end or repair a wrong direction label.

Only the last facing projection is reset during unit-vector→circular migration.
Class-prior edits also reset the final size-residual projection. All other
compatible weights load strictly. Optimizer, schedule, best metrics and epoch
are NOT resumed. Cost: +198 parameters, about +0.0283G convolution MACs per
four-camera frame (~0.26% of the model). No learned attention, Softmax or dynamic
sampling is added to deployment graphs; circular decoding is host-only.

## Offline teacher: no ROSA launch required

Use **`teacher/raybev_heading_teacher_train_reviewv4.json`**, supplied here.
The original teacher is first validated against original TRAIN labels; each
record is then matched to unchanged geometry, class IDs remapped and ignored
records removed. Teacher yaw, confidence and original prediction/checkpoint
hashes are unchanged. 370 cached class IDs were remapped and 95 ignored records
dropped; 32,112 teacher centers and 38,434 track references remain after gating.
It is yaw-only supervision, not reused teacher class probabilities.

Never bypass the annotation hash check or use the original teacher with edited
TRAIN labels. Rebuilding this binding, if needed, uses cached yaw only:

```bash
python scripts/rebind_heading_teacher.py --root /absolute/data/root \
  --reviewed-ann datasets/heavy_review_v4/train.pkl \
  --output teacher/a_new_reviewed_teacher.json
```

## Run the paired next experiment

Copy updated `raybev/`, `scripts/`, **`datasets/heavy_review_v4/`** and
**`teacher/raybev_heading_teacher_train_reviewv4.json`** to the server project.
Keep v5 `best_heading.pt` in its existing run directory. No new dependencies or
pretrained download are required; Ultralytics 8.4.182 and 8.4.208 remain supported.

From the project directory, in the existing training environment:

```bash
# Data + regularization control: original two-vector facing branch.
bash scripts/launch_v5_1_review_heading.sh control

# Same data, initialization and recipe; only the facing formulation changes.
bash scripts/launch_v5_1_review_heading.sh heading
```

Run these sequentially on the two GPUs. ROOT is auto-detected as in the v5
launcher; override with `ROOT=/absolute/data/root` if necessary. Outputs are
separate `raybev_v5_1_yolo26n_{control,heading}_reviewv4_48m_clidar_2x3080ti` dirs.
The launcher verifies the annotation manifest before training.

Both arms start weights-only from **v5 best_heading, epoch 3**, before the severe
confidence drift: 6 epochs, LR 1e-4, batch 4/GPU, accumulation 1, warmup 200.
Freeze /8 and /16 YOLO stages **including BN statistics**. /32 stays trainable
at 0.1× backbone LR; the neck and BEV/head parameters use the main LR.
Class/scoring weights, ±48 m/0.25 m geometry, OpenCV and CLIDAR conventions,
height fusion and teacher weights remain unchanged. Confidence is addressed by
removing conflicting supervision and constraining overfitting, not score boosts.

**Do not `--resume` v5 or start from `last.pt`.** For an interrupted *new* arm,
use its same launcher/mode and its own checkpoint:

```bash
bash scripts/launch_v5_1_review_heading.sh heading \
  --resume work_dirs/raybev_v5_1_yolo26n_heading_reviewv4_48m_clidar_2x3080ti/last.pt
```

Compare heavy AP, recall and precision, true-class probabilities vs objectness,
confusion and GT/predicted-class size ratios. Compare heading flip rate/AOE at
matched recall, including score≥0.01/0.1/0.3 and by range/bag. Do not pick a low
AOE checkpoint merely because its recall collapses. If both arms' validation
probabilities collapse again, further TRAIN geometry/identity review is needed;
do not just extend training or increase class weights.

Validation includes circular-angle identities and gradients, ignore-region
positive/negative loss isolation, old-checkpoint compatibility, migration,
teacher rebinding, single-epoch teacher/track training, ONNX parity, two-rank CPU
DDP, and four native full-resolution GPU AMP recovery updates using the final
v4 annotations/teacher: cargo Truck, tram Bus, ambulance Van and ignored target.
All 273 unit/integration tests pass. Final smoke artifacts:
`analysis/v5_1_reviewv4_native_smoke.json` and
`analysis/v5_1_reviewv4_native_export.json`. Reproduce with:

```bash
python tests/smoke_v5_1_native.py --root /absolute/data/root \
  --output /tmp/v51_native_smoke.json --export-output /tmp/v51_native_export.json
```

These are structural checks, not retraining results. No production training was
started here. TIDL compilation/INT8 accuracy and TDA4VH latency remain unmeasured.
