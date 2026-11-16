# Heavy-vehicle annotation review — 2026-11-16, release v4

## Result and limits

Applied **577 class corrections** across TRAIN and VAL,
including the user's nine confirmed passenger-car annotations. Original files
were not overwritten. The nine user-confirmed cars now have user-prescribed
length × width × height **4.0 × 1.8 × 1.5 m**. No other boxes were resized;
none were deleted, ignored, moved or rotated. Centers, yaw, track IDs, image
paths, frame order and split membership are unchanged.

**This is a partial correction, not a certified clean dataset.**
1,059 annotations remain unresolved and 35
require a taxonomy decision. These labels remain in training/evaluation, with their
original classes; they have not been silently removed to improve metrics.
Original Car/Pedestrian/Cyclist labels were not visually audited. Headings and 3D
cuboid accuracy are not certified by this review.

## Coverage and class findings

All **4,555** original Bus/Truck/Van annotations (4,180 TRAIN, 375 VAL), across
**486 track IDs**, were inventoried. First pass: all 41 representative-track
sheets. Second pass: **all 114 generated candidate sheets**, one tile per source
annotation, with temporal context and LiDAR point-count diagnostics. Corrections
were revised when full sequences contradicted a representative crop. Individual
occluded frames are not independently identifiable just because track identity
is supported elsewhere. Counts below describe those reviewed track identities,
not an independent high-resolution, four-camera cuboid certification per frame.

| Original class | Source boxes | Supported same class | Corrected class | Unresolved | Taxonomy pending |
|---|---:|---:|---:|---:|---:|
| Bus | 551 | 313 | 119 | 119 | 0 |
| Truck | 2,302 | 1,473 | 309 | 504 | 16 |
| Van | 1,702 | 1,098 | 149 | 436 | 19 |

Supported identities: 2,884 source boxes. Applied
corrections: 577. Unresolved + taxonomy pending:
1,094. Track IDs are not unique
physical vehicles: the same vehicle may have fragmented IDs.

- Bus → Car: 4
- Bus → Truck: 95
- Bus → Van: 20
- Truck → Bus: 6
- Truck → Car: 4
- Truck → Van: 299
- Van → Car: 53
- Van → Truck: 96

Class policy retained: passenger trams → Bus; minibuses and ambulances → Van;
separate cab/cargo body, flatbed, tanker and articulated cargo vehicles → Truck;
SUV/crossover → Car. Existing construction→Van policy was not broadened, and
pickup mapping was not guessed. See `unresolved.csv` for the 35 construction/pickup
policy entries and other unresolved cases.

## User-confirmed cars

Bag `11_09_09_40_48`: Bus track `1516`, **4 TRAIN boxes**, and Van track `1748`,
**5 TRAIN boxes**, are all now `Car` with mesh `detzero.car` (A4172–A4180).
All nine now use **LWH 4.0 × 1.8 × 1.5 m**, as explicitly requested by the user.
The original centers and yaw are preserved, including Z-center: changing height
with fixed center changes the box bottom. These are prescribed car sizes, not a
claim of calibrated cuboid fitting. The manifest lists all nine old/new sizes.
Quality point-count/bottom-Z diagnostics refer to the original source cuboids;
they must not be interpreted as measurements of the new nine cuboids.
The refreshed sheet is `second_pass_pages/all_114.jpg`. Old audit/source sheets
still show old labels for provenance; they are not the new training inputs.

## YOLO weak extents

`train_yolo_extents_v1.json` is synchronized to the revised TRAIN pickle:

- All **5,201 records** validated against runtime `VerifiedExtents`.
- **27 linked GT class IDs** changed. Not every corrected 3D box has a
  retained weak YOLO extent; association/confidence/occlusion gates leave only
  this subset in the extent file.
- Measured 2D boxes, camera indices, edge-valid masks, occlusion/truncation flags,
  weights and original YOLO/manifest evidence hashes are unchanged.
- TRAIN fingerprint is rebuilt. Never pair this JSON with the old TRAIN pickle,
  or the old JSON with this TRAIN pickle.
- None of the nine resized cars has a retained extent record. Existing 2D
  associations remain geometrically valid; the migration rejects size changes
  on an associated box rather than reusing stale association evidence.
- Detector-cache predictions are preserved as evidence, not relabeled to pretend
  the detector originally predicted the corrected classes. No detector rerun needed.

New TRAIN fingerprint: `3e47895020ec3e89664575b848bb09bc4df3fa0bd776d8b2105673ec3cca3d72`.
These remain **weak** 2D boundaries, not human-traced ground truth.

## Other problems exposed by the sweep

The following point-count and bottom-Z numbers describe **original source
geometry**, before the nine authorized size edits.

- **33 TRAIN heavy boxes contain zero LiDAR points**; **129** heavy boxes have
  fewer than 10 points (126 TRAIN / 3 VAL). Sparse support is a flag, not proof
  of non-existence: occlusion and range can explain it.
- **280** heavy boxes have bottom Z outside ±1 m in BASELINK (265 TRAIN / 15 VAL).
  Road slope/calibration can contribute; inspect before editing heights.
- `09_25_13_28_18` Truck IDs `4179/4180/4182` project around the ego test car and
  sky/building, with no identifiable truck in the chosen views. They need full
  calibrated camera/LiDAR association review, not a guessed Car relabel.
- Some tracks contain competing visible identities. Example: Van `2932` in
  `09_17_08_33_47` (71 TRAIN boxes) has white-van versus tractor/occluder evidence.
- The validation Bus labels are only **23 frames**: 16 are one passenger tram
  (`09_17_08_33_47/2894`), 1 is a visible coach, and 6 remain occluded/unresolved.
  A correct class taxonomy does not establish correct heading/size for these boxes.
- Validation Van track `11_09_09_00_30/1400` contributes **117 frames**, over half
  of the revised VAL Van count (225). This is not broad vehicle/scene coverage.

`quality_flags.csv` contains exact frame/index addresses for those flags.
Do not use the reviewed VAL labels to tune class rules until metrics look good;
keep this revision fixed and publish its manifest with the experiment.

## Files and restart

Use this directory as an annotation override; **keep the image ROOT unchanged**:

```
datasets/combined_bags_camera_ray_heavy_review_v4/
  fisheye3dod_infos_train.pkl
  fisheye3dod_infos_val.pkl
  train_yolo_extents_v1.json
  correction_manifest.json
  annotation_review.csv / annotation_review.jsonl
  track_findings.json / unresolved.csv / quality_flags.csv
  verification_report.json / second_pass_report.json
  second_pass_pages/
```

On the training machine copy this directory under its RayBEV project directory
(e.g. `/home/mustafa/mmdetection3d/projects/raybev/datasets/combined_bags_camera_ray_heavy_review_v4/`
if that is its checkout). Copy the new `scripts/launch_v8_heavy_review.sh` and
`scripts/prepare_heavy_annotation_review.py` too. Existing image/source data and
source annotation pickles must remain available. If the original extent JSON
was deleted, the updated launcher allows its absence while still enforcing the
v4 JSON hash and runtime GT/calibration checks. It warns and records the original
extent comparison as unavailable; it never accepts a wrong existing source JSON.

From the RayBEV project directory, in the existing training Python environment:

```bash
VERIFY_ONLY=1 bash scripts/launch_v8_heavy_review.sh
DRY_RUN=1 bash scripts/launch_v8_heavy_review.sh
bash scripts/launch_v8_heavy_review.sh
```

Default: medium backbone, fixes objective, microbatch **4 per GPU**, accumulation
**2**, **2 GPUs**, effective batch **16**, 64 epochs, new output directory
`work_dirs/raybev_v8_medium_heavy_review_v4`. No ROSA teacher and no old RayBEV
checkpoint is resumed. Backbone initialization is the existing local COCO weights.
Use `ROOT=/actual/image/root` if the image data path differs. Training was **not** launched.
The loss/backbone settings are unchanged; the experiment changes reviewed labels
and the nine explicitly requested sizes.
Compare against the old checkpoint only after reevaluating it on this same fixed
VAL revision; its old validation metrics are not directly comparable.

For a genuinely clean next benchmark, resolve `unresolved.csv` with calibrated
full-camera + LiDAR cuboid review; settle construction/pickup policy, then audit
the original Car class for missing heavy vehicles. Refit/verify geometry and
directed heading separately. Keep difficult VAL objects; do not delete unknowns
to manufacture a clean score.

## Verification

Post-write consistency sweep checked **119,264 boxes**
in all **10,993** TRAIN/VAL frames, including non-heavy annotations for unintended
changes, and every YOLO extent record. Source and release hashes are recorded;
only the nine authorized LWH edits differ in geometry; centers, yaw, identities,
splits and non-target fields match the originals.
TRAIN/VAL frame-token sets are disjoint. This is data integrity evidence, not a
claim that every visual annotation is spot on. Targeted regression tests passed.

| Split | Car | Truck | Bus | Van |
|---|---:|---:|---:|---:|
| TRAIN | 102,131 | 2,070 | 415 | 1,647 |
| VAL | 8,606 | 114 | 23 | 225 |
