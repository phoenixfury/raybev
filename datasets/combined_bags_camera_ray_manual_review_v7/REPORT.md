# V7 annotation release

Source: `combined_bags_camera_ray_manual_review_v6`. V6 stays unchanged.

## Changes

- Seven visually identified delivery/box/tow Trucks changed from Car to Truck.
- Two integrated panel Vans changed from Car to Van.
- Thirteen flagged/reclassified Vans receive default LWH **5.46 x 2.14 x 2.26 m**.
  This is the rounded coordinate-wise median of 1,517 plausible TRAIN Vans, not a
  physical measurement or a fit. Unflagged Vans retain their original dimensions.
- Two passenger SUV boxes receive size priors of 4.8 x 1.9 x 1.75 m and
  4.8 x 1.9 x 1.85 m instead of the previous 3 m prescribed height.
- Nine ambiguous object/frame annotations excluded: six TRAIN, three VAL.
  These are frame-only removals. No images, frames or complete tracks removed.
- Seventeen per-frame track IDs changed to unused IDs to separate conflicting
  class/identity groups without forcing a majority class. The exact source-index
  mapping and rationale are in `track_id_revisions.json`.

The track-ID changes are conservative review segments, **not certified physical
tracks**. They remove class-inconsistent ID grouping but do not prove every
within-class association is correct. Use this same frozen release for control
and treatment runs and qualify seen/unseen-track metrics accordingly.

## Verification

Independent source-index comparison checked every retained object against V6,
all explicit decisions, all ID revisions, all untouched fields and image paths.
TRAIN: **109,831 objects / 9,959 frames**.
VAL: **9,295 objects / 1,034 frames**.
Total: **119,126 objects**.

All **5,200 YOLO extent records** verified. One linked GT class changed from Car
to Truck; retained pixel measurements/evidence are unchanged. No linked extent
needed dropping for these removals or size edits.

Remaining automated class/dimension plausibility flags: **0**.
Remaining mixed-class track IDs: **0**.
These are consistency/plausibility results, not certification of all dataset
labels, object associations or metric geometry. Do not enable endpoint geometry
loss from this review: centers/yaw were not refitted and no independent LiDAR
geometry approvals were created.

## Files and training

Use this directory's `fisheye3dod_infos_train.pkl`,
`fisheye3dod_infos_val.pkl` and `train_yolo_extents_v1.json` together.
Never mix V7 PKLs with old extents. Keep all review metadata and
`track_id_revisions.json` with the release for verification.

The existing V10 launcher defaults to V5. Prepare a **new V7 audit** and set
both `REVIEW_DIR` and `AUDIT_DIR`; do not bypass its old V5 hash checks.
Training was not launched during annotation creation.

Review evidence, exact correction plan and independent full audit:
`analysis/v10_manual_review_v7/{previews/,correction_plan.json,annotation_audit.json,build_report.json}`.

Rebuild into a new output/analysis directory with
`scripts/prepare_manual_review_v7.py --root /path/to/data --plan /path/to/correction_plan.json`.
The builder refuses existing output/state paths and verifies staged output
before publishing it. V6 source hashes are bound in the correction plan.
