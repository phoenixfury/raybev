# Precomputed ROSA training teacher — ready for the server

## Current V5.1 experiment

Use `raybev_heading_teacher_train_reviewv4.json` with
`datasets/heavy_review_v4/` and `scripts/launch_v5_1_review_heading.sh`.
See [TASK_V5_1.md](../TASK_V5_1.md) for the reviewed taxonomy and paired runs.
The original cache below is for the original TRAIN labels; reviewv3 is retained
for reproducibility. Do not mix caches and annotation revisions.

All three compact caches are tracked in Git (about 6.4 MiB each).
The bulky `*_rosa_inference/` directories are ignored and are not needed for
training or teacher rebinding. No Git LFS or ROSA installation is required to
use these caches. `SHA256SUMS` covers all three files.

## Original-label experiment

Copy **`raybev_heading_teacher_train.json`** with the updated RayBEV project.
This is the only teacher data file required for training: no ROSA model,
checkpoint, native ROSA images or ROSA inference launch is needed on the server.
The JSON is 6,746,924 bytes (about 6.7 MB).

From the updated project directory on the server:

```bash
BATCH_SIZE=1 ACCUM_STEPS=4 bash scripts/launch_2x3080ti.sh
```

The launcher defaults to `teacher/raybev_heading_teacher_train.json`. Its
dataset ROOT default is unchanged; override `ROOT` if your dataset is elsewhere.
Use a fresh output directory/run, not `--resume` from MobileNet weights.

## Contents and coverage

Generated locally on 2026-11-09 with ROSA's
`best_fisheye_bev_mAP_epoch_6.pth`, using its original native dataset/pipeline.
All **3,787 TRAIN frames** were inferred; 3,620 have an accepted teacher match.
None of the 674 RayBEV validation tokens appear in the cache.

| Class | Accepted teacher matches | In-range TRAIN GT |
|---|---:|---:|
| Car | 30,048 | 38,155 |
| Pedestrian | 1,013 | 1,849 |
| Cyclist | 95 | 144 |
| Truck | 630 | 794 |
| Van | 54 | 58 |
| Bus | 402 | 587 |
| Total | 32,242 | 41,587 |

These are training-object exposures, not unique tracks or validation accuracy.
Matches require the same class, ROSA score >=0.1, XY center distance <=1 m,
and directed heading disagreement with GT <=30 degrees. The observed maximum
accepted heading difference is 29.956 degrees. Raw ROSA detections originate in
CLIDAR; cache GT centers and teacher headings have been converted once into
BASELINK with the active calibration. Input labels remain CLIDAR.

The training loader retains 32,207 matches after exact regression-owner/cell
collision filtering; it does not supervise competing owners in the same cell.
Loading this actual file and backward checks on cached Truck/Bus examples
passed with finite axis/facing gradients. No RayBEV training run was started.

## Verification and portability

```bash
cd teacher
sha256sum -c SHA256SUMS
```

Cache SHA256:
`2bcce021a36b51d549768aa65cd06f89a51db4c34affe3f8d18abdbbf5fc710e`

ROSA checkpoint SHA256:
`308a4105a29c34d9571bbddf02fd0e49c408750079dd00f76e479296b47157eb`

Raw prediction JSON SHA256:
`413bf4fb37d622515e4aa30f06bb29e0afa1f95fa9c5560119ffce7220c9aecc`

TRAIN token/BASELINK-box signature:
`b12ee0604eed6951f909ee8b5707c51845138830f1f2ef5e8c87265d97932db5`

The updated loader verifies tokens, ordered GT boxes, label convention and
camera/LiDAR calibration. Pickle serialization, frame ordering and machine
paths may differ without invalidating identical labels. Changed GT or
calibration is rejected. The local checkpoint path recorded in the JSON is
provenance only; the loader never opens it. Keep the JSON unchanged so resume
can verify its exact content hash.

## Optional local audit files — not needed on the server

If separately retained or regenerated, full raw detections and logs belong in
this ignored directory (not included in a Git clone):

```
raybev_heading_teacher_train_rosa_inference/
  best_fisheye_bev_mAP_epoch_6/predictions.json
  best_fisheye_bev_mAP_epoch_6/predictions.pkl
```

The raw prediction JSON is approximately 113 MiB and uses ROSA's class names
and CLIDAR gravity-center boxes. Do not pass that raw JSON as `--heading-teacher`;
training consumes the small, filtered `raybev_heading_teacher_train.json` above.
Regeneration instructions are in [TASK_V4.md](../TASK_V4.md).
