# Bus/Van review v1 — conservative class corrections

This is a partial cleanup, not a certified clean dataset. Original files remain unchanged.
Based on the 2026-11-13 image audit; 343 class changes across four tracks:

| Bag | Track | Change | TRAIN | VAL |
|---|---|---|---:|---:|
| 09_30_10_10_54 | 10995 | Van -> Car | 0 | 100 |
| 11_06_10_07_06 | 6568 | Van -> Car | 170 | 36 |
| 09_30_10_10_54 | 10499 | Bus -> Truck | 8 | 0 |
| 11_09_08_01_03 | 7304 | Bus -> Truck | 29 | 0 |

Track-wide class changes use the reviewed representative images; every temporal frame
was not individually certified. No box coordinates, dimensions, headings, IDs, sample
splits, camera paths or calibration were changed. Matching gt_mesh strings were updated.
Suspected ghost Bus tracks 7702/7991, floating boxes and other uncertain candidates
remain unchanged. Do not interpret this package as completing all geometry cleanup.

## Server placement

Copy this entire directory to:

    /home/mustafa/mmdetection3d/projects/raybev/datasets/combined_bags_bus_van_review_v1/

Required files:

- fisheye3dod_infos_train.pkl
- fisheye3dod_infos_val.pkl
- train_yolo_extents_v1.json (rebuilt for these labels; do not use the previous cache)
- correction_manifest.json (provenance and exact changes)

Keep images in the existing data/Custom/combined_bags root. Do not overwrite its
original annotations. Also copy scripts/launch_v7_1_bus_van_review.sh into the server repo.
The launcher resolves paths from its own project location, whether named raybev or raybev_lite.

From the server project directory:

```bash
bash scripts/launch_v7_1_bus_van_review.sh
```

Uses pretrained weights/yolo26n.pt, batch 16 per GPU, accumulation 1, 2 GPUs,
the existing V7 architecture and 96-epoch recipe. No teacher. Training is not launched
by preparing this package. Override ROOT if the server image root differs.

## Initialization recommendation

Use a fresh run with the pretrained YOLO backbone and freshly initialized remaining
network, not fully random YOLO weights and not --resume from old labels. This avoids
retaining optimizer state and class priors fitted to the noisy annotations and gives
a controlled baseline. Fine-tuning is a potentially cheaper later comparison, not
guaranteed to be worse; the current trainer does not support V7 changed-data --init-from.
Do not bypass its provenance checks. Old and new VAL scores are not directly comparable;
evaluate any old checkpoint on the same corrected VAL set for comparison.

For a final-quality retrain, first resolve the remaining ghost/geometry candidates.
This package is ready for a conservative class-correction experiment, not proof the
Bus/Van data-quality problem is solved.
