# Bus/Van review v2

Includes all 343 class corrections from v1. Additionally removes all Bus objects
in the two bags the user confirmed contain no buses:

| Dataset sequence | TRAIN removed | VAL removed |
|---|---:|---:|
| 11_09_09_40_48 | 124 | 86 |
| 09_25_13_28_18 (BAG_2026_09_25_13_28_18_converted) | 101 | 11 |

322 boxes removed, not converted to ignore regions. Images, frames, other classes,
and retained box geometry are unchanged. Both target bags now have zero Bus labels.
Original source annotations and review v1 remain intact for recovery.
correction_manifest.json records every removal and includes the prior corrections.
Other unresolved ghost/floating labels remain unchanged; this is not full cleanup.

Copy this entire folder, including both PKLs and the newly rebuilt
train_yolo_extents_v1.json, to the server:

    /home/mustafa/mmdetection3d/projects/raybev/datasets/combined_bags_bus_van_review_v2/

Also copy the updated scripts/launch_v7_1_bus_van_review.sh. From the project root:

```bash
bash scripts/launch_v7_1_bus_van_review.sh
```

The launcher now defaults to v2 and a new v2 output directory. Images stay in the
existing combined_bags root. Do not mix v1 extent annotations with v2 PKLs.
Start a fresh run with the pretrained YOLO backbone, not --resume from older labels.
Reevaluate old checkpoints on this same validation version for a fair comparison.
No training has been launched by preparing these files.
