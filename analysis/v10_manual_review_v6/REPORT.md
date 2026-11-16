# Manual v6 annotation audit

## Training readiness

Structural/source-index/YOLO verification passed for all 119,135 retained objects.
All 43,972 referenced camera image files exist. All 126 original follow-up keys
have exactly one saved decision: 82 reviewed, 44 removed, none uncertain/missing.
The v5 inputs and v6 annotation files were not modified by these checks.

V6 is technically loadable, but is not yet a clean controlled-training baseline:
30 class/dimension plausibility flags remain (19 Car, 11 Van; 25 TRAIN, 5 VAL).
These are warnings requiring association/geometry evidence, not automatic proof
that the manually confirmed class is wrong. Examples include a retained Van on
09_16_09_58_56/1837 with LWH 12.24 x 2.94 x 3.74 m, and two prescribed Car cuboids
of 4.0 x 2.5 x 3.0 m. The five oversized Pedestrian boxes were all removed.

Six bag/track IDs now contain mixed classes:

| Bag | Track | Retained labels |
| --- | --- | --- |
| 08_27_10_12_18 | 5051 | TRAIN: Bus 33, Truck 2 |
| 09_24_08_52_50 | 5806 | TRAIN: Car 6, Truck 1 |
| 09_24_08_52_50 | 5809 | TRAIN: Car 3, Truck 1 |
| 09_24_08_52_50 | 6118 | TRAIN: Car 8, Truck 1 |
| 11_09_08_01_03 | 1114 | TRAIN: Car 1, Truck 1 |
| 09_30_10_10_54 | 4909 | VAL: Car 5, Truck 1 |

Track 5053's former Van/Truck conflict is resolved by the explicit Van-frame
removals. Track 5051 remains mixed. The five new conflicts correspond to the
five Car-to-Truck changes; check whether these are partial corrections or
real track-ID switches. Do not majority-vote or propagate labels blindly.

YOLO: 5,200 verified extent records remain; one stale geometry association was
dropped, zero linked class records required changes, and all retained pixel
evidence is unchanged. No removed object had a linked extent.

Recommendation: resolve/document the remaining identity and geometry warnings,
then freeze a new release/audit for the V10 control versus shapes comparison.
Do not reuse the frozen v5 audit for v6 or bypass launcher guards silently.
Endpoint loss remains disallowed without independent geometry approvals.
No training was launched. These checks are exhaustive consistency checks,
not independent visual certification of every reviewed cuboid.

Exhaustive source/decision/export consistency, not independent visual certification of every user label.

All edits/removals and all review queue source identities checked; originals untouched.

## Remaining flags

- {"sequence": "08_27_10_12_18", "track_id": "5051", "classes": ["bus", "truck"]}
- {"sequence": "09_24_08_52_50", "track_id": "5806", "classes": ["car", "truck"]}
- {"sequence": "09_24_08_52_50", "track_id": "5809", "classes": ["car", "truck"]}
- {"sequence": "09_24_08_52_50", "track_id": "6118", "classes": ["car", "truck"]}
- {"sequence": "11_09_08_01_03", "track_id": "1114", "classes": ["car", "truck"]}
- {"sequence": "09_30_10_10_54", "track_id": "4909", "classes": ["car", "truck"]}

Geometry flags are not automatic class corrections. Class review alone does not certify cuboid association/geometry. Inspect `annotation_audit.json` and the final-index `followup_review.jsonl`. The controlled launcher is guarded. See `scripts/README_v10_controlled.md` for commands and experiment details.

```json
{
  "status": "passed",
  "reviewed": 126,
  "decision_statuses": {
    "removed": 44,
    "reviewed": 82
  },
  "splits": {
    "train": {
      "frames": 9959,
      "objects": {
        "car": 102285,
        "truck": 2020,
        "van": 1528,
        "pedestrian": 3567,
        "bus": 315,
        "cyclist": 122
      },
      "removed": 44
    },
    "val": {
      "frames": 1034,
      "objects": {
        "car": 8646,
        "van": 170,
        "pedestrian": 342,
        "truck": 115,
        "bus": 23,
        "cyclist": 2
      },
      "removed": 0
    }
  },
  "transitions": {
    "train:car->truck": 4,
    "val:car->truck": 1
  },
  "yolo": {
    "records": 5200,
    "changed": 0,
    "removed": 0,
    "geometry_dropped": 1,
    "pixel_evidence_preserved": true
  },
  "seen_unseen_val": {
    "car": {
      "unseen": 3794,
      "seen": 4842
    },
    "pedestrian": {
      "unseen": 342
    },
    "cyclist": {
      "unseen": 2
    },
    "truck": {
      "unseen": 115
    },
    "van": {
      "unseen": 32,
      "seen": 138
    },
    "bus": {
      "unseen": 23
    }
  },
  "credible_geometry": {
    "approved_boxes": 0,
    "endpoint_training_allowed": false,
    "reason": "Class confirmation does not certify cuboid centers/dimensions/yaw. No geometry approval ledger supplied."
  },
  "track_class_conflicts": 6,
  "geometry_flags": 30,
  "geometry_flags_by_class": {
    "car": 19,
    "van": 11
  }
}
```
