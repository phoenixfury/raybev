# Manual v5 annotation audit

Exhaustive source/decision/export consistency, not independent visual certification of every user label.

All edits/removals and all review queue source identities checked; originals untouched.

## Remaining flags

Do not interpret export consistency as clean geometry. The cross-class audit flags
49 Car, 35 Van and 5 Pedestrian cuboids (89 total). Two flagged Van frames overlap
the mixed-class track checks, so `followup_review.jsonl` contains 126 unique objects.

Most urgent: five Van→Pedestrian edits on `11_09_08_01_03/211` retain identical
11.81 × 3.05 × 3.37 m vehicle-sized boxes. See `pedestrian_211_preview.jpg`:
the region contains a pedestrian and a passing foreground van, so neither class
nor association should be decided from the foreground alone. No linked YOLO
extent was changed to Pedestrian. The retained geometry is not credible Pedestrian
supervision and needs image/LiDAR/temporal verification. No automatic relabel,
resize, deletion, or physical-identity assumption was applied.

`van_4946_preview.jpg` provides another example: the Van label is visually plausible
for the blue delivery van, but two retained cuboids are about 16.5/16.1 m long,
3.06 m wide and 4.3/4.24 m high. This illustrates why class confirmation must not
be treated as geometry verification. Inspect metric depth/center and temporal
association as well as dimensions; do not automatically shrink the boxes.

- {"sequence": "08_27_10_12_18", "track_id": "5051", "classes": ["bus", "truck"]}
- {"sequence": "08_27_10_12_18", "track_id": "5053", "classes": ["truck", "van"]}

Geometry flags are not automatic class corrections. Class review alone does not certify cuboid association/geometry. Inspect `annotation_audit.json` and the final-index `followup_review.jsonl`. The controlled launcher is guarded. See `scripts/README_v10_controlled.md` for commands and experiment details.

```json
{
  "status": "passed",
  "reviewed": 4494,
  "decision_statuses": {
    "reviewed": 4409,
    "removed": 85
  },
  "splits": {
    "train": {
      "frames": 9959,
      "objects": {
        "car": 102312,
        "truck": 2016,
        "van": 1544,
        "pedestrian": 3572,
        "bus": 315,
        "cyclist": 122
      },
      "removed": 71
    },
    "val": {
      "frames": 1034,
      "objects": {
        "car": 8647,
        "van": 170,
        "pedestrian": 342,
        "truck": 114,
        "bus": 23,
        "cyclist": 2
      },
      "removed": 14
    }
  },
  "transitions": {
    "train:bus->truck": 45,
    "train:bus->van": 18,
    "train:truck->van": 75,
    "train:van->truck": 59,
    "train:van->car": 129,
    "train:truck->car": 52,
    "train:van->pedestrian": 5,
    "val:truck->car": 6,
    "val:van->truck": 20,
    "val:van->car": 35
  },
  "yolo": {
    "records": 5201,
    "changed": 5,
    "removed": 0,
    "geometry_dropped": 0,
    "pixel_evidence_preserved": true
  },
  "seen_unseen_val": {
    "car": {
      "unseen": 3795,
      "seen": 4842
    },
    "pedestrian": {
      "unseen": 342
    },
    "cyclist": {
      "unseen": 2
    },
    "truck": {
      "unseen": 114
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
  "track_class_conflicts": 2,
  "geometry_flags": 89,
  "geometry_flags_by_class": {
    "car": 49,
    "van": 35,
    "pedestrian": 5
  }
}
```
