# V7 audit

Release: `datasets/combined_bags_camera_ray_manual_review_v7`.
See its `REPORT.md` for the complete change summary, limitations and training
input instructions. Source V6 remains intact.

`annotation_audit.json`: independent source/decision/track-ID/YOLO verification
of all 119,126 retained annotations, with zero remaining automatic geometry
flags and zero mixed-class IDs. `correction_plan.json` binds exact decisions
to V6 input hashes. `previews/` contains the four-camera review evidence.
`build_report.json` records plan hash and published manifest.

Image review, default dimensions and conservative track segmentation are not
independent metric-geometry/physical-identity certification. Endpoint geometry
supervision remains disabled pending credible geometry approvals.

Validation: 38 tests passed, two display-only tests skipped in the headless
run. Includes ID collision avoidance, post-removal source-index mapping,
independent full audit, and tamper rejection. No training launched.
