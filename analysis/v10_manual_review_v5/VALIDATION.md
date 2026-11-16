# Preparation validation

- Independent real-data audit: all 4,494 decisions and review-queue identities,
  all 119,179 retained boxes, all 85 removals, and all 5,201 YOLO records passed.
  Reviewed/source annotation files were not modified.
- TRAIN-only control and shape-hypothesis initialization generated with seed 3407.
  Same learned-anchor XY Car coverage; shape proportions recorded separately.
- Focused regression suite: **113 passed, 2 skipped**, covering V10, V9 repairs,
  heading experiments, sparse training/refinement/observations and desktop review.
- Real trainer one-epoch CPU smoke passed, including mixed empty/nonempty frames,
  checkpoint/history writing, objectness/semantic losses, context sampling, early
  weighting and per-stage TRAIN/VAL diagnostics. This is not a V10 accuracy result.
- Actual local COCO YOLO26-medium checkpoint loaded strictly and completed a
  four-camera CPU forward pass. Backbone source SHA256:
  `401cea9ab23ad19246ff7744859816bc599f350e93c9dd30367b6f0a0745d0b7`.
- Fresh medium launcher dry run and shell/Python syntax checks passed. Its frozen
  annotation/decision/queue hashes match V5. Track/geometry-flag acknowledgement was
  used only for a **dry run**, not for training or changing annotations.
- Endpoint gradient test confirms unapproved heavy cuboids receive no endpoint
  gradient; empty/stale approval files are rejected.
- Cross-class plausibility flags are separate from consistency: 49 Car, 35 Van,
  5 Pedestrian boxes; 126 unique follow-up entries after overlap with mixed tracks.
  Targeted contact sheets were inspected, not all 4,494 user labels independently
  visually certified. Geometry/association flags were not automatically repaired.

No full experiment training, GPU memory benchmark or new two-GPU DDP run was
performed. The `context` arm's extra sampling may need a lower microbatch; use
the same microbatch/accumulation in its paired control. See the protocol README.
