# RayBEV V9 heavy-vehicle update

V9 keeps the V8 six classes, YOLO26m default, four calibrated 800×640 cameras, Q225/C96, four decoder stages, and the current ±48 m XY evaluation range. V8's actual range is retained for comparability. The network does not add velocity, a dense BEV map, or inference-time LiDAR. No checkpoints, data, or labels are included or changed.

## 1. Diagnose your existing checkpoint before retraining

From the project folder in the existing training environment. Replace the paths with your own and use the exact VAL revision saved in the checkpoint.

```bash
ROOT=/path/to/combined_bags_camera_ray_dedup_v2
REVIEW=/path/to/combined_bags_camera_ray_heavy_review_v4
CHECKPOINT=/path/to/raybev_v8_medium_heavy_review_v4/last.pt

python scripts/infer_yolo_rosa_sparse.py \
  --root "$ROOT" --ann "$REVIEW/fisheye3dod_infos_val.pkl" \
  --checkpoint "$CHECKPOINT" --amp --batch-size 2 \
  --audit --sweep --car-recall .8 \
  --raw-queries analysis/v8_raw_queries.npz \
  --output analysis/v8_audit.json
```

- `query_audit` records the first candidate-level failure for each Truck/Van/Bus at XY <2 m and score ≥.1, including wrong argmax, low score, camera support/range, and NMS/cap. It also reports signed longitudinal/lateral and ego-radial errors, LWH ratios, yaw, rank and track. A retained candidate is not proof of a one-to-one true positive. Signed ego-radial error is a diagnostic proxy, not a selected camera-ray measurement.
- `operating.errors` splits false positives into duplicates, near wrong-class, mislocalized and background/unlabeled buckets. The standard evaluator's matching does not change. Review clips before treating suspected unannotated objects as background. `--raw-queries` saves unfiltered boxes, class/quality logits, support, axis and facing in frame-token order.
- `scoring_sweep` replays quality powers 0, .5, 1, same-class NMS at .3/.4, and a conservative cross-vehicle duplicate option. It reports Car FPs/frame at the requested Car recall (or marks it unreachable) and Truck/Van/Bus recall at .1. All variants use one forward pass. The benchmark fields always use the checkpoint's original decode, .01 retention floor and .1 operating point. The sweep is analysis, not a new benchmark.
- To apply an operational setting, add `--quality-power`, `--nms-iou`, or `--vehicle-duplicates`. Optional `--class-bias` and `--class-thresholds` take six values in Car/Pedestrian/Cyclist/Truck/Van/Bus order. Fit offsets or thresholds on an independent calibration split. Operating thresholds are lower-bounded by `--operating-floor` (default .1). To test thresholds below .1, lower that floor explicitly. The conservative cross-class option defaults off because it can still suppress a true nearby detection.

## 2. Launch V9 on the reviewed split

```bash
ROOT="$ROOT" REVIEW_DIR="$REVIEW" DRY_RUN=1 bash scripts/launch_v9_heavy.sh
ROOT="$ROOT" REVIEW_DIR="$REVIEW" bash scripts/launch_v9_heavy.sh
```

The inherited launcher uses two GPUs, microbatch 4 per GPU, two accumulation steps, and 64 epochs. Source labels, reviewed label releases, extent annotations, and fresh YOLO26m weights must already be present at their normal paths. Override `YOLO26_WEIGHTS` if needed. The existing review verification still runs. Use the original project location or pass the full dataset paths after extracting elsewhere.

For a fresh optimizer/scheduler initialized from a compatible architecture-5 checkpoint on exactly the same dataset/calibration:

```bash
ROOT="$ROOT" REVIEW_DIR="$REVIEW" INIT_FROM="$CHECKPOINT" bash scripts/launch_v9_heavy.sh
```

`INIT_FROM` is not resume. It transfers verified state with strict key/shape checks. The added final-box classifier starts as a zero residual. Its geometry input is detached, so the new class path cannot directly distort boxes to fit a class. Development controls are `V9_ARM=proposal`, `V9_ARM=matching` and `V9_ARM=resample`; the latter two each add only their named change above the proposal arm. Use distinct OUT paths when repeating a run. Resume a V9 run with the same arm/launch recipe and its own `--resume .../last.pt`.

The full arm enables:

1. **Proposal assignment.** Choose the closest GT *inside* each visible half-open lattice cell and average each frame loss once, even if a frame is empty. If two GT centers share a cell, the nearest still wins because the proposal has only one offset.
2. **Geometry-first matching.** Scale the class cost by .05 for three epochs, then ramp to 1 over five more; keep one-to-one assignment. Log final heavy assignment regret and count at >.5 m in `history.jsonl`. Validation uses the full matching class cost throughout.
3. **Post-refinement class evidence.** Six fixed box-relative points sampled at stride 8 only after the last refinement, fused with detached updated LWH and visibility. There is no additional full decoder stage.
4. **Facing.** Accept each visible longitudinal end independently in the camera-specific branch. Retain axis and directed yaw measurements separately.
5. **Quality/ranking.** Use a smooth target reflecting the fraction of .5/1/2/4 m center-distance criteria passed. Mine high-scoring unmatched background or close, unambiguous duplicates when a nearby owner is already assigned. Ignore regions, unsupported candidates and near-only unmatched hypotheses are excluded from this new term.
6. **Epoch audits.** Add grouped heavy candidate failure counts and FP categories to the JSONL history. Standard AP, score/NMS defaults and annotation splits are unchanged. Threshold and cross-vehicle NMS changes are optional replay/operating controls.

The implementation also exposes `--heavy-endpoint-weight .1` for an unordered lengthwise-endpoint loss. It is **off by default**. Enable only for an annotation variant you have verified geometrically and only if the Truck audit shows a signed end bias/underestimated length. It applies to matched annotated Truck/Van/Bus training boxes and is invariant to a global π flip of either box axis. It is not a substitute for LiDAR visibility, time synchronization, or accurate cuboids.

## 3. Review suspected data issues rather than auto-relabeling

```bash
python scripts/audit_v9_training_labels.py \
  --train-ann "$REVIEW/fisheye3dod_infos_train.pkl" \
  --output analysis/v9_training_label_review.json
```

The queue includes Car tracks with heavy-looking dimensions, mixed vehicle labels within a track, and heavy boxes with unusual BASELINK bottom Z. The thresholds are review heuristics, not a class definition. Check multiple cameras, timestamps, and calibrated point clouds. Do not use a large box alone to relabel an SUV, pickup, or van. Use the pre-existing `scripts/audit_heading_heavy_data.py` for world-heading flip candidates on the source split. Review the single-tram concentration in the Bus benchmark and keep tram/coach numbers distinct where track labels support it. This update cannot correct or certify missing annotations, track identities, calibration, or Bus heading without the real data.

Primary comparison: Truck AP at .5/1/2 m, raw .5 m coverage, per-track Truck/Bus→Car, Bus directed versus axis error, and Car FPs/frame at matched recall. Benchmark the full arm against the proposal-only arm before attributing a gain to any particular component. None of these code changes guarantees higher real-world AP or TDA4VH throughput.

## Verification performed

- Nine new V9 regression/integration tests passed, including an inference CLI run with query export, scoring sweep and label queue.
- Broad CPU suite: 376 passed, 16 skipped, four legacy V5/V6 launcher assertions failed identically in the unmodified source.
- A synthetic YOLO26m forward, validation loss and decode passed using four 800×640 images at Q225/C96 and all four stages. A separate small synthetic one-epoch training run wrote the checkpoint, validation metrics and audit history.
- All four V9 launcher variants passed shell syntax and argument-only dry-runs. Two-rank CPU execution could not initialize Gloo under the local networking restrictions. CUDA/DDP, real-bag accuracy and TDA4VH deployment were not verified.
