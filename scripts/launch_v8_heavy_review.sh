#!/usr/bin/env bash
# Reviewed heavy labels plus nine user-prescribed car sizes. Originals stay intact.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
export ROOT="${ROOT:-$PROJECT_DIR/../../data/Custom/combined_bags_camera_ray_dedup_v2}"
REVIEW="${REVIEW_DIR:-$PROJECT_DIR/datasets/combined_bags_camera_ray_heavy_review_v4}"
export TRAIN_ANN="$REVIEW/fisheye3dod_infos_train.pkl"
export VAL_ANN="$REVIEW/fisheye3dod_infos_val.pkl"
export EXTENT_ANNOTATIONS="$REVIEW/train_yolo_extents_v1.json"
export OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v8_medium_heavy_review_v4}"
export EXPERIMENT="${EXPERIMENT:-medium_fixes}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-2}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
echo 'Reviewed revision: unresolved labels retained; nine car LWH edits. Read datasets/combined_bags_camera_ray_heavy_review_v4/REPORT.md before interpreting metrics.' >&2
"${PYTHON:-python}" scripts/prepare_heavy_annotation_review.py --verify \
  --root "$ROOT" --output "$REVIEW" \
  --source-extents "$PROJECT_DIR/datasets/combined_bags_camera_ray_dedup_v2/train_yolo_extents_v1.json" \
  --calibration "$PROJECT_DIR/configs/calibration_baselink.json"
if [[ "${VERIFY_ONLY:-0}" == 1 ]]; then exit 0; fi
exec bash scripts/launch_v8_heading_heavy.sh "$@"
