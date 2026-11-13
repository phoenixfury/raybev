#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
REVIEW_DIR="$PROJECT_DIR/datasets/combined_bags_bus_van_review_v2"
export TRAIN_ANN="${TRAIN_ANN:-$REVIEW_DIR/fisheye3dod_infos_train.pkl}"
export VAL_ANN="${VAL_ANN:-$REVIEW_DIR/fisheye3dod_infos_val.pkl}"
export EXTENT_ANNOTATIONS="${EXTENT_ANNOTATIONS:-$REVIEW_DIR/train_yolo_extents_v1.json}"
export OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v7_1_yolo_rosa_bus_van_review_v2}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
for file in "$TRAIN_ANN" "$VAL_ANN" "$EXTENT_ANNOTATIONS"; do
  [[ -f "$file" ]] || { echo "Missing reviewed data: $file" >&2; exit 1; }
done
# New optimizer/schedule and fresh detection network with pretrained YOLO.
# --resume is only valid later for this same corrected-data experiment.
exec bash "$PROJECT_DIR/scripts/launch_v7_yolo_rosa_observe.sh" "$@"
