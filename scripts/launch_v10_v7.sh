#!/usr/bin/env bash
# Frozen v7 data, with separate outputs from the previous v5 experiments.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
export REVIEW_DIR="${REVIEW_DIR:-$PROJECT_DIR/datasets/combined_bags_camera_ray_manual_review_v7}"
export AUDIT_DIR="${AUDIT_DIR:-$PROJECT_DIR/analysis/v10_manual_review_v7}"
export OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v10_medium_v7_${V10_ARM:-control}}"
export V10_ARM="${V10_ARM:-control}"
if [[ ! -f "$AUDIT_DIR/input_contract.json" ]]; then
  echo 'Missing V7 training preparation. Run prepare_v10_experiment.py with --release datasets/combined_bags_camera_ray_manual_review_v7 --source datasets/combined_bags_camera_ray_manual_review_v6 --output analysis/v10_manual_review_v7 (and --root pointing to the images).' >&2
  exit 1
fi
exec bash "$PROJECT_DIR/scripts/launch_v10_controlled.sh" "$@"
