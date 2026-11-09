#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
MODE="${1:-early}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in early|none) ;; *) echo 'Usage: launch_v4_4_review_experiment.sh [early|none] [trainer args]' >&2; exit 2;; esac
export ROOT="${ROOT:-/home/mustafa/mmdetection3d/data/raybev_4bags_native_800x640_v2}"
export OUT="${OUT:-work_dirs/raybev_v4_4_r18_${MODE}_reviewv1_48m_clidar_2x3080ti}"
export EPOCHS="${EPOCHS:-6}"
export LR="${LR:-0.0001}"
# Match the actual v4.3 run. Batch 1 + accumulation 4 is NOT equivalent for image BN.
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
REVISION="$PROJECT_DIR/datasets/heavy_review_v1"
python scripts/prepare_label_revision.py --root "$ROOT" --output "$REVISION" --verify
bash scripts/launch_2x3080ti.sh \
  --init-from work_dirs/raybev_v4_3_r18_geometry_48m_clidar_2x3080ti/best_map.pt \
  --train-ann "$REVISION/train.pkl" --val-ann "$REVISION/val.pkl" \
  --height-fusion concat --freeze-image-stages "$MODE" --warmup-steps 200 "$@"
