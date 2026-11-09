#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
MODE="${1:-heading}"
if [[ $# -gt 0 ]]; then shift; fi
case "$MODE" in
  control) FACING=unit_vector ;;
  heading) FACING=circular_bins ;;
  *) echo 'Usage: launch_v5_1_review_heading.sh [control|heading] [trainer arguments]' >&2; exit 2 ;;
esac
DATA_ROOT="$PROJECT_DIR/../../data/raybev_4bags_native_800x640_v2"
if [[ ! -d "$DATA_ROOT" ]]; then DATA_ROOT="$PROJECT_DIR/../../data/Custom/raybev_4bags_native_800x640_v2"; fi
export ROOT="${ROOT:-$DATA_ROOT}"
export OUT="${OUT:-work_dirs/raybev_v5_1_yolo26n_${MODE}_reviewv4_48m_clidar_2x3080ti}"
export EPOCHS="${EPOCHS:-6}"
export LR="${LR:-0.0001}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
export HEADING_TEACHER="${HEADING_TEACHER:-teacher/raybev_heading_teacher_train_reviewv4.json}"
REVISION="$PROJECT_DIR/datasets/heavy_review_v4"
INIT_FROM="${INIT_FROM:-work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti/best_heading.pt}"
INITIALIZATION=(--init-from "$INIT_FROM")
RESUMING=0
for arg in "$@"; do
  case "$arg" in --resume|--resume=*) INITIALIZATION=(); RESUMING=1 ;; esac
done
if [[ ( ! -f "$INIT_FROM" && "$RESUMING" == 0 ) || ! -f "$HEADING_TEACHER" ]]; then
  echo 'Missing v5 epoch-3 best_heading.pt or reviewed TRAIN-only teacher. Copy the supplied files; see TASK_V5_1.md.' >&2
  exit 1
fi
# The annotation fork keeps all images/boxes. Explicit TRAIN-only ignores remove
# uncertain targets AND their dense-background penalties, not validation GT.
python scripts/prepare_label_revision.py --root "$ROOT" --output "$REVISION" --verify
bash scripts/launch_2x3080ti.sh \
  --backbone yolo26n "${INITIALIZATION[@]}" \
  --train-ann "$REVISION/train.pkl" --val-ann "$REVISION/val.pkl" \
  --facing-loss "$FACING" --height-fusion concat \
  --freeze-image-stages early --backbone-lr-mult 0.1 --warmup-steps 200 "$@"
