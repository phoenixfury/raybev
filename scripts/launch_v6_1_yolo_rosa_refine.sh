#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
DATA_ROOT="$PROJECT_DIR/../../data/raybev_4bags_native_800x640_v2"
if [[ ! -d "$DATA_ROOT" ]]; then DATA_ROOT="$PROJECT_DIR/../../data/Custom/raybev_4bags_native_800x640_v2"; fi
ROOT="${ROOT:-$DATA_ROOT}"
OUT="${OUT:-work_dirs/raybev_v6_1_yolo_rosa_refine_48m_clidar_b16a1}"
TEACHER="${HEADING_TEACHER:-teacher/raybev_heading_teacher_train_reviewv4.json}"
WEIGHTS="${YOLO26_WEIGHTS:-weights/yolo26n.pt}"
REVISION="$PROJECT_DIR/datasets/heavy_review_v4"
if [[ ! -f "$TEACHER" ]]; then echo 'Missing reviewed TRAIN teacher cache' >&2; exit 1; fi
RESUMING=0
CLI_INITIALIZING=0
for arg in "$@"; do
  case "$arg" in
    --resume|--resume=*) RESUMING=1 ;;
    --init-from|--init-from=*) CLI_INITIALIZING=1 ;;
    --architecture-version|--architecture-version=*|--no-pretrained|--backbone-weights|--backbone-weights=*)
      echo 'This launcher fixes architecture 4 and verified initialization; use INIT_FROM or YOLO26_WEIGHTS.' >&2; exit 2 ;;
  esac
done
INITIALIZATION=()
if [[ "$RESUMING" == 0 ]]; then
  if [[ -n "${INIT_FROM:-}" ]]; then
    if [[ "$CLI_INITIALIZING" == 1 ]]; then echo 'Choose INIT_FROM or --init-from, not both' >&2; exit 2; fi
    if [[ ! -f "$INIT_FROM" ]]; then echo 'Missing sparse warm-start checkpoint' >&2; exit 1; fi
    INITIALIZATION=(--init-from "$INIT_FROM")
  elif [[ "$CLI_INITIALIZING" == 0 ]]; then
    if [[ ! -f "$WEIGHTS" ]]; then echo 'Copy verified official weights/yolo26n.pt before fresh training' >&2; exit 1; fi
    INITIALIZATION=(--backbone-weights "$WEIGHTS")
  fi
fi
python scripts/prepare_label_revision.py --root "$ROOT" --output "$REVISION" --verify
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_yolo_rosa_sparse.py \
  --architecture-version 4 --root "$ROOT" --train-ann "$REVISION/train.pkl" --val-ann "$REVISION/val.pkl" \
  --out "$OUT" --heading-teacher "$TEACHER" "${INITIALIZATION[@]}" \
  --epochs "${EPOCHS:-96}" --batch-size "${BATCH_SIZE:-16}" --accum-steps "${ACCUM_STEPS:-1}" \
  --workers "${WORKERS:-4}" --lr "${LR:-0.0002}" --amp --progress always "$@"
