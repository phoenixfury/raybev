#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
DATA_ROOT="$PROJECT_DIR/../../data/raybev_4bags_native_800x640_v2"
if [[ ! -d "$DATA_ROOT" ]]; then DATA_ROOT="$PROJECT_DIR/../../data/Custom/raybev_4bags_native_800x640_v2"; fi
ROOT="${ROOT:-$DATA_ROOT}"
OUT="${OUT:-work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti}"
WEIGHTS="${YOLO26_WEIGHTS:-weights/yolo26n.pt}"
REVISION="$PROJECT_DIR/datasets/heavy_review_v4"
RESUMING=0
for arg in "$@"; do case "$arg" in --resume|--resume=*) RESUMING=1 ;; esac; done
INITIALIZATION=()
if [[ "$RESUMING" == 0 ]]; then
  if [[ ! -f "$WEIGHTS" ]]; then echo 'Copy verified official weights/yolo26n.pt before training' >&2; exit 1; fi
  INITIALIZATION=(--backbone-weights "$WEIGHTS")
fi
python scripts/prepare_label_revision.py --root "$ROOT" --output "$REVISION" --verify
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_yolo_rosa_sparse.py \
  --root "$ROOT" --train-ann "$REVISION/train.pkl" --val-ann "$REVISION/val.pkl" \
  --out "$OUT" "${INITIALIZATION[@]}" \
  --epochs "${EPOCHS:-24}" --batch-size "${BATCH_SIZE:-2}" --accum-steps "${ACCUM_STEPS:-2}" \
  --workers "${WORKERS:-4}" --lr "${LR:-0.0002}" --amp --progress always "$@"
