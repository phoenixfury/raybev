#!/usr/bin/env bash
set -euo pipefail

# Paths and batch sizes can be customized using environment variables.
ROOT="${ROOT:-/home/mustafa/mmdetection3d/data/raybev_4bags_native_800x640_v2}"
OUT="${OUT:-work_dirs/raybev_v4_r18_axis_48m_clidar_2x3080ti}"
HEADING_TEACHER="${HEADING_TEACHER:-teacher/raybev_heading_teacher_train.json}"
BOX_FRAME="${BOX_FRAME:-CLIDAR}" # Source DetZero/ROSA labels; network targets use BASELINK.
BATCH_SIZE="${BATCH_SIZE:-1}"       # per GPU microbatch
ACCUM_STEPS="${ACCUM_STEPS:-4}"     # 2*1*4 = 8 effective images/optimizer step
WORKERS="${WORKERS:-4}"             # per GPU => 8 total worker processes
EPOCHS="${EPOCHS:-24}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ ! -f "$HEADING_TEACHER" ]]; then
  echo "Missing TRAIN-only ROSA heading cache: $HEADING_TEACHER" >&2
  echo "Copy the generated teacher JSON with this project, or prepare it locally; see TASK_V4.md. Validation caches are forbidden." >&2
  exit 1
fi

# Extra args pass through. Corrected loss recipe requires a fresh run, no resume.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_combined_rosa_ddp.py \
  --root "$ROOT" --box-frame "$BOX_FRAME" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --accum-steps "$ACCUM_STEPS" --workers "$WORKERS" \
  --backbone resnet18 --heading-teacher "$HEADING_TEACHER" \
  --amp --progress always --out "$OUT" "$@"
