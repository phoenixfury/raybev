#!/usr/bin/env bash
set -euo pipefail

# Paths and batch sizes can be customized using environment variables.
ROOT="${ROOT:-/home/mustafa/mmdetection3d/data/raybev_4bags_native_800x640_v2}"
OUT="${OUT:-work_dirs/raybev_v2_48m_clidar_2x3080ti}"
BOX_FRAME="${BOX_FRAME:-CLIDAR}" # Source DetZero/ROSA labels; network targets use BASELINK.
BATCH_SIZE="${BATCH_SIZE:-1}"       # per GPU microbatch
ACCUM_STEPS="${ACCUM_STEPS:-4}"     # 2*1*4 = 8 effective images/optimizer step
WORKERS="${WORKERS:-4}"             # per GPU => 8 total worker processes
EPOCHS="${EPOCHS:-24}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

# Extra args pass through. Resume only a corrected CLIDAR run with identical calibration.
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
  scripts/train_combined_rosa_ddp.py \
  --root "$ROOT" --box-frame "$BOX_FRAME" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
  --accum-steps "$ACCUM_STEPS" --workers "$WORKERS" \
  --amp --progress always --out "$OUT" "$@"
