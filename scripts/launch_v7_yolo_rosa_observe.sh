#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
DATA_ROOT="$PROJECT_DIR/../../data/Custom/combined_bags"
ROOT="${ROOT:-$DATA_ROOT}"
TRAIN_ANN="${TRAIN_ANN:-$ROOT/annotations/fisheye3dod_infos_train.pkl}"
VAL_ANN="${VAL_ANN:-$ROOT/annotations/fisheye3dod_infos_val.pkl}"
WEIGHTS="${YOLO26_WEIGHTS:-weights/yolo26n.pt}"
OUT="${OUT:-work_dirs/raybev_v7_yolo_rosa_observe_q225_c96_p2_8bags_48m_clidar}"
EXTENTS="${EXTENT_ANNOTATIONS:-datasets/combined_8bags/train_yolo_extents_v1.json}"
OPTIONS=()
if [[ "${STRIDE4:-1}" == 1 ]]; then OPTIONS+=(--stride4); fi
if [[ "${OBSERVATION_CONDITIONING:-1}" == 1 ]]; then OPTIONS+=(--observation-conditioning); fi
if [[ "${MULTI_CAMERA_FACING:-1}" == 0 ]]; then OPTIONS+=(--no-multi-camera-facing); fi
if [[ "${EXTENT_WEIGHT:-1}" != 0 ]]; then
  if [[ ! -f "$EXTENTS" ]]; then echo "Missing TRAIN extent annotations: $EXTENTS" >&2; exit 1; fi
  OPTIONS+=(--extent-annotations "$EXTENTS" --extent-weight "${EXTENT_WEIGHT:-1}")
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# Fresh decoder/neck/proposals; verified pretrained YOLO backbone. Architecture
# 4 checkpoints cannot resume into changed widths, query counts and objectives.
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_yolo_rosa_sparse.py \
  --architecture-version 5 --root "$ROOT" --train-ann "$TRAIN_ANN" --val-ann "$VAL_ANN" \
  --expected-sequences 8 --resize-native-images \
  --queries 225 --channels "${CHANNELS:-96}" --stages 4 --proposal-queries "${PROPOSAL_QUERIES:-64}" \
  --ground-z "${GROUND_Z:-0}" --facing-cosine-prior "${FACING_COSINE_PRIOR:-0}" \
  --out "$OUT" --backbone-weights "$WEIGHTS" "${OPTIONS[@]}" \
  --epochs "${EPOCHS:-96}" --batch-size "${BATCH_SIZE:-4}" --accum-steps "${ACCUM_STEPS:-4}" \
  --workers "${WORKERS:-4}" --lr "${LR:-0.0002}" --amp --progress always "$@"
