#!/usr/bin/env bash
# Fresh, controlled backbone x objective experiments on the combined split.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
EXPERIMENT="${EXPERIMENT:-medium_fixes}"
case "$EXPERIMENT" in
  nano_control|nano_fixes) BACKBONE=yolo26n; DEFAULT_WEIGHTS="$PROJECT_DIR/weights/yolo26n.pt" ;;
  medium_control|medium_fixes) BACKBONE=yolo26m; DEFAULT_WEIGHTS="$PROJECT_DIR/../../checkpoints/yolo/yolo26m.pt" ;;
  *) echo "EXPERIMENT must be nano_control, nano_fixes, medium_control or medium_fixes" >&2; exit 1 ;;
esac
ROOT="${ROOT:-$PROJECT_DIR/../../data/Custom/combined_bags_camera_ray_dedup_v2}"
TRAIN_ANN="${TRAIN_ANN:-$ROOT/annotations/fisheye3dod_infos_train.pkl}"
VAL_ANN="${VAL_ANN:-$ROOT/annotations/fisheye3dod_infos_val.pkl}"
WEIGHTS="${YOLO26_WEIGHTS:-$DEFAULT_WEIGHTS}"
EXTENTS="${EXTENT_ANNOTATIONS:-$PROJECT_DIR/datasets/combined_bags_camera_ray_dedup_v2/train_yolo_extents_v1.json}"
OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v8_$EXPERIMENT}"
OPTIONS=()
if [[ "$EXPERIMENT" == *_fixes ]]; then
  OPTIONS+=(--matched-class-weight "${MATCHED_CLASS_WEIGHT:-1}" \
    --quality-background-weight "${QUALITY_BACKGROUND_WEIGHT:-0.25}" \
    --heading-hard-weight "${HEADING_HARD_WEIGHT:-2}" \
    --facing-binary-weight "${FACING_BINARY_WEIGHT:-1}" \
    --facing-label-smoothing "${FACING_LABEL_SMOOTHING:-0.05}")
fi
for file in "$TRAIN_ANN" "$VAL_ANN" "$EXTENTS" "$WEIGHTS"; do
  [[ -f "$file" ]] || { echo "Missing experiment input: $file" >&2; exit 1; }
done
NPROC="${NPROC_PER_NODE:-2}"
MICROBATCH="${BATCH_SIZE:-1}"
GLOBAL_BATCH="${GLOBAL_BATCH_SIZE:-32}"
if [[ -n "${ACCUM_STEPS:-}" ]]; then
  ACCUMULATION="$ACCUM_STEPS"
else
  (( GLOBAL_BATCH % (NPROC * MICROBATCH) == 0 )) || {
    echo "GLOBAL_BATCH_SIZE must be divisible by NPROC_PER_NODE * BATCH_SIZE" >&2; exit 1;
  }
  ACCUMULATION=$((GLOBAL_BATCH / (NPROC * MICROBATCH)))
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
COMMAND=(torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" scripts/train_yolo_rosa_sparse.py \
  --architecture-version 5 --backbone "$BACKBONE" --backbone-weights "$WEIGHTS" \
  --root "$ROOT" --train-ann "$TRAIN_ANN" --val-ann "$VAL_ANN" \
  --calibration "$PROJECT_DIR/configs/calibration_baselink.json" \
  --resize-native-images --stride4 --observation-conditioning \
  --queries 225 --channels 96 --stages 4 --proposal-queries 64 \
  --ground-z 0 --facing-cosine-prior 0 \
  --extent-annotations "$EXTENTS" --extent-weight 1 \
  --epochs "${EPOCHS:-64}" --batch-size "$MICROBATCH" --accum-steps "$ACCUMULATION" \
  --workers "${WORKERS:-4}" --lr "${LR:-0.0002}" --backbone-lr-mult .1 \
  --sampling track --seed 3407 --amp --progress always --out "$OUT" "${OPTIONS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"; printf '\n'
else
  exec "${COMMAND[@]}"
fi
