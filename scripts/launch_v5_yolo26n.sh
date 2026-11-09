#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
# Separate output and fresh initialization; the current R18 run is untouched.
DATA_ROOT="$PROJECT_DIR/../../data/raybev_4bags_native_800x640_v2"
if [[ ! -d "$DATA_ROOT" ]]; then DATA_ROOT="$PROJECT_DIR/../../data/Custom/raybev_4bags_native_800x640_v2"; fi
export ROOT="${ROOT:-$DATA_ROOT}"
export OUT="${OUT:-work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti}"
export EPOCHS="${EPOCHS:-24}"
export LR="${LR:-0.0002}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export ACCUM_STEPS="${ACCUM_STEPS:-1}"
WEIGHTS="${YOLO26_WEIGHTS:-$PROJECT_DIR/weights/yolo26n.pt}"
RESUMING=0
# Refuse incompatible R18 recovery or accidental random initialization.
for arg in "$@"; do
  case "$arg" in
    --resume|--resume=*) RESUMING=1 ;;
    --init-from|--init-from=*|--no-pretrained|--backbone|--backbone=*|--backbone-weights|--backbone-weights=*)
      echo 'YOLO launcher fixes the pretrained backbone; use YOLO26_WEIGHTS for a local official checkpoint. No R18 recovery.' >&2
      exit 2 ;;
  esac
done
if [[ ! -f "$WEIGHTS" && "$RESUMING" == 0 ]]; then
  echo 'Missing YOLO26n weights. Run: python scripts/prepare_yolo26n.py; copy weights/yolo26n.pt to offline servers.' >&2
  exit 1
fi
REVISION="$PROJECT_DIR/datasets/heavy_review_v1"
python scripts/prepare_label_revision.py --root "$ROOT" --output "$REVISION" --verify
bash scripts/launch_2x3080ti.sh \
  --backbone yolo26n --backbone-weights "$WEIGHTS" \
  --train-ann "$REVISION/train.pkl" --val-ann "$REVISION/val.pkl" \
  --height-fusion concat --freeze-image-stages none "$@"
