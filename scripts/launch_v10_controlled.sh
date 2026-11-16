#!/usr/bin/env bash
# Fresh COCO-medium backbone; every arm differs from control in one stated factor.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
REVIEW="${REVIEW_DIR:-$PROJECT_DIR/datasets/combined_bags_camera_ray_manual_review_v5}"
AUDIT="${AUDIT_DIR:-$PROJECT_DIR/analysis/v10_manual_review_v5}"
ROOT="${ROOT:-$PROJECT_DIR/../../data/Custom/combined_bags_camera_ray_dedup_v2}"
ARM="${V10_ARM:-shapes}"
OPTIONS=()
case "$ARM" in
  control) ;;
  shapes) OPTIONS+=(--shape-balanced-anchors --shape-balanced-proposals) ;;
  anchors) OPTIONS+=(--shape-balanced-anchors) ;;
  proposals) OPTIONS+=(--shape-balanced-proposals) ;;
  foreground_ce) OPTIONS+=(--matched-class-weight 1) ;;
  foreground_balanced) OPTIONS+=(--matched-class-weight 1 --foreground-balance) ;;
  objectness) OPTIONS+=(--semantic-objectness) ;;
  objectness_balanced) OPTIONS+=(--semantic-objectness --foreground-balance) ;;
  context) OPTIONS+=(--object-aligned-context) ;;
  early) OPTIONS+=(--early-stage-weight 2) ;;
  matching) OPTIONS+=(--match-class-start .05 --match-warmup-epochs 3 --match-ramp-epochs 5) ;;
  endpoint)
    [[ -n "${GEOMETRY_APPROVALS:-}" ]] || { echo 'Endpoint arm blocked: provide independently reviewed GEOMETRY_APPROVALS, not class decisions.' >&2; exit 1; }
    OPTIONS+=(--heavy-endpoint-weight .25 --geometry-approvals "$GEOMETRY_APPROVALS") ;;
  *) echo "Unknown V10_ARM=$ARM" >&2; exit 1 ;;
esac
for option in "$@"; do
  case "$option" in --resume*|--init-from*|--out*|--train-ann*|--val-ann*)
    echo 'Controlled launcher forbids resume/warm-start/data/output overrides in arguments. Use a new OUT/REVIEW_DIR and re-prepare its audit.' >&2; exit 1 ;;
  esac
done
"${PYTHON:-python}" - "$REVIEW" "$AUDIT" "$PROJECT_DIR/configs/calibration_baselink.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
review,audit,calibration=map(Path,sys.argv[1:])
contract=json.loads((audit/'input_contract.json').read_text())
for name,expected in contract['files'].items():
    if hashlib.sha256((review/name).read_bytes()).hexdigest()!=expected:raise SystemExit('Frozen V5 release changed: '+name)
if hashlib.sha256(calibration.read_bytes()).hexdigest()!=contract['calibration_sha256']:raise SystemExit('Calibration changed')
if contract['track_class_conflicts'] and os.environ.get('ALLOW_TRACK_CONFLICTS')!='1':
    raise SystemExit('Two mixed-class bag/track IDs remain. Review the audit, then resolve them or explicitly acknowledge with ALLOW_TRACK_CONFLICTS=1. No labels were auto-fixed.')
if contract['geometry_flags'] and os.environ.get('ALLOW_GEOMETRY_FLAGS')!='1':
    raise SystemExit('Class/dimension flags remain, including vehicle-sized Pedestrian/Van boxes. Resolve the follow-up queue or explicitly acknowledge with ALLOW_GEOMETRY_FLAGS=1. This never enables unapproved endpoint loss.')
PY
export ROOT TRAIN_ANN="$REVIEW/fisheye3dod_infos_train.pkl" VAL_ANN="$REVIEW/fisheye3dod_infos_val.pkl"
export EXTENT_ANNOTATIONS="$REVIEW/train_yolo_extents_v1.json"
export OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v10_medium_v5_$ARM}"
export EXPERIMENT=medium_control BATCH_SIZE="${BATCH_SIZE:-4}" ACCUM_STEPS="${ACCUM_STEPS:-2}" NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
exec bash scripts/launch_v8_heading_heavy.sh --unit-class-weights --proposal-cell-targets --heavy-query-audit \
  --stage-diagnostics --train-audit-every 50 --tram-tracks "$AUDIT/tram_tracks.json" "${OPTIONS[@]}" "$@"
