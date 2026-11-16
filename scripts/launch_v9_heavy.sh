#!/usr/bin/env bash
# V9 experiment on the same reviewed split and hardware budget as V8.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
export OUT="${OUT:-$PROJECT_DIR/work_dirs/raybev_v9_medium_heavy_${V9_ARM:-full}}"
export EXPERIMENT="${EXPERIMENT:-medium_fixes}"
ARGS=(--proposal-cell-targets --heavy-query-audit)
case "${V9_ARM:-full}" in
  proposal) ;;
  matching) ARGS+=(--match-class-start .05 --match-warmup-epochs 3 --match-ramp-epochs 5) ;;
  resample) ARGS+=(--fresh-box-classification) ;;
  full) ARGS+=(--match-class-start .05 --match-warmup-epochs 3 --match-ramp-epochs 5
              --fresh-box-classification --independent-facing-views --quality-distance-target
              --quality-hard-negative-weight .2) ;;
  *) echo 'V9_ARM must be proposal, matching, resample, or full' >&2; exit 1 ;;
esac
if [[ -n "${INIT_FROM:-}" ]]; then ARGS+=(--init-from "$INIT_FROM"); fi
exec bash scripts/launch_v8_heavy_review.sh "${ARGS[@]}" "$@"
