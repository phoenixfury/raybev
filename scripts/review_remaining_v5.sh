#!/usr/bin/env bash
# Review only the unresolved audit flags against the already corrected v5 release.
set -euo pipefail
PROJECT_DIR="$(realpath "$(dirname "$0")/..")"
cd "$PROJECT_DIR"
exec "${PYTHON:-python}" scripts/review_heavy_annotations.py \
  --remaining-issues \
  --root "${ROOT:-$PROJECT_DIR/../../data/Custom/combined_bags_camera_ray_dedup_v2}" \
  "$@"
