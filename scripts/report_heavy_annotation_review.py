#!/usr/bin/env python3
"""Summarize the completed class review without claiming cuboid certification."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from prepare_heavy_annotation_review import dump, sha, verify_release
from raybev.config import DEFAULT_CALIBRATION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--audit', type=Path, required=True)
    p.add_argument('--source-extents', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=Path(DEFAULT_CALIBRATION))
    p.add_argument('--allow-missing-source-extents', action='store_true')
    a = p.parse_args(); output = a.review.resolve()
    verified = verify_release(a.root.resolve(), output, a.source_extents.resolve(), a.calibration.resolve(),
                              allow_missing_source_extents=a.allow_missing_source_extents)
    manifest = json.loads((output / 'correction_manifest.json').read_text())
    records = [json.loads(line) for line in (output / 'annotation_review.jsonl').read_text().splitlines()]
    tracks = json.loads((a.audit / 'index.json').read_text())['tracks']
    by_status = Counter(r['status'] for r in records)
    track_statuses = Counter()
    for group in ({r['track_number']: r for r in records}).values():
        track_statuses[group['status']] += 1
    flags = []; flagged_counts = Counter()
    for r in records:
        reasons = []
        if 'new_dimensions_lwh' in r:
            reasons.append('user_prescribed_dimensions_center_yaw_not_certified')
        if r['lidar']['points_in_box'] == 0:
            reasons.append('zero_lidar_points')
        if r['lidar']['points_in_box'] < 10:
            reasons.append('fewer_than_10_lidar_points')
        if abs(r['bottom_z']) > 1:
            reasons.append('bottom_z_outside_plus_minus_1m')
        if not r['views']:
            reasons.append('no_eligible_camera_view')
        if r['status'] in ('unresolved', 'taxonomy_policy'):
            reasons.append(r['status'])
        if reasons:
            flags.append(dict(id=r['id'], split=r['split'], sequence=r['sequence'], track_id=r['track_id'],
                token=r['token'], index=r['index'], label=r['new_label'], flags=';'.join(reasons),
                source_points_in_box=r['lidar']['points_in_box'], source_bottom_z=r['bottom_z'], page=r['all_frames_page']))
            flagged_counts.update(reasons)
    with (output / 'quality_flags.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['id', 'split', 'sequence', 'track_id', 'token', 'index', 'label', 'flags', 'source_points_in_box', 'source_bottom_z', 'page'])
        writer.writeheader(); writer.writerows(flags)
    # These are the actual generated candidate pages inspected in the second pass.
    reviewed_pages = output.parent / 'combined_bags_camera_ray_heavy_review_v1/second_pass_pages'
    page_hashes = {f'all_{i:03d}.jpg': sha(reviewed_pages / f'all_{i:03d}.jpg') for i in range(1, 115)}
    sweep = dict(visual_review=json.loads((a.audit / 'second_pass_review.json').read_text()),
        inspected_candidate_page_sha256=page_hashes, released_annotation_sha256={s: d['output_sha256'] for s, d in manifest['splits'].items()},
        release_extent_sha256=manifest['extents']['output_sha256'], review_statuses=dict(by_status),
        track_review_statuses=dict(track_statuses), potential_quality_flags=dict(flagged_counts),
        note='Point counts and bottom Z describe original source cuboids, not the nine resized boxes. Flags are review candidates, not automatic proof of an invalid box. No flags cause deletion/ignore/relabel.')
    dump(output / 'second_pass_report.json', sweep)
    table = ['| Original class | Source boxes | Supported same class | Corrected class | Unresolved | Taxonomy pending |',
             '|---|---:|---:|---:|---:|---:|']
    for label in ('bus', 'truck', 'van'):
        subset = [r for r in records if r['old_label'] == label]; count = Counter(r['status'] for r in subset)
        table.append(f"| {label.title()} | {len(subset):,} | {count['supported_class']:,} | {count['corrected_class']:,} | {count['unresolved']:,} | {count['taxonomy_policy']:,} |")
    transitions = Counter((r['old_label'].title(), r['new_label']) for r in records if r['status'] == 'corrected_class')
    transitions_md = '\n'.join(f'- {old} → {new}: {n}' for (old, new), n in sorted(transitions.items()))
    final_counts = ['| Split | Car | Truck | Bus | Van |', '|---|---:|---:|---:|---:|']
    for split, info in manifest['splits'].items():
        c = info['after']; final_counts.append(f"| {split.upper()} | {c['Car']:,} | {c['Truck']:,} | {c['Bus']:,} | {c['Van']:,} |")
    text = f'''# Heavy-vehicle annotation review — 2026-11-16, release v4

## Result and limits

Applied **{by_status['corrected_class']:,} class corrections** across TRAIN and VAL,
including the user's nine confirmed passenger-car annotations. Original files
were not overwritten. The nine user-confirmed cars now have user-prescribed
length × width × height **4.0 × 1.8 × 1.5 m**. No other boxes were resized;
none were deleted, ignored, moved or rotated. Centers, yaw, track IDs, image
paths, frame order and split membership are unchanged.

**This is a partial correction, not a certified clean dataset.**
{by_status['unresolved']:,} annotations remain unresolved and {by_status['taxonomy_policy']}
require a taxonomy decision. These labels remain in training/evaluation, with their
original classes; they have not been silently removed to improve metrics.
Original Car/Pedestrian/Cyclist labels were not visually audited. Headings and 3D
cuboid accuracy are not certified by this review.

## Coverage and class findings

All **4,555** original Bus/Truck/Van annotations (4,180 TRAIN, 375 VAL), across
**486 track IDs**, were inventoried. First pass: all 41 representative-track
sheets. Second pass: **all 114 generated candidate sheets**, one tile per source
annotation, with temporal context and LiDAR point-count diagnostics. Corrections
were revised when full sequences contradicted a representative crop. Individual
occluded frames are not independently identifiable just because track identity
is supported elsewhere. Counts below describe those reviewed track identities,
not an independent high-resolution, four-camera cuboid certification per frame.

{chr(10).join(table)}

Supported identities: {by_status['supported_class']:,} source boxes. Applied
corrections: {by_status['corrected_class']:,}. Unresolved + taxonomy pending:
{by_status['unresolved'] + by_status['taxonomy_policy']:,}. Track IDs are not unique
physical vehicles: the same vehicle may have fragmented IDs.

{transitions_md}

Class policy retained: passenger trams → Bus; minibuses and ambulances → Van;
separate cab/cargo body, flatbed, tanker and articulated cargo vehicles → Truck;
SUV/crossover → Car. Existing construction→Van policy was not broadened, and
pickup mapping was not guessed. See `unresolved.csv` for the 35 construction/pickup
policy entries and other unresolved cases.

## User-confirmed cars

Bag `11_09_09_40_48`: Bus track `1516`, **4 TRAIN boxes**, and Van track `1748`,
**5 TRAIN boxes**, are all now `Car` with mesh `detzero.car` (A4172–A4180).
All nine now use **LWH 4.0 × 1.8 × 1.5 m**, as explicitly requested by the user.
The original centers and yaw are preserved, including Z-center: changing height
with fixed center changes the box bottom. These are prescribed car sizes, not a
claim of calibrated cuboid fitting. The manifest lists all nine old/new sizes.
Quality point-count/bottom-Z diagnostics refer to the original source cuboids;
they must not be interpreted as measurements of the new nine cuboids.
The refreshed sheet is `second_pass_pages/all_114.jpg`. Old audit/source sheets
still show old labels for provenance; they are not the new training inputs.

## YOLO weak extents

`train_yolo_extents_v1.json` is synchronized to the revised TRAIN pickle:

- All **{manifest['extents']['records']:,} records** validated against runtime `VerifiedExtents`.
- **{manifest['extents']['changed_gt_records']} linked GT class IDs** changed. Not every corrected 3D box has a
  retained weak YOLO extent; association/confidence/occlusion gates leave only
  this subset in the extent file.
- Measured 2D boxes, camera indices, edge-valid masks, occlusion/truncation flags,
  weights and original YOLO/manifest evidence hashes are unchanged.
- TRAIN fingerprint is rebuilt. Never pair this JSON with the old TRAIN pickle,
  or the old JSON with this TRAIN pickle.
- None of the nine resized cars has a retained extent record. Existing 2D
  associations remain geometrically valid; the migration rejects size changes
  on an associated box rather than reusing stale association evidence.
- Detector-cache predictions are preserved as evidence, not relabeled to pretend
  the detector originally predicted the corrected classes. No detector rerun needed.

New TRAIN fingerprint: `{manifest['extents']['new_training_boxes_sha256']}`.
These remain **weak** 2D boundaries, not human-traced ground truth.

## Other problems exposed by the sweep

The following point-count and bottom-Z numbers describe **original source
geometry**, before the nine authorized size edits.

- **33 TRAIN heavy boxes contain zero LiDAR points**; **129** heavy boxes have
  fewer than 10 points (126 TRAIN / 3 VAL). Sparse support is a flag, not proof
  of non-existence: occlusion and range can explain it.
- **280** heavy boxes have bottom Z outside ±1 m in BASELINK (265 TRAIN / 15 VAL).
  Road slope/calibration can contribute; inspect before editing heights.
- `09_25_13_28_18` Truck IDs `4179/4180/4182` project around the ego test car and
  sky/building, with no identifiable truck in the chosen views. They need full
  calibrated camera/LiDAR association review, not a guessed Car relabel.
- Some tracks contain competing visible identities. Example: Van `2932` in
  `09_17_08_33_47` (71 TRAIN boxes) has white-van versus tractor/occluder evidence.
- The validation Bus labels are only **23 frames**: 16 are one passenger tram
  (`09_17_08_33_47/2894`), 1 is a visible coach, and 6 remain occluded/unresolved.
  A correct class taxonomy does not establish correct heading/size for these boxes.
- Validation Van track `11_09_09_00_30/1400` contributes **117 frames**, over half
  of the revised VAL Van count (225). This is not broad vehicle/scene coverage.

`quality_flags.csv` contains exact frame/index addresses for those flags.
Do not use the reviewed VAL labels to tune class rules until metrics look good;
keep this revision fixed and publish its manifest with the experiment.

## Files and restart

Use this directory as an annotation override; **keep the image ROOT unchanged**:

```
datasets/combined_bags_camera_ray_heavy_review_v4/
  fisheye3dod_infos_train.pkl
  fisheye3dod_infos_val.pkl
  train_yolo_extents_v1.json
  correction_manifest.json
  annotation_review.csv / annotation_review.jsonl
  track_findings.json / unresolved.csv / quality_flags.csv
  verification_report.json / second_pass_report.json
  second_pass_pages/
```

On the training machine copy this directory under its RayBEV project directory
(e.g. `/home/mustafa/mmdetection3d/projects/raybev/datasets/combined_bags_camera_ray_heavy_review_v4/`
if that is its checkout). Copy the new `scripts/launch_v8_heavy_review.sh` and
`scripts/prepare_heavy_annotation_review.py` too. Existing image/source data and
source annotation pickles must remain available. The updated launcher uses
`--verify --release-only`: no original extent JSON is required. It still enforces
the v4 JSON hash and runtime GT/calibration checks, and explicitly records that
the historical extent comparison was not requested. Full historical comparison
is available by verifying with `--source-extents` instead of `--release-only`.

From the RayBEV project directory, in the existing training Python environment:

```bash
VERIFY_ONLY=1 bash scripts/launch_v8_heavy_review.sh
DRY_RUN=1 bash scripts/launch_v8_heavy_review.sh
bash scripts/launch_v8_heavy_review.sh
```

Default: medium backbone, fixes objective, microbatch **4 per GPU**, accumulation
**2**, **2 GPUs**, effective batch **16**, 64 epochs, new output directory
`work_dirs/raybev_v8_medium_heavy_review_v4`. No ROSA teacher and no old RayBEV
checkpoint is resumed. Backbone initialization is the existing local COCO weights.
Use `ROOT=/actual/image/root` if the image data path differs. Training was **not** launched.
The loss/backbone settings are unchanged; the experiment changes reviewed labels
and the nine explicitly requested sizes.
Compare against the old checkpoint only after reevaluating it on this same fixed
VAL revision; its old validation metrics are not directly comparable.

For a genuinely clean next benchmark, resolve `unresolved.csv` with calibrated
full-camera + LiDAR cuboid review; settle construction/pickup policy, then audit
the original Car class for missing heavy vehicles. Refit/verify geometry and
directed heading separately. Keep difficult VAL objects; do not delete unknowns
to manufacture a clean score.

## Verification

Post-write consistency sweep checked **{verified['total_boxes_checked']:,} boxes**
in all **10,993** TRAIN/VAL frames, including non-heavy annotations for unintended
changes, and every YOLO extent record. Source and release hashes are recorded;
only the nine authorized LWH edits differ in geometry; centers, yaw, identities,
splits and non-target fields match the originals.
TRAIN/VAL frame-token sets are disjoint. This is data integrity evidence, not a
claim that every visual annotation is spot on. Targeted regression tests passed.

{chr(10).join(final_counts)}
'''
    (output / 'REPORT.md').write_text(text)
    print(json.dumps(dict(statuses=dict(by_status), track_statuses=dict(track_statuses), quality_flags=dict(flagged_counts)), indent=2))


if __name__ == '__main__':
    main()
