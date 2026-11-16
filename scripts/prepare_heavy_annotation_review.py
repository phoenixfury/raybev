#!/usr/bin/env python3
"""Build a non-destructive reviewed revision and synchronize weak YOLO extents.

Review findings are manual observations, not automatic size-based class rules.
Unresolved identities, pickup policy and construction vehicles remain unchanged.
This is not a geometry/heading or all-class ground-truth certification.
"""
import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import pickle
import runpy
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from raybev.config import CLASSES, DEFAULT_CALIBRATION
from raybev.rosa_dataset import RosaCombinedDataset, load_annotations, training_boxes_sha256
from raybev.sparse_config import SparseConfig
from raybev.sparse_extent import VerifiedExtents

MAPPING = dict(bus='Bus', tram='Bus', truck='Truck', van='Van', car='Car')


def approved_dimensions(decision):
    dimensions = decision.get('new_dimensions_lwh')
    if dimensions is None:
        return None
    dimensions = np.asarray(dimensions, dtype=float)
    if (not decision.get('geometry_authority') or dimensions.shape != (3,)
            or not np.isfinite(dimensions).all() or (dimensions <= 0).any()):
        raise ValueError('Invalid or unauthorized dimension edit')
    return dimensions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def revise_document(original, decisions):
    """Change only approved names/meshes and explicit sizes; exact addressing."""
    revised = copy.deepcopy(original)
    seen = set()
    for row in revised['data_list']:
        token = str(row['token']); ann = row['annos']
        names = list(map(str, ann['gt_names']))
        meshes = list(map(str, ann['gt_mesh']))
        touched = False
        for i, name in enumerate(names):
            key = (token, i)
            if key not in decisions:
                continue
            d = decisions[key]; seen.add(key)
            if name.lower() != d['old_label'] or str(ann['gt_ids'][i]) != d['track_id']:
                raise ValueError('Annotation identity differs from visual review: ' + str(key))
            if not np.array_equal(np.asarray(ann['gt_boxes'])[i, :7], d['box_CLIDAR']):
                raise ValueError('Reviewed box geometry changed: ' + str(key))
            dimensions = approved_dimensions(d)
            if dimensions is not None:
                ann['gt_boxes'][i, 3:6] = dimensions
            if name.lower() != d['new_label'].lower():
                if meshes[i] != 'detzero.' + name.lower():
                    raise ValueError('Unexpected source mesh: ' + str(key))
                names[i] = d['new_label']; meshes[i] = 'detzero.' + names[i].lower()
                touched = True
        if touched:
            ann['gt_names'] = np.asarray(names, dtype=np.result_type(ann['gt_names'].dtype, 'U10'))
            ann['gt_mesh'] = np.asarray(meshes, dtype=np.result_type(ann['gt_mesh'].dtype, 'U18'))
    if seen != set(decisions):
        raise ValueError('Missing reviewed annotation')
    verify_document(original, revised, decisions)
    return revised


def verify_document(original, revised, decisions):
    if original.keys() != revised.keys() or len(original['data_list']) != len(revised['data_list']):
        raise ValueError('Document/frame structure changed')
    for key in original:
        if key != 'data_list' and pickle.dumps(original[key], 4) != pickle.dumps(revised[key], 4):
            raise ValueError('Document metadata changed')
    checked = 0
    for src, dst in zip(original['data_list'], revised['data_list']):
        if src.keys() != dst.keys() or src['annos'].keys() != dst['annos'].keys():
            raise ValueError('Frame/annotation fields changed')
        token = str(src['token'])
        for key in src:
            if key != 'annos' and pickle.dumps(src[key], 4) != pickle.dumps(dst[key], 4):
                raise ValueError('Frame metadata/image path changed')
        for field, values in src['annos'].items():
            if field == 'gt_boxes':
                expected_boxes = np.asarray(values).copy()
                for i in range(len(expected_boxes)):
                    dimensions = approved_dimensions(decisions.get((token, i), {}))
                    if dimensions is not None:
                        expected_boxes[i, 3:6] = dimensions
                if not np.array_equal(expected_boxes, dst['annos'][field]):
                    raise ValueError('Unapproved geometry edit')
            elif field not in ('gt_names', 'gt_mesh') and not np.array_equal(values, dst['annos'][field]):
                raise ValueError('Geometry/track/split/other annotation field changed: ' + field)
        for i, name in enumerate(src['annos']['gt_names']):
            decision = decisions.get((token, i)); checked += 1
            expected = decision['new_label'] if decision else str(name)
            if str(dst['annos']['gt_names'][i]) != expected:
                raise ValueError('Incorrect revised name or string truncation')
            old_mesh = str(src['annos']['gt_mesh'][i])
            expected_mesh = 'detzero.' + expected.lower() if expected != str(name) else old_mesh
            if str(dst['annos']['gt_mesh'][i]) != expected_mesh:
                raise ValueError('Name/mesh mismatch')
    return checked


def synchronize_extents(document, old_dataset, new_dataset, allow_unreferenced_size_changes=False):
    """Preserve independent measured 2D evidence, update only linked GT classes."""
    doc = copy.deepcopy(document)
    if doc['training_boxes_sha256'] != training_boxes_sha256(old_dataset):
        raise ValueError('Source extents do not match source TRAIN labels')
    old = {str(r['token']): g.numpy() for r, g in zip(old_dataset.rows, old_dataset.boxes)}
    new = {str(r['token']): g.numpy() for r, g in zip(new_dataset.rows, new_dataset.boxes)}
    if old.keys() != new.keys():
        raise ValueError('Extent revision changed TRAIN frames')
    size_changes = 0
    for token in old:
        if old[token].shape != new[token].shape:
            raise ValueError('Extent migration requires unchanged ordered boxes')
        if not np.array_equal(old[token][:, :7], new[token][:, :7]):
            if (not allow_unreferenced_size_changes or
                    not np.array_equal(old[token][:, [0, 1, 2, 6]], new[token][:, [0, 1, 2, 6]])):
                raise ValueError('Extent migration requires unchanged centers/headings and approved sizes')
            size_changes += int(np.any(old[token][:, 3:6] != new[token][:, 3:6], axis=1).sum())
    changed = 0; total = 0; counts = Counter()
    for token, records in doc['frames'].items():
        for record in records:
            gt = np.asarray(record['gt'], np.float32)
            match = np.isclose(old[token], gt, atol=1e-4, rtol=0).all(-1)
            candidates = new[token][match]
            if not len(candidates) or len(np.unique(candidates[:, 7])) != 1:
                raise ValueError('Missing/ambiguous old extent GT association: ' + token)
            if not np.isclose(candidates[:, :7], gt[:7], atol=1e-4, rtol=0).all():
                raise ValueError('Geometry-edited extent needs fresh association evidence: ' + token)
            replacement = candidates[0].tolist()
            if record['gt'][7] != replacement[7]:
                record['gt'][7] = replacement[7]; changed += 1
            counts[CLASSES[int(replacement[7])]] += 1; total += 1
    for name in CLASSES:
        doc['counts'].pop(name, None)
    doc['counts'].update(counts)
    doc['training_boxes_sha256'] = training_boxes_sha256(new_dataset)
    doc['label_revision'] = dict(policy='manual_heavy_class_review_20261116', changed_gt_records=changed,
        unreferenced_size_changes=size_changes,
        note='Only linked GT class IDs updated. Measured bbox, edge masks, camera, weights and YOLO evidence hashes unchanged; weak boundaries are not manually certified.')
    return doc, dict(records=total, changed_gt_records=changed, unreferenced_size_changes=size_changes, classes=dict(counts))


def make_pages(audit, output, records):
    destination = output / 'second_pass_pages'; destination.mkdir()
    ordered = sorted(records, key=lambda r: (r['key'], int(r['token'])))
    for start in range(0, len(ordered), 40):
        canvas = np.full((1680, 1400, 3), 12, np.uint8)
        for j, r in enumerate(ordered[start:start + 40]):
            tile = cv2.imread(str(audit / r['crop_path']))
            if tile is None:
                raise FileNotFoundError(r['crop_path'])
            tile[:18] = 20
            text = f"A{r['id']:04d} {r['split']} {r['old_label']}->{r['new_label']}"
            color = (70, 220, 255) if r['status'] == 'unresolved' else (110, 255, 150)
            cv2.putText(tile, text, (3, 14), cv2.FONT_HERSHEY_SIMPLEX, .37, color, 1, cv2.LINE_AA)
            if 'new_dimensions_lwh' in r:
                tile[18:46] = 20
                cv2.putText(tile, 'USER LWH: 4.0 x 1.8 x 1.5 m', (3, 29), cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
                cv2.putText(tile, 'Center/yaw kept; source-view crop', (3, 42), cv2.FONT_HERSHEY_SIMPLEX, .32, color, 1, cv2.LINE_AA)
            canvas[(j // 5) * 210:(j // 5 + 1) * 210, (j % 5) * 280:(j % 5 + 1) * 280] = tile
        cv2.imwrite(str(destination / f'all_{start // 40 + 1:03d}.jpg'), canvas)


def build(root, audit, source_extents, output, calibration):
    if output.exists():
        raise FileExistsError('Choose a new versioned output: ' + str(output))
    index = json.loads((audit / 'index.json').read_text())
    findings = runpy.run_path(str(audit / 'review_findings.py'))['FINDINGS']
    if set(findings) != set(range(1, index['track_count'] + 1)):
        raise ValueError('Every indexed track needs an explicit review decision')
    tracks = {t['key']: t for t in index['tracks']}
    records = [json.loads(line) for line in (audit / 'annotations.jsonl').read_text().splitlines()]
    if len(records) != index['annotation_count'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Incomplete/duplicate audit inventory')
    decisions = dict(train={}, val={}); transitions = Counter(); statuses = Counter()
    for r in records:
        t = tracks[r['key']]; f = findings[t['number']]
        r['old_label'] = r['label']; r['new_label'] = MAPPING.get(f['physical_class'], r['label'].title())
        r['physical_class'] = f['physical_class']; r['track_number'] = t['number']; r['reason'] = f['note']
        if 'dimensions_lwh' in f:
            r['new_dimensions_lwh'] = f['dimensions_lwh']
            r['geometry_authority'] = f['geometry_authority']
            approved_dimensions(r)
        r['status'] = ('unresolved' if f['physical_class'] == 'unresolved' else
            'taxonomy_policy' if f['physical_class'] in ('pickup', 'construction') else
            'corrected_class' if r['new_label'].lower() != r['label'] else 'supported_class')
        r['review_scope'] = 'Track identity from temporal crops; individual box geometry/heading not certified'
        key = (r['token'], r['index'])
        if key in decisions[r['split']]:
            raise ValueError('Duplicate annotation address')
        decisions[r['split']][key] = r
        statuses[(r['split'], r['status'])] += 1
        if r['status'] == 'corrected_class':
            transitions[(r['split'], r['old_label'], r['new_label'])] += 1
    output.mkdir(parents=True)
    geometry_edits = [dict(id=r['id'], split=r['split'], token=r['token'], index=r['index'],
        track_id=r['track_id'], old_dimensions_lwh=r['box_CLIDAR'][3:6],
        new_dimensions_lwh=r['new_dimensions_lwh'], authority=r['geometry_authority'])
        for r in records if 'new_dimensions_lwh' in r]
    manifest = dict(type='raybev_manual_heavy_revision_v4', scope='Bus/Truck/Van class corrections plus nine user-prescribed car sizes',
        geometry_edits=geometry_edits,
        limitations=['Unresolved annotations retained, not silently deleted/ignored.',
            'Car/Pedestrian/Cyclist source annotations were not visually audited.',
            'Nine explicitly authorized LWH edits; centers, yaw and all other geometry unchanged. Cuboid fitting is not certified.',
            'Tram remains Bus; construction->Van existing policy retained; pickup unchanged pending taxonomy decision.'],
        audit_annotation_count=len(records), audit_track_count=len(tracks), findings_sha256=sha(audit / 'review_findings.py'),
        audit_index_sha256=sha(audit / 'index.json'), audit_records_sha256=sha(audit / 'annotations.jsonl'),
        calibration_sha256=sha(calibration), splits={})
    for split in ('train', 'val'):
        source = root / f'annotations/fisheye3dod_infos_{split}.pkl'
        if sha(source) != index['annotation_sha256'][split]:
            raise ValueError('Source differs from visually reviewed version: ' + str(source))
        original = load_annotations(source); revised = revise_document(original, decisions[split])
        path = output / source.name
        with path.open('wb') as stream:
            pickle.dump(revised, stream, protocol=4)
        checked = verify_document(original, load_annotations(path), decisions[split])
        before = Counter(str(n) for row in original['data_list'] for n in row['annos']['gt_names'])
        after = Counter(str(n) for row in revised['data_list'] for n in row['annos']['gt_names'])
        manifest['splits'][split] = dict(source=str(source), source_sha256=sha(source), output_sha256=sha(path),
            frames=len(original['data_list']), all_boxes_checked=checked, before=dict(before), after=dict(after),
            review_statuses={s: n for (sp, s), n in statuses.items() if sp == split},
            transitions={f'{old}->{new}': n for (sp, old, new), n in transitions.items() if sp == split})
    torch.set_num_threads(2); cfg = SparseConfig(architecture_version=5)
    old_dataset = RosaCombinedDataset(root, (root / 'annotations/fisheye3dod_infos_train.pkl').resolve(), cfg=cfg, calibration=calibration)
    VerifiedExtents(source_extents, old_dataset, calibration)
    new_dataset = RosaCombinedDataset(root, (output / 'fisheye3dod_infos_train.pkl').resolve(), cfg=cfg, calibration=calibration)
    extent_doc, extent_stats = synchronize_extents(json.loads(source_extents.read_text()), old_dataset, new_dataset,
        allow_unreferenced_size_changes=bool(geometry_edits))
    extent_path = output / 'train_yolo_extents_v1.json'; dump(extent_path, extent_doc)
    VerifiedExtents(extent_path, new_dataset, calibration)
    manifest['extents'] = dict(source=str(source_extents), source_sha256=sha(source_extents), output_sha256=sha(extent_path),
        old_training_boxes_sha256=training_boxes_sha256(old_dataset), new_training_boxes_sha256=training_boxes_sha256(new_dataset), **extent_stats)
    dump(output / 'correction_manifest.json', manifest)
    dump(output / 'track_findings.json', findings)
    with (output / 'annotation_review.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps(r, allow_nan=False) + '\n')
    fields = ['id', 'split', 'sequence', 'track_id', 'track_number', 'token', 'index', 'old_label', 'new_label', 'physical_class', 'status', 'reason', 'new_dimensions_lwh', 'geometry_authority', 'crop_path', 'all_frames_page']
    for name, selected in [('annotation_review.csv', records), ('unresolved.csv', [r for r in records if r['status'] in ('unresolved', 'taxonomy_policy')])]:
        with (output / name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(selected)
    make_pages(audit, output, records)
    print(json.dumps(dict(splits=manifest['splits'], extents=manifest['extents']), indent=2), flush=True)


def verify_release(root, output, source_extents, calibration):
    """Independent post-write sweep over ALL boxes and EVERY extent record."""
    report = json.loads((output / 'correction_manifest.json').read_text())
    if sha(calibration) != report['calibration_sha256']:
        raise ValueError('Calibration hash changed')
    records = [json.loads(line) for line in (output / 'annotation_review.jsonl').read_text().splitlines()]
    if len(records) != report['audit_annotation_count'] or len({r['id'] for r in records}) != len(records):
        raise ValueError('Incomplete/duplicate review ledger')
    checked = {}; split_tokens = {}
    for split in ('train', 'val'):
        source = root / f'annotations/fisheye3dod_infos_{split}.pkl'
        path = output / source.name; expected = report['splits'][split]
        if sha(source) != expected['source_sha256'] or sha(path) != expected['output_sha256']:
            raise ValueError('Annotation file hash differs: ' + split)
        decisions = {(r['token'], r['index']): r for r in records if r['split'] == split}
        original = load_annotations(source); revised = load_annotations(path)
        checked[split] = verify_document(original, revised, decisions)
        split_tokens[split] = {str(row['token']) for row in revised['data_list']}
        for row in revised['data_list']:
            if row['metainfo']['temporal_split'] != split:
                raise ValueError('Split metadata mismatch')
    if split_tokens['train'] & split_tokens['val']:
        raise ValueError('TRAIN/VAL token overlap')
    extent_path = output / 'train_yolo_extents_v1.json'
    if sha(source_extents) != report['extents']['source_sha256'] or sha(extent_path) != report['extents']['output_sha256']:
        raise ValueError('Extent file hash differs')
    old = json.loads(source_extents.read_text()); new = json.loads(extent_path.read_text())
    if old['frames'].keys() != new['frames'].keys():
        raise ValueError('Extent frame set changed')
    preserved = 0; changes = 0
    for token in old['frames']:
        if len(old['frames'][token]) != len(new['frames'][token]):
            raise ValueError('Extent record count changed')
        for src, dst in zip(old['frames'][token], new['frames'][token]):
            if src.keys() != dst.keys() or src['gt'][:7] != dst['gt'][:7]:
                raise ValueError('Extent structure/GT geometry changed')
            if any(src[k] != dst[k] for k in src if k != 'gt'):
                raise ValueError('Measured 2D evidence changed')
            changes += int(src['gt'][7] != dst['gt'][7]); preserved += 1
    torch.set_num_threads(2)
    dataset = RosaCombinedDataset(root, (output / 'fisheye3dod_infos_train.pkl').resolve(),
        cfg=SparseConfig(architecture_version=5), calibration=calibration)
    VerifiedExtents(extent_path, dataset, calibration)
    if preserved != report['extents']['records'] or changes != report['extents']['changed_gt_records']:
        raise ValueError('Extent migration counts differ')
    # Verify source data were not modified while building this revision.
    result = dict(all_annotation_boxes_checked=checked, total_boxes_checked=sum(checked.values()),
        heavy_review_records_checked=len(records), yolo_extent_records_checked=preserved,
        yolo_extent_class_changes=changes,
        authorized_size_edits=sum('new_dimensions_lwh' in r for r in records),
        centers_yaw_identity_and_nonrequested_geometry_unchanged=True,
        measured_yolo_evidence_unchanged=True, train_val_tokens_disjoint=True,
        runtime_extent_validator='passed', certification='Data consistency passed; not all-class visual/3D correctness certification')
    dump(output / 'verification_report.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--audit', type=Path, help='Required when building; not needed for --verify')
    p.add_argument('--source-extents', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', type=Path, default=Path(DEFAULT_CALIBRATION))
    p.add_argument('--verify', action='store_true', help='Check an existing release without rewriting annotations/extents')
    a = p.parse_args()
    if a.verify:
        verify_release(a.root.resolve(), a.output.resolve(), a.source_extents.resolve(), a.calibration.resolve())
    else:
        if a.audit is None:
            p.error('--audit is required when building')
        build(a.root.resolve(), a.audit.resolve(), a.source_extents.resolve(), a.output.resolve(), a.calibration.resolve())


if __name__ == '__main__':
    main()
