#!/usr/bin/env python3
"""Desktop TRAIN/VAL class review. Autosave decisions; export new PKLs + extents.

No inference, YOLO installation, CUDA, browser, or running training job required.
Images and trusted annotation PKLs must be available locally. Sources are never
overwritten. Projected boxes identify the target, not the visible foreground
occluder. An image alone cannot certify a 3D cuboid fit.
"""
import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from raybev.config import CAMERAS, CLASSES, DEFAULT_CALIBRATION
from raybev.geometry import load_cameras, project_fisheye, letterbox_params
from raybev.rosa_dataset import (load_annotations, boxes_to_baselink, read_T_base_clidar,
                                camera_entry, names_to_ids, RosaCombinedDataset, training_boxes_sha256)
from raybev.sparse_config import SparseConfig
from raybev.sparse_extent import VerifiedExtents
from prepare_heavy_annotation_review import (sha, revise_document, verify_document,
                                            synchronize_extents)

NAMES = dict(zip(CLASSES, ('Car', 'Pedestrian', 'Cyclist', 'Truck', 'Van', 'Bus')))
REVIEW_DEFAULT = PROJECT / 'datasets/combined_bags_camera_ray_heavy_review_v4'
FOLLOWUP_DEFAULT = PROJECT / 'analysis/v10_manual_review_v5'
MANUAL_V5 = PROJECT / 'datasets/combined_bags_camera_ray_manual_review_v5'
REVIEW_UI_VERSION = 'font-v4'


def named_ui_fonts(tkfont, root):
    """Use the default Tk root; older nametofont() accepts only a name."""
    return {name: tkfont.nametofont(name) for name in tkfont.names(root)}


def scalable_font_family(tkfont, root):
    """Reject bitmap fallbacks that silently ignore every requested size."""
    available = {name.lower(): name for name in tkfont.families(root)}
    preferred = ('Liberation Sans', 'DejaVu Sans', 'Noto Sans', 'Segoe UI', 'Arial', 'Helvetica')
    candidates = [available[name.lower()] for name in preferred if name.lower() in available]
    candidates += [name for name in sorted(available.values()) if name not in candidates]
    for family in candidates:
        probe = tkfont.Font(root=root, family=family, size=-16)
        small = probe.metrics('linespace')
        probe.configure(size=-40)
        large = probe.metrics('linespace')
        if large >= 30 and large >= small * 1.5:
            return probe.actual('family')
    raise RuntimeError('Tk exposes no scalable font. Install Liberation Sans or DejaVu Sans and restart the desktop session.')


def apply_text_fonts(widget, ui_font, heading_font):
    """Set widget fonts explicitly rather than relying on theme inheritance."""
    options = widget.keys()
    settings = {}
    if 'font' in options and str(widget.cget('font')) != str(heading_font):
        settings['font'] = ui_font
    kind = widget.winfo_class()
    if 'style' in options and kind in ('TLabel', 'TButton', 'TCheckbutton', 'TEntry', 'TCombobox'):
        settings['style'] = 'Review.' + kind
    if settings:
        widget.configure(**settings)
    for child in widget.winfo_children():
        apply_text_fonts(child, ui_font, heading_font)


def flow_layout(widths, available, gap=8):
    """Wrap a control row instead of clipping larger-font widgets offscreen."""
    row = column = used = 0
    result = []
    for width in widths:
        if column and used + width + gap > available:
            row += 1; column = used = 0
        result.append((row, column))
        used += width + gap; column += 1
    return result


def overlay_font(size):
    for name in ('DejaVuSans.ttf', 'Arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def fit_preview(image, width, height):
    """Fit the panel in either direction: thumbnail() never enlarges crops."""
    width, height = max(1, width), max(1, height)
    scale = min(width / image.width, height / image.height)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def valid_track_id(track_id):
    value = str(track_id).strip().lower()
    return value not in ('', 'none', 'nan', 'unknown', '-1') and not value.startswith('-')


def remove_frame_annotations(document, removals):
    """Drop exact source indices, keeping frames and all object fields aligned."""
    if not any(removals.values()):return document
    revised = copy.deepcopy(document)
    for row in revised['data_list']:
        ann = row['annos']; token = str(row['token']); count = len(ann['gt_boxes'])
        removed = removals.get(token, set())
        if not removed:continue
        if any(index < 0 or index >= count for index in removed):
            raise ValueError('Invalid removal index: ' + token)
        keep = [index for index in range(count) if index not in removed]
        for name, values in list(ann.items()):
            if isinstance(values, np.ndarray) and values.ndim and len(values) == count:
                ann[name] = values[keep].copy()
            elif isinstance(values, (list, tuple)) and len(values) == count:
                ann[name] = type(values)(values[index] for index in keep)
            elif name in ('gt_boxes', 'gt_names', 'gt_ids', 'gt_mesh'):
                raise ValueError('Unaligned annotation field: ' + name)
    # Independently compare every retained object/metadata field with its source.
    for source, target in zip(document['data_list'], revised['data_list']):
        count = len(source['annos']['gt_boxes'])
        keep = [i for i in range(count) if i not in removals.get(str(source['token']), set())]
        for name, values in source['annos'].items():
            actual = target['annos'][name]
            if isinstance(values, np.ndarray) and values.ndim and len(values) == count:
                np.testing.assert_array_equal(actual, np.take(values, keep, axis=0))
            elif isinstance(values, (list, tuple)) and len(values) == count:
                if pickle.dumps(actual,4)!=pickle.dumps(type(values)(values[i] for i in keep),4):
                    raise ValueError('Removal changed retained annotations: '+name)
            else:
                if pickle.dumps(actual,4)!=pickle.dumps(values,4):
                    raise ValueError('Removal changed annotation metadata: '+name)
        for name in source:
            if name != 'annos':
                if pickle.dumps(target[name],4)!=pickle.dumps(source[name],4):
                    raise ValueError('Removal changed frame metadata: '+name)
    return revised


def synchronize_removed_extents(document, old, new, removals):
    """Associate through raw source indices, never shifted post-removal indices."""
    if document['training_boxes_sha256'] != training_boxes_sha256(old):
        raise ValueError('Source extents do not match source TRAIN labels')
    doc = copy.deepcopy(document)
    old_rows = {str(row['token']):(row, boxes.numpy(), indices)
                for row, boxes, indices in zip(old.rows, old.boxes, old.box_source_indices)}
    new_rows = {str(row['token']):(boxes.numpy(), indices)
                for row, boxes, indices in zip(new.rows, new.boxes, new.box_source_indices)}
    if old_rows.keys() != new_rows.keys():raise ValueError('Removal changed TRAIN frames')
    lookup = {}; size_changes = 0
    for token, (row, boxes, indices) in old_rows.items():
        removed = removals.get(token, set())
        kept_indices = [i for i in range(len(row['annos']['gt_boxes'])) if i not in removed]
        new_boxes, new_indices = new_rows[token]
        lookup[token] = {kept_indices[int(index)]:box for index,box in zip(new_indices,new_boxes)}
        for index, box in zip(indices, boxes):
            replacement = lookup[token].get(int(index))
            if replacement is not None:
                if not np.array_equal(box[[0,1,2,6]], replacement[[0,1,2,6]]):
                    raise ValueError('Removal changed a retained center/heading')
                size_changes += int(not np.array_equal(box[3:6], replacement[3:6]))
    dropped = geometry_dropped = changed = total = 0; counts = Counter()
    for token, records in doc['frames'].items():
        retained = []; _, boxes, indices = old_rows[token]
        for record in records:
            matches = np.flatnonzero(np.isclose(boxes, np.asarray(record['gt'], np.float32), atol=1e-4, rtol=0).all(-1))
            if not len(matches):raise ValueError('Missing old extent association: ' + token)
            raw_indices = [int(indices[i]) for i in matches]
            deleted = [i in removals.get(token, set()) for i in raw_indices]
            if all(deleted):dropped += 1;continue
            if any(deleted):raise ValueError('Ambiguous removed/retained extent association: ' + token)
            candidates = [lookup[token].get(i) for i in raw_indices]
            if any(box is None for box in candidates):raise ValueError('Missing retained extent association: ' + token)
            if any(not np.isclose(box[:7], record['gt'][:7], atol=1e-4, rtol=0).all() for box in candidates):
                geometry_dropped += 1;continue
            if len({int(box[7]) for box in candidates}) != 1:
                raise ValueError('Ambiguous revised extent class: ' + token)
            replacement = candidates[0]
            changed += int(record['gt'][7] != replacement[7]); record['gt'][7] = float(replacement[7])
            counts[CLASSES[int(replacement[7])]] += 1;total += 1;retained.append(record)
        doc['frames'][token] = retained
    for name in CLASSES:doc['counts'].pop(name,None)
    doc['counts'].update(counts);doc['training_boxes_sha256'] = training_boxes_sha256(new)
    stats = dict(records=total,changed_gt_records=changed,unreferenced_size_changes=size_changes,classes=dict(counts))
    doc['label_revision'] = dict(policy='desktop_manual_review',**stats,
        removed_object_records_dropped=dropped,geometry_edited_records_dropped=geometry_dropped)
    return doc, stats, dropped, geometry_dropped


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def valid_dimensions(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all() or (array <= 0).any():
        raise ValueError('Length, width and height must be three positive finite metres.')
    return array.tolist()


def verify_export(root, directory, calibration):
    """Server-side release verification; original input JSON/state not required."""
    directory = Path(directory).resolve(); calibration = Path(calibration).resolve()
    manifest = json.loads((directory / 'review_manifest.json').read_text())
    if manifest.get('type') != 'raybev_desktop_manual_review_v1':
        raise ValueError('Not a desktop-review release.')
    identity = manifest['source_identity']
    if 'track_id_revisions' in manifest:
        if sha(directory/'track_id_revisions.json') != manifest['track_id_revisions']['sha256']:
            raise ValueError('Export track revision hash mismatch.')
    if sha(calibration) != identity['calibration']:
        raise ValueError('Export calibration hash mismatch.')
    datasets = {}; tokens = {}; checked = 0
    for split in ('train', 'val'):
        path = directory / f'fisheye3dod_infos_{split}.pkl'
        expected = manifest['splits'][split]
        if sha(path) != expected['output_sha256']:
            raise ValueError('Export annotation hash mismatch: ' + split)
        doc = load_annotations(path); rows = doc['data_list']
        if len(rows) != expected['frames'] or any(r['metainfo'].get('temporal_split') != split for r in rows):
            raise ValueError('Export frame/split structure mismatch: ' + split)
        tokens[split] = {str(r['token']) for r in rows}
        count = sum(len(r['annos']['gt_boxes']) for r in rows)
        if len(tokens[split]) != len(rows) or count != expected['all_boxes_checked']:
            raise ValueError('Export token/box count mismatch: ' + split)
        checked += count
        datasets[split] = RosaCombinedDataset(root, path, cfg=SparseConfig(architecture_version=5),
            calibration=calibration, box_frame=identity['box_frame'], z_reference=identity['z_reference'])
    if tokens['train'] & tokens['val']:
        raise ValueError('Export TRAIN/VAL token overlap.')
    extents = directory / 'train_yolo_extents_v1.json'
    if sha(extents) != manifest['extents']['output_sha256']:
        raise ValueError('Export extent hash mismatch.')
    validated = VerifiedExtents(extents, datasets['train'], calibration)
    records = sum(map(len, validated.frames.values()))
    if records != manifest['extents']['records']:
        raise ValueError('Export extent count mismatch.')
    return dict(status='passed', all_boxes_checked=checked, extent_records_checked=records,
                note='Hash/GT/calibration consistency, not a certification of human review correctness.')


class ReviewSession:
    def __init__(self, root, train_ann, val_ann, extents, calibration, state,
                 classes=('bus', 'van', 'truck'), box_frame='CLIDAR', z_reference='center', review_ledger=None):
        self.root = Path(root).resolve()
        self.paths = dict(train=Path(train_ann).resolve(), val=Path(val_ann).resolve())
        self.extents = Path(extents).resolve(); self.calibration = Path(calibration).resolve()
        self.state_path = Path(state).resolve()
        self.issue_records = None
        self.box_frame = box_frame; self.z_reference = z_reference
        self.cameras = load_cameras(self.calibration)
        self.transform = read_T_base_clidar(self.calibration) if box_frame == 'CLIDAR' else None
        self.identity = dict(version=1, annotations={s: sha(p) for s, p in self.paths.items()},
            extents=sha(self.extents), calibration=sha(self.calibration),
            classes=sorted(classes), box_frame=box_frame, z_reference=z_reference)
        self.documents = {s: load_annotations(p) for s, p in self.paths.items()}
        selected_tracks = set()
        for doc in self.documents.values():
            for row in doc['data_list']:
                ann = row['annos']; sequence = str(row['metainfo']['sequence_id'])
                for index, label in enumerate(names_to_ids(ann['gt_names'])):
                    track_id = str(ann['gt_ids'][index])
                    if CLASSES[int(label)] in classes and valid_track_id(track_id):
                        selected_tracks.add((sequence, track_id))
        self.records = []; self.by_id = {}; self.by_track = defaultdict(list)
        for split, doc in self.documents.items():
            seen_tokens = set()
            for row_index, row in enumerate(doc['data_list']):
                token = str(row['token']); ann = row['annos']
                if token in seen_tokens:
                    raise ValueError('Duplicate frame token in ' + split)
                seen_tokens.add(token)
                if row['metainfo'].get('temporal_split') != split:
                    raise ValueError('TRAIN/VAL metadata mismatch: ' + token)
                for field in ('gt_names', 'gt_boxes', 'gt_mesh', 'gt_ids'):
                    if field not in ann:
                        raise ValueError('Annotation missing field: ' + field)
                labels = names_to_ids(ann['gt_names'])
                base = boxes_to_baselink(ann['gt_boxes'], box_frame, self.transform, z_reference)
                for index, label in enumerate(labels):
                    category = CLASSES[int(label)]
                    sequence = str(row['metainfo']['sequence_id']); track_id = str(ann['gt_ids'][index])
                    if category not in classes and (sequence, track_id) not in selected_tracks:
                        continue
                    key = f'{split}:{token}:{index}'
                    rec = dict(id=key, split=split, token=token, index=index, row_index=row_index,
                        sequence=sequence, track_id=track_id,
                        original_class=category, old_label=str(ann['gt_names'][index]).lower(),
                        box_CLIDAR=np.asarray(ann['gt_boxes'])[index, :7].tolist(),
                        box_BASELINK=base[index].tolist())
                    self.records.append(rec); self.by_id[key] = rec
                    if valid_track_id(track_id):
                        self.by_track[sequence, track_id].append(key)
        self.records.sort(key=lambda r: (r['sequence'], r['track_id'], r['split'], r['token'], r['index']))
        self.review_ledger = Path(review_ledger) if review_ledger is not None else self.paths['train'].parent / 'annotation_review.jsonl'
        if review_ledger is not None or self.review_ledger.exists():
            with self.review_ledger.open() as stream:
                for line in stream:
                    if not line.strip():continue
                    entry = json.loads(line)
                    key = f"{entry['split']}:{entry['token']}:{entry['index']}"
                    rec = self.by_id.get(key)
                    if rec and (rec['sequence'], rec['track_id']) == (str(entry['sequence']), str(entry['track_id'])):
                        rec['previous_review_status'] = entry.get('status', '')
                        rec['previous_review_reason'] = entry.get('reason', '')
        self.decisions = {}; self.undo_stack = []; self.last_id = None
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            if saved['identity'] != self.identity:
                raise ValueError('Progress belongs to different annotations/calibration/extents. Use a NEW --state file.')
            for key, decision in saved['decisions'].items():
                self.validate_decision(key, decision)
            self.decisions = saved['decisions']; self.last_id = saved.get('last_id')
            self.undo_stack = saved.get('undo', [])

    def validate_decision(self, key, decision):
        if key not in self.by_id or decision.get('label') not in CLASSES:
            raise ValueError('Unknown annotation/class in progress file.')
        if decision.get('status') not in ('reviewed', 'uncertain', 'removed'):
            raise ValueError('Invalid review status.')
        valid_dimensions(decision.get('dimensions_lwh'))
        if not isinstance(decision.get('note', ''), str) or len(decision.get('note', '')) > 2000:
            raise ValueError('Review note must be at most 2000 characters.')

    def current(self, rec):
        return self.decisions.get(rec['id'], dict(label=rec['original_class'], status='unreviewed', note=''))

    def unknown(self, rec):
        status = self.current(rec)['status']
        return status == 'uncertain' or (status not in ('reviewed','removed') and
            rec.get('previous_review_status') in ('unresolved', 'taxonomy_policy'))

    def review_priority(self, rec):
        return 0 if self.unknown(rec) else 1 if self.current(rec)['status'] == 'unreviewed' else 2

    def save(self):
        atomic_json(self.state_path, dict(identity=self.identity, decisions=self.decisions,
            last_id=self.last_id, undo=self.undo_stack[-100:]))

    def targets(self, key, track=False, both_splits=False):
        rec = self.by_id[key]
        if not track or not valid_track_id(rec['track_id']):
            return [key]
        return [k for k in self.by_track[rec['sequence'], rec['track_id']]
                if (both_splits or self.by_id[k]['split'] == rec['split']) and self.current(self.by_id[k])['status']!='removed']

    def apply(self, key, label=None, uncertain=False, note='', dimensions=None,
              track=False, both_splits=False):
        dimensions = valid_dimensions(dimensions)
        if dimensions is not None and self.z_reference != 'center':
            raise ValueError('Size editing requires center-referenced input boxes. Bottom-referenced inputs are class-only.')
        ids = self.targets(key, track, both_splits)
        if not ids:raise ValueError('No retained annotations in this track.')
        changes = {}
        for item in ids:
            old = self.current(self.by_id[item])
            decision = dict(label=label or old['label'], status='uncertain' if uncertain else 'reviewed', note=note)
            size = dimensions if dimensions is not None else old.get('dimensions_lwh')
            if size is not None:
                decision['dimensions_lwh'] = size
            self.validate_decision(item, decision); changes[item] = decision
        previous = {item: copy.deepcopy(self.decisions.get(item)) for item in ids}
        self.decisions.update(changes); self.undo_stack.append(previous)
        self.last_id = key
        try:
            self.save()
        except Exception:
            self._restore(previous); self.undo_stack.pop(); raise
        return len(ids)

    def remove(self, key, note=''):
        """Tombstone exactly one annotation; track scope is deliberately absent."""
        previous = {key:copy.deepcopy(self.decisions.get(key))}
        choice = dict(self.current(self.by_id[key]), status='removed', note=note)
        self.validate_decision(key,choice)
        self.decisions[key] = choice;self.undo_stack.append(previous);self.last_id = key
        try:
            self.save()
        except Exception:
            self._restore(previous);self.undo_stack.pop();raise
        return 1

    def _restore(self, previous):
        for key, decision in previous.items():
            if decision is None:
                self.decisions.pop(key, None)
            else:
                self.decisions[key] = decision

    def undo(self):
        if not self.undo_stack:
            return False
        snapshot = copy.deepcopy(self.decisions); previous = self.undo_stack.pop()
        self._restore(previous)
        try:
            self.save()
        except Exception:
            self.decisions = snapshot; self.undo_stack.append(previous); raise
        return True

    def summary(self):
        records = self.queue_records()
        decisions = {r['id']:self.decisions[r['id']] for r in records if r['id'] in self.decisions}
        return dict(queued=len(records), reviewed=len(decisions),
            unknown_remaining=sum(self.unknown(r) for r in records),
            review_ledger=str(self.review_ledger) if self.review_ledger.exists() else None,
            uncertain=sum(d['status'] == 'uncertain' for d in decisions.values()),
            removed=sum(d['status'] == 'removed' for d in decisions.values()),
            class_changes=sum(d['status']!='removed' and d['label'] != self.by_id[k]['original_class'] for k, d in decisions.items()),
            dimension_edits=sum(d['status']!='removed' and 'dimensions_lwh' in d for d in decisions.values()),
            splits=dict(Counter(r['split'] for r in records)),
            original_classes=dict(Counter(r['original_class'] for r in records)))

    def queue_records(self):
        return self.records if self.issue_records is None else self.issue_records

    def remaining_issues(self, ledger, contract):
        """Bind final-v5 indices to frozen inputs; narrow only the UI, never exports."""
        contract = json.loads(Path(contract).read_text())
        for split, filename in [('train','fisheye3dod_infos_train.pkl'),('val','fisheye3dod_infos_val.pkl')]:
            if self.identity['annotations'][split] != contract['files'][filename]:
                raise ValueError('Remaining-issues ledger requires the audited v5 ' + split + ' annotations.')
        if (self.identity['extents'] != contract['files']['train_yolo_extents_v1.json'] or
                self.identity['calibration'] != contract['calibration_sha256']):
            raise ValueError('Remaining-issues extents/calibration differ from the audited v5 inputs.')
        selected = {}; ledger = Path(ledger)
        for line in ledger.read_text().splitlines():
            if not line.strip():continue
            entry = json.loads(line)
            key = f"{entry['split']}:{entry['token']}:{entry['index']}"
            rec = self.by_id.get(key)
            if rec is None or (rec['sequence'],rec['track_id']) != (str(entry['sequence']),str(entry['track_id'])):
                raise ValueError('Remaining issue does not match its final-v5 object: ' + key)
            if key in selected:raise ValueError('Duplicate remaining issue: ' + key)
            rec['previous_review_status'] = 'unresolved'
            rec['previous_review_reason'] = entry['reason']
            selected[key] = rec
        if not selected:raise ValueError('Remaining-issues ledger is empty.')
        # Impossible pedestrian geometry first, then oversized vans; stable track order within groups.
        priority = {'pedestrian':0,'van':1}
        self.issue_records = sorted(selected.values(),key=lambda r:(priority.get(r['original_class'],2),
            r['sequence'],r['track_id'],r['split'],r['token'],r['index']))
        self.review_ledger = ledger

    def export(self, destination):
        """Export the decision snapshot; no source paths are changed."""
        destination = Path(destination).resolve()
        if destination.exists():
            raise FileExistsError('Output already exists. Choose a NEW directory: ' + str(destination))
        if any(sha(p) != self.identity['annotations'][s] for s, p in self.paths.items()):
            raise ValueError('Source annotations changed during review.')
        if sha(self.extents) != self.identity['extents'] or sha(self.calibration) != self.identity['calibration']:
            raise ValueError('Source extents/calibration changed during review.')
        decisions = dict(train={}, val={});removals = dict(train=defaultdict(set),val=defaultdict(set))
        for key, choice in self.decisions.items():
            rec = self.by_id[key]; d = dict(rec, new_label=NAMES[choice['label']])
            if choice['status']=='removed':
                removals[rec['split']][rec['token']].add(rec['index']);continue
            if 'dimensions_lwh' in choice:
                d.update(new_dimensions_lwh=choice['dimensions_lwh'], geometry_authority='Explicit desktop reviewer LWH edit; center/yaw preserved')
            decisions[rec['split']][rec['token'], rec['index']] = d
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=destination.name + '.building-', dir=destination.parent))
        try:
            manifest = dict(type='raybev_desktop_manual_review_v1', source_identity=self.identity,
                review_summary=self.summary(), splits={}, limitations=[
                    'Unreviewed and uncertain annotations retained. Only explicit single-frame object removals are applied.',
                    'Centers/yaw unchanged. Optional sizes are prescribed, not fitted from images.',
                    'Geometry-edited weak extent associations are dropped, never fabricated.'])
            for split, original in self.documents.items():
                revised = revise_document(original, decisions[split])
                source_checked = verify_document(original, revised, decisions[split])
                revised = remove_frame_annotations(revised,removals[split])
                path = staging / f'fisheye3dod_infos_{split}.pkl'
                with path.open('wb') as stream:
                    pickle.dump(revised, stream, protocol=4)
                if pickle.dumps(load_annotations(path),4)!=pickle.dumps(revised,4):
                    raise ValueError('Export serialization changed revised annotations')
                checked = sum(len(row['annos']['gt_boxes']) for row in revised['data_list'])
                manifest['splits'][split] = dict(source_sha256=self.identity['annotations'][split],
                    output_sha256=sha(path), all_boxes_checked=checked, source_boxes_checked=source_checked,
                    removed_boxes=source_checked-checked, frames=len(original['data_list']))
            cfg = SparseConfig(architecture_version=5)
            options = dict(cfg=cfg, calibration=self.calibration, box_frame=self.box_frame, z_reference=self.z_reference)
            old = RosaCombinedDataset(self.root, self.paths['train'], **options)
            new = RosaCombinedDataset(self.root, staging / 'fisheye3dod_infos_train.pkl', **options)
            VerifiedExtents(self.extents, old, self.calibration)
            ext = json.loads(self.extents.read_text()); dropped = 0
            removed_extents = 0
            if removals['train']:
                ext,stats,removed_extents,dropped = synchronize_removed_extents(ext,old,new,removals['train'])
            else:
                ext,stats,dropped = self.synchronize_retained_extents(ext,old,new)
            ext['label_revision']['policy'] = 'desktop_manual_review'
            ext['label_revision']['geometry_edited_records_dropped'] = dropped
            ext['label_revision']['removed_object_records_dropped'] = removed_extents
            atomic_json(staging / 'train_yolo_extents_v1.json', ext)
            VerifiedExtents(staging / 'train_yolo_extents_v1.json', new, self.calibration)
            manifest['extents'] = dict(**stats, geometry_edited_records_dropped=dropped,
                removed_object_records_dropped=removed_extents,output_sha256=sha(staging / 'train_yolo_extents_v1.json'))
            atomic_json(staging / 'review_manifest.json', manifest)
            manifest['verification'] = verify_export(self.root, staging, self.calibration)
            atomic_json(staging / 'review_manifest.json', manifest)
            atomic_json(staging / 'review_decisions.json', dict(identity=self.identity, decisions=self.decisions))
            with (staging / 'review_queue.jsonl').open('w') as stream:
                for rec in self.records:
                    stream.write(json.dumps(dict(rec, decision=self.current(rec)), allow_nan=False) + '\n')
            if destination.exists():raise FileExistsError(destination)
            staging.rename(destination)
        except Exception:
            # Cleanup is restricted to this export's freshly allocated staging directory.
            shutil.rmtree(staging)
            raise
        return manifest

    @staticmethod
    def synchronize_retained_extents(ext,old,new):
        dropped = 0
        old_boxes = {str(r['token']): b.numpy() for r, b in zip(old.rows, old.boxes)}
        new_boxes = {str(r['token']): b.numpy() for r, b in zip(new.rows, new.boxes)}
        for token, records in ext['frames'].items():
            retained = []
            for record in records:
                match = np.isclose(old_boxes[token], np.asarray(record['gt'], np.float32), atol=1e-4, rtol=0).all(-1)
                candidates = new_boxes[token][match]
                if not len(candidates):
                    raise ValueError('Old extent association cannot be found: ' + token)
                if not np.isclose(candidates[:, :7], record['gt'][:7], atol=1e-4, rtol=0).all():
                    dropped += 1
                else:
                    retained.append(record)
            ext['frames'][token] = retained
        ext, stats = synchronize_extents(ext, old, new, allow_unreferenced_size_changes=True)
        return ext,stats,dropped

    @lru_cache(maxsize=8)
    def source_image(self, split, row_index, camera):
        row = self.documents[split]['data_list'][row_index]
        relative = Path(camera_entry(row['cam_info']['cam_fisheye'], camera)['cam_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe camera image path: ' + str(relative))
        # Open the path ourselves: Ultralytics can monkey-patch Image.open and
        # turn a missing file into an unrelated optional HEIF import failure.
        with (self.root / relative).open('rb') as stream, Image.open(stream) as image:
            return image.convert('RGB')

    def render(self, key, crop=True, context=.6):
        rec = self.by_id[key]; choice = self.current(rec)
        raw = np.asarray(rec['box_CLIDAR'], dtype=float).copy()
        if 'dimensions_lwh' in choice:
            raw[3:6] = choice['dimensions_lwh']
        box = boxes_to_baselink(raw[None], self.box_frame, self.transform, self.z_reference)[0]
        offsets = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
        points = offsets * box[3:6]; c, s = np.cos(box[6]), np.sin(box[6])
        points[:, :2] = points[:, :2] @ np.array([[c, s], [-s, c]])
        points += box[:3]
        edges = [(i, j) for i in range(8) for j in range(i+1, 8) if (offsets[i] != offsets[j]).sum() == 1]
        output = []
        for camera in self.cameras:
            full = self.source_image(rec['split'], rec['row_index'], camera.name).copy()
            draw = ImageDraw.Draw(full)
            width, height, left, top = letterbox_params(camera.width, camera.height, full.width, full.height)
            scale = np.array([width / camera.width, height / camera.height])
            def project(value):
                uv, valid, _ = project_fisheye(value, camera)
                return (uv + .5) * scale - .5 + [left, top], valid
            pixels = []
            for i, j in edges:
                sample = points[i] + np.linspace(0, 1, 17)[:, None] * (points[j]-points[i])
                uv, valid = project(sample)
                pixels.extend(uv[valid].tolist())
                for k in range(len(uv)-1):
                    if valid[k] and valid[k+1]:
                        draw.line([tuple(uv[k]), tuple(uv[k+1])], fill='#53ff9b', width=2)
            center, center_valid = project(box[None, :3])
            if center_valid[0]:
                x, y = center[0]; draw.line((x-9,y,x+9,y), fill='yellow', width=3)
                draw.line((x,y-9,x,y+9), fill='yellow', width=3)
            font_size = max(18, round(full.width / 55))
            font = overlay_font(font_size)
            draw.rectangle((0,0,full.width,font_size+14), fill='black')
            draw.text((6,5), f"{camera.name} | {choice['label']} | selected 3D box", fill='white', font=font)
            preview = full
            if crop and pixels:
                xy = np.asarray(pixels); lo = xy.min(0); hi = xy.max(0)
                pad = np.maximum((hi-lo)*context, 35*scale)
                lo = np.maximum(np.floor(lo-pad), [0,0]).astype(int)
                hi = np.minimum(np.ceil(hi+pad), full.size).astype(int)
                if (hi > lo).all():
                    preview = full.crop((*lo, *hi))
            elif not pixels:
                draw.text((8,font_size+20), 'Selected cuboid is outside this camera/FOV.', fill='yellow', font=font)
            output.append((camera.name, preview, full))
        return output


class ReviewWindow:
    def __init__(self, session, output, font_size=20, unknown_first=True):
        import tkinter as tk
        from tkinter import font as tkfont
        from tkinter import ttk, messagebox
        from PIL import ImageTk
        self.tk = tk; self.ttk = ttk; self.messagebox = messagebox; self.ImageTk = ImageTk
        self.session = session; self.output = Path(output); self.exporting = False
        self.unknown_first = unknown_first
        self.root = tk.Tk(); self.root.title('RayBEV heavy annotation review — local, autosaved')
        self.root.tk.call('tk', 'scaling', max(float(self.root.tk.call('tk','scaling')), 96/72))
        self.fonts = named_ui_fonts(tkfont, self.root)
        self.font_family = scalable_font_family(tkfont, self.root)
        self.ui_font = tkfont.Font(root=self.root, family=self.font_family)
        self.camera_font = tkfont.Font(root=self.root, family=self.font_family, weight='bold')
        self.style=ttk.Style(self.root)
        self.font_readout=tk.StringVar(master=self.root)
        self.flow_refresh = []
        self.font_size = font_size
        self.set_font_size(font_size)
        style=self.style
        style.configure('TButton',padding=(10,6));style.configure('TCheckbutton',padding=(4,4))
        style.configure('TEntry',padding=4);style.configure('TCombobox',padding=4)
        self.root.option_add('*TCombobox*Listbox.font',self.ui_font)
        self.root.option_add('*Font',self.ui_font)
        width=min(1500,max(700,self.root.winfo_screenwidth()-60))
        height=min(1000,max(500,self.root.winfo_screenheight()-100))
        self.root.geometry(f'{width}x{height}'); self.images = []; self.photos = []
        self.root.rowconfigure(3,weight=1);self.root.columnconfigure(0,weight=1)
        self.root.bind('<Control-plus>',lambda e:self.set_font_size(self.font_size+2))
        self.root.bind('<Control-equal>',lambda e:self.set_font_size(self.font_size+2))
        self.root.bind('<Control-minus>',lambda e:self.set_font_size(self.font_size-2))
        self.root.bind('<Control-KP_Add>',lambda e:self.set_font_size(self.font_size+2))
        self.root.bind('<Control-KP_Subtract>',lambda e:self.set_font_size(self.font_size-2))
        self.split = tk.StringVar(value='all'); self.category = tk.StringVar(value='all')
        self.status = tk.StringVar(value='unknown' if session.issue_records is not None else 'all'); self.search = tk.StringVar()
        self.track = tk.BooleanVar(value=session.issue_records is None); self.both = tk.BooleanVar(value=True)
        self.crop = tk.BooleanVar(value=True); self.advance = tk.BooleanVar(value=True)
        self.context = tk.DoubleVar(value=.6); self.resize = tk.BooleanVar(value=False)
        self.dims = [tk.StringVar() for _ in range(3)]; self.note = tk.StringVar()
        self.progress=ttk.Label(self.root,wraplength=width-30)
        self.progress.grid(row=0,column=0,sticky='ew',padx=10,pady=(5,0))
        self.info=ttk.Label(self.root,wraplength=width-30)
        self.info.grid(row=1,column=0,sticky='ew',padx=10,pady=5)
        self.issue=ttk.Label(self.root,wraplength=width-30,foreground='#922000')
        self.issue.grid(row=2,column=0,sticky='ew',padx=10,pady=5)
        workspace=ttk.Frame(self.root);workspace.grid(row=3,column=0,sticky='nsew')
        workspace.rowconfigure(0,weight=1);workspace.columnconfigure(0,weight=1)
        views=ttk.Frame(workspace);views.grid(row=0,column=0,sticky='nsew')
        sidebar=ttk.Frame(workspace);sidebar.grid(row=0,column=1,sticky='ns',padx=(5,0))
        font_bar=ttk.Frame(sidebar,padding=4);font_bar.pack(side='top',fill='x')
        ttk.Button(font_bar,text='−',width=3,command=lambda:self.set_font_size(self.font_size-2)).pack(side='left')
        ttk.Label(font_bar,textvariable=self.font_readout).pack(side='left',padx=5)
        ttk.Button(font_bar,text='+',width=3,command=lambda:self.set_font_size(self.font_size+2)).pack(side='left')
        self.controls_canvas=tk.Canvas(sidebar,width=min(460,max(320,width//3)),highlightthickness=0)
        scrollbar=ttk.Scrollbar(sidebar,orient='vertical',command=self.controls_canvas.yview)
        self.controls_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right',fill='y');self.controls_canvas.pack(side='left',fill='both',expand=True)
        controls=ttk.Frame(self.controls_canvas)
        controls_item=self.controls_canvas.create_window(0,0,window=controls,anchor='nw')
        self.controls_canvas.bind('<Configure>',lambda e:self.controls_canvas.itemconfigure(controls_item,width=e.width))
        controls.bind('<Configure>',lambda e:self.controls_canvas.configure(scrollregion=self.controls_canvas.bbox('all')))
        ttk.Label(controls,text='REVIEW CONTROLS — scroll ↓',font=self.camera_font).pack(fill='x',padx=8,pady=8)
        top = ttk.Frame(controls, padding=7)
        for title, variable, values in [('Split',self.split,('all','train','val')),
            ('Original class',self.category,('all','bus','van','truck','car','pedestrian','cyclist')),
            ('Status',self.status,('all','unknown','unreviewed','reviewed','uncertain','changed','removed'))]:
            ttk.Label(top,text=title).pack(side='left',padx=4)
            combo=ttk.Combobox(top,textvariable=variable,values=values,width=12,state='readonly')
            combo.pack(side='left'); combo.bind('<<ComboboxSelected>>',lambda e:self.filter())
        ttk.Label(top,text='Find bag / track / token:').pack(side='left',padx=5)
        entry=ttk.Entry(top,textvariable=self.search,width=24); entry.pack(side='left')
        entry.bind('<Return>',lambda e:self.filter())
        ttk.Button(top,text='Filter',command=self.filter).pack(side='left',padx=4)
        self.canvases=[]
        for n in range(4):
            canvas=tk.Canvas(views,bg='#151515',highlightthickness=1,highlightbackground='#555')
            canvas.grid(row=n//2,column=n%2,sticky='nsew',padx=2,pady=2)
            canvas.bind('<Configure>',lambda e:self.paint())
            canvas.bind('<Double-Button-1>',lambda e,n=n:self.open_camera(n))
            self.canvases.append(canvas)
        for n in range(2):views.rowconfigure(n,weight=1);views.columnconfigure(n,weight=1)
        settings=ttk.Frame(controls,padding=6)
        ttk.Checkbutton(settings,text='Object crops',variable=self.crop,command=self.show).pack(side='left')
        ttk.Label(settings,text='Context:').pack(side='left')
        scale=ttk.Scale(settings,from_=.1,to=2.,variable=self.context,command=lambda v:None)
        scale.pack(side='left');scale.bind('<ButtonRelease-1>',lambda e:self.show())
        ttk.Label(settings,text='Double-click a camera for full-resolution inspection.').pack(side='left',padx=10)
        ttk.Checkbutton(settings,text='Label entire bag + track',variable=self.track).pack(side='left')
        ttk.Checkbutton(settings,text='Across TRAIN + VAL',variable=self.both).pack(side='left')
        edit=ttk.Frame(controls,padding=6)
        ttk.Label(edit,text='Note:').pack(side='left');ttk.Entry(edit,textvariable=self.note,width=24).pack(side='left',padx=5)
        ttk.Checkbutton(edit,text='Also set L/W/H (m):',variable=self.resize).pack(side='left',padx=5)
        for d in self.dims:ttk.Entry(edit,textvariable=d,width=7).pack(side='left',padx=2)
        ttk.Label(edit,text='Center/yaw stay fixed. Save with a class/uncertain button.').pack(side='left',padx=5)
        buttons=ttk.Frame(controls,padding=6)
        self.shortcuts={'1':'car','2':'truck','3':'van','4':'bus','5':'pedestrian','6':'cyclist'}
        for key,label in self.shortcuts.items():
            ttk.Button(buttons,text=f'{key}: {label.title()}',command=lambda l=label:self.commit(l)).pack(side='left',padx=3)
        ttk.Button(buttons,text='K: Confirm current',command=lambda:self.commit()).pack(side='left',padx=3)
        ttk.Button(buttons,text='U: Uncertain',command=lambda:self.commit(uncertain=True)).pack(side='left',padx=3)
        ttk.Button(buttons,text='Remove from this frame',command=self.remove_frame).pack(side='left',padx=3)
        nav=ttk.Frame(controls,padding=6)
        ttk.Button(nav,text='← Previous',command=lambda:self.navigate(-1)).pack(side='left',padx=3)
        ttk.Button(nav,text='Next →',command=lambda:self.navigate(1)).pack(side='left',padx=3)
        ttk.Button(nav,text='Undo last action',command=self.undo).pack(side='left',padx=3)
        ttk.Checkbutton(nav,text='Next after saving',variable=self.advance).pack(side='left',padx=8)
        ttk.Label(nav,text='Go to filtered item:').pack(side='left')
        self.goto=tk.StringVar(); box=ttk.Entry(nav,textvariable=self.goto,width=8);box.pack(side='left')
        box.bind('<Return>',lambda e:self.jump())
        ttk.Button(nav,text='Export new annotations + YOLO',command=self.export).pack(side='right',padx=5)
        self.notice=ttk.Label(self.root,wraplength=width-30,text='Green box identifies the target, not necessarily the foreground vehicle. Use Uncertain when occluded.')
        self.notice.grid(row=4,column=0,sticky='ew',padx=10,pady=5)
        for frame in (buttons,nav,top,settings,edit):
            frame.pack(fill='x')
            self.wrap_controls(frame)
        self.root.bind('<Configure>',self.window_resized)
        self.root.bind('<Key>',self.keypress)
        self.root.protocol('WM_DELETE_WINDOW',self.close)
        apply_text_fonts(self.root,self.ui_font,self.camera_font)
        self.ids=[];self.position=0;self.filter(initial=True)
        print(f'Desktop UI {REVIEW_UI_VERSION} | script={Path(__file__).resolve()} | '
              f'font={self.font_family}, {self.font_size} pt / {self.font_pixels} px | '
              f'actual line height={self.ui_font.metrics("linespace")} px',flush=True)

    def set_font_size(self, size):
        self.font_size=min(48,max(10,int(size)))
        # A negative Tk size means actual pixels, independent of desktop DPI.
        # Keep the CLI in points, using a predictable 96-DPI conversion.
        self.font_pixels=round(self.font_size*96/72)
        for font in self.fonts.values():font.configure(family=self.font_family,size=-self.font_pixels)
        self.ui_font.configure(size=-self.font_pixels)
        self.camera_font.configure(size=-(self.font_pixels+2))
        self.style.configure('.',font=self.ui_font)
        for kind in ('TLabel','TButton','TCheckbutton','TEntry','TCombobox'):
            self.style.configure('Review.'+kind,font=self.ui_font)
        apply_text_fonts(self.root,self.ui_font,self.camera_font)
        self.font_readout.set(f'{self.font_size} pt')
        self.root.title(f'RayBEV review [{REVIEW_UI_VERSION}] — {self.font_size} pt / {self.font_pixels} px')
        if hasattr(self,'canvases'):
            for refresh in self.flow_refresh:self.root.after_idle(refresh)
            self.paint()

    def wrap_controls(self, frame):
        children=frame.winfo_children();previous=[None];rows=[]
        for child in children:child.pack_forget()
        def arrange(event=None):
            available=max(frame.winfo_width()-12,300)
            for child in children:
                if isinstance(child,self.ttk.Label):child.configure(wraplength=max(160,min(700,available-16)))
            widths=[child.winfo_reqwidth() for child in children]
            signature=(available,tuple(widths),tuple(child.winfo_reqheight() for child in children))
            if signature==previous[0]:return
            previous[0]=signature
            layout=flow_layout(widths,available)
            needed=max((row for row,column in layout),default=-1)+1
            while len(rows)<needed:rows.append(self.ttk.Frame(frame))
            for child in children:child.pack_forget()
            for i,row_frame in enumerate(rows):
                row_frame.pack_forget()
                if i<needed:row_frame.pack(fill='x')
            for child,(row,column) in zip(children,layout):
                child.pack(in_=rows[row],side='left',padx=4,pady=3)
                # These widgets were created before the row containers. Without
                # raising them, the later-created frames cover every control.
                child.lift()
        frame.bind('<Configure>',arrange)
        self.flow_refresh.append(arrange)
        self.root.after_idle(arrange)

    def window_resized(self,event):
        if event.widget is self.root:
            for label in (self.progress,self.info,self.issue,self.notice):label.configure(wraplength=max(event.width-30,100))

    def filter(self, initial=False, next_pending=False):
        old=self.ids[self.position] if self.ids else self.session.last_id
        text=self.search.get().strip().lower();ids=[]
        for rec in self.session.queue_records():
            d=self.session.current(rec);status=self.status.get()
            if d['status']=='removed' and status!='removed':continue
            if self.split.get()!='all' and rec['split']!=self.split.get():continue
            if self.category.get()!='all' and rec['original_class']!=self.category.get():continue
            if status=='unknown' and not self.session.unknown(rec):continue
            if status=='changed' and d['label']==rec['original_class'] and 'dimensions_lwh' not in d:continue
            if status not in ('all','changed','unknown') and d['status']!=status:continue
            if text and text not in (rec['sequence']+' '+rec['track_id']+' '+rec['token']).lower():continue
            ids.append(rec['id'])
        if self.unknown_first:
            ids.sort(key=lambda key:self.session.review_priority(self.session.by_id[key]))
            if initial and ids and (old not in ids or self.session.review_priority(self.session.by_id[old]) >
                                    self.session.review_priority(self.session.by_id[ids[0]])):
                old=None;self.position=0
        if next_pending:
            old=next((key for key in ids if self.session.current(self.session.by_id[key])['status'] not in ('reviewed','removed')),old)
        self.ids=ids;self.position=ids.index(old) if old in ids else min(self.position,max(0,len(ids)-1))
        self.show()

    def show(self):
        summary=self.session.summary()
        self.progress.config(text=f"Queue {self.position+1 if self.ids else 0}/{len(self.ids)} | "
            f"Total {summary['queued']} | Reviewed {summary['reviewed']} | Uncertain {summary['uncertain']} | "
            f"Unknown remaining {summary['unknown_remaining']} | "
            f"Removed {summary['removed']} | "
            f"Class changes {summary['class_changes']} | Autosave: {self.session.state_path}")
        if not self.ids:
            self.info.config(text='No annotations match this filter.');self.issue.config(text='No pending issues in this filter. Select Status: all to revisit saved items.' if self.session.issue_records is not None else '')
            self.images=[];self.paint();return
        rec=self.session.by_id[self.ids[self.position]];d=self.session.current(rec)
        self.session.last_id=rec['id'];self.session.save()
        lwh=d.get('dimensions_lwh',rec['box_CLIDAR'][3:6])
        self.info.config(text=f"{rec['split'].upper()} | {rec['sequence']} | track {rec['track_id']} | "
            f"token {rec['token']} | annotation index {rec['index']} | {rec['original_class']} → {d['label']} | "
            f"{d['status']} | LWH {lwh[0]:.2f} × {lwh[1]:.2f} × {lwh[2]:.2f} m")
        self.issue.config(text='')
        if self.session.issue_records is not None:
            self.issue.config(text='FLAG: '+rec.get('previous_review_reason','')+' Frame-only edits by default; enable track labeling only after confirming identity. Geometry needs image/LiDAR evidence, not a class change alone.')
        elif rec.get('previous_review_status'):
            self.info.config(text=self.info.cget('text')+f" | Prior: {rec['previous_review_status']} — {rec.get('previous_review_reason','')}")
        self.note.set(d.get('note',''));self.resize.set(False)
        for var,value in zip(self.dims,lwh):var.set(f'{value:.4f}')
        self.goto.set(str(self.position+1))
        try:
            self.images=self.session.render(rec['id'],self.crop.get(),self.context.get())
            self.paint()
        except Exception as error:
            self.images=[];self.paint();self.notice.config(text='IMAGE ERROR: '+str(error))

    def paint(self):
        self.photos=[]
        for n,canvas in enumerate(self.canvases):
            canvas.delete('all')
            if n>=len(self.images):continue
            name,image,_=self.images[n]
            header=self.camera_font.metrics('linespace')+12
            small=fit_preview(image,canvas.winfo_width()-8,canvas.winfo_height()-header-8)
            photo=self.ImageTk.PhotoImage(small);self.photos.append(photo)
            canvas.create_image(canvas.winfo_width()/2,(canvas.winfo_height()+header)/2,image=photo)
            canvas.create_text(8,header/2,text=name,anchor='w',fill='white',font=self.camera_font)

    def open_camera(self, n):
        if n>=len(self.images):return
        name,_,image=self.images[n];win=self.tk.Toplevel(self.root);win.title(name+' — native resolution')
        win.geometry('1150x850');canvas=self.tk.Canvas(win)
        horizontal=self.ttk.Scrollbar(win,orient='horizontal',command=canvas.xview)
        vertical=self.ttk.Scrollbar(win,orient='vertical',command=canvas.yview)
        canvas.configure(xscrollcommand=horizontal.set,yscrollcommand=vertical.set,scrollregion=(0,0,*image.size))
        horizontal.pack(side='bottom',fill='x');vertical.pack(side='right',fill='y');canvas.pack(fill='both',expand=True)
        photo=self.ImageTk.PhotoImage(image);canvas.create_image(0,0,image=photo,anchor='nw');canvas.photo=photo

    def navigate(self, delta):
        if self.ids:self.position=min(max(self.position+delta,0),len(self.ids)-1);self.show()

    def jump(self):
        try:self.position=min(max(int(self.goto.get())-1,0),len(self.ids)-1);self.show()
        except ValueError:self.messagebox.showerror('Invalid item','Enter a numeric item position.')

    def commit(self,label=None,uncertain=False):
        if not self.ids or self.exporting:return
        key=self.ids[self.position]
        try:
            dimensions=[float(v.get()) for v in self.dims] if self.resize.get() else None
            track=self.track.get() and not uncertain
            if track and label is None:
                label=self.session.current(self.session.by_id[key])['label']
            count=len(self.session.targets(key,track,self.both.get()))
            if count>1 and not self.messagebox.askyesno('Track-wide change',
                f'Apply to {count} queued annotations in this track? '+
                ('TRAIN and VAL are included.' if self.both.get() else 'Only the current split is included.')):return
            changed=self.session.apply(key,label,uncertain,self.note.get(),dimensions,track,self.both.get())
            self.notice.config(text=f'Saved {changed} annotation(s). Originals unchanged. '+('Flagged uncertain; label retained.' if uncertain else ''))
            if track and self.advance.get():self.filter(next_pending=True)
            elif self.status.get() in ('unreviewed','uncertain','changed','unknown'):self.filter()
            elif self.advance.get():self.navigate(1)
            else:self.show()
        except Exception as error:self.messagebox.showerror('Not saved',str(error))

    def undo(self):
        if self.exporting:return
        try:
            self.session.undo();self.filter();self.notice.config(text='Last action undone and autosaved.')
        except Exception as error:self.messagebox.showerror('Undo failed',str(error))

    def remove_frame(self):
        if not self.ids or self.exporting:return
        key=self.ids[self.position];rec=self.session.by_id[key]
        if self.session.current(rec)['status']=='removed':return
        if not self.messagebox.askyesno('Remove this object from one frame?',
            f"Remove annotation {rec['index']} from {rec['split'].upper()} frame {rec['token']}?\n"
            f"Bag {rec['sequence']}, track {rec['track_id']}.\n"
            'Other frames of this track are NOT removed. Original files stay intact. Undo is available.'):return
        try:
            self.session.remove(key,self.note.get())
            self.filter(next_pending=self.advance.get())
            self.notice.config(text='Removed one object from this frame only. Other track frames unchanged. Undo restores it; Export applies the removal.')
        except Exception as error:self.messagebox.showerror('Not removed',str(error))

    def keypress(self,event):
        if event.widget.winfo_class() in ('Entry','TEntry','Text','TCombobox'):return
        key=event.keysym.lower()
        if key in self.shortcuts:self.commit(self.shortcuts[key])
        elif key=='k':self.commit()
        elif key=='u':self.commit(uncertain=True)
        elif key=='left':self.navigate(-1)
        elif key in ('right','space'):self.navigate(1)
        elif key=='z':self.undo()

    def export(self):
        from tkinter import filedialog
        if self.exporting:return
        path=filedialog.asksaveasfilename(title='Choose a NEW export directory name',
            initialdir=str(self.output.parent),initialfile=self.output.name)
        if not path:return
        summary=self.session.summary()
        if not self.messagebox.askyesno('Export reviewed dataset',
            f"Reviewed {summary['reviewed']}/{summary['queued']}; uncertain {summary['uncertain']}; "
            f"explicit single-frame removals {summary['removed']}. "
            'Unreviewed/uncertain objects are retained. Export to a new directory?'):return
        self.exporting=True;self.notice.config(text='Exporting and verifying every box + YOLO association. Please wait…')
        def work():
            try:
                report=self.session.export(path)
                message=f"Exported to {path}\n"+json.dumps(report['review_summary'],indent=2)+\
                    f"\nYOLO associations dropped after size edits: {report['extents']['geometry_edited_records_dropped']}"+\
                    f"\nYOLO associations removed with objects: {report['extents']['removed_object_records_dropped']}"
                self.root.after(0,lambda:self.export_done(message,None))
            except Exception as error:
                self.root.after(0,lambda message=str(error):self.export_done(None,message))
        threading.Thread(target=work,daemon=True).start()

    def export_done(self,message,error):
        self.exporting=False
        if error:self.messagebox.showerror('Export failed',error);self.notice.config(text='Export failed; input files remain unchanged.')
        else:self.messagebox.showinfo('Export verified',message);self.notice.config(text='Export complete. Copy the entire exported folder to the training server.')

    def close(self):
        if self.exporting:
            self.messagebox.showwarning('Export in progress','Wait until export finishes.');return
        try:self.session.save()
        except Exception as error:self.messagebox.showerror('Could not save',str(error));return
        self.root.destroy()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True,help='Local dataset directory containing images/')
    parser.add_argument('--train-ann',type=Path)
    parser.add_argument('--val-ann',type=Path)
    parser.add_argument('--extents',type=Path)
    parser.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    parser.add_argument('--state',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--classes',nargs='+',choices=CLASSES)
    parser.add_argument('--remaining-issues',action='store_true',help='Review only the audited final-v5 issues; separate autosave and v6 export')
    parser.add_argument('--issue-contract',type=Path,default=FOLLOWUP_DEFAULT/'input_contract.json',help='Frozen v5 input hashes for --remaining-issues')
    parser.add_argument('--box-frame',choices=['CLIDAR','BASELINK'],default='CLIDAR')
    parser.add_argument('--z-reference',choices=['center','bottom'],default='center')
    parser.add_argument('--summary-only',action='store_true',help='Headless inventory/progress check; opens no window')
    parser.add_argument('--export-only',action='store_true',help='Headless export of existing saved decisions; no images required')
    parser.add_argument('--verify-export',type=Path,help='Headless check of a completed desktop export; no original extent JSON/state needed')
    parser.add_argument('--font-size',type=int,default=20,help='Desktop UI font size in points (default: 20); Ctrl +/- adjusts it live')
    parser.add_argument('--review-ledger',type=Path,help='Prior annotation_review.jsonl; defaults to the TRAIN annotation directory')
    parser.add_argument('--start-with',choices=['unknown','all'],default='unknown',help='Unknown/unresolved objects first (default), or normal track order')
    args=parser.parse_args()
    source = MANUAL_V5 if args.remaining_issues else REVIEW_DEFAULT
    args.train_ann = args.train_ann or source/'fisheye3dod_infos_train.pkl'
    args.val_ann = args.val_ann or source/'fisheye3dod_infos_val.pkl'
    args.extents = args.extents or source/'train_yolo_extents_v1.json'
    args.state = args.state or PROJECT/('work_dirs/manual_v5_followup/decisions.json' if args.remaining_issues else 'work_dirs/heavy_manual_review/decisions.json')
    args.output = args.output or PROJECT/('datasets/combined_bags_camera_ray_manual_review_v6' if args.remaining_issues else 'datasets/combined_bags_camera_ray_manual_review_v1')
    args.classes = args.classes or (list(CLASSES) if args.remaining_issues else ['bus','van','truck'])
    if args.remaining_issues:
        args.review_ledger = args.review_ledger or FOLLOWUP_DEFAULT/'followup_review.jsonl'
    if not 10 <= args.font_size <= 48:
        parser.error('--font-size must be between 10 and 48 points')
    if sum([args.summary_only,args.export_only,args.verify_export is not None]) > 1:
        parser.error('Choose only one of --summary-only, --export-only, --verify-export')
    if args.verify_export is not None:
        print(json.dumps(verify_export(args.root,args.verify_export,args.calibration),indent=2),flush=True)
        return
    if args.state.resolve() in {args.train_ann.resolve(),args.val_ann.resolve(),args.extents.resolve(),args.calibration.resolve()}:
        parser.error('--state must not overwrite an input file')
    session=ReviewSession(args.root,args.train_ann,args.val_ann,args.extents,args.calibration,args.state,
                          args.classes,args.box_frame,args.z_reference,args.review_ledger)
    if args.remaining_issues:
        session.remaining_issues(args.review_ledger,args.issue_contract)
    print(json.dumps(session.summary(),indent=2),flush=True)
    if args.export_only:
        print(json.dumps(session.export(args.output),indent=2),flush=True)
    elif not args.summary_only:
        try:
            import tkinter as tk
            window=ReviewWindow(session,args.output,args.font_size,args.start_with=='unknown')
        except ImportError as error:
            parser.exit(1,'Desktop UI unavailable: '+str(error)+'\nInstall Tkinter/Pillow and run on your laptop desktop.\n')
        except tk.TclError as error:
            parser.exit(1,'Desktop display unavailable: '+str(error)+'\nRun this command on your laptop desktop, not a headless SSH session.\n')
        window.root.mainloop()


if __name__=='__main__':
    main()
