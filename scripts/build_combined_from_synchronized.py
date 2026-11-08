#!/usr/bin/env python3
"""Build a NEW combined ROSA/RayBEV dataset DIRECTLY from synchronized bags.

No --base or pre-existing combined PKLs. Frame annotation conversion is delegated
ONLY to the user's original DetZero->ROSA converter, via:
  append_improved_detzero_rosa_dataset.load_sequence()
  append_improved_detzero_rosa_dataset.assign_chronological_split()
  build_improved_detzero_rosa_dataset.build_rosa_anchor()
Those existing functions are intentionally not reimplemented with guessed label
semantics: this wrapper builds native/resized camera paths and PKL views safely.

The source synchronized dataset should contain manifest.csv, four camera image
folders, detzero_output/improved_results.json, and its per-bag template info PKL.
The converter modules (plus ocamcamera.py) must be importable, e.g. --builders-dir.

Images and labels are synchronized by exact LiDAR SENSOR timestamps first,
falling back to exact recorder timestamps. Do not guess by row index unless the
source supplies a frame index and an acquisition timestamp agreeing with it.

The source bag folders are read-only. The output must not exist already.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

CAMERAS = ("ARC_F", "ARC_R", "ARC_SL", "ARC_SR")
MANIFEST_COLS = {"ARC_F": "arc_f_path", "ARC_R": "arc_r_path",
                 "ARC_SL": "arc_sl_path", "ARC_SR": "arc_sr_path"}
CLASS_NAMES = ("Car", "Pedestrian", "Cyclist", "Truck", "Van", "Bus")
PATH_KEYS = ("cam_path", "img_path", "image_path", "data_path")
SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,90}$")


def safe_rel(value):
    p = Path(str(value))
    if p.is_absolute() or '..' in p.parts or str(p) in ('.', ''):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return p


def checksum(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def publish_pickle(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(obj, f, protocol=4)


def alias_pickle(canonical, alias):
    alias.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(canonical, alias)
    except OSError:
        shutil.copy2(canonical, alias)


def link_or_copy(src, dst, storage):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise FileExistsError(dst)
    if storage == 'copy':
        shutil.copy2(src, dst)
    elif storage == 'hardlink':
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError(storage)


def dataset_root(path):
    p = Path(path).expanduser().resolve()
    if (p / 'manifest.csv').is_file():
        return p
    if (p / 'dataset' / 'manifest.csv').is_file():
        return p / 'dataset'
    raise FileNotFoundError(f"No manifest.csv in {p} or {p / 'dataset'}")


class Manifest:
    def __init__(self, root):
        self.root = dataset_root(root)
        with (self.root / 'manifest.csv').open(newline='') as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"Empty manifest: {self.root}")
        self.by_sensor = {}
        self.by_recorder = {}
        for i, row in enumerate(self.rows):
            for key, mapping in (("lidar_sensor_timestamp_ns", self.by_sensor),
                                 ("timestamp_ns", self.by_recorder)):
                stamp = row.get(key)
                if not stamp:
                    continue
                n = int(stamp)
                if n in mapping:
                    raise ValueError(f"Duplicate {key}={n} at row {i} in {self.root}")
                mapping[n] = row
        for row in self.rows:
            for cam, key in MANIFEST_COLS.items():
                if not row.get(key):
                    raise ValueError(f"Missing manifest {key} for {cam}, timestamp={row.get('timestamp_ns')}")

    def match(self, sample):
        meta = sample.get('metainfo') or {}
        for k, mapping in (("acquisition_timestamp_ns", self.by_sensor),
                           ("lidar_sensor_timestamp_ns", self.by_sensor),
                           ("timestamp_ns", self.by_recorder),
                           ("lidar_recorder_timestamp_ns", self.by_recorder)):
            stamp = meta.get(k)
            if stamp is not None:
                row = mapping.get(int(stamp))
                if row is not None:
                    return row
        # Some converters make their sample token equal to the recorder timestamp.
        token = str(sample.get('token', ''))
        if token.isdecimal():
            row = self.by_recorder.get(int(token))
            if row is not None:
                return row
        # Only accept source_frame_id when timestamp cross-validation is possible.
        idx = meta.get('source_frame_id')
        if idx is not None and 0 <= int(idx) < len(self.rows):
            row = self.rows[int(idx)]
            stamp = meta.get('acquisition_timestamp_ns') or meta.get('lidar_sensor_timestamp_ns')
            if stamp is not None and int(row['lidar_sensor_timestamp_ns']) == int(stamp):
                return row
        raise KeyError(f"No verified manifest row for token={token} in {self.root}; "
                       f"metainfo keys={sorted(meta)}")


def resized_image(src, dst, native_size, target_size):
    with Image.open(src) as im:
        # Keep camera sensor pixel orientation unchanged: transforms/calibration
        # refer to RAW pixel coordinates, not auto-transposed JPEG display.
        orientation = im.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError(f'EXIF orientation {orientation} in {src}; calibrate/normalize it explicitly')
        im = im.convert('RGB')
        if native_size and tuple(im.size) != tuple(native_size):
            raise ValueError(f"Native camera size mismatch: {src}: {im.size} != {native_size}")
        sw, sh = im.size
        tw, th = target_size
        if sw * th != sh * tw:
            raise ValueError(f"Aspect ratio mismatch for {src}: {sw}x{sh} -> {tw}x{th}. "
                             'Recalibrate / explicitly implement letterboxing rather than stretch.')
        dst.parent.mkdir(parents=True, exist_ok=True)
        im.resize((tw, th), Image.Resampling.BILINEAR).save(
            dst, 'JPEG', quality=93, subsampling=0)


def set_paths(sample, path_for_camera, manifest_row, source_root, size, native_size, camera_calibration=None):
    info = sample.setdefault('cam_info', {})
    if not isinstance(info, dict):
        raise TypeError(f"cam_info must be dict for token {sample.get('token')}")
    cams = info.setdefault('cam_fisheye', {})
    if not isinstance(cams, dict):
        raise TypeError(f"cam_info.cam_fisheye must be dict for token {sample.get('token')}")
    scale_x = size[0] / native_size[0] if native_size else None
    scale_y = size[1] / native_size[1] if native_size else None
    for cam in CAMERAS:
        record = cams.setdefault(cam, {})
        if not isinstance(record, dict):
            raise TypeError(f"Camera record {cam} is not a dict for token={sample.get('token')}")
        rel = str(path_for_camera[cam])
        for key in PATH_KEYS:
            if key == 'cam_path' or key in record:
                record[key] = rel
        for tag in ('cam_timestamp_ns', 'camera_timestamp_ns', 'camera_delta_ns'):
            value = manifest_row.get(cam.lower() + ('_timestamp_ns' if 'timestamp' in tag else '_delta_ns'))
            if value:
                record[tag] = int(value)
        record['image_size_wh'] = list(size)
        if native_size:
            record['native_image_size_wh'] = list(native_size)
            record['native_to_image_scale_xy'] = [scale_x, scale_y]
        # Explicit sensor->BASELINK extrinsic provenance when supplied.
        # Do not overwrite converter-defined ROSA calibration keys or pretend
        # that the separate OpenCV approximation is identical to OCam.
        if camera_calibration is not None and cam in camera_calibration:
            entry = camera_calibration[cam]
            T = np.asarray(entry['matrix'], dtype=np.float64)
            if T.shape != (4, 4) or not np.allclose(T[3], [0, 0, 0, 1]):
                raise ValueError(f'Invalid extrinsic T_BASELINK_{cam}')
            if entry.get('from') != cam or entry.get('to') != 'BASELINK':
                raise ValueError(f'Extrinsics frame mismatch for {cam}')
            record['T_BASELINK_CAMERA'] = T.tolist()
            record['T_CAMERA_BASELINK'] = np.linalg.inv(T).tolist()
            record['calibration_reference'] = 'calibration_baselink.json'
    return sample


def source_counts(sample):
    """Validate converted box/label consistency before publishing its PKLs."""
    ann = sample.get('annos') or {}
    names = ann.get('gt_names')
    if names is None:
        raise KeyError(f"Missing annos.gt_names for {sample.get('token')}")
    boxes = next((ann[key] for key in ('gt_boxes', 'gt_boxes_3d', 'boxes_3d')
                  if key in ann), None)
    if boxes is None:
        raise KeyError(f"Missing annos.gt_boxes for {sample.get('token')}")
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        if len(names):
            raise ValueError(f"Empty boxes but nonempty classes for {sample.get('token')}")
    elif boxes.ndim != 2 or boxes.shape[1] < 7 or boxes.shape[0] != len(names):
        raise ValueError(f"GT box shape/class mismatch token={sample.get('token')}: "
                         f"boxes={boxes.shape}, names={len(names)}")
    if 'gt_ids' in ann and len(ann['gt_ids']) != len(names):
        raise ValueError(f"GT ids/classes mismatch for {sample.get('token')}")
    counts = Counter(str(x) for x in names)
    unknown = set(counts) - set(CLASS_NAMES)
    if unknown:
        raise ValueError(f"Unknown class labels for {sample.get('token')}: {sorted(unknown)}. "
                         'Configure a verified source taxonomy rather than silently map.')
    return counts


def split_publish(root, rows, meta):
    subsets = {'all': rows,
               'train': [s for s in rows if s['metainfo']['temporal_split'] == 'train'],
               'val': [s for s in rows if s['metainfo']['temporal_split'] == 'val']}
    results = {}
    for split, seq in subsets.items():
        rel = Path('annotations') / f'fisheye3dod_infos_{split}.pkl'
        canon = root / rel
        publish_pickle(canon, {'metainfo': copy.deepcopy(meta), 'data_list': seq})
        for view in ('rosa', 'fisheye3dod', 'ImageSets-improved-detzero'):
            alias_pickle(canon, root / view / canon.name)
        results[split] = checksum(canon)
    return results


def import_existing_builders(directory):
    if directory:
        sys.path.insert(0, str(Path(directory).expanduser().resolve()))
    try:
        builder = importlib.import_module('build_improved_detzero_rosa_dataset')
        appender = importlib.import_module('append_improved_detzero_rosa_dataset')
        from ocamcamera import OcamCamera
    except ImportError as exc:
        raise SystemExit('Missing your existing DetZero->ROSA builder modules. '
                         'Pass --builders-dir DIRECTORY containing '
                         'build_improved_detzero_rosa_dataset.py, '
                         'append_improved_detzero_rosa_dataset.py, and ocamcamera.py '
                         '(or install the ocamcamera package). ' + str(exc)) from exc
    for key in ('load_sequence', 'assign_chronological_split'):
        if not callable(getattr(appender, key, None)):
            raise RuntimeError(f'Missing appender.{key}')
    if not callable(getattr(builder, 'build_rosa_anchor', None)):
        raise RuntimeError('Missing builder.build_rosa_anchor')
    return builder, appender, OcamCamera


def build(args):
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    if not args.native_dataset:
        raise ValueError('Supply at least one --native-dataset BAG PATH')
    names = [name for name, _ in args.native_dataset]
    if len(names) != len(set(names)) or any(not SAFE.fullmatch(name) for name in names):
        raise ValueError(f'Duplicate/unsafe bag names: {names}')
    sources = {name: Manifest(path) for name, path in args.native_dataset}
    for name, source in sources.items():
        if output == source.root or source.root in output.parents:
            raise ValueError('Output must not be inside any source dataset')
        for rel in ('detzero_output/improved_results.json',
                    'fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl'):
            if not (source.root / rel).is_file():
                raise FileNotFoundError(f"{name} missing {rel} under {source.root}")
    calibration = Path(args.ocam_calibration).expanduser().resolve()
    if not calibration.is_file():
        raise FileNotFoundError(calibration)
    if not 0 < args.val_fraction < 1 or args.purge_seconds < 0 or args.max_scene_gap_seconds <= 0:
        raise ValueError('Invalid split settings')
    if args.native_width < 1 or args.native_height < 1:
        raise ValueError('Invalid native image dimensions')
    builder, appender, OcamCamera = import_existing_builders(args.builders_dir)
    ocam = OcamCamera(filename=str(calibration), fov=220.)
    cal_json = None
    if args.calibration_json:
        cal_json = json.loads(Path(args.calibration_json).expanduser().read_text())
        if not isinstance(cal_json.get('extrinsics'), dict):
            raise ValueError('Expected calibration JSON with extrinsics map')
        if any(cam not in cal_json['extrinsics'] for cam in CAMERAS):
            raise ValueError('Calibration JSON missing some ARC_* extrinsics')
    explicit = {}
    for name, train_frames, val_start in args.sequence_boundaries:
        if name not in sources or name in explicit:
            raise ValueError(f'Unknown or duplicate split boundary name {name}')
        explicit[name] = int(train_frames), int(val_start)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '.building-', dir=output.parent))
    try:
        all_rows, by_bag, next_id = [], {}, 0
        used_tokens = set()
        meta = None
        stats = {}
        all_native_rows = []
        for name, source in sources.items():
            # Matches the exact legacy build convention from the user's script.
            new, entry, next_id = appender.load_sequence(
                name, source.root, staging, ocam, args.storage, False, next_id, False)
            if not isinstance(new, list) or not new:
                raise ValueError(f'No annotations returned for {name} by load_sequence()')
            if not isinstance(entry, dict):
                raise TypeError(f'load_sequence metadata entry should be a dict: {name}')
            # Preserve the actual metainfo from the per-bag source template where available.
            with (source.root / 'fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl').open('rb') as f:
                template = pickle.load(f)  # trusted user's own converter pickle only
            if not isinstance(template, dict) or 'metainfo' not in template:
                raise ValueError(f"Unexpected template annotation metadata for {name}")
            if meta is None:
                meta = copy.deepcopy(template['metainfo'])
            ordered = sorted(new, key=lambda s: int(s['metainfo']['acquisition_timestamp_ns']))
            if name in explicit:
                ntrain, vstart = explicit[name]
                if not 0 < ntrain <= vstart < len(ordered):
                    raise ValueError(f'Invalid boundaries for {name}: {ntrain} {vstart} / {len(ordered)}')
                for i, s in enumerate(ordered):
                    s['metainfo']['temporal_split'] = ('train' if i < ntrain else
                                                      'gap' if i < vstart else 'val')
            else:
                appender.assign_chronological_split(
                    ordered, args.val_fraction, args.purge_seconds, args.max_scene_gap_seconds)
            converted, native_samples = [], []
            perbag_counts = Counter()
            for s in ordered:
                token = str(s.get('token', ''))
                if not token or token in used_tokens:
                    raise ValueError(f'Duplicate/empty cross-bag sample token: {name}/{token}')
                used_tokens.add(token)
                row = source.match(s)
                cam_out, cam_native = {}, {}
                token_file = (token if SAFE.fullmatch(token) else
                              hashlib.sha256(token.encode()).hexdigest()[:32])
                for cam, key in MANIFEST_COLS.items():
                    original_rel = safe_rel(row[key]); source_img = source.root / original_rel
                    if not source_img.is_file():
                        raise FileNotFoundError(f'Missing {name}/{cam} image: {source_img}')
                    cam_folder = cam.lower()
                    native_rel = Path('native_images') / name / cam_folder / f'{token_file}.jpg'
                    resized_rel = Path('images_800x640') / name / cam_folder / f'{token_file}.jpg'
                    link_or_copy(source_img, staging / native_rel, args.storage)
                    resized_image(source_img, staging / resized_rel,
                                  (args.native_width, args.native_height), (800, 640))
                    cam_out[cam], cam_native[cam] = resized_rel, native_rel
                r = copy.deepcopy(s)
                n = copy.deepcopy(s)
                set_paths(r, cam_out, row, source.root, (800, 640),
                          (args.native_width, args.native_height),
                          cal_json['extrinsics'] if cal_json else None)
                set_paths(n, cam_native, row, source.root,
                          (args.native_width, args.native_height),
                          (args.native_width, args.native_height),
                          cal_json['extrinsics'] if cal_json else None)
                if r['metainfo']['temporal_split'] not in ('train', 'val', 'gap'):
                    raise ValueError(f'Invalid split for {name}/{token}')
                perbag_counts.update(source_counts(r))
                converted.append(r)
                native_samples.append(n)
            by_bag[name] = converted
            all_rows.extend(converted)
            all_native_rows.extend(native_samples)
            stats[name] = {'frames': len(converted), 'labels': dict(perbag_counts),
                           'input_frames': len(source.rows),
                           'split_counts': dict(Counter(s['metainfo']['temporal_split'] for s in converted)),
                           'source': str(source.root)}
            print(f"{name}: {len(converted)} usable frames, {sum(perbag_counts.values())} labels", flush=True)
        membership = {str(s['token']): s['metainfo']['temporal_split'] for s in all_rows}
        train, val, purged = appender.assign_chronological_split(
            all_rows, args.val_fraction, args.purge_seconds,
            args.max_scene_gap_seconds, membership)
        if not train or not val:
            raise ValueError('Empty train or validation split')
        # Ensure membership was not silently rewritten by the split helper.
        if any(s['metainfo']['temporal_split'] != membership[str(s['token'])] for s in all_rows):
            raise AssertionError('Existing split membership changed')
        meta = dict(meta or {})
        meta['classes'] = CLASS_NAMES
        meta['version'] = 'raybev-built-directly-from-synchronized-v1'
        hashes = split_publish(staging, all_rows, meta)
        # Native views are rooted at the SAME dataset directory, so all cam_path
        # fields resolve without fragile ../ references or symlinks.
        for split, rows_native in (('all', all_native_rows),
                                   ('train', [s for s in all_native_rows if s['metainfo']['temporal_split'] == 'train']),
                                   ('val', [s for s in all_native_rows if s['metainfo']['temporal_split'] == 'val'])):
            native_pkl = staging / 'annotations' / 'native' / f'fisheye3dod_infos_{split}.pkl'
            publish_pickle(native_pkl, {'metainfo': copy.deepcopy(meta), 'data_list': rows_native})
            alias_pickle(native_pkl, staging / 'rosa_native' / native_pkl.name)
        native_by_bag = defaultdict(list)
        for s in all_native_rows:
            native_by_bag[str(s['metainfo']['sequence_id'])].append(s)
        for name, rows in by_bag.items():
            publish_pickle(staging / 'annotations' / 'by_bag_native' / f'{name}.pkl',
                           {'metainfo': copy.deepcopy(meta), 'data_list': native_by_bag[name]})
            publish_pickle(staging / 'annotations' / 'by_bag' / f'{name}.pkl',
                           {'metainfo': copy.deepcopy(meta), 'data_list': rows})
            for split in ('train', 'val'):
                sub = [s for s in rows if s['metainfo']['temporal_split'] == split]
                publish_pickle(staging / 'annotations' / 'by_bag' / f'{name}_{split}.pkl',
                               {'metainfo': copy.deepcopy(meta), 'data_list': sub})
        anchor, training_boxes = builder.build_rosa_anchor(train, 300, 3407)
        anchor_dir = staging / 'rosa' / 'anchors'
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_path = anchor_dir / 'kmeans300_seed3407.npy'
        np.save(anchor_path, anchor)
        shutil.copy2(anchor_path, staging / 'improved_detzero_kmeans300_seed3407.npy')
        shutil.copy2(calibration, staging / 'calib_results.txt')
        if args.calibration_json:
            shutil.copy2(Path(args.calibration_json), staging / 'calibration_baselink.json')
        summary = {'output': str(output), 'source': 'individual synchronized dataset/manifest.csv + improved_results.json',
                   'frames': len(all_rows), 'train_frames': len(train), 'val_frames': len(val),
                   'purged_frames': int(purged),
                   'native_images': len(all_rows) * len(CAMERAS),
                   'resized_images': len(all_rows) * len(CAMERAS),
                   'class_counts': dict(sum((source_counts(s) for s in all_rows), Counter())),
                   'train_class_counts': dict(sum((source_counts(s) for s in train), Counter())),
                   'val_class_counts': dict(sum((source_counts(s) for s in val), Counter())),
                   'bag_stats': stats, 'camera_order': list(CAMERAS),
                   'native_image_size_wh': [args.native_width, args.native_height],
                   'train_image_size_wh': [800, 640],
                   'annotation_sha256': hashes,
                   'ocam_calibration_sha256': checksum(calibration),
                   'anchor_sha256': checksum(anchor_path),
                   'rosa_anchor_shape': list(anchor.shape),
                   'anchor_training_boxes': int(training_boxes),
                   'metric_note': 'Omit classes with zero validation GT from mAP averaging; report their AP as N/A.',
                   'split_note': 'New split computed independently unless explicit per-sequence boundaries supplied.'}
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(output)
        os.rename(staging, output)
        print('Published ' + str(output), flush=True)
        print(json.dumps({k: summary[k] for k in ('frames', 'train_frames', 'val_frames',
                     'purged_frames', 'class_counts', 'val_class_counts')}, indent=2))
        return summary
    except Exception:
        print('Build failed; staging preserved for inspection: ' + str(staging), file=sys.stderr)
        raise


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--native-dataset', nargs=2, action='append', required=True,
                   metavar=('BAG', 'SYNCHRONIZED_DATASET'))
    p.add_argument('--output', required=True)
    p.add_argument('--builders-dir', default=None,
                   help='directory with build_improved_detzero_rosa_dataset.py and append_improved_detzero_rosa_dataset.py')
    p.add_argument('--ocam-calibration', required=True, help='calib_results.txt (not an old combined annotation)')
    p.add_argument('--calibration-json', default=None, help='optional ARC_* extrinsics provenance JSON')
    p.add_argument('--storage', choices=('hardlink', 'copy'), default='hardlink')
    p.add_argument('--val-fraction', type=float, default=0.15)
    p.add_argument('--purge-seconds', type=float, default=5.0)
    p.add_argument('--max-scene-gap-seconds', type=float, default=4.0)
    p.add_argument('--sequence-boundaries', nargs=3, action='append', default=[],
                   metavar=('BAG', 'TRAIN_FRAMES', 'VAL_START_FRAME'))
    p.add_argument('--native-width', type=int, default=1920)
    p.add_argument('--native-height', type=int, default=1536)
    return p


if __name__ == '__main__':
    build(make_parser().parse_args())
