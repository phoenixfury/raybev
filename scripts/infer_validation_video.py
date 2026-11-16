#!/usr/bin/env python3
"""Run a sparse RayBEV checkpoint and render four fisheye views plus LiDAR BEV."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from raybev.config import CLASSES, DEFAULT_CALIBRATION
from raybev.geometry import load_cameras
from raybev.metrics import evaluate_detections
from raybev.rosa_dataset import RosaCombinedDataset, camera_entry, collate_samples
from raybev.sparse_model import decode_sparse
from raybev.sparse_training import sparse_annotation_sha256
from raybev.training import calibration_sha256
from compare_validation_rosa import (
    CAMERA_ORDER,
    COLORS,
    bev_base,
    cloud,
    draw_bev,
    draw_camera,
    frame_counts,
    sha,
    text,
)
from infer_yolo_rosa_sparse import load_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True,
                        help='Combined dataset root containing images, annotations and summary.json')
    parser.add_argument('--ann', type=Path, default=Path('annotations/fisheye3dod_infos_val.pkl'),
                        help='Validation annotation path, absolute or relative to --root')
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--lidar-calibration', type=Path,
                        help='Defaults to --calibration')
    parser.add_argument('--output', type=Path, required=True, help='Destination .mp4 file')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--fps', type=float, default=2.0, help='Playback FPS, not inference FPS')
    parser.add_argument('--score-threshold', type=float, default=0.3)
    parser.add_argument('--max-frames', type=int,
                        help='Render only the first N validation frames (useful for a smoke test)')
    parser.add_argument('--save-predictions', type=Path,
                        help='Optional JSON output containing all decoded detections')
    return parser.parse_args()


def resolve_annotation(root: Path, ann: Path) -> Path:
    return ann if ann.is_absolute() else root / ann


def load_inputs(args):
    model, state = load_model(args.checkpoint, args.calibration, args.device)
    state['_video_architecture'] = model.cfg.architecture
    lidar_calibration = args.lidar_calibration or args.calibration
    if state.get('lidar_calibration_sha256') != calibration_sha256(lidar_calibration):
        raise ValueError('Sparse checkpoint LiDAR calibration mismatch')

    annotation = resolve_annotation(args.root, args.ann)
    dataset = RosaCombinedDataset(
        args.root,
        annotation,
        cfg=model.cfg,
        box_frame=state['box_frame'],
        strict_size=not state.get('recipe', {}).get('resize_native_images', False),
        native_image_sizes={c.name: (c.width, c.height) for c in load_cameras(args.calibration)}
        if state.get('recipe', {}).get('resize_native_images', False) else None,
        z_reference=state['z_reference'],
        calibration=lidar_calibration,
        expected_size=(model.cfg.image_width, model.cfg.image_height),
    )
    if any(row['metainfo'].get('temporal_split') != 'val' for row in dataset.rows):
        raise ValueError('The video script accepts validation annotations only')
    expected = state.get('semantic_annotation_sha256', {}).get('val')
    if expected is not None and sparse_annotation_sha256(dataset) != expected:
        raise ValueError('Validation labels differ from the checkpoint contract')
    return model, state, dataset


def infer(args, model, dataset):
    predictions, tokens = [], []
    latencies = []
    limit = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    inference_dataset = Subset(dataset, range(limit))
    loader = DataLoader(inference_dataset, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate_samples)
    with torch.inference_mode():
        for images, _boxes, batch_tokens in loader:
            images = images.to(args.device)
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            started = time.perf_counter()
            context = torch.autocast('cuda') if args.amp and str(args.device).startswith('cuda') else nullcontext()
            with context:
                output = model(images)
            if str(args.device).startswith('cuda'):
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - started, len(images)))
            predictions.extend(p.cpu().numpy() for p in decode_sparse(output, model.cfg, 0.01))
            tokens.extend(map(str, batch_tokens))
            if len(tokens) % 50 < len(batch_tokens):
                print(f'Inference {len(tokens)}/{limit}', flush=True)
    return predictions, tokens, latencies


def point_cloud_sources(root: Path):
    summary = json.loads((root / 'summary.json').read_text())
    if 'sequences' in summary:
        return {str(item['name']): Path(item['dataset']) for item in summary['sequences']}
    return {str(name): Path(item['source']) for name, item in summary['bag_stats'].items()}


def point_cloud_path(root: Path, sources, row):
    relative = Path(row['pc_info']['pts_path'])
    bag = str(row['metainfo']['sequence_id'])
    candidates = (root / relative, sources[bag] / relative,
                  sources[bag] / 'lidar_pcd' / relative.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f'LiDAR cloud for token {row["token"]}: {candidates}')


def video_writer(path: Path, fps: float):
    return subprocess.Popen([
        'ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', '1280x900', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(path),
    ], stdin=subprocess.PIPE)


def render(args, state, dataset, predictions, metrics):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}')
    partial = args.output.with_name(args.output.stem + '.partial.mp4')
    if partial.exists():
        raise FileExistsError(f'Partial output already exists: {partial}')

    sources = point_cloud_sources(args.root)
    cameras = {camera.name: camera for camera in load_cameras(args.calibration)}
    limit = len(dataset) if args.max_frames is None else min(args.max_frames, len(dataset))
    process = video_writer(partial, args.fps)
    previous_scene = None
    rendered = 0
    started = time.monotonic()
    cv2.setNumThreads(2)
    try:
        for index in range(limit):
            row = dataset.rows[index]
            meta = row['metainfo']
            gt = dataset.boxes[index].numpy()
            pred = predictions[index]
            pred = pred[pred[:, 8] >= args.score_threshold]

            points = cloud(point_cloud_path(args.root, sources, row)).copy()
            transform = dataset.T_base_clidar
            if transform is not None:
                points[:, :3] = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
            base = bev_base(points)

            canvas = np.full((900, 1280, 3), 15, np.uint8)
            bag = str(meta['sequence_id'])
            scene = str(meta.get('scene_name', bag))
            cut = scene != previous_scene
            previous_scene = scene
            text(canvas, f'VALIDATION {index + 1}/{limit} | bag {bag} | frame {meta.get("source_frame_id", row["token"])}'
                 + (' | SCENE CUT' if cut else ''), (18, 28), .7)
            text(canvas, f'RayBEV {state["_video_architecture"]} | checkpoint epoch {state["epoch"]}',
                 (18, 67), .8, thickness=2)
            text(canvas, f'mAP {100 * metrics["mAP"]:.2f}% | display score >= {args.score_threshold:.2f}',
                 (18, 100), .55)
            canvas[140:780, 0:640] = draw_bev(base, gt, pred)

            for camera_index, name in enumerate(CAMERA_ORDER):
                entry = camera_entry(row['cam_info']['cam_fisheye'], name)
                image = cv2.imread(str(args.root / entry['cam_path']))
                if image is None:
                    raise FileNotFoundError(f'{row["token"]}/{name}: {entry["cam_path"]}')
                image = draw_camera(image, cameras[name], gt, pred)
                image = cv2.resize(image, (320, 256), interpolation=cv2.INTER_AREA)
                left = 640 + (camera_index % 2) * 320
                top = 150 + (camera_index // 2) * 276
                canvas[top:top + 256, left:left + 320] = image
                text(canvas, name, (left + 8, top + 20), .55, thickness=2)

            text(canvas, frame_counts(pred, gt), (18, 815), .65)
            text(canvas, 'BASELINK +/-48m | white dashed: GT | colors: RayBEV predictions', (18, 847), .5)
            for class_index, class_name in enumerate(CLASSES):
                text(canvas, class_name, (15 + class_index * 170, 880), .5,
                     COLORS[class_index], 2)
            process.stdin.write(canvas.tobytes())
            rendered += 1
            if index % 50 == 0:
                print(f'Render {index + 1}/{limit} elapsed={time.monotonic() - started:.1f}s', flush=True)
    finally:
        process.stdin.close()
        code = process.wait()
    if code or rendered != limit:
        raise RuntimeError(f'Incomplete video: ffmpeg={code}, frames={rendered}/{limit}')
    partial.replace(args.output)
    return rendered


def main():
    args = parse_args()
    args.root = args.root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.calibration = args.calibration.resolve()
    if args.lidar_calibration is not None:
        args.lidar_calibration = args.lidar_calibration.resolve()
    args.output = args.output.resolve()
    if args.output.suffix.lower() != '.mp4':
        raise ValueError('--output must end in .mp4')
    if not 0 <= args.score_threshold <= 1:
        raise ValueError('--score-threshold must be between 0 and 1')
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError('--max-frames must be positive')
    manifest_path = args.output.with_suffix('.manifest.json')
    for destination in (args.output, manifest_path, args.save_predictions):
        if destination is not None and destination.exists():
            raise FileExistsError(f'Output already exists: {destination}')
    torch.set_num_threads(2)

    model, state, dataset = load_inputs(args)
    predictions, tokens, latencies = infer(args, model, dataset)
    truth = [dataset.boxes[index].numpy() for index in range(len(predictions))]
    metrics = evaluate_detections(predictions, truth, score_threshold=0.1)
    rendered = render(args, state, dataset, predictions, metrics)

    if args.save_predictions is not None:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(json.dumps({
            'checkpoint_sha256': sha(args.checkpoint),
            'epoch': state['epoch'],
            'frames': [dict(token=token, detections=prediction.tolist())
                       for token, prediction in zip(tokens, predictions)],
        }, allow_nan=False) + '\n')

    measured = latencies[1:] or latencies
    network_ms = 1000 * sum(seconds for seconds, _count in measured) / max(
        sum(count for _seconds, count in measured), 1)
    manifest = {
        'video': str(args.output),
        'video_sha256': sha(args.output),
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': sha(args.checkpoint),
        'checkpoint_epoch': state['epoch'],
        'annotation': str(resolve_annotation(args.root, args.ann)),
        'annotation_sha256': sha(resolve_annotation(args.root, args.ann)),
        'calibration': str(args.calibration),
        'calibration_sha256': calibration_sha256(args.calibration),
        'validation_frames': len(dataset),
        'rendered_frames': rendered,
        'fps': args.fps,
        'display_score_threshold': args.score_threshold,
        'network_ms_per_four_camera_frame': network_ms,
        'network_latency_scope': 'network only; excludes image IO, transfer, decode, metrics and rendering',
        'metrics_at_score_0.1': metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
    print(f'Completed video: {args.output}', flush=True)
    print(f'Manifest: {manifest_path}', flush=True)


if __name__ == '__main__':
    main()
