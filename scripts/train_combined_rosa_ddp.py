#!/usr/bin/env python3
"""Train RayBEV-Lite with torchrun DistributedDataParallel (e.g. 2x RTX 3080 Ti).

One process per GPU; AMP + optional gradient accumulation. Training shards are
reshuffled each epoch. Validation uses *exact* disjoint shards (no padding),
then gathers and evaluates each frame exactly once on rank 0. Epoch metrics,
checkpoints and TensorBoard are written only on rank 0.

Launch: torchrun --standalone --nproc_per_node=2 scripts/train_combined_rosa_ddp.py ...
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from raybev import CLASSES, ModelConfig, RayBEVLite
from raybev.config import DEFAULT_BOX_FRAME,DEFAULT_CALIBRATION
from raybev.detection import make_targets, detection_loss, decode_detections
from raybev.metrics import evaluate_detections,heading_diagnostics
from raybev.heading_supervision import HeadingSupervision
from raybev.rosa_dataset import RosaCombinedDataset, collate_samples
from raybev.training import add_training_options,balanced_training_view,build_optimizer,build_scheduler,check_resume,calibration_sha256,fit_training_config,heading_selection_score,initialize_recovery

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


class ExactEvalSampler(Sampler[int]):
    """Disjoint, non-padded shards for unbiased validation over all examples."""
    def __init__(self, dataset, rank: int, world_size: int):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed sampler configuration")
        self.n = len(dataset)
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, self.n, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.n, self.world_size))


def combine_exact_predictions(per_rank: list, length: int):
    """Reconstruct dataset order and reject duplicates/missing items."""
    predictions = [None] * length
    truths = [None] * length
    seen = set()
    for shard in per_rank:
        for idx, pred, gt in shard:
            if not 0 <= idx < length or idx in seen:
                raise ValueError(f"Duplicate or invalid validation index {idx}")
            seen.add(idx)
            predictions[idx] = pred
            truths[idx] = gt
    if len(seen) != length:
        missing = sorted(set(range(length)) - seen)
        raise ValueError(f"Missing {len(missing)} validation frames: {missing[:10]}")
    return predictions, truths


class Progress:
    """A rank-0-only terminal bar or periodic batch progress for redirected logs."""
    def __init__(self, phase, epoch, epochs, count, mode, log_every):
        self.phase, self.epoch, self.epochs = phase, epoch, epochs
        self.count, self.mode, self.log_every = count, mode, log_every
        self.n, self.images = 0, 0
        self.started = time.monotonic()
        use_tqdm = tqdm is not None and (mode == 'always' or (mode == 'auto' and sys.stderr.isatty()))
        self.bar = tqdm(total=count, desc=f'Epoch {epoch}/{epochs} {phase} GPU0',
                        unit='batch', dynamic_ncols=True, mininterval=.5) if use_tqdm else None
        if self.bar is None:
            print(f'[Epoch {epoch}/{epochs}] {phase} GPU0 started ({count} local batches)', flush=True)

    def update(self, batch_size, stats):
        self.n += 1
        self.images += batch_size
        elapsed = max(time.monotonic()-self.started, 1.e-6)
        msg = dict(stats, gpu0_img_s=f'{self.images/elapsed:.1f}')
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix(msg, refresh=False)
        elif self.n == 1 or self.n % self.log_every == 0 or self.n == self.count:
            eta = elapsed / self.n * (self.count-self.n)
            fields = ' '.join(f'{k}={v}' for k, v in msg.items())
            print(f'[{self.phase} epoch={self.epoch}] {self.n}/{self.count} '
                  f'{self.n/self.count:.1%} ETA={eta:.0f}s {fields}', flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
        print(f'[{self.phase} epoch={self.epoch}] GPU0 processed {self.images} images '
              f'in {time.monotonic()-self.started:.1f}s', flush=True)


def init_distributed(allow_cpu=False):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ or 'LOCAL_RANK' not in os.environ:
        raise RuntimeError('Launch with torchrun --standalone --nproc_per_node=2 (not python).')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 2:
        raise ValueError(f'This launch requires exactly two processes, got world_size={world}')
    if torch.cuda.is_available():
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(f'LOCAL_RANK={local_rank} exceeds visible CUDA devices')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        backend = 'nccl'
    elif allow_cpu:
        device = torch.device('cpu')
        backend = 'gloo'
    else:
        raise RuntimeError('CUDA GPUs not available (use --allow-cpu only for local testing)')
    dist.init_process_group(backend=backend, init_method='env://', timeout=timedelta(minutes=30))
    return rank, world, local_rank, device


def sync_bn_running_buffers(model: torch.nn.Module):
    """Align per-rank BatchNorm running stats with rank 0 for identical validation."""
    for name, buffer in model.named_buffers():
        if name.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
            dist.broadcast(buffer, src=0)


def all_reduce_loss_sums(local: list[float], device: torch.device):
    vals = torch.tensor(local, device=device, dtype=torch.float64)
    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
    return vals.cpu().tolist()


def make_flat_metrics(row):
    metrics = row['metrics']
    flat = {k: row[k] for k in ('epoch','lr','train_loss','train_heatmap_loss',
             'train_regression_loss','val_loss','val_heatmap_loss','val_regression_loss',
             'seconds','world_size','global_batch_size','train_samples_seen')}
    for phase in ('train','val'):
        for component in ('heading_bin','center_classification','size_regression','heading_teacher','heading_track','heading_axis','heading_direction','heavy_classification'):
            key=f'{phase}_{component}_loss'
            if key in row:flat[key]=row[key]
    for key, value in metrics.items():
        if key not in ('class_results','active_classes','excluded_classes',
                       'AP_distance_thresholds','per_bag_metrics'):
            flat[key] = value
    for bag,bm in row['per_bag_metrics'].items():
        for key in ('mAP','mATE','mASE','mAOE'):
            flat[f'bag_{bag}_{key}'] = bm[key]
    for name,cm in metrics['class_results'].items():
        for key in ('AP','gt_count','prediction_count','ATE','ASE','AOE','precision','recall','TP','FP','FN',
                    'heading_matches','axis_AOE','heading_flip_rate','heading_within15_rate'):
            flat[f'{key}_{name}'] = cm.get(key)
        for threshold, ap in (cm.get('AP_by_distance') or {}).items():
            flat[f'AP_{name}_at_{threshold}m'] = ap
    for score,classes in row.get('heading_diagnostics',{}).get('by_score',{}).items():
        for name,values in classes.items():
            for key in ('matched_count','AOE_deg','axis_AOE_deg','flip_rate','recall'):
                flat[f'heading_score{score}_{key}_{name}']=values[key]
    return flat


def save_checkpoint(path: Path, state: dict):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--train-ann', default='rosa/fisheye3dod_infos_train.pkl')
    p.add_argument('--val-ann', default='rosa/fisheye3dod_infos_val.pkl')
    p.add_argument('--calibration', default=DEFAULT_CALIBRATION)
    p.add_argument('--box-frame', choices=['BASELINK','CLIDAR'], default=DEFAULT_BOX_FRAME,
                   help='Input labels default to CLIDAR; the network uses BASELINK')
    p.add_argument('--lidar-calibration', default=None,
                   help='CLIDAR transform JSON; defaults to the same file as --calibration')
    p.add_argument('--z-reference', choices=['center','bottom'], default='center')
    p.add_argument('--epochs', type=int, default=24)
    p.add_argument('--batch-size', type=int, default=1, help='PER GPU microbatch, not global batch')
    p.add_argument('--accum-steps', type=int, default=2, help='Gradient accumulation steps per GPU')
    p.add_argument('--workers', type=int, default=2, help='DataLoader workers PER GPU (0 disables multiprocessing)')
    p.add_argument('--sharing-strategy', choices=['file_system','file_descriptor'], default='file_system',
                   help='CPU tensor IPC: file_system avoids file-descriptor exhaustion on Linux')
    p.add_argument('--prefetch-factor', type=int, default=1,
                   help='Batches prefetched per worker when --workers > 0')
    p.add_argument('--persistent-workers', action='store_true',
                   help='Keep DataLoader workers between epochs (off by default to limit FDs)')
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--weight-decay', type=float, default=.001)
    p.add_argument('--score-threshold', type=float, default=.3)
    p.add_argument('--map-min-score', type=float, default=.01)
    p.add_argument('--eval-topk', type=int, default=300)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--progress', choices=['auto','always','off'], default='auto')
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--out', default='work_dirs/raybev_v4_1_r18_norm_heavy_48m_clidar_ddp')
    p.add_argument('--resume', default=None)
    p.add_argument('--allow-cpu', action='store_true', help=argparse.SUPPRESS)
    add_training_options(p)
    return p



def create_grad_scaler(enabled: bool):
    """Portable FP16 GradScaler for older and newer PyTorch versions."""
    modern_amp = getattr(torch, 'amp', None)
    modern_scaler = getattr(modern_amp, 'GradScaler', None)
    if modern_scaler is not None:
        return modern_scaler('cuda', enabled=enabled)
    # torch.amp.GradScaler is absent in some PyTorch 1.x / 2.x releases.
    return torch.cuda.amp.GradScaler(enabled=enabled)


def cuda_autocast(enabled: bool):
    """Only enter CUDA autocast when AMP is requested on a CUDA device."""
    if not enabled:
        return nullcontext()
    # Modern syntax, with a backwards-compatible fallback.
    if hasattr(torch, 'autocast'):
        return torch.autocast(device_type='cuda', enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def load_trusted_checkpoint(path, map_location):
    """Load locally produced optimizer checkpoints across torch.load API versions.

    Checkpoints contain pickled optimizer metadata. NEVER load checkpoints from
    untrusted sources with weights_only=False.
    """
    import inspect
    if 'weights_only' in inspect.signature(torch.load).parameters:
        return torch.load(path, map_location=map_location, weights_only=False)
    return torch.load(path, map_location=map_location)

def main():
    args = build_parser().parse_args()
    if args.resume and args.init_from:raise ValueError('--init-from and --resume are mutually exclusive')
    args.lidar_calibration = args.lidar_calibration or args.calibration
    if args.epochs < 1 or args.batch_size < 1 or args.accum_steps < 1 or args.workers < 0 or args.log_every < 1 or args.prefetch_factor < 1:
        raise ValueError('Invalid training controls')
    # Each torchrun rank needs this BEFORE constructing worker processes.
    # file_descriptor IPC may keep one FD open for every shared tensor storage;
    # each sample also carries a variable-length box tensor, amplifying FD usage.
    torch.multiprocessing.set_sharing_strategy(args.sharing_strategy)
    rank, world, local_rank, device = init_distributed(args.allow_cpu)
    primary = rank == 0
    writer = None
    try:
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed)  # identical parameter init across GPUs
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(args.seed + rank)
            torch.backends.cudnn.benchmark = False  # conservative VRAM; enable only after profiling
        cfg = ModelConfig(backbone=args.backbone,bev_normalization=args.bev_normalization,
                          heavy_classification_weight=args.heavy_classification_weight).validate()
        if primary:
            print('Foreground/box class weights:',dict(zip(CLASSES,zip(cfg.class_positive_weights,cfg.class_regression_weights))),flush=True)
            print(f'Task recipe 4.1: backbone={cfg.backbone}; heading={cfg.heading_encoding}; BEV_norm={cfg.bev_normalization}; '
                  f'size={cfg.size_encoding}; loss_norm={cfg.loss_normalization}; heavy_margin_weight={cfg.heavy_classification_weight}; macro_losses={cfg.class_balanced_losses}; '
                  f'center_CE={cfg.center_classification_weight}; sampling={cfg.sampling_mode}',flush=True)
            print(f'Score = class_confidence * quality ** {cfg.quality_score_power}; peaks={cfg.score_peak_mode}',flush=True)
        ds_args = dict(root=args.root, box_frame=args.box_frame, z_reference=args.z_reference,cfg=cfg,
                       calibration=args.lidar_calibration if args.box_frame == 'CLIDAR' else None)
        train = RosaCombinedDataset(ann_file=args.train_ann,augment=not args.no_augmentation,**ds_args)
        val = RosaCombinedDataset(ann_file=args.val_ann, **ds_args)
        if len(train) == 0 or len(val) == 0 or len(train) < world:
            raise ValueError('Empty/inadequate train or validation split')
        cfg=fit_training_config(cfg,train.boxes)
        if primary:print(f'TRAIN mean objects/frame={cfg.mean_training_objects:.6f}; shared size reference={cfg.size_reference}',flush=True)
        heading_supervision=HeadingSupervision(train,cfg,args)
        if primary:print('TRAIN-only class size priors:',dict(zip(CLASSES,cfg.class_size_priors)),flush=True)
        if any(s['metainfo'].get('temporal_split')!='train' for s in train.rows):
            raise ValueError('Train/validation leakage in training annotation PKL')
        if any(s['metainfo'].get('temporal_split')!='val' for s in val.rows):
            raise ValueError('Train/validation leakage in validation annotation PKL')
        training_data=train if args.no_balance else balanced_training_view(train,args.seed,cfg)
        train_sampler = DistributedSampler(training_data, num_replicas=world, rank=rank,
                                           shuffle=True, seed=args.seed, drop_last=True)
        eval_sampler = ExactEvalSampler(val,rank,world)
        pin = device.type == 'cuda'
        loader_opts = dict(num_workers=args.workers, collate_fn=collate_samples,
                           pin_memory=pin)
        if args.workers > 0:
            loader_opts['prefetch_factor'] = args.prefetch_factor
            # Avoid retaining train and validation worker pools simultaneously.
            loader_opts['persistent_workers'] = args.persistent_workers
        train_loader = DataLoader(training_data, batch_size=args.batch_size,
                                  sampler=train_sampler, **loader_opts)
        val_loader = DataLoader(val, batch_size=args.batch_size,
                                sampler=eval_sampler, **loader_opts)
        ckpt=load_trusted_checkpoint(args.resume,map_location='cpu') if args.resume else None
        if ckpt is not None:check_resume(ckpt,cfg,args)
        if ckpt is not None and ckpt.get('world_size')!=world:
            raise ValueError('Resume world size differs from checkpoint')
        model_options=dict(pretrained=not args.no_pretrained and not args.resume and not args.init_from,
                           backbone_weights=args.backbone_weights if not args.resume and not args.init_from else None)
        # Rank 0 populates the official weight cache before the second rank opens it.
        if primary:local_model=RayBEVLite(cfg,args.calibration,**model_options).to(device)
        dist.barrier()
        if not primary:local_model=RayBEVLite(cfg,args.calibration,**model_options).to(device)
        dist.barrier()
        recovery=initialize_recovery(local_model,cfg,args) if args.init_from else (ckpt.get('warm_start') if ckpt else None)
        optimizer=build_optimizer(local_model,args.lr,args.weight_decay,args.backbone_lr_mult)
        scaler = create_grad_scaler(enabled=args.amp and pin)
        steps_per_epoch=math.ceil(len(train_loader)/args.accum_steps)
        scheduler=build_scheduler(optimizer,steps_per_epoch*args.epochs,args.warmup_steps)
        out = Path(args.out)
        if primary:
            out.mkdir(parents=True, exist_ok=True)
            if not args.resume and any((out / f).exists() for f in ('history.jsonl','epochs.csv')):
                raise FileExistsError(f'Existing metrics at {out}; use a fresh --out or --resume')
        dist.barrier()
        best, start = -1.0, 0
        best_heading=float('inf')
        if args.resume:
            if tuple(ckpt['classes']) != CLASSES:
                raise ValueError('Checkpoint class list does not match model')
            if ckpt.get('box_frame') != args.box_frame:
                raise ValueError('Checkpoint box-frame does not match requested frame')
            local_model.load_state_dict(ckpt['state_dict'])
            optimizer.load_state_dict(ckpt['optimizer'])
            scheduler.load_state_dict(ckpt['scheduler'])
            if 'scaler' in ckpt:
                scaler.load_state_dict(ckpt['scaler'])
            start, best = int(ckpt['epoch']), float(ckpt['best_map'])
            best_heading=ckpt.get('best_heading_aoe',float('inf'))
        # The non-persistent geometric LUT buffers are large and fixed. They are
        # identical on every rank, so DO NOT broadcast them on every forward.
        model = DDP(local_model, device_ids=[local_rank] if pin else None,
                    output_device=local_rank if pin else None,
                    broadcast_buffers=False, find_unused_parameters=False)
        if primary:
            try:
                import resource
                fd_soft, fd_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            except (ImportError, OSError, ValueError):
                fd_soft, fd_hard = 'unknown', 'unknown'
            print(f'DataLoader IPC={torch.multiprocessing.get_sharing_strategy()}, '
                  f'workers_per_rank={args.workers}, prefetch_factor={args.prefetch_factor}, '
                  f'persistent_workers={args.persistent_workers}, '
                  f'file_descriptor_soft_limit={fd_soft}, hard_limit={fd_hard}',flush=True)
            print(f'DDP ready: world={world}, {device}, '\
                  f'train={len(train)} frames, val={len(val)} frames, '
                  f'microbatch/GPU={args.batch_size}, accumulation={args.accum_steps}, '
                  f'effective batch={world*args.batch_size*args.accum_steps}, '
                  f'train_batches/GPU={len(train_loader)}, val_batches/GPU0={len(val_loader)}',flush=True)
            print('Train class counts:',train.gt_counts,flush=True)
            print(f'Range-filtered boxes train/val: {train.range_filtered_boxes}/{val.range_filtered_boxes}; balanced entries={len(training_data)}',flush=True)
            print('Validation counts:',val.gt_counts,flush=True)
            print('Validation classes excluded due to zero GT:',
                  [n for n,c in val.gt_counts.items() if c==0],flush=True)
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(str(out/'tensorboard'))
            except ImportError:
                pass
        # Start all ranks only after rank 0 creates output and initializes reporting.
        dist.barrier()
        for epoch in range(start,args.epochs):
            start_time = time.monotonic()
            train_sampler.set_epoch(epoch)  # different deterministic shuffle each epoch
            model.train()
            if pin:
                torch.cuda.reset_peak_memory_stats(device)
            local_sums = [0.]*11
            optimizer.zero_grad(set_to_none=True)
            progress = Progress('train',epoch+1,args.epochs,len(train_loader),args.progress,args.log_every) if primary else None
            for i,(images,boxes,tokens) in enumerate(train_loader):
                images = images.to(device, non_blocking=pin)
                targets = make_targets(boxes, cfg, device=device,return_class_ids=True)
                group_start = (i // args.accum_steps) * args.accum_steps
                group_size = min(args.accum_steps, len(train_loader)-group_start)
                is_last_in_group = (i-group_start+1) == group_size
                # Prevent redundant NCCL gradient reductions on microsteps.
                sync_context = nullcontext() if is_last_in_group else model.no_sync()
                with sync_context:
                    with cuda_autocast(enabled=args.amp and pin):
                        output = model(images)
                        loss, parts = detection_loss(output, targets,cfg=cfg)
                        auxiliary,aux_parts=heading_supervision.loss(output,targets,tokens)
                        loss=loss+auxiliary;parts.update(aux_parts)
                    scaler.scale(loss/group_size).backward()
                n = len(images)
                local_sums[0] += float(loss.detach()) * n
                local_sums[1] += float(parts['heatmap']) * n
                local_sums[2] += float(parts['regression']) * n
                local_sums[3] += n
                for j,key in enumerate(('heading_bin','center_classification','size_regression','heading_teacher','heading_track','heading_axis','heavy_classification'),4):
                    local_sums[j]+=float(parts[key])*n
                if is_last_in_group:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(local_model.parameters(),10.)
                    old_scale=scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.get_scale()>=old_scale:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                if progress:
                    progress.update(n, {'loss':f'{local_sums[0]/local_sums[3]:.4f}',
                                        'box':f'{local_sums[2]/local_sums[3]:.4f}',
                                        'lr':f'{optimizer.param_groups[0]["lr"]:.2e}'})
            if progress: progress.close()
            totals = all_reduce_loss_sums(local_sums, device)
            train_loss,train_heat,train_reg = [v / max(totals[3],1) for v in totals[:3]]
            # Use rank-0 BatchNorm running stats on ALL ranks; avoid broadcasting
            # fixed multi-megabyte projection buffers.
            sync_bn_running_buffers(local_model)
            local_model.eval()
            val_sums = [0.]*11
            local_predictions = []
            val_progress = Progress('val',epoch+1,args.epochs,len(val_loader),args.progress,args.log_every) if primary else None
            pos = 0
            with torch.inference_mode():
                for images,boxes,_ in val_loader:
                    images = images.to(device, non_blocking=pin)
                    targets = make_targets(boxes,cfg,device=device,return_class_ids=True)
                    with cuda_autocast(enabled=args.amp and pin):
                        output = local_model(images)  # no DDP collectives in uneven eval
                        loss, parts = detection_loss(output, targets,cfg=cfg)
                    n = len(images)
                    val_sums[0] += float(loss.detach()) * n
                    val_sums[1] += float(parts['heatmap']) * n
                    val_sums[2] += float(parts['regression']) * n
                    val_sums[3] += n
                    for j,key in ((4,'heading_bin'),(5,'center_classification'),(6,'size_regression'),(9,'heading_axis'),(10,'heavy_classification')):
                        val_sums[j]+=float(parts[key])*n
                    decoded = decode_detections(output,cfg,score_threshold=args.map_min_score,topk=args.eval_topk)
                    # ExactEvalSampler yields rank, rank+world, rank+2*world...
                    indices = [rank+world*(pos+j) for j in range(n)]
                    for idx,pred,gt in zip(indices,decoded,boxes):
                        local_predictions.append((idx,pred.float().cpu().numpy(),gt.float().cpu().numpy()))
                    pos += n
                    if val_progress:
                        val_progress.update(n,{'loss':f'{val_sums[0]/val_sums[3]:.4f}',
                                               'box':f'{val_sums[2]/val_sums[3]:.4f}'})
            if val_progress: val_progress.close()
            assert pos == len(eval_sampler), (pos,len(eval_sampler))
            val_totals = all_reduce_loss_sums(val_sums,device)
            val_loss,val_heat,val_reg = [v/max(val_totals[3],1) for v in val_totals[:3]]
            # Full validation results MUST be merged across ranks before AP:
            # per-rank APs cannot be averaged to obtain true global mAP.
            gathered = [None]*world
            dist.all_gather_object(gathered,local_predictions)
            # Advance LR on every rank before saving so a resumed scheduler starts
            # at the correct epoch, including when resuming a 1-GPU checkpoint.
            epoch_lr = optimizer.param_groups[0]['lr']
            if primary:
                predictions,ground_truth = combine_exact_predictions(gathered,len(val))
                print(f'[Epoch {epoch+1}/{args.epochs}] Evaluating global '
                      f'mAP on {len(val)} distinct validation frames...',flush=True)
                metrics = evaluate_detections(predictions,ground_truth,
                                              score_threshold=args.score_threshold)
                diagnostics=heading_diagnostics(predictions,ground_truth,[str(r['metainfo']['sequence_id']) for r in val.rows])
                heading_score=heading_selection_score(diagnostics)
                bag_groups = defaultdict(list)
                for idx,sample in enumerate(val.rows):
                    bag_groups[str(sample['metainfo']['sequence_id'])].append(idx)
                per_bag={}
                for bag,ids in bag_groups.items():
                    if any(len(ground_truth[i]) for i in ids):
                        print(f'  Evaluating bag {bag} ({len(ids)} frames)',flush=True)
                        per_bag[bag] = evaluate_detections(
                            [predictions[i] for i in ids],
                            [ground_truth[i] for i in ids],
                            score_threshold=args.score_threshold)
                elapsed = time.monotonic()-start_time
                row = dict(epoch=epoch+1, lr=epoch_lr,
                           train_loss=train_loss,train_heatmap_loss=train_heat,
                           train_regression_loss=train_reg,val_loss=val_loss,
                           val_heatmap_loss=val_heat,val_regression_loss=val_reg,
                           seconds=elapsed,world_size=world,
                           global_batch_size=world*args.batch_size*args.accum_steps,
                           train_samples_seen=int(totals[3]),metrics=metrics,
                           per_bag_metrics=per_bag,heading_diagnostics=diagnostics)
                for phase,sums in (('train',totals),('val',val_totals)):
                    for j,key in enumerate(('heading_bin','center_classification','size_regression','heading_teacher','heading_track','heading_axis','heavy_classification'),4):
                        row[f'{phase}_{key}_loss']=sums[j]/max(sums[3],1)
                    row[f'{phase}_heading_direction_loss']=row[f'{phase}_heading_bin_loss']
                with (out/'history.jsonl').open('a') as fh:
                    fh.write(json.dumps(row,allow_nan=False)+'\n')
                flat = make_flat_metrics(row)
                csvpath=out/'epochs.csv'
                with csvpath.open('a',newline='') as fh:
                    dw=csv.DictWriter(fh,fieldnames=list(flat),extrasaction='ignore')
                    if fh.tell()==0:dw.writeheader()
                    dw.writerow(flat)
                if writer:
                    for key,value in flat.items():
                        if isinstance(value,(int,float)) and math.isfinite(value):
                            writer.add_scalar(key,value,epoch+1)
                    writer.flush()
                print(f'[Epoch {epoch+1}/{args.epochs}] train={train_loss:.4f}, '
                      f'val={val_loss:.4f}, mAP={metrics["mAP"]:.4f}, '
                      f'mATE={metrics["mATE"]:.4f}, mASE={metrics["mASE"]:.4f}, '
                      f'mAOE={metrics["mAOE"]:.4f}, NDS=unavailable, '
                      f'active_classes={metrics["active_classes"]}',flush=True)
                print('AP per class:', ' '.join(f'{name}={value["AP"]:.4f}'
                      if value['AP'] is not None else f'{name}=N/A'
                      for name,value in metrics['class_results'].items()),flush=True)
                print('Heading car score>=0.3:',diagnostics['by_score']['0.3']['car'],flush=True)
                print(f'Prediction/GT counts (decode floor={args.map_min_score}):',
                      ' '.join(f'{name}={value["prediction_count"]}/{value["gt_count"]}'
                               for name,value in metrics['class_results'].items()),flush=True)
                # Store unwrapped model keys for compatibility with 1-GPU checkpoints.
                state={'state_dict':local_model.state_dict(),
                       'optimizer':optimizer.state_dict(), 'scheduler':scheduler.state_dict(),
                       'scaler':scaler.state_dict(),'epoch':epoch+1,
                       'best_map':max(best,metrics['mAP']),
                       'classes':list(CLASSES),'cfg':vars(cfg),
                       'metrics':metrics,'box_frame':args.box_frame,
                       'recipe':vars(args),
                       'warm_start':recovery,
                       'lidar_calibration_sha256':calibration_sha256(args.lidar_calibration) if args.box_frame=='CLIDAR' else None,
                       'calibration_sha256':calibration_sha256(args.calibration),
                       'backbone_initialization':ckpt.get('backbone_initialization') if ckpt else local_model.image_encoder.initialization,
                       'heading_teacher_sha256':heading_supervision.teacher_sha256,
                       'heading_diagnostics':diagnostics,
                       'best_heading_aoe':min(best_heading,heading_score),
                       'world_size':world,'global_batch_size':world*args.batch_size*args.accum_steps}
                save_checkpoint(out/'last.pt',state)
                if metrics['mAP']>best:
                    best=metrics['mAP']
                    save_checkpoint(out/'best_map.pt',state)
                if heading_score<best_heading:
                    best_heading=heading_score;save_checkpoint(out/'best_heading.pt',state)
                if pin:
                    print(f'[GPU0] CUDA peak allocated: '
                          f'{torch.cuda.max_memory_allocated(device)/1024**2:.0f} MiB',flush=True)
            dist.barrier()   # ensure rank 0 evaluation/checkpoint finished
        if primary:
            print(f'Training complete. Metrics and checkpoints saved under {out}',flush=True)
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
