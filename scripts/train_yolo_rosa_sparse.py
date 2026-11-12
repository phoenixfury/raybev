#!/usr/bin/env python3
"""Train versioned YOLO-ROSA-Lite, standalone or torchrun DDP. No dense-model resume."""
import argparse
import csv
from dataclasses import replace
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from raybev.config import CLASSES,DEFAULT_CALIBRATION
from raybev.geometry import load_cameras
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.sparse_loss import SparseCriterion
from raybev.sparse_training import fit_sparse_anchors,TrackBalancedSampler,sparse_annotation_sha256
from raybev.training import configure_image_training,build_optimizer,build_scheduler,calibration_sha256,heading_selection_score,accumulation_weight
from raybev.metrics import evaluate_detections,heading_diagnostics
from train_combined_rosa_ddp import ExactEvalSampler,combine_exact_predictions,sync_bn_running_buffers,create_grad_scaler,cuda_autocast,save_checkpoint,Progress

PROJECT=Path(__file__).resolve().parents[1]
COMPONENTS=('classification','center','size','axis','facing','quality','teacher','denoising','facing_binary','extent','proposal')


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--train-ann',default=str(PROJECT/'datasets/heavy_review_v4/train.pkl'))
    p.add_argument('--val-ann',default=str(PROJECT/'datasets/heavy_review_v4/val.pkl'))
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--lidar-calibration')
    p.add_argument('--box-frame',choices=['CLIDAR','BASELINK'],default='CLIDAR')
    p.add_argument('--z-reference',choices=['center','bottom'],default='center')
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--accum-steps',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=.001)
    p.add_argument('--backbone-lr-mult',type=float,default=.1)
    p.add_argument('--freeze-image-stages',choices=['none','early','all'],default='none')
    p.add_argument('--warmup-steps',type=int,default=500)
    p.add_argument('--queries',type=int,default=100)
    p.add_argument('--stages',type=int,default=4)
    p.add_argument('--channels',type=int,default=64)
    p.add_argument('--architecture-version',type=int,choices=[3,4,5],default=3,
                   help='3 reproduces V6; 4 refines geometry; 5 enables proposals and richer observations')
    p.add_argument('--image-width',type=int,default=800)
    p.add_argument('--stride4',action='store_true')
    p.add_argument('--observation-conditioning',action='store_true')
    p.add_argument('--proposal-queries',type=int,default=0)
    p.add_argument('--proposal-grid',type=int,default=24)
    p.add_argument('--ground-z',type=float,default=0.)
    p.add_argument('--proposal-weight',type=float,default=1.)
    p.add_argument('--facing-cosine-prior',type=float,default=1.)
    p.add_argument('--facing-temperature',type=float,default=1.)
    p.add_argument('--no-multi-camera-facing',action='store_true')
    p.add_argument('--extent-annotations')
    p.add_argument('--extent-weight',type=float,default=0.)
    p.add_argument('--image-height',type=int,default=640)
    p.add_argument('--dn-queries',type=int,default=24)
    p.add_argument('--dn-weight',type=float,default=.5)
    p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--resize-native-images',action='store_true',help='Resize matching-aspect native images to the calibrated network size')
    p.add_argument('--expected-sequences',type=int,help='Fail if TRAIN/VAL do not both contain this many bags')
    p.add_argument('--sampling',choices=['track','uniform'],default='track')
    p.add_argument('--no-pretrained',action='store_true',help='Explicit random-backbone ablation only')
    p.add_argument('--backbone-weights')
    initialization=p.add_mutually_exclusive_group()
    initialization.add_argument('--resume',help='Only the same sparse run/architecture/recipe')
    initialization.add_argument('--init-from',help='Audited sparse weights only; starts a NEW optimizer and schedule')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--allow-cpu',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--progress',choices=['auto','always','off'],default='auto')
    p.add_argument('--log-every',type=int,default=50)
    p.add_argument('--out',default='work_dirs/raybev_v6_yolo_rosa_q100_l4_48m_clidar_2x3080ti')
    return p


def contracts(train,val,args):
    return dict(classes=list(CLASSES),box_frame=args.box_frame,z_reference=args.z_reference,
                calibration_sha256=calibration_sha256(args.calibration),
                lidar_calibration_sha256=calibration_sha256(args.lidar_calibration or args.calibration),
                annotation_sha256={k:calibration_sha256(d.annotation_path) for k,d in [('train',train),('val',val)]},
                semantic_annotation_sha256={k:sparse_annotation_sha256(d) for k,d in [('train',train),('val',val)]})


def check_resume(state,cfg,args,contract,teacher_sha,world):
    source=SparseConfig.from_checkpoint(state)
    # Older teacher-free runs stored an inactive .2 weight. Normalizing that
    # metadata does not change the objective or invalidate optimizer state.
    if state.get('heading_teacher_sha256') is None and teacher_sha is None and cfg.teacher_weight==0:
        source=replace(source,teacher_weight=0.)
    if state.get('heading_teacher_sha256')!=teacher_sha:raise ValueError('Sparse resume teacher changed; cannot exactly resume a teacher-trained run without its teacher')
    if source!=cfg:raise ValueError('Sparse resume architecture/config mismatch')
    for key in ('classes','box_frame','z_reference','calibration_sha256','lidar_calibration_sha256','semantic_annotation_sha256'):
        if state.get(key)!=contract[key]:raise ValueError('Sparse resume data/calibration mismatch: '+key)
    if state.get('world_size')!=world:raise ValueError('Sparse resume world size changed')
    for key in ('epochs','lr','weight_decay','batch_size','accum_steps','backbone_lr_mult','freeze_image_stages',
                'warmup_steps','sampling','no_augmentation','seed','amp','workers'):
        if state['recipe'].get(key)!=getattr(args,key):raise ValueError('Sparse resume recipe changed: '+key)
    if state['recipe'].get('resize_native_images',False)!=getattr(args,'resize_native_images',False):
        raise ValueError('Sparse resume image resize policy changed')


def check_warm_start(state,cfg,contract,teacher_sha):
    """Allow an explicit new recipe without silently changing data or geometry."""
    source=SparseConfig.from_checkpoint(state)
    if not ((source.architecture_version==3 and cfg.architecture_version==4) or
            source.architecture_version==cfg.architecture_version==4):
        raise ValueError('Sparse warm start supports only architecture 3 -> 4 or 4 -> 4')
    for key in ('architecture','backbone','image_width','image_height','channels','queries','stages','heads',
                'attention_channels','ffn_channels','num_classes','theta_max_degrees','bev_x_min','bev_x_max',
                'bev_y_min','bev_y_max','min_range','max_range','heading_bins','facing_loss','heading_encoding',
                'mean_training_objects','training_class_counts'):
        if getattr(source,key)!=getattr(cfg,key):raise ValueError('Sparse warm-start geometry/config mismatch: '+key)
    for key in ('classes','box_frame','z_reference','calibration_sha256','lidar_calibration_sha256','semantic_annotation_sha256'):
        if state.get(key)!=contract[key]:raise ValueError('Sparse warm-start data/calibration mismatch: '+key)
    if state.get('heading_teacher_sha256')!=teacher_sha:raise ValueError('Sparse warm-start teacher changed')
    return source


def transfer_sparse_weights(model,state):
    """Strict, enumerated architecture-3 migration, never a permissive strict=False.

    Axis and extent outputs change their learned objective/parameterization. The
    final two centre heads change from angular/log-range to metric corrections.
    Their fresh initialization must be retained rather than reinterpreting old
    numbers. Existing point slots retain identity; six new slots stay initialized.
    """
    source_cfg=SparseConfig.from_checkpoint(state);cfg=model.cfg
    source=state['state_dict'];destination=model.state_dict()
    if source_cfg.architecture_version==cfg.architecture_version==4:
        model.load_state_dict(source,strict=True)
        return dict(source_architecture_version=4,target_architecture_version=4,
                    copied=list(source),expanded=[],reset=[],new=[])
    if source_cfg.architecture_version!=3 or cfg.architecture_version!=4:
        raise ValueError('Unsupported sparse weight transfer')
    new_pattern=re.compile(r'^(?:image_encoder\.top_down\.|layers\.\d+\.(?:head_state|class_context|size_context|evidence\.offsets)\.)')
    extra=sorted(set(source)-set(destination))
    missing=sorted(set(destination)-set(source))
    if extra or any(not new_pattern.match(key) for key in missing):
        raise ValueError('Unexpected sparse warm-start keys: extra='+str(extra)+' missing='+str(missing))
    updated={key:value.clone() for key,value in destination.items()}
    report=dict(source_architecture_version=3,target_architecture_version=4,copied=[],expanded=[],reset=[],new=missing)
    for key,value in source.items():
        target=destination[key]
        match=re.match(r'^layers\.(\d+)\.(axis|size|center)\.(weight|bias)$',key)
        reset=match and (match.group(2) in ('axis','size') or
                        (match.group(2)=='center' and int(match.group(1))>=max(1,cfg.stages-2)))
        if reset:
            if target.shape!=value.shape:raise ValueError('Unexpected reset-head shape: '+key)
            report['reset'].append(key);continue
        if re.match(r'^layers\.\d+\.evidence\.point_query\.(weight|bias)$',key):
            if (value.shape[0]!=7*cfg.heads or target.shape[0]!=13*cfg.heads or
                    value.shape[1:]!=target.shape[1:]):raise ValueError('Unexpected point-query expansion: '+key)
            updated[key][:value.shape[0]].copy_(value)
            report['expanded'].append(key);continue
        if value.shape!=target.shape:raise ValueError('Unexpected sparse warm-start tensor shape: '+key)
        updated[key]=value;report['copied'].append(key)
    model.load_state_dict(updated,strict=True)
    return report


def measured_geometry(metrics):
    """Distinguish actual matched errors from the inherited no-match=1 sentinel."""
    matched={name:values for name,values in metrics['class_results'].items()
             if values.get('heading_matches',0)>0}
    return dict(classes=list(matched),matched_count=sum(m['heading_matches'] for m in matched.values()),
                **{'m'+key:float(np.mean([m[key] for m in matched.values()])) if matched else None
                   for key in ('ATE','ASE','AOE')})


def flat_metrics(row):
    metrics=row['metrics'];geometry=row.get('measured_geometry',measured_geometry(metrics))
    flat={key:row.get(key) for key in ('epoch','train_loss','val_loss','lr','seconds','world_size',
                                     'global_batch_size','train_samples_seen','val_samples_seen')}
    for key in ('mAP','mATE','mASE','mAOE','precision_at_score_threshold','recall_at_score_threshold','score_threshold'):
        flat[key]=metrics.get(key)
    flat['mAOE_deg']=math.degrees(metrics['mAOE'])
    for key in ('mATE','mASE','mAOE','matched_count'):
        flat['measured_'+key]=geometry[key]
    flat['measured_mAOE_deg']=math.degrees(geometry['mAOE']) if geometry['mAOE'] is not None else None
    flat['geometry_macro_includes_no_match_sentinel']=True
    for phase in ('train','val'):
        for name in COMPONENTS:flat[f'{phase}_{name}_loss']=row[f'{phase}_components'][name]
    for cls,values in metrics['class_results'].items():
        for key in ('AP','ATE','ASE','AOE','precision','recall','TP','FP','FN','gt_count','prediction_count',
                    'heading_matches','axis_AOE','heading_flip_rate','heading_within15_rate','status'):
            flat[f'{cls}_{key}']=values.get(key)
        count=values.get('heading_matches',0)
        flat[f'{cls}_geometry_status']='measured' if count else ('no_matches' if values['gt_count'] else 'no_ground_truth')
        for key in ('ATE','ASE','AOE'):
            flat[f'{cls}_measured_{key}']=values.get(key) if count else None
        flat[f'{cls}_measured_AOE_deg']=math.degrees(values['AOE']) if count else None
        for distance in (.5,1.,2.,4.):
            flat[f'{cls}_AP_{distance:g}m']=(values.get('AP_by_distance') or {}).get(str(distance))
    for key,values in row['heading_diagnostics']['by_score'].items():
        for cls,values in values.items():
            for name in ('AOE_deg','axis_AOE_deg','flip_rate','recall','matched_count','gt_count','prediction_count','precision'):
                flat[f'heading_{key}_{cls}_{name}']=values.get(name)
    for stage,records in row.get('raw_query_coverage',{}).items():
        for key,values in records.items():
            for field in ('recall','matched','total'):
                flat[f'raw_{stage}_{key.replace("/","_")}_{field}']=values[field]
    for name,values in row.get('facing_reliability',{}).get('by_class',{}).items():
        for key,value in values.items():flat[f'facing_calibration_{name}_{key}']=value
    return flat


def append_metrics_csv(out,flat):
    """Never append a new schema below an old header when resuming a V6 run."""
    out=Path(out);fields=list(flat);version=1
    while True:
        path=out/('epochs.csv' if version==1 else f'epochs_metrics_v{version}.csv')
        if not path.exists() or not path.stat().st_size:break
        with path.open(newline='') as stream:existing=next(csv.reader(stream))
        if existing==fields:break
        version+=1
    new=not path.exists() or not path.stat().st_size
    with path.open('a',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if new:writer.writeheader()
        writer.writerow(flat)
    return path


def print_epoch_metrics(row):
    metrics=row['metrics'];geometry=row['measured_geometry']
    def number(value,digits=3):return 'NA' if value is None else f'{value:.{digits}f}'
    print(f'Epoch {row["epoch"]}: TRAIN={row["train_loss"]:.4f} VAL={row["val_loss"]:.4f} '
          f'mAP={metrics["mAP"]:.6f} mATE={metrics["mATE"]:.3f}m mASE={metrics["mASE"]:.3f} '
          f'mAOE={metrics["mAOE"]:.3f}rad ({math.degrees(metrics["mAOE"]):.1f}deg)',flush=True)
    print('  Macro geometry above includes inherited 1.0 placeholders for active classes with no matches; '
          f'measured-only: ATE={number(geometry["mATE"])}m ASE={number(geometry["mASE"])} '
          f'AOE={number(math.degrees(geometry["mAOE"]) if geometry["mAOE"] is not None else None)}deg '
          f'({geometry["matched_count"]} raw-score matches, {len(geometry["classes"])} classes).',flush=True)
    for phase in ('train','val'):
        print('  '+phase.upper()+' components: '+' '.join(f'{key}={row[phase+"_components"][key]:.4f}' for key in COMPONENTS),flush=True)
    print('  class          AP       GT     TP     FP     FN  geom_n   ATE_m    ASE   AOE_deg',flush=True)
    for name,values in metrics['class_results'].items():
        count=values.get('heading_matches',0)
        ate=values.get('ATE') if count else None;ase=values.get('ASE') if count else None
        aoe=math.degrees(values['AOE']) if count else None
        print(f'  {name:12s} {number(values["AP"],4):>7} {values["gt_count"]:6d} '
              f'{str(values.get("TP","NA")):>6} {str(values.get("FP","NA")):>6} {str(values.get("FN","NA")):>6} '
              f'{count:7d} {number(ate):>7} {number(ase):>6} {number(aoe,1):>9}',flush=True)
    print('  TP/FP/FN use score >= 0.1 and 2m; geom_n uses retained score >= 0.01 and 2m. '
          'Teacher disabled; VAL excludes denoising/extent losses; TRAIN/VAL total objectives differ.',flush=True)
    coverage=row.get('raw_query_coverage',{})
    if coverage:
        stages=[key for key in coverage if key.startswith('stage_')]
        final=max(stages,key=lambda x:int(x.split('_')[-1]))
        print('  Raw class/score-agnostic XY recall @0.5/1/2/4m: initial -> final (ranges in history/CSV)',flush=True)
        for name in CLASSES:
            def recalls(stage):return '/'.join(number(coverage[stage][f'{name}/all/{t:g}m']['recall']) for t in (.5,1.,2.,4.))
            print(f'  {name:12s} {recalls("initial")} -> {recalls(final)}',flush=True)


def main():
    args=build_parser().parse_args()
    if min(args.epochs,args.batch_size,args.accum_steps,args.log_every)<1 or args.workers<0:
        raise ValueError('Invalid sparse training counts')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    primary=rank==0
    if torch.cuda.is_available():
        torch.cuda.set_device(local);device=torch.device('cuda',local)
    elif args.allow_cpu:device=torch.device('cpu')
    else:raise RuntimeError('CUDA unavailable; --allow-cpu is for verification only')
    distributed=world>1
    if distributed:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',timeout=timedelta(minutes=30))
    try:
        torch.multiprocessing.set_sharing_strategy('file_system')
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
        cfg=SparseConfig(architecture_version=args.architecture_version,image_width=args.image_width,image_height=args.image_height,queries=args.queries,
            stages=args.stages,channels=args.channels,ffn_channels=2*args.channels,max_detections=args.queries,
            stride4=args.stride4,observation_conditioning=args.observation_conditioning,
            proposal_queries=args.proposal_queries,proposal_grid=args.proposal_grid,ground_z=args.ground_z,
            proposal_weight=args.proposal_weight,facing_cosine_prior=args.facing_cosine_prior,
            facing_temperature=args.facing_temperature,multi_camera_facing=not args.no_multi_camera_facing,extent_weight=args.extent_weight,
            dn_queries=args.dn_queries,dn_weight=args.dn_weight,teacher_weight=0.).validate()
        options=dict(root=args.root,cfg=cfg,box_frame=args.box_frame,z_reference=args.z_reference,
                     strict_size=not args.resize_native_images,
                     native_image_sizes={c.name:(c.width,c.height) for c in load_cameras(args.calibration)} if args.resize_native_images else None,
                     calibration=args.lidar_calibration or args.calibration,expected_size=(cfg.image_width,cfg.image_height))
        train=RosaCombinedDataset(ann_file=args.train_ann,augment=not args.no_augmentation,**options)
        val=RosaCombinedDataset(ann_file=args.val_ann,**options)
        if not len(train) or not len(val):raise ValueError('TRAIN and VAL must be nonempty')
        train_sequences={str(r['metainfo']['sequence_id']) for r in train.rows}
        val_sequences={str(r['metainfo']['sequence_id']) for r in val.rows}
        if args.expected_sequences is not None and (len(train_sequences)!=args.expected_sequences or train_sequences!=val_sequences):
            raise ValueError('Expected bag coverage mismatch in TRAIN/VAL')
        if primary:print('Dataset bags:',sorted(train_sequences),'TRAIN:',len(train),'VAL:',len(val),flush=True)
        if any(r['metainfo'].get('temporal_split')!='train' for r in train.rows):raise ValueError('Sparse TRAIN split leakage')
        if any(r['metainfo'].get('temporal_split')!='val' for r in val.rows):raise ValueError('Sparse VAL split leakage')
        if {str(r['token']) for r in train.rows}&{str(r['token']) for r in val.rows}:raise ValueError('TRAIN/VAL token overlap')
        cfg,anchors=fit_sparse_anchors(train,cfg,args.seed)
        contract=contracts(train,val,args)
        teacher_sha=None  # Preserve checkpoint provenance schema; no teacher is loaded.
        from raybev.sparse_extent import VerifiedExtents
        extents=VerifiedExtents(args.extent_annotations,train,args.calibration) if args.extent_annotations else None
        if bool(extents)!=bool(cfg.extent_weight):raise ValueError('Set both --extent-annotations and positive --extent-weight')
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
        if checkpoint:check_resume(checkpoint,cfg,args,contract,teacher_sha,world)
        if checkpoint and checkpoint.get('extent_annotations_sha256')!=(extents.sha256 if extents else None):
            raise ValueError('Resume extent annotations changed')
        initializer=torch.load(args.init_from,map_location='cpu',weights_only=False) if args.init_from else None
        if initializer:check_warm_start(initializer,cfg,contract,teacher_sha)
        out=Path(args.out)
        if not args.resume and any((out/f).exists() for f in ('history.jsonl','last.pt','epochs.csv')):
            raise FileExistsError('Existing sparse output; choose a new --out or resume its own checkpoint')
        if primary:out.mkdir(parents=True,exist_ok=True)
        if distributed:dist.barrier()
        # Rank 0 validates/downloads the official COCO weights before other ranks.
        existing_weights=bool(args.resume or args.init_from)
        model_options=dict(pretrained=not args.no_pretrained and not existing_weights,
                           backbone_weights=args.backbone_weights if not existing_weights else None)
        if primary:model=YOLOROSALite(cfg,args.calibration,anchors,**model_options).to(device)
        if distributed:dist.barrier()
        if not primary:model=YOLOROSALite(cfg,args.calibration,anchors,**model_options).to(device)
        warm_start=checkpoint.get('warm_start') if checkpoint else None
        if initializer:
            transfer_report=transfer_sparse_weights(model,initializer)
            model.image_encoder.initialization=initializer['backbone_initialization']
            warm_start=dict(checkpoint=str(Path(args.init_from).resolve()),checkpoint_sha256=calibration_sha256(args.init_from),
                            epoch=initializer['epoch'],transfer_report=transfer_report,
                            optimizer='fresh',scheduler='fresh',source_cfg=initializer['cfg'])
        configure_image_training(model,args.freeze_image_stages)
        optimizer=build_optimizer(model,args.lr,args.weight_decay,args.backbone_lr_mult)
        criterion=SparseCriterion(cfg,teacher=None,projector=model.projector,extents=extents)
        if args.sampling=='track':sampler=TrackBalancedSampler(train,args.seed,rank,world)
        else:sampler=DistributedSampler(train,num_replicas=world,rank=rank,seed=args.seed,drop_last=False)
        eval_sampler=ExactEvalSampler(val,rank,world)
        loader_options=dict(batch_size=args.batch_size,num_workers=args.workers,collate_fn=collate_samples,
                            pin_memory=device.type=='cuda')
        if args.workers:loader_options['prefetch_factor']=1
        loader=DataLoader(train,sampler=sampler,**loader_options)
        evaluation=DataLoader(val,sampler=eval_sampler,**loader_options)
        steps=math.ceil(len(loader)/args.accum_steps)
        scheduler=build_scheduler(optimizer,steps*args.epochs,args.warmup_steps)
        scaler=create_grad_scaler(args.amp and device.type=='cuda')
        start=0;best=-1.;best_heading=float('inf')
        if checkpoint:
            model.load_state_dict(checkpoint['state_dict'],strict=True)
            model.image_encoder.initialization=checkpoint['backbone_initialization']
            optimizer.load_state_dict(checkpoint['optimizer']);scheduler.load_state_dict(checkpoint['scheduler']);scaler.load_state_dict(checkpoint['scaler'])
            start=checkpoint['epoch'];best=checkpoint['best_map'];best_heading=checkpoint['best_heading_aoe']
        wrapped=DDP(model,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False) if distributed else model
        components=COMPONENTS
        if primary:
            print('YOLO-ROSA-Lite:',cfg.to_dict(),'TRAIN counts:',train.gt_counts,'VAL counts:',val.gt_counts,flush=True)
            print('Backbone:',model.image_encoder.initialization,'sampling:',args.sampling,'freeze:',args.freeze_image_stages,flush=True)
            print(f'Only {cfg.queries} matching queries at inference; denoising is training-only. Teacher disabled (weight=0). No dense BEV.',flush=True)
            if warm_start:
                report=warm_start['transfer_report']
                print(f'Warm start: source epoch={warm_start["epoch"]} sha256={warm_start["checkpoint_sha256"]}; '
                      f'copied={len(report["copied"])} expanded={len(report["expanded"])} '
                      f'reset={len(report["reset"])} new={len(report["new"])}; '
                      f'optimizer/schedule are {"resumed" if checkpoint else "fresh"}.',flush=True)
        for epoch in range(start,args.epochs):
            # Epoch-boundary resume is deterministic without carrying worker RNG
            # pools: every epoch reconstructs its seeded loader workers.
            torch.manual_seed(args.seed+1000*epoch+rank);np.random.seed(args.seed+1000*epoch+rank);random.seed(args.seed+1000*epoch+rank)
            sampler.set_epoch(epoch);wrapped.train();configure_image_training(model,args.freeze_image_stages)
            sums=torch.zeros(1+len(components)+1+len(CLASSES),device=device,dtype=torch.float64)
            optimizer.zero_grad(set_to_none=True);started=time.monotonic()
            progress=Progress('sparse train',epoch+1,args.epochs,len(loader),args.progress,args.log_every) if primary else None
            for i,(images,boxes,tokens) in enumerate(loader):
                targets=[b.to(device) for b in boxes];images=images.to(device)
                last=(i+1)%args.accum_steps==0 or i+1==len(loader)
                from contextlib import nullcontext
                context=wrapped.no_sync() if distributed and not last else nullcontext()
                with context:
                    with cuda_autocast(args.amp and device.type=='cuda'):
                        output=wrapped(images,targets)
                        loss,parts=criterion(output,targets,tokens)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sparse TRAIN loss')
                    weight=accumulation_weight(i,len(images),args.batch_size,args.accum_steps,len(sampler))
                    scaler.scale(loss*weight).backward()
                if last:
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                    scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
                    if scaler.get_scale()>=scale:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                n=len(images);sums[0]+=float(loss.detach())*n
                for j,k in enumerate(components,1):sums[j]+=float(parts[k])*n
                sums[1+len(components)]+=n
                for box in boxes:
                    labels=box[box[:,7]>=0,7].long()
                    sums[2+len(components):]+=torch.bincount(labels,minlength=len(CLASSES)).to(device)
                if progress:progress.update(n,dict(loss=f'{float(loss.detach()):.3f}',
                    **{key:f'{float(parts[key]):.3f}' for key in ('classification','center','size','axis','facing')}))
            if progress:progress.close()
            if distributed:dist.all_reduce(sums);sync_bn_running_buffers(model)
            model.eval();val_sums=torch.zeros(2+len(components),device=device,dtype=torch.float64);records=[];raw_records=[];position=0;ids=list(eval_sampler)
            from raybev.sparse_diagnostics import output_raw_counts,merge_raw_counts,facing_samples,summarize_facing
            facing_records=[]
            val_progress=Progress('sparse validation',epoch+1,args.epochs,len(evaluation),args.progress,args.log_every) if primary else None
            with torch.inference_mode():
                for images,boxes,tokens in evaluation:
                    targets=[b.to(device) for b in boxes]
                    with cuda_autocast(args.amp and device.type=='cuda'):
                        output=model(images.to(device));loss,parts=criterion(output,targets)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sparse VAL loss')
                    predictions=decode_sparse(output,cfg,.01)
                    raw_records.extend(output_raw_counts(output,boxes,cfg))
                    facing_records.extend(facing_samples(output,boxes,cfg))
                    val_sums[:2]+=val_sums.new_tensor([float(loss)*len(images),len(images)])
                    for j,key in enumerate(components,2):val_sums[j]+=float(parts[key])*len(images)
                    for pred,gt in zip(predictions,boxes):
                        records.append((ids[position],pred.cpu().numpy(),gt.numpy()));position+=1
                    if val_progress:val_progress.update(len(images),dict(loss=f'{float(loss):.3f}',
                        center=f'{float(parts["center"]):.3f}',axis=f'{float(parts["axis"]):.3f}'))
            if val_progress:val_progress.close()
            if position!=len(ids):raise RuntimeError('Sparse validation shard coverage mismatch')
            if distributed:
                dist.all_reduce(val_sums);gathered=[None]*world;dist.all_gather_object(gathered,records)
            else:gathered=[records]
            if distributed:
                gathered_raw=[None]*world;dist.all_gather_object(gathered_raw,raw_records)
            else:gathered_raw=[raw_records]
            if distributed:
                gathered_facing=[None]*world;dist.all_gather_object(gathered_facing,facing_records)
            else:gathered_facing=[facing_records]
            if primary:
                predictions,truths=combine_exact_predictions(gathered,len(val))
                print(f'[Epoch {epoch+1}/{args.epochs}] Evaluating {len(val)} distinct validation frames globally...',flush=True)
                metrics=evaluate_detections(predictions,truths,score_threshold=.1)
                sequences=[str(r['metainfo']['sequence_id']) for r in val.rows]
                heading=heading_diagnostics(predictions,truths,sequences)
                per_bag={}
                for bag in sorted(set(sequences)):
                    selected=[i for i,s in enumerate(sequences) if s==bag]
                    per_bag[bag]=evaluate_detections([predictions[i] for i in selected],[truths[i] for i in selected],score_threshold=.1)
                count=float(sums[1+len(components)])
                row=dict(epoch=epoch+1,train_loss=float(sums[0])/count,val_loss=float(val_sums[0]/val_sums[1]),
                    raw_query_coverage=merge_raw_counts([r for shard in gathered_raw for r in shard]),
                    facing_reliability=summarize_facing([r for shard in gathered_facing for r in shard]),
                    metrics=metrics,heading_diagnostics=heading,per_bag_metrics=per_bag,
                    train_components={k:float(sums[j])/count for j,k in enumerate(components,1)},
                    val_components={k:float(val_sums[j]/val_sums[1]) for j,k in enumerate(components,2)},
                    measured_geometry=measured_geometry(metrics),
                    train_class_exposures=dict(zip(CLASSES,map(int,sums[2+len(components):].cpu().tolist()))),
                    world_size=world,global_batch_size=world*args.batch_size*args.accum_steps,
                    train_samples_seen=int(count),val_samples_seen=int(val_sums[1]),
                    seconds=time.monotonic()-started,lr=optimizer.param_groups[0]['lr'])
                with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                csv_path=append_metrics_csv(out,flat_metrics(row))
                if csv_path.name!='epochs.csv':print('Extended metric CSV:',csv_path,flush=True)
                improved=metrics['mAP']>best;best=max(best,metrics['mAP']);h=heading_selection_score(heading)
                improved_heading=h<best_heading;best_heading=min(best_heading,h)
                state=dict(architecture=cfg.architecture,cfg=cfg.to_dict(),state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                    epoch=epoch+1,best_map=best,best_heading_aoe=best_heading,metrics=metrics,heading_diagnostics=heading,
                    world_size=world,recipe=vars(args),heading_teacher_sha256=teacher_sha,
                    extent_annotations_sha256=extents.sha256 if extents else None,
                    backbone_initialization=model.image_encoder.initialization,backbone_profile=model.image_encoder.backbone_profile,
                    warm_start=warm_start,**contract)
                save_checkpoint(out/'last.pt',state)
                if improved:save_checkpoint(out/'best_map.pt',state)
                if improved_heading:save_checkpoint(out/'best_heading.pt',state)
                print_epoch_metrics(row)
            if distributed:dist.barrier()
    finally:
        if distributed and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
