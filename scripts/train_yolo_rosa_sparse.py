#!/usr/bin/env python3
"""Train Q100/L4 YOLO-ROSA-Lite, standalone or torchrun DDP. No dense-model resume."""
import argparse
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
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
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.sparse_loss import SparseCriterion,SparseYawTeacher
from raybev.sparse_training import fit_sparse_anchors,TrackBalancedSampler,sparse_annotation_sha256
from raybev.training import configure_image_training,build_optimizer,build_scheduler,calibration_sha256,heading_selection_score,accumulation_weight
from raybev.metrics import evaluate_detections,heading_diagnostics
from train_combined_rosa_ddp import ExactEvalSampler,combine_exact_predictions,sync_bn_running_buffers,create_grad_scaler,cuda_autocast,save_checkpoint,Progress

PROJECT=Path(__file__).resolve().parents[1]


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
    p.add_argument('--image-width',type=int,default=800)
    p.add_argument('--image-height',type=int,default=640)
    p.add_argument('--dn-queries',type=int,default=24)
    p.add_argument('--dn-weight',type=float,default=.5)
    p.add_argument('--heading-teacher')
    p.add_argument('--heading-teacher-weight',type=float,default=.2)
    p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--sampling',choices=['track','uniform'],default='track')
    p.add_argument('--no-pretrained',action='store_true',help='Explicit random-backbone ablation only')
    p.add_argument('--backbone-weights')
    p.add_argument('--resume',help='Only the same sparse run/architecture/recipe')
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
    if SparseConfig.from_checkpoint(state)!=cfg:raise ValueError('Sparse resume architecture/config mismatch')
    for key in ('classes','box_frame','z_reference','calibration_sha256','lidar_calibration_sha256','semantic_annotation_sha256'):
        if state.get(key)!=contract[key]:raise ValueError('Sparse resume data/calibration mismatch: '+key)
    if state.get('heading_teacher_sha256')!=teacher_sha:raise ValueError('Sparse resume teacher changed')
    if state.get('world_size')!=world:raise ValueError('Sparse resume world size changed')
    for key in ('epochs','lr','weight_decay','batch_size','accum_steps','backbone_lr_mult','freeze_image_stages',
                'warmup_steps','sampling','no_augmentation','seed','amp','workers'):
        if state['recipe'].get(key)!=getattr(args,key):raise ValueError('Sparse resume recipe changed: '+key)


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
        cfg=SparseConfig(image_width=args.image_width,image_height=args.image_height,queries=args.queries,
            stages=args.stages,channels=args.channels,ffn_channels=2*args.channels,max_detections=args.queries,
            dn_queries=args.dn_queries,dn_weight=args.dn_weight,teacher_weight=args.heading_teacher_weight).validate()
        options=dict(root=args.root,cfg=cfg,box_frame=args.box_frame,z_reference=args.z_reference,
                     calibration=args.lidar_calibration or args.calibration,expected_size=(cfg.image_width,cfg.image_height))
        train=RosaCombinedDataset(ann_file=args.train_ann,augment=not args.no_augmentation,**options)
        val=RosaCombinedDataset(ann_file=args.val_ann,**options)
        if not len(train) or not len(val):raise ValueError('TRAIN and VAL must be nonempty')
        if any(r['metainfo'].get('temporal_split')!='train' for r in train.rows):raise ValueError('Sparse TRAIN split leakage')
        if any(r['metainfo'].get('temporal_split')!='val' for r in val.rows):raise ValueError('Sparse VAL split leakage')
        if {str(r['token']) for r in train.rows}&{str(r['token']) for r in val.rows}:raise ValueError('TRAIN/VAL token overlap')
        cfg,anchors=fit_sparse_anchors(train,cfg,args.seed)
        contract=contracts(train,val,args)
        teacher=SparseYawTeacher(args.heading_teacher,train,args.calibration,args.lidar_calibration) if args.heading_teacher else None
        teacher_sha=teacher.sha256 if teacher else None
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
        if checkpoint:check_resume(checkpoint,cfg,args,contract,teacher_sha,world)
        out=Path(args.out)
        if not args.resume and any((out/f).exists() for f in ('history.jsonl','last.pt','epochs.csv')):
            raise FileExistsError('Existing sparse output; choose a new --out or resume its own checkpoint')
        if primary:out.mkdir(parents=True,exist_ok=True)
        if distributed:dist.barrier()
        # Rank 0 validates/downloads the official COCO weights before other ranks.
        model_options=dict(pretrained=not args.no_pretrained and not args.resume,
                           backbone_weights=args.backbone_weights if not args.resume else None)
        if primary:model=YOLOROSALite(cfg,args.calibration,anchors,**model_options).to(device)
        if distributed:dist.barrier()
        if not primary:model=YOLOROSALite(cfg,args.calibration,anchors,**model_options).to(device)
        configure_image_training(model,args.freeze_image_stages)
        optimizer=build_optimizer(model,args.lr,args.weight_decay,args.backbone_lr_mult)
        criterion=SparseCriterion(cfg,teacher)
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
        components=('classification','center','size','axis','facing','quality','teacher','denoising')
        if primary:
            print('YOLO-ROSA-Lite:',cfg.to_dict(),'TRAIN counts:',train.gt_counts,'VAL counts:',val.gt_counts,flush=True)
            print('Backbone:',model.image_encoder.initialization,'sampling:',args.sampling,'freeze:',args.freeze_image_stages,flush=True)
            print('Only 100 matching queries at inference; denoising/teacher are training-only. No dense BEV.',flush=True)
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
                if progress:progress.update(n,{'loss':f'{float(loss.detach()):.3f}'})
            if progress:progress.close()
            if distributed:dist.all_reduce(sums);sync_bn_running_buffers(model)
            model.eval();val_sums=torch.zeros(2,device=device,dtype=torch.float64);records=[];position=0;ids=list(eval_sampler)
            with torch.inference_mode():
                for images,boxes,tokens in evaluation:
                    targets=[b.to(device) for b in boxes]
                    with cuda_autocast(args.amp and device.type=='cuda'):
                        output=model(images.to(device));loss,_=criterion(output,targets)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite sparse VAL loss')
                    predictions=decode_sparse(output,cfg,.01)
                    val_sums+=val_sums.new_tensor([float(loss)*len(images),len(images)])
                    for pred,gt in zip(predictions,boxes):
                        records.append((ids[position],pred.cpu().numpy(),gt.numpy()));position+=1
            if distributed:
                dist.all_reduce(val_sums);gathered=[None]*world;dist.all_gather_object(gathered,records)
            else:gathered=[records]
            if primary:
                predictions,truths=combine_exact_predictions(gathered,len(val))
                metrics=evaluate_detections(predictions,truths,score_threshold=.1)
                sequences=[str(r['metainfo']['sequence_id']) for r in val.rows]
                heading=heading_diagnostics(predictions,truths,sequences)
                per_bag={}
                for bag in sorted(set(sequences)):
                    selected=[i for i,s in enumerate(sequences) if s==bag]
                    per_bag[bag]=evaluate_detections([predictions[i] for i in selected],[truths[i] for i in selected],score_threshold=.1)
                count=float(sums[1+len(components)])
                row=dict(epoch=epoch+1,train_loss=float(sums[0])/count,val_loss=float(val_sums[0]/val_sums[1]),
                    metrics=metrics,heading_diagnostics=heading,per_bag_metrics=per_bag,
                    train_components={k:float(sums[j])/count for j,k in enumerate(components,1)},
                    train_class_exposures=dict(zip(CLASSES,map(int,sums[2+len(components):].cpu().tolist()))),
                    world_size=world,train_samples_seen=int(count),seconds=time.monotonic()-started,lr=optimizer.param_groups[0]['lr'])
                with (out/'history.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                flat=dict(epoch=epoch+1,train_loss=row['train_loss'],val_loss=row['val_loss'],mAP=metrics['mAP'],lr=row['lr'])
                for cls,m in metrics['class_results'].items():
                    for key in ('AP','precision','recall','TP','FP','FN'):flat[f'{cls}_{key}']=m.get(key)
                for key,values in heading['by_score'].items():
                    for cls,m in values.items():
                        for metric in ('AOE_deg','axis_AOE_deg','flip_rate','recall'):flat[f'heading_{key}_{cls}_{metric}']=m[metric]
                new_csv=not (out/'epochs.csv').exists()
                with (out/'epochs.csv').open('a',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(flat))
                    if new_csv:writer.writeheader()
                    writer.writerow(flat)
                improved=metrics['mAP']>best;best=max(best,metrics['mAP']);h=heading_selection_score(heading)
                improved_heading=h<best_heading;best_heading=min(best_heading,h)
                state=dict(architecture=cfg.architecture,cfg=cfg.to_dict(),state_dict=model.state_dict(),
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                    epoch=epoch+1,best_map=best,best_heading_aoe=best_heading,metrics=metrics,heading_diagnostics=heading,
                    world_size=world,recipe=vars(args),heading_teacher_sha256=teacher_sha,
                    backbone_initialization=model.image_encoder.initialization,backbone_profile=model.image_encoder.backbone_profile,**contract)
                save_checkpoint(out/'last.pt',state)
                if improved:save_checkpoint(out/'best_map.pt',state)
                if improved_heading:save_checkpoint(out/'best_heading.pt',state)
                print(f'Epoch {epoch+1}: TRAIN={row["train_loss"]:.4f} VAL={row["val_loss"]:.4f} mAP={metrics["mAP"]:.6f}',flush=True)
            if distributed:dist.barrier()
    finally:
        if distributed and dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
