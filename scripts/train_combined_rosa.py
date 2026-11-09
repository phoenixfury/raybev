#!/usr/bin/env python3
"""Train and evaluate RayBEV-Lite on untouched ROSA/Fisheye3DOD annotation layout.

Runs full validation after EACH epoch, logs JSONL, CSV, optional TensorBoard,
last/best checkpoints. Validation mAP omits only classes without val GT.
NDS is NULL: velocity and attributes were intentionally removed from the head.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
import time

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from raybev import CLASSES,ModelConfig,RayBEVLite
from raybev.config import DEFAULT_BOX_FRAME,DEFAULT_CALIBRATION
from raybev.detection import make_targets,detection_loss,decode_detections
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples
from raybev.metrics import evaluate_detections,heading_diagnostics
from raybev.diagnostics import center_statistics,summarize_centers
from raybev.training import configure_image_training,annotation_hashes
from raybev.heading_supervision import HeadingSupervision
from raybev.training import add_training_options,balanced_training_view,build_optimizer,build_scheduler,check_resume,calibration_sha256,fit_training_config,heading_selection_score,initialize_recovery


class BatchProgress:
    """Interactive tqdm bars, or useful periodic logs for redirected output.

    The fallback also works without tqdm installed. No background threads or
    dependence on model output structure. Created after the DataLoader so a
    visible bar exists while the first batch is being prepared.
    """

    def __init__(self, phase, epoch, epochs, batches, mode='auto', log_every=50):
        if log_every < 1:
            raise ValueError('log_every must be >= 1')
        self.phase = phase
        self.epoch = epoch
        self.epochs = epochs
        self.batches = batches
        self.log_every = log_every
        self.index = 0
        self.samples = 0
        self.since = time.monotonic()
        use_bar = tqdm is not None and (mode == 'always' or (mode == 'auto' and sys.stderr.isatty()))
        self.bar = tqdm(total=batches, desc=f'Epoch {epoch}/{epochs} {phase}',
                        dynamic_ncols=True, unit='batch', mininterval=0.5,
                        leave=True, file=sys.stderr) if use_bar else None
        if self.bar is None:
            why = 'non-interactive output' if mode == 'auto' and not sys.stderr.isatty() else 'bar disabled/unavailable'
            print(f'[Epoch {epoch}/{epochs}] {phase} starting: {batches} batches ({why}); '
                  f'log every {log_every} batches', flush=True)

    def update(self, batch_size, stats):
        self.index += 1
        self.samples += batch_size
        elapsed = max(time.monotonic() - self.since, 1e-9)
        fps = self.samples / elapsed
        info = {**stats, 'images/s': f'{fps:.1f}'}
        if self.bar is not None:
            self.bar.update(1)
            self.bar.set_postfix(info, refresh=False)
        elif self.index == 1 or self.index % self.log_every == 0 or self.index == self.batches:
            remaining = elapsed / self.index * (self.batches - self.index)
            details = ' '.join(f'{key}={val}' for key, val in info.items())
            print(f'[Epoch {self.epoch}/{self.epochs} {self.phase}] '
                  f'{self.index}/{self.batches} ({self.index/self.batches:.1%}) '
                  f'{details} ETA={remaining:.0f}s', flush=True)

    def close(self):
        if self.bar is not None:
            self.bar.close()
        print(f'[Epoch {self.epoch}/{self.epochs}] {self.phase} finished: '
              f'{self.samples} samples in {time.monotonic()-self.since:.1f}s', flush=True)


def build_parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='new output of prepare_combined_rosa.py')
    p.add_argument('--train-ann',default='rosa/fisheye3dod_infos_train.pkl')
    p.add_argument('--val-ann',default='rosa/fisheye3dod_infos_val.pkl')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION,
                   help='4 camera OpenCV fisheye calibration, validated for these bag sessions')
    p.add_argument('--box-frame',choices=['BASELINK','CLIDAR'],default=DEFAULT_BOX_FRAME,
                   help='Input label frame (default: CLIDAR); converted once into BASELINK')
    p.add_argument('--lidar-calibration',default=None,
                   help='CLIDAR transform JSON; defaults to the same file as --calibration')
    p.add_argument('--z-reference',choices=['center','bottom'],default='center')
    p.add_argument('--epochs',type=int,default=24)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight-decay',type=float,default=.001)
    p.add_argument('--score-threshold',type=float,default=.3)
    p.add_argument('--map-min-score',type=float,default=.01)
    p.add_argument('--eval-topk',type=int,default=300)
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--amp',action='store_true',help='automatic mixed precision on CUDA')
    p.add_argument('--progress',choices=['auto','always','off'],default='auto',
                   help='tqdm bars in a terminal; periodic logs in non-interactive output')
    p.add_argument('--log-every',type=int,default=50,
                   help='print progress every N batches if bars are disabled (default: 50)')
    p.add_argument('--out',default='work_dirs/raybev_v4_3_r18_geometry_48m_clidar')
    p.add_argument('--resume',default=None)
    add_training_options(p)
    return p


def main():
    args=build_parser().parse_args()
    if args.resume and args.init_from:raise ValueError('--init-from and --resume are mutually exclusive')
    args.lidar_calibration=args.lidar_calibration or args.calibration
    if args.epochs<1 or args.batch_size<1 or args.workers<0 or args.log_every<1:
        raise ValueError('Invalid training controls')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    cfg=ModelConfig(backbone=args.backbone,bev_normalization=args.bev_normalization,height_fusion=args.height_fusion,
                    heavy_classification_weight=args.heavy_classification_weight).validate()
    print('Foreground/box class weights:',dict(zip(CLASSES,zip(cfg.class_positive_weights,cfg.class_regression_weights))),flush=True)
    print(f'Task recipe 4.3: backbone={cfg.backbone}; lattice_origin={cfg.feature_pixel_origin}; heights={cfg.height_fusion}; facing_loss={cfg.facing_loss}; classification={cfg.classification_mode}; heading={cfg.heading_encoding}; BEV_norm={cfg.bev_normalization}; '
          f'size={cfg.size_encoding}; loss_norm={cfg.loss_normalization}; heavy_margin_weight={cfg.heavy_classification_weight}; macro_losses={cfg.class_balanced_losses}; '
          f'center_CE={cfg.center_classification_weight}; sampling={cfg.sampling_mode}',flush=True)
    print(f'Score = objectness * conditional_class_probability * quality ** {cfg.quality_score_power}; peaks={cfg.score_peak_mode}',flush=True)
    ds_args=dict(root=args.root,box_frame=args.box_frame,z_reference=args.z_reference,cfg=cfg,
                 calibration=args.lidar_calibration if args.box_frame=='CLIDAR' else None)
    train=RosaCombinedDataset(ann_file=args.train_ann,augment=not args.no_augmentation,**ds_args)
    val=RosaCombinedDataset(ann_file=args.val_ann,**ds_args)
    if not train or not val:raise ValueError('Empty train/val split')
    cfg=fit_training_config(cfg,train.boxes)
    print(f'TRAIN mean objects/frame={cfg.mean_training_objects:.6f}; shared size reference={cfg.size_reference}',flush=True)
    heading_supervision=HeadingSupervision(train,cfg,args)
    print('TRAIN-only class size priors:',dict(zip(CLASSES,cfg.class_size_priors)),flush=True)
    print('Train boxes:',train.gt_counts,flush=True)
    print('Validation boxes:',val.gt_counts,flush=True)
    active=[k for k,n in val.gt_counts.items() if n>0]
    print(f'Validation active classes: {active}; excluded: {[k for k,n in val.gt_counts.items() if n==0]}',flush=True)
    # Persistent held-out examples, never train on validation/purge membership.
    for row in train.rows:
        if row['metainfo'].get('temporal_split')!='train':raise ValueError('Non-train sample leaked into training split')
    for row in val.rows:
        if row['metainfo'].get('temporal_split')!='val':raise ValueError('Non-val sample leaked into validation split')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin=(device.type=='cuda')
    training_data=train if args.no_balance else balanced_training_view(train,args.seed,cfg)
    print(f'Range-filtered boxes train/val: {train.range_filtered_boxes}/{val.range_filtered_boxes}; balanced entries={len(training_data)}',flush=True)
    dl_train=DataLoader(training_data,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                       collate_fn=collate_samples,pin_memory=pin,persistent_workers=args.workers>0)
    dl_val=DataLoader(val,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,
                     collate_fn=collate_samples,pin_memory=pin,persistent_workers=args.workers>0)
    state=torch.load(args.resume,map_location='cpu',weights_only=False) if args.resume else None
    if state is not None:check_resume(state,cfg,args)
    if state is not None and state.get('world_size')!=1:
        raise ValueError('Resume world size differs from checkpoint')
    model=RayBEVLite(cfg,args.calibration,pretrained=not args.no_pretrained and not args.resume and not args.init_from,
                     backbone_weights=args.backbone_weights if not args.resume and not args.init_from else None).to(device)
    recovery=initialize_recovery(model,cfg,args) if args.init_from else (state.get('warm_start') if state else None)
    initialization=state.get('backbone_initialization') if state else model.image_encoder.initialization
    print('Backbone initialization:',initialization,flush=True)
    configure_image_training(model,args.freeze_image_stages)
    annotation_sha=annotation_hashes(train,val)
    print('Image stage freeze:',args.freeze_image_stages,'annotation SHA256:',annotation_sha,flush=True)
    optimizer=build_optimizer(model,args.lr,args.weight_decay,args.backbone_lr_mult)
    scaler=torch.cuda.amp.GradScaler(enabled=args.amp and pin)
    scheduler=build_scheduler(optimizer,len(dl_train)*args.epochs,args.warmup_steps)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if not args.resume and any((out/f).exists() for f in ('history.jsonl','epochs.csv','last.pt')):
        raise FileExistsError(f'Existing run at {out}; choose a fresh --out or resume a v2 checkpoint')
    writer=None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(str(out/'tensorboard'))
    except ImportError:pass
    best=-1.0;start=0;best_heading=float('inf')
    if args.resume:
        if tuple(state['classes'])!=CLASSES:raise ValueError('Checkpoint class list mismatch')
        if state.get('box_frame')!=args.box_frame:raise ValueError('Checkpoint box frame mismatch')
        model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:scaler.load_state_dict(state['scaler'])
        start=int(state['epoch']);best=float(state['best_map'])
        best_heading=state.get('best_heading_aoe',float('inf'))
    print(f'Device={device}; train={len(train)} frames/{len(dl_train)} batches; '
          f'val={len(val)} frames/{len(dl_val)} batches; epochs={args.epochs}; '
          f'progress={args.progress}',flush=True)
    for epoch in range(start,args.epochs):
        train_centers=torch.zeros(len(CLASSES),18,device=device)
        val_centers=torch.zeros_like(train_centers)
        train_frames=val_frames=0
        t0=time.monotonic();model.train();train_total=0.;train_heat=0.;train_reg=0.
        configure_image_training(model,args.freeze_image_stages)
        train_aux={k:0. for k in ('heading_bin','center_classification','size_regression','heading_teacher','heading_track','heading_axis','heavy_classification','objectness')}
        if pin: torch.cuda.reset_peak_memory_stats(device)
        train_progress=BatchProgress('train',epoch+1,args.epochs,len(dl_train),args.progress,args.log_every)
        for images,boxes,tokens in dl_train:
            images=images.to(device,non_blocking=pin)
            targets=make_targets(boxes,cfg,device=device,return_class_ids=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=args.amp and pin):
                output=model(images);loss,parts=detection_loss(output,targets,cfg=cfg)
                auxiliary,aux_parts=heading_supervision.loss(output,targets,tokens)
                loss=loss+auxiliary;parts.update(aux_parts)
            train_centers+=center_statistics(output,targets,cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scaler.get_scale()>=old_scale:scheduler.step()
            n=len(images);train_frames+=n
            train_total+=float(loss.item())*n;train_heat+=float(parts['heatmap'])*n;train_reg+=float(parts['regression'])*n
            for key in train_aux:train_aux[key]+=float(parts[key])*n
            train_progress.update(n,{'loss':f'{train_total/train_frames:.4f}',
                'heat':f'{train_heat/train_frames:.4f}',
                'box':f'{train_reg/train_frames:.4f}',
                'lr':f'{optimizer.param_groups[0]["lr"]:.2e}'})
        train_progress.close()
        model.eval();val_total=0.;val_heat=0.;val_reg=0.;predictions=[];ground_truth=[]
        val_aux={k:0. for k in ('heading_bin','center_classification','size_regression','heading_axis','heavy_classification','objectness')}
        val_progress=BatchProgress('val',epoch+1,args.epochs,len(dl_val),args.progress,args.log_every)
        with torch.no_grad():
            for images,boxes,_ in dl_val:
                images=images.to(device,non_blocking=pin)
                targets=make_targets(boxes,cfg,device=device,return_class_ids=True)
                output=model(images);loss,parts=detection_loss(output,targets,cfg=cfg)
                val_centers+=center_statistics(output,targets,cfg)
                n=len(images);val_frames+=n
                val_total+=float(loss.item())*n;val_heat+=float(parts['heatmap'])*n;val_reg+=float(parts['regression'])*n
                for key in val_aux:val_aux[key]+=float(parts[key])*n
                det=decode_detections(output,cfg,score_threshold=args.map_min_score,topk=args.eval_topk)
                predictions.extend([x.float().cpu().numpy() for x in det])
                ground_truth.extend([x.float().cpu().numpy() for x in boxes])
                val_progress.update(n,{'loss':f'{val_total/val_frames:.4f}',
                    'heat':f'{val_heat/val_frames:.4f}',
                    'box':f'{val_reg/val_frames:.4f}'})
        val_progress.close()
        print(f'[Epoch {epoch+1}/{args.epochs}] Computing mAP and 3D errors from '
              f'{len(predictions)} validation frames...',flush=True)
        metrics=evaluate_detections(predictions,ground_truth,score_threshold=args.score_threshold)
        diagnostics=heading_diagnostics(predictions,ground_truth,[str(r['metainfo']['sequence_id']) for r in val.rows])
        heading_score=heading_selection_score(diagnostics)
        print(f'[Epoch {epoch+1}/{args.epochs}] Global mAP={metrics["mAP"]:.4f}; computing per-bag results...',flush=True)
        # Optional per-bag diagnostics, never mix their per-bag class sets into the global mAP.
        bag_groups={}
        for idx,sample in enumerate(val.rows):
            bag_groups.setdefault(str(sample['metainfo']['sequence_id']),[]).append(idx)
        per_bag={}
        for bag,ids in bag_groups.items():
            print(f'[Epoch {epoch+1}/{args.epochs}] Evaluating bag {bag}: {len(ids)} frames',flush=True)
            if any(len(ground_truth[i]) for i in ids):
                per_bag[bag]=evaluate_detections([predictions[i] for i in ids],
                    [ground_truth[i] for i in ids],score_threshold=args.score_threshold)
        steps=train_frames;vs=val_frames
        row={'epoch':epoch+1,'lr':optimizer.param_groups[0]['lr'],
             'train_loss':train_total/steps,'train_heatmap_loss':train_heat/steps,'train_regression_loss':train_reg/steps,
             'val_loss':val_total/vs,'val_heatmap_loss':val_heat/vs,'val_regression_loss':val_reg/vs,
             'seconds':time.monotonic()-t0,**metrics,'per_bag_metrics':per_bag,'heading_diagnostics':diagnostics}
        centers={'train':summarize_centers(train_centers),'val':summarize_centers(val_centers)}
        row['center_diagnostics']=centers
        row.update({phase+'_annotation_sha256':value for phase,value in annotation_sha.items()})
        for phase,values,denom in (('train',train_aux,steps),('val',val_aux,vs)):
            row.update({f'{phase}_{key}_loss':value/denom for key,value in values.items()})
            row[f'{phase}_heading_direction_loss']=row[f'{phase}_heading_bin_loss']
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        flat={k:v for k,v in row.items() if k not in ('class_results','active_classes','excluded_classes','AP_distance_thresholds','per_bag_metrics','heading_diagnostics','center_diagnostics')}
        for phase,classes in centers.items():
            for name,values in classes.items():
                for key,value in values.items():flat[f'{phase}_center_{key}_{name}']=value
        for score,classes in diagnostics['by_score'].items():
            for cname,values in classes.items():
                for key in ('matched_count','AOE_deg','axis_AOE_deg','flip_rate','recall'):
                    flat[f'heading_score{score}_{key}_{cname}']=values[key]
        for bag,bm in per_bag.items():
            flat[f'bag_{bag}_mAP']=bm['mAP']
            flat[f'bag_{bag}_mATE']=bm['mATE']
            flat[f'bag_{bag}_mASE']=bm['mASE']
            flat[f'bag_{bag}_mAOE']=bm['mAOE']
        for cname,cm in metrics['class_results'].items():
            flat[f'AP_{cname}']=cm['AP']
            flat[f'GT_{cname}']=cm['gt_count']
            flat[f'prediction_count_{cname}']=cm['prediction_count']
            flat[f'ATE_{cname}']=cm['ATE']
            flat[f'ASE_{cname}']=cm['ASE']
            flat[f'AOE_{cname}']=cm['AOE']
            flat[f'precision_{cname}']=cm['precision']
            flat[f'recall_{cname}']=cm['recall']
            flat[f'TP_{cname}']=cm.get('TP')
            flat[f'FP_{cname}']=cm.get('FP')
            flat[f'FN_{cname}']=cm.get('FN')
            for key in ('heading_matches','axis_AOE','heading_flip_rate','heading_within15_rate'):
                flat[f'{key}_{cname}']=cm.get(key)
            for th,ap in (cm['AP_by_distance'] or {}).items():flat[f'AP_{cname}_at_{th}m']=ap
        csvpath=out/'epochs.csv'
        fields=list(flat)
        if csvpath.exists() and csvpath.stat().st_size:
            with csvpath.open(newline='') as previous:fields=next(csv.reader(previous))
        with csvpath.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
            if f.tell()==0:w.writeheader()
            w.writerow(flat)
        if writer:
            for k,v in flat.items():
                if isinstance(v,(int,float)) and math.isfinite(v):writer.add_scalar(k,v,epoch+1)
        print(f'Epoch {epoch+1}/{args.epochs} train_loss={row["train_loss"]:.4f} '
              f'val_loss={row["val_loss"]:.4f} mAP(active)={metrics["mAP"]:.4f} '
              f'geometry_3term={metrics["geometry_score_3term_nonstandard"]:.4f} '
              f'mATE={metrics["mATE"]:.4f} mASE={metrics["mASE"]:.4f} mAOE={metrics["mAOE"]:.4f} '
              f'NDS=unavailable classes={metrics["active_classes"]}',flush=True)
        ap_summary=' '.join(f'{name}={result["AP"]:.3f}' if result['AP'] is not None else f'{name}=N/A'
                            for name,result in metrics['class_results'].items())
        print(f'[Epoch {epoch+1}/{args.epochs}] Per-class AP: {ap_summary}',flush=True)
        print('Heading car score>=0.3:',diagnostics['by_score']['0.3']['car'],flush=True)
        print('Validation oracle centers (not detection recall):', {k:centers['val'][k] for k in ('car','truck','bus')},flush=True)
        print(f'Prediction/GT counts (decode floor={args.map_min_score}):',
              ' '.join(f'{name}={result["prediction_count"]}/{result["gt_count"]}'
                       for name,result in metrics['class_results'].items()),flush=True)
        print(f'[Epoch {epoch+1}/{args.epochs}] Saving checkpoint(s)...',flush=True)
        state={'state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'epoch':epoch+1,'best_map':max(best,metrics['mAP']),'classes':list(CLASSES),
               'cfg':vars(cfg),'metrics':metrics,'box_frame':args.box_frame,'recipe':vars(args),'world_size':1,
               'scaler':scaler.state_dict(),'backbone_initialization':initialization,'warm_start':recovery,
               'heading_teacher_sha256':heading_supervision.teacher_sha256,
               'heading_diagnostics':diagnostics,
               'annotation_sha256':annotation_sha,
               'center_diagnostics':centers,
               'best_heading_aoe':min(best_heading,heading_score),
               'lidar_calibration_sha256':calibration_sha256(args.lidar_calibration) if args.box_frame=='CLIDAR' else None,
               'calibration_sha256':calibration_sha256(args.calibration)}
        torch.save(state,out/'last.pt')
        if metrics['mAP']>best:
            best=metrics['mAP'];torch.save(state,out/'best_map.pt')
        if heading_score<best_heading:
            best_heading=heading_score;torch.save(state,out/'best_heading.pt')
        if pin:
            print(f'[Epoch {epoch+1}/{args.epochs}] CUDA peak allocated: '
                  f'{torch.cuda.max_memory_allocated(device)/1024**2:.0f} MiB',flush=True)
    if writer:writer.close()
    print(f'Saved metrics and checkpoints in {out}')

if __name__=='__main__':main()
