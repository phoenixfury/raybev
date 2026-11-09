#!/usr/bin/env python3
"""Probe a trusted checkpoint at GT centers; optionally reevaluate cached ROSA.

No training, no teacher inference, no checkpoint changes. Sampled oracle-center
statistics are not detection AP. Logged full-validation metrics are labeled as
such; corrected ASE requires reevaluating detections, not editing historical logs.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CLASSES
from raybev.model import RayBEVLite
from raybev.rosa_dataset import RosaCombinedDataset
from raybev.detection import make_targets
from raybev.diagnostics import center_statistics,summarize_centers
from raybev.training import calibration_sha256


def select(dataset,per_bag,rare):
    groups=defaultdict(list)
    for i,row in enumerate(dataset.rows):groups[str(row['metainfo']['sequence_id'])].append(i)
    chosen=set()
    for indices in groups.values():
        chosen.update(indices[j] for j in np.linspace(0,len(indices)-1,min(per_bag,len(indices)),dtype=int))
    for ci in (1,2,3,5):
        indices=[i for i,b in enumerate(dataset.boxes) if bool((b[:,7]==ci).any())]
        if indices:chosen.update(indices[j] for j in np.linspace(0,len(indices)-1,min(rare,len(indices)),dtype=int))
    return sorted(chosen)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    p.add_argument('--lidar-calibration',type=Path)
    p.add_argument('--output',type=Path,required=True,help='New report directory')
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--val-per-bag',type=int,default=12)
    p.add_argument('--train-per-bag',type=int,default=8)
    p.add_argument('--rare-per-class',type=int,default=4)
    p.add_argument('--rosa-analysis',type=Path,help='Existing verified ROSA cache directory, not a ROSA launch')
    args=p.parse_args()
    if min(args.threads,args.val_per_bag,args.train_per_bag)<1 or args.rare_per_class<0:
        raise ValueError('Invalid probe sizes')
    args.lidar_calibration=args.lidar_calibration or args.calibration
    if args.output.exists():raise ValueError('Choose a new report directory; existing reports are never overwritten')
    torch.set_num_threads(args.threads)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_checkpoint(state)
    if tuple(state.get('classes',()))!=CLASSES or state.get('calibration_sha256')!=calibration_sha256(args.calibration):
        raise ValueError('Checkpoint class/camera calibration mismatch')
    if state['box_frame']=='CLIDAR' and state.get('lidar_calibration_sha256')!=calibration_sha256(args.lidar_calibration):
        raise ValueError('Checkpoint CLIDAR calibration mismatch')
    device=torch.device(args.device)
    model=RayBEVLite(cfg,args.calibration).to(device).eval()
    model.load_state_dict(state['state_dict'],strict=True)
    report=dict(checkpoint=str(args.checkpoint),sha256=calibration_sha256(args.checkpoint),epoch=state['epoch'],
                cfg=vars(cfg),logged_full_validation_metrics=state['metrics'],
                logged_metric_version=state['metrics'].get('metric_version','legacy_envelope_ASE'),
                finite_parameters=all(not v.is_floating_point() or bool(torch.isfinite(v).all()) for v in state['state_dict'].values()),
                probe_note='Unaugmented, stratified samples, oracle GT centers; means, not medians. NOT detection AP/recall.',
                center_diagnostics={},sample_tokens={})
    for split,per_bag in (('val',args.val_per_bag),('train',args.train_per_bag)):
        ds=RosaCombinedDataset(args.root,'rosa/fisheye3dod_infos_'+split+'.pkl',cfg=cfg,
                              box_frame=state['box_frame'],calibration=args.lidar_calibration,
                              z_reference=state.get('recipe',{}).get('z_reference','center'))
        if any(r['metainfo']['temporal_split']!=split for r in ds.rows):raise ValueError('Invalid dataset split')
        indices=select(ds,per_bag,args.rare_per_class)
        stats=torch.zeros(len(CLASSES),18,device=device)
        with torch.inference_mode():
            for j,i in enumerate(indices):
                images,boxes,_=ds[i]
                targets=make_targets([boxes],cfg,device=device,return_class_ids=True)
                stats+=center_statistics(model(images[None].to(device)),targets,cfg)
                if (j+1)%12==0:print(split,j+1,'/',len(indices),flush=True)
        report['center_diagnostics'][split]=summarize_centers(stats)
        report['sample_tokens'][split]=[str(ds.rows[i]['token']) for i in indices]
    args.output.mkdir(parents=True)
    if args.rosa_analysis:
        from compare_validation_rosa import prepare,as_rosa_array
        from raybev.metrics import evaluate_detections,heading_diagnostics
        from types import SimpleNamespace
        cache_args=SimpleNamespace(checkpoint=args.checkpoint,root=args.root,calibration=args.calibration,
                                   lidar_calibration=args.lidar_calibration,rosa_analysis=args.rosa_analysis,
                                   output=args.output/'rosa_provenance')
        _,_,ds,_,rosa=prepare(cache_args)
        predictions=[as_rosa_array(rosa[str(r['token'])],cfg,ds.T_base_clidar) for r in ds.rows]
        gt=[b.numpy() for b in ds.boxes]
        report['ROSA_cached_full_validation']=evaluate_detections(predictions,gt,score_threshold=.3)
        report['ROSA_heading_diagnostics']=heading_diagnostics(predictions,gt,[str(r['metainfo']['sequence_id']) for r in ds.rows])
        report['ROSA_note']='Identical current GT/frame/range/evaluator; native NMS ROSA=.5, RayBEV=.25; not official ROSA FDS/NDS.'
    destination=args.output/'audit.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Wrote',destination,flush=True)


if __name__=='__main__':main()
