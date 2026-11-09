#!/usr/bin/env python3
"""Run each checkpoint once; score identical predictions against old/reviewed GT.

A label correction is NOT a learned-model improvement. Both result sets are
retained, with input hashes. Optional ROSA uses existing cached predictions only.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CLASSES
from raybev.model import RayBEVLite
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples
from raybev.detection import make_targets,decode_detections
from raybev.diagnostics import center_statistics,summarize_centers
from raybev.metrics import evaluate_detections,heading_diagnostics
from raybev.training import calibration_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--revision',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    p.add_argument('--device',default='cpu')
    p.add_argument('--amp',action='store_true')
    p.add_argument('--rosa-analysis',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new evaluation output directory')
    torch.set_num_threads(4)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=ModelConfig.from_checkpoint(state)
    if tuple(state['classes'])!=CLASSES or state['box_frame']!='CLIDAR':raise ValueError('Unexpected checkpoint class/frame contract')
    h=calibration_sha256(a.calibration)
    if state['calibration_sha256']!=h or state['lidar_calibration_sha256']!=h:raise ValueError('Calibration mismatch')
    from prepare_label_revision import verify_revision
    manifest=verify_revision(a.root,a.revision,a.calibration)
    if manifest['calibration_sha256']!=h:raise ValueError('Review calibration mismatch')
    old=RosaCombinedDataset(a.root,'rosa/fisheye3dod_infos_val.pkl',cfg=cfg,calibration=a.calibration)
    new=RosaCombinedDataset(a.root,(a.revision/'val.pkl').resolve(),cfg=cfg,calibration=a.calibration)
    if len(old)!=len(new):raise ValueError('Revision changed validation length')
    for r,s,b,c in zip(old.rows,new.rows,old.boxes,new.boxes):
        if str(r['token'])!=str(s['token']) or not torch.equal(b[:,:7],c[:,:7]):raise ValueError('Revision changed token/geometry')
    device=torch.device(a.device)
    model=RayBEVLite(cfg,a.calibration).to(device).eval();model.load_state_dict(state['state_dict'],strict=True)
    predictions=[];stats={k:torch.zeros(6,18,device=device) for k in ('original','reviewed')}
    with torch.inference_mode():
        for i,(images,boxes,_) in enumerate(DataLoader(old,batch_size=1,num_workers=0,collate_fn=collate_samples)):
            context=torch.autocast('cuda') if a.amp and device.type=='cuda' else nullcontext()
            with context:output=model(images.to(device))
            output=tuple(v.float() for v in output)
            predictions.append(decode_detections(output,cfg,score_threshold=.01,topk=300)[0].cpu().numpy())
            for key,gt in (('original',boxes),('reviewed',[new.boxes[i]])):
                stats[key]+=center_statistics(output,make_targets(gt,cfg,device=device,return_class_ids=True),cfg)
            if (i+1)%50==0:print('Validation',i+1,'/',len(old),flush=True)
    report=dict(checkpoint=str(a.checkpoint),checkpoint_sha256=calibration_sha256(a.checkpoint),epoch=state['epoch'],
                calibration_sha256=h,review_rules_sha256=manifest['rules_sha256'],
                original_val_sha256=manifest['splits']['val']['source_sha256'],reviewed_val_sha256=manifest['splits']['val']['output_sha256'],
                note='Same predictions, two GT revisions. Metric changes are annotation effects, not training gains.',models={})
    sequences=[str(r['metainfo']['sequence_id']) for r in old.rows]
    gts={'original':[b.numpy() for b in old.boxes],'reviewed':[b.numpy() for b in new.boxes]}
    def evaluate(ps):
        return {key:dict(metrics=evaluate_detections(ps,gt),heading=heading_diagnostics(ps,gt,sequences)) for key,gt in gts.items()}
    report['models']['RayBEV']=evaluate(predictions)
    for key,values in stats.items():report['models']['RayBEV'][key]['center_diagnostics']=summarize_centers(values)
    a.output.mkdir(parents=True)
    if a.rosa_analysis:
        from compare_validation_rosa import prepare,as_rosa_array
        from types import SimpleNamespace
        cache_args=SimpleNamespace(checkpoint=a.checkpoint,root=a.root,calibration=a.calibration,lidar_calibration=a.calibration,
                                   rosa_analysis=a.rosa_analysis,output=a.output/'rosa_provenance')
        _,_,ds,_,rosa=prepare(cache_args)
        ps=[as_rosa_array(rosa[str(r['token'])],cfg,ds.T_base_clidar) for r in old.rows]
        report['models']['ROSA']=evaluate(ps)
    payload=dict(checkpoint_sha256=report['checkpoint_sha256'],calibration_sha256=h,
                 frames=[dict(token=str(row['token']),detections=pred.tolist()) for row,pred in zip(old.rows,predictions)])
    (a.output/'predictions.json').write_text(json.dumps(payload,allow_nan=False)+'\n')
    (a.output/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for name,variants in report['models'].items():
        for key,result in variants.items():print(name,key,'mAP',result['metrics']['mAP'],flush=True)


if __name__=='__main__':main()
