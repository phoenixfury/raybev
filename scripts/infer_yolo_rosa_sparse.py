#!/usr/bin/env python3
"""Validate/export predictions from a trusted sparse checkpoint, never retrain."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from raybev.config import CLASSES,DEFAULT_CALIBRATION
from raybev.geometry import load_cameras
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples
from raybev.sparse_training import sparse_annotation_sha256
from raybev.training import calibration_sha256
from raybev.metrics import evaluate_detections,heading_diagnostics


def load_model(checkpoint,calibration=DEFAULT_CALIBRATION,device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=SparseConfig.from_checkpoint(state)
    if tuple(state.get('classes',()))!=CLASSES or state.get('calibration_sha256')!=calibration_sha256(calibration):
        raise ValueError('Sparse checkpoint class/camera contract mismatch')
    model=YOLOROSALite(cfg,calibration).to(device).eval();model.load_state_dict(state['state_dict'],strict=True)
    return model,state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--ann',help='Defaults to combined annotations/VAL for eight-bag runs, reviewed four-bag VAL for legacy runs')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--lidar-calibration')
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--amp',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--audit',action='store_true',help='Per-query misses and FP breakdown')
    p.add_argument('--raw-queries',type=Path,help='Optional compressed NPZ with pre-NMS queries')
    p.add_argument('--sweep',action='store_true',help='Replay quality exponents and NMS variants at fixed Car recall')
    p.add_argument('--car-recall',type=float,default=.8,help='Car recall operating point for the sweep')
    p.add_argument('--quality-power',type=float)
    p.add_argument('--nms-iou',type=float)
    p.add_argument('--vehicle-duplicates',action='store_true')
    p.add_argument('--class-bias',type=float,nargs=len(CLASSES),help='Logit offsets fitted on a separate calibration split, in CLASSES order')
    p.add_argument('--operating-floor',type=float,default=.1,help='Base operating/audit score floor, independent of benchmark AP')
    p.add_argument('--class-thresholds',type=float,nargs=len(CLASSES),help='Operating thresholds in CLASSES order; benchmark AP stays unfiltered')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new sparse inference output')
    if a.raw_queries and (a.raw_queries.exists() or a.raw_queries.resolve()==a.output.resolve()):
        raise FileExistsError('Choose a new, distinct raw query path')
    if not 0<=a.operating_floor<=1:raise ValueError('Operating floor must be in [0,1]')
    if not 0<a.car_recall<=1:raise ValueError('Car recall must be in (0,1]')
    torch.set_num_threads(2);model,state=load_model(a.checkpoint,a.calibration,a.device)
    lidar=a.lidar_calibration or a.calibration
    if state['lidar_calibration_sha256']!=calibration_sha256(lidar):raise ValueError('Sparse LiDAR calibration changed')
    ann=a.ann or ('annotations/fisheye3dod_infos_val.pkl' if state.get('recipe',{}).get('expected_sequences')==8 else
                 str(Path(__file__).resolve().parents[1]/'datasets/heavy_review_v4/val.pkl'))
    dataset=RosaCombinedDataset(a.root,ann,cfg=model.cfg,box_frame=state['box_frame'],
        strict_size=not state.get('recipe',{}).get('resize_native_images',False),
        native_image_sizes={c.name:(c.width,c.height) for c in load_cameras(a.calibration)} if state.get('recipe',{}).get('resize_native_images',False) else None,
        z_reference=state['z_reference'],calibration=lidar,expected_size=(model.cfg.image_width,model.cfg.image_height))
    if not len(dataset):raise ValueError('Validation annotations are empty')
    if any(r['metainfo'].get('temporal_split')!='val' for r in dataset.rows):raise ValueError('Validation-only inference command')
    if sparse_annotation_sha256(dataset)!=state['semantic_annotation_sha256']['val']:
        raise ValueError('Sparse VAL labels changed; use a separately named/reviewed evaluation baseline')
    from raybev.sparse_diagnostics import output_raw_counts,merge_raw_counts,facing_samples,summarize_facing
    from raybev.sparse_audit import (audit_queries,summarize_query_audit,raw_snapshot,
        false_positive_audit,fixed_car_recall,summarize_tracks)
    import numpy as np
    predictions=[];tokens=[];raw=[];facing=[];latencies=[]
    snapshots=[];audit=[];operating=[]
    options=dict(quality_power=a.quality_power,nms_iou=a.nms_iou,
                 vehicle_duplicates=a.vehicle_duplicates,class_bias=a.class_bias)
    sweep=[]
    if a.sweep:
        for power in (0.,.5,1.):
            sweep.append((dict(quality_power=power),[]))
        for iou in (.3,.4):
            sweep.append((dict(nms_iou=iou),[]))
        sweep.append((dict(vehicle_duplicates=True),[]))
    with torch.inference_mode():
        for images,boxes,batch_tokens in DataLoader(dataset,batch_size=a.batch_size,num_workers=a.workers,collate_fn=collate_samples):
            context=torch.autocast('cuda') if a.amp and str(a.device).startswith('cuda') else nullcontext()
            images=images.to(a.device)
            if str(a.device).startswith('cuda'):torch.cuda.synchronize()
            started=time.perf_counter()
            with context:output=model(images)
            if str(a.device).startswith('cuda'):torch.cuda.synchronize()
            latencies.append((time.perf_counter()-started,len(images)))
            raw.extend(output_raw_counts(output,boxes,model.cfg));facing.extend(facing_samples(output,boxes,model.cfg))
            decoded=decode_sparse(output,model.cfg,.01,return_indices=True)
            predictions.extend(p.cpu().numpy() for p,_ in decoded);tokens.extend(batch_tokens)
            run_operating=a.audit or any(v is not None and v is not False for v in options.values()) or a.class_thresholds is not None or a.operating_floor!=.1
            if run_operating:
                op_opts=dict(options,class_thresholds=a.class_thresholds)
                variants=decode_sparse(output,model.cfg,a.operating_floor,return_indices=True,**op_opts)
                operating.extend(p.cpu().numpy() for p,_ in variants)
                if a.audit:audit.extend(audit_queries(output,model.cfg,boxes,variants,score_threshold=a.operating_floor,**op_opts))
            if a.raw_queries:snapshots.append(raw_snapshot(output,model.cfg))
            for setting,frames in sweep:
                frames.extend(p.cpu().numpy() for p in decode_sparse(output,model.cfg,0.,**setting))
            if len(tokens)%50<a.batch_size:print('Sparse VAL',len(tokens),'/',len(dataset),flush=True)
    truth=[b.numpy() for b in dataset.boxes]
    report=dict(architecture=model.cfg.architecture,checkpoint_sha256=calibration_sha256(a.checkpoint),epoch=state['epoch'],
        calibration_sha256=calibration_sha256(a.calibration),annotation_sha256=calibration_sha256(dataset.annotation_path),
        metrics=evaluate_detections(predictions,truth,score_threshold=.1),
        raw_query_coverage=merge_raw_counts(raw),facing_reliability=summarize_facing(facing),
        network_latency=dict(device=a.device,amp=a.amp,batch_size=a.batch_size,
            ms_per_four_camera_frame=1000*sum(t for t,n in latencies[1:])/max(sum(n for t,n in latencies[1:]),1),
            scope='encoder+proposal+decoder; first batch excluded; no image IO, transfer, NMS or metrics'),
        heading=heading_diagnostics(predictions,truth,[str(r['metainfo']['sequence_id']) for r in dataset.rows]),
        frames=[dict(token=t,detections=d.tolist()) for t,d in zip(tokens,predictions)])
    if operating:
        operational,fp_frames=false_positive_audit(operating,truth,a.operating_floor)
        report['operating']=dict(settings=dict(options,class_thresholds=a.class_thresholds,score_floor=a.operating_floor),
            errors=operational,
            class_results={k:{m:v[m] for m in ('TP','FP','FN','precision','recall') if m in v}
                           for k,v in evaluate_detections(operating,truth,a.operating_floor)['class_results'].items()})
        if a.audit:
            tracks=[]
            for row,indices in zip(dataset.rows,dataset.box_source_indices):
                ids=row['annos'].get('gt_ids')
                bag=str(row['metainfo']['sequence_id'])
                tracks.append([(bag,str(ids[i])) if ids is not None else None for i in indices])
            report['query_audit']=dict(candidates_not_one_to_one=True,
                summary=summarize_query_audit(audit),by_track=summarize_tracks(audit,tracks),
                frames=[dict(token=t,heavy=q,false_positives=f)
                        for t,q,f in zip(tokens,audit,fp_frames)])
    if sweep:
        report['scoring_sweep']=[]
        for setting,frames in sweep:
            at_score=evaluate_detections(frames,truth,.1)['class_results']
            report['scoring_sweep'].append(dict(settings=setting,target_car_recall=a.car_recall,
                car_at_fixed_recall=fixed_car_recall(frames,truth,a.car_recall),
                recall_at_0_1={n:at_score[n]['recall'] for n in ('truck','van','bus')}))
    if a.raw_queries:
        a.raw_queries.parent.mkdir(parents=True,exist_ok=True)
        with a.raw_queries.open('xb') as stream:
            np.savez_compressed(stream,tokens=np.asarray(tokens),classes=np.asarray(CLASSES),
                **{k:torch.cat([batch[k] for batch in snapshots]).numpy() for k in snapshots[0]})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,allow_nan=False)+'\n')
    print('Sparse validation mAP:',report['metrics']['mAP'],'->',a.output,flush=True)


if __name__=='__main__':main()
