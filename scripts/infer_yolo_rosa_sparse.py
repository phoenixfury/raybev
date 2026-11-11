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
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new sparse inference output')
    torch.set_num_threads(2);model,state=load_model(a.checkpoint,a.calibration,a.device)
    lidar=a.lidar_calibration or a.calibration
    if state['lidar_calibration_sha256']!=calibration_sha256(lidar):raise ValueError('Sparse LiDAR calibration changed')
    ann=a.ann or ('annotations/fisheye3dod_infos_val.pkl' if state.get('recipe',{}).get('expected_sequences')==8 else
                 str(Path(__file__).resolve().parents[1]/'datasets/heavy_review_v4/val.pkl'))
    dataset=RosaCombinedDataset(a.root,ann,cfg=model.cfg,box_frame=state['box_frame'],
        strict_size=not state.get('recipe',{}).get('resize_native_images',False),
        native_image_sizes={c.name:(c.width,c.height) for c in load_cameras(a.calibration)} if state.get('recipe',{}).get('resize_native_images',False) else None,
        z_reference=state['z_reference'],calibration=lidar,expected_size=(model.cfg.image_width,model.cfg.image_height))
    if any(r['metainfo'].get('temporal_split')!='val' for r in dataset.rows):raise ValueError('Validation-only inference command')
    if sparse_annotation_sha256(dataset)!=state['semantic_annotation_sha256']['val']:
        raise ValueError('Sparse VAL labels changed; use a separately named/reviewed evaluation baseline')
    from raybev.sparse_diagnostics import output_raw_counts,merge_raw_counts,facing_samples,summarize_facing
    predictions=[];tokens=[];raw=[];facing=[];latencies=[]
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
            predictions.extend(p.cpu().numpy() for p in decode_sparse(output,model.cfg,.01));tokens.extend(batch_tokens)
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
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,allow_nan=False)+'\n')
    print('Sparse validation mAP:',report['metrics']['mAP'],'->',a.output,flush=True)


if __name__=='__main__':main()
