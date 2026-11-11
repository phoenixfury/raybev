#!/usr/bin/env python3
"""Link cached native YOLO boxes to reviewed TRAIN GT; no detector rerun.

These are weak labels, not manually verified boundaries. Geometry only
associates the independent YOLO measurement; it never generates its bbox.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from raybev.config import CAMERAS,CLASSES,DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import OpenCVProjector
from raybev.sparse_extent import extent_points,VerifiedExtents
from raybev.rosa_dataset import RosaCombinedDataset,training_boxes_sha256
from raybev.training import calibration_sha256


def iou_matrix(a,b):
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum((a[:,2:]-a[:,:2]).clip(0).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).clip(0).prod(-1)[None]-inter,1e-6)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='Prepared image root (not read during conversion)')
    p.add_argument('--train-ann',help='Defaults to ROOT/annotations/fisheye3dod_infos_train.pkl')
    p.add_argument('--source',action='append',default=[],help='sequence_id=/path/to/original/dataset; repeat per bag')
    p.add_argument('--source-summary',type=Path,help='Combined summary.json contains every source dataset path')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.output.exists():raise FileExistsError('Choose a new extent annotation file')
    cfg=SparseConfig(architecture_version=5)
    ann=a.train_ann or str(Path(a.root).resolve()/'annotations/fisheye3dod_infos_train.pkl')
    dataset=RosaCombinedDataset(a.root,ann,cfg=cfg,calibration=a.calibration)
    if any(r['metainfo'].get('temporal_split')!='train' for r in dataset.rows):raise ValueError('TRAIN only')
    projector=OpenCVProjector(cfg,a.calibration);sources={};counts=Counter();frames={}
    source_list=list(a.source)
    if a.source_summary or not source_list:
        summary=json.loads((a.source_summary or Path(a.root)/'summary.json').read_text())
        source_list.extend(s['name']+'='+s['dataset'] for s in summary['sequences'])
    for item in source_list:
        name,path=item.split('=',1);path=Path(path)
        cache=path/'detzero_output/yolo_detections.json';manifest=path/'manifest.csv'
        with manifest.open(newline='') as stream:rows=list(csv.DictReader(stream))
        sources[name]=dict(detections=json.loads(cache.read_text()),rows=rows,
            cache_sha256=calibration_sha256(cache),manifest_sha256=calibration_sha256(manifest))
    if set(sources)!={str(r['metainfo']['sequence_id']) for r in dataset.rows}:
        raise ValueError('Source caches must cover exactly all TRAIN bags')
    with torch.inference_mode():
        for row,gt,ignored in zip(dataset.rows,dataset.boxes,dataset.ignore_boxes):
            sequence=str(row['metainfo']['sequence_id']);source=sources[sequence]
            frame=int(row['metainfo']['source_frame_id']);native=source['rows'][frame]
            if str(native['timestamp_ns'])!=str(row['token']):raise ValueError('Native frame/timestamp mismatch')
            if not len(gt):continue
            all_boxes=torch.cat((gt,ignored),0)
            uv,visible,cues=projector(extent_points(all_boxes[None,:,:7]))
            projected=torch.cat((uv.amin(2),uv.amax(2)),-1)[0].numpy()
            _,_,center_cues=projector(all_boxes[None,:,None,:3])
            distances=center_cues[0,:,0,:,5].numpy()
            records=[]
            for camera,name in enumerate(CAMERAS):
                # Reject large synchronization offsets instead of fabricating
                # a motion correction without object/camera-time state.
                if abs(int(native[name.lower()+'_delta_ns']))>30_000_000:
                    counts['camera_time_rejected']+=1;continue
                detections=[d for d in source['detections'].get(f'{frame}:{name}',[]) if d['confidence']>=.6]
                if not detections:continue
                raw=np.asarray([d['xyxy'] for d in detections],np.float32)
                scale=projector.resize_scale[camera].numpy();pad=projector.padding[camera].numpy()
                measured=(raw.reshape(-1,2,2)+.5)*scale-.5+pad
                measured=measured.reshape(-1,4)
                geometry=projected[:len(gt),camera];scores=iou_matrix(geometry,measured)
                # Never overwrite reviewed Truck/Bus/Van taxonomy using YOLO.
                # All vehicle types are compatible for geometric association.
                for gi,g in enumerate(gt):
                    for di,d in enumerate(detections):
                        target=CLASSES[int(g[7])];observed=d['class'].lower()
                        compatible=target==observed or (target in ('car','truck','bus','van') and observed in ('car','truck','bus','van'))
                        if not compatible:scores[gi,di]=0
                gi,di=linear_sum_assignment(-scores)
                for g,d in zip(gi,di):
                    overlap=float(scores[g,d])
                    alternatives=np.concatenate((np.delete(scores[g],d),np.delete(scores[:,d],g)))
                    if overlap<.4 or (len(alternatives) and overlap-float(alternatives.max())<.15):
                        counts['ambiguous_or_low_overlap']+=1;continue
                    # Reject lens-invalid support; image truncation is handled
                    # per measured edge below. Occluders include ignored GT.
                    if not bool((cues[0,g,:,camera,3]<=1).all()):continue
                    box=measured[d];area=max(float(np.prod(box[2:]-box[:2])),1.)
                    others=projected[:,camera]
                    inter=np.maximum(np.minimum(others[:,2:],box[2:])-np.maximum(others[:,:2],box[:2]),0).prod(-1)
                    nearer=distances[:,camera]<distances[g,camera]-.01
                    if np.any(nearer&(inter/area>.1)):
                        counts['potential_occlusion_rejected']+=1;continue
                    width,height=projector.raw_size[camera].numpy()
                    edges=[bool(raw[d,0]>3),bool(raw[d,1]>3),bool(raw[d,2]<width-3),bool(raw[d,3]<height-3)]
                    # If a projected edge is already outside the native image,
                    # the measured boundary cannot constrain that box side.
                    content=np.r_[pad,projector.raw_size[camera].numpy()*scale+pad-1]
                    edges=np.asarray(edges)&np.r_[geometry[g,:2]>content[:2]+1,geometry[g,2:]<content[2:]-1]
                    if not edges.any():continue
                    bbox=np.clip(box,0,[cfg.image_width,cfg.image_height]*2)
                    record=dict(gt=gt[g].tolist(),camera=camera,bbox=bbox.tolist(),edge_valid=edges.tolist(),
                        occluded=False,truncated=not bool(edges.all()),verified=True,source='postprocess_yolo',
                        weight=.25*float(detections[d]['confidence']),
                        verification=dict(policy='unique_geometry_confidence_v1',iou=overlap,
                            confidence=float(detections[d]['confidence']),cache_sha256=source['cache_sha256'],
                            manifest_sha256=source['manifest_sha256'],frame=frame,
                            visibility='conservative projected-GT occlusion test; not human verified'))
                    records.append(record);counts[CLASSES[int(gt[g,7])]]+=1
            if records:frames[str(row['token'])]=records
    doc=dict(type='raybev_verified_extents_v1',split='train',coordinates='network_pixels',
        image_size=[cfg.image_width,cfg.image_height],camera_order=list(CAMERAS),
        training_boxes_sha256=training_boxes_sha256(dataset),camera_calibration_sha256=calibration_sha256(a.calibration),
        note='Weak YOLO extents, confidence/uniqueness gated, partial edges masked. No reviewed 3D labels changed.',
        counts=dict(counts),sequences=sorted(sources),frames=frames)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(doc,separators=(',',':'),allow_nan=False)+'\n')
    VerifiedExtents(a.output,dataset,a.calibration)
    print(json.dumps(dict(frames=len(frames),counts=dict(counts),output=str(a.output)),indent=2))


if __name__=='__main__':main()
