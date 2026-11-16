#!/usr/bin/env python3
"""Read-only TRAIN review queue for heavy-looking Car labels and heading/geometry."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raybev.config import CLASSES,DEFAULT_CALIBRATION
from raybev.rosa_dataset import load_annotations,parse_box_annotations,read_T_base_clidar


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-ann',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--box-frame',choices=['CLIDAR','BASELINK'],default='CLIDAR')
    p.add_argument('--z-reference',choices=['bottom','center'],default='center')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--car-length',type=float,default=5.5)
    p.add_argument('--car-height',type=float,default=2.1)
    p.add_argument('--bottom-limit',type=float,default=1.)
    a=p.parse_args()
    if min(a.car_length,a.car_height,a.bottom_limit)<=0:raise ValueError('Review thresholds must be positive')
    rows=load_annotations(a.train_ann)['data_list']
    if any(r['metainfo'].get('temporal_split')!='train' for r in rows):
        raise ValueError('Review queue uses TRAIN annotations only')
    transform=read_T_base_clidar(a.calibration) if a.box_frame=='CLIDAR' else None
    tracks=defaultdict(list);samples=[]
    for r in rows:
        boxes,ids=parse_box_annotations(r,a.box_frame,transform,a.z_reference,return_indices=True)
        native_ids=r['annos'].get('gt_ids')
        bag=str(r['metainfo']['sequence_id']);token=str(r['token'])
        for box,i in zip(boxes.numpy(),ids):
            c=int(box[7])
            if c not in (0,3,4,5):continue
            uid=str(native_ids[i]) if native_ids is not None else token+'/'+str(i)
            bottom=float(box[2]-box[5]/2)
            flags=[]
            if c==0 and (box[3]>=a.car_length or box[5]>=a.car_height):flags.append('car_heavy_dimensions')
            if c!=0 and abs(bottom)>a.bottom_limit:flags.append('heavy_bottom_z_review')
            entry=dict(bag=bag,track_id=uid,token=token,annotation_index=int(i),
                       class_name=CLASSES[c],box_baselink=box[:7].tolist(),bottom_z=bottom,
                       flags=flags)
            if flags:samples.append(entry)
            tracks[bag,uid].append(entry)
    queue=[]
    for (bag,uid),group in sorted(tracks.items()):
        flags=sorted({f for entry in group for f in entry['flags']})
        classes=sorted({entry['class_name'] for entry in group})
        if len(classes)>1:flags.append('track_changes_class')
        if flags:
            queue.append(dict(bag=bag,track_id=uid,classes=classes,flags=flags,
                              frames=len(group),flagged_frames=sum(bool(e['flags']) for e in group),
                              review_frames=group))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:
        json.dump(dict(policy='Review candidates only. Size is not a class label. No annotations changed.',
                       frames=len(rows),flagged_tracks=len(queue),flagged_frames=len(samples),
                       thresholds=dict(car_length=a.car_length,car_height=a.car_height,bottom_limit=a.bottom_limit),
                       review_queue=queue),f,indent=2,allow_nan=False)
        f.write('\n')
    print(f'{len(queue)} tracks queued for review -> {a.output}')

if __name__=='__main__':main()
