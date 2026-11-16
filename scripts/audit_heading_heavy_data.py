#!/usr/bin/env python3
"""Audit class/track diversity and world-frame heading discontinuities without relabeling."""
import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raybev.config import CLASSES,DEFAULT_CALIBRATION
from raybev.detection import in_bev_range
from raybev.rosa_dataset import RosaCombinedDataset,boxes_to_baselink,names_to_ids
from raybev.sparse_config import SparseConfig
from raybev.sparse_training import sparse_annotation_sha256


def audit_dataset(dataset):
    groups=defaultdict(list);class_tracks=defaultdict(set)
    for row in dataset.rows:
        ann=row['annos'];meta=row['metainfo'];raw=np.asarray(ann['gt_boxes'])
        boxes=boxes_to_baselink(raw,dataset.box_frame,dataset.T_base_clidar,dataset.z_reference)
        labels=names_to_ids(ann['gt_names']);ids=np.asarray(ann.get('gt_ids',np.arange(len(boxes))))
        keep=in_bev_range(boxes,dataset.cfg)&np.isfinite(boxes).all(1)&(boxes[:,3:6]>0).all(1)
        keep &= ~np.asarray(ann.get('gt_ignore',np.zeros(len(boxes),bool)))
        for i in np.flatnonzero(keep):
            key=(str(meta['sequence_id']),str(ids[i]));name=CLASSES[int(labels[i])]
            class_tracks[name].add(key)
            record=dict(token=str(row['token']),class_name=name,time_ns=meta.get('acquisition_timestamp_ns'))
            transform=meta.get('T_world_CLIDAR')
            if transform is not None and dataset.box_frame=='CLIDAR':
                transform=np.asarray(transform);yaw=float(raw[i,6])
                direction=transform[:3,:3]@np.array([math.cos(yaw),math.sin(yaw),0.])
                record['world_yaw']=float(math.atan2(direction[1],direction[0]))
            groups[key].append(record)
    mixed=[];flips=[];classes=Counter()
    for key,records in groups.items():
        counts=Counter(r['class_name'] for r in records);classes.update(counts)
        if len(counts)>1:
            mixed.append(dict(sequence=key[0],track_id=key[1],class_counts=dict(counts),
                              example_tokens=[r['token'] for r in records[:5]]))
        records=sorted((r for r in records if r.get('time_ns') is not None and 'world_yaw' in r),key=lambda r:r['time_ns'])
        for left,right in zip(records,records[1:]):
            dt=(right['time_ns']-left['time_ns'])/1e9
            error=abs(math.atan2(math.sin(right['world_yaw']-left['world_yaw']),math.cos(right['world_yaw']-left['world_yaw'])))
            if 0<dt<=2 and error>math.radians(160):
                flips.append(dict(sequence=key[0],track_id=key[1],class_name=right['class_name'],
                                  tokens=[left['token'],right['token']],dt_seconds=dt,world_heading_change_deg=math.degrees(error)))
    result=dict(frames=len(dataset),boxes=dict(classes),unique_tracks={name:len(class_tracks[name]) for name in CLASSES},
                semantic_annotation_sha256=sparse_annotation_sha256(dataset),
                tracks_with_multiple_class_labels=mixed,world_heading_flip_pairs=flips,
                note='Flip/class-change candidates require review; track switches and real motion can also cause them. No annotations changed.')
    return result,class_tracks,groups


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    report={};tracks={};groups={}
    for split in ('train','val'):
        dataset=RosaCombinedDataset(a.root,f'annotations/fisheye3dod_infos_{split}.pkl',cfg=SparseConfig(),
                                    calibration=a.calibration,box_frame='CLIDAR',strict_size=False)
        report[split],tracks[split],groups[split]=audit_dataset(dataset)
    report['train_seen_validation_tracks']={name:len(tracks['train'][name]&tracks['val'][name]) for name in CLASSES}
    report['train_seen_validation_boxes']={name:sum(sum(r['class_name']==name for r in groups['val'][key])
        for key in tracks['train'][name]&tracks['val'][name]) for name in CLASSES}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for split in ('train','val'):
        print(split,report[split]['boxes'],'unique tracks',report[split]['unique_tracks'])
        print('  mixed-class tracks:',len(report[split]['tracks_with_multiple_class_labels']),
              'world-yaw flip candidates:',len(report[split]['world_heading_flip_pairs']))
    print('Wrote',a.output)


if __name__=='__main__':main()
