#!/usr/bin/env python3
"""Rebind an existing TRAIN-only yaw cache after a reviewed class/ignore edit.

No ROSA launch. Geometry, tokens and track IDs MUST be identical. Only cached
GT class IDs are remapped; ignored targets are dropped. Teacher yaws, scores
and original checkpoint/prediction hashes stay unchanged, with edit provenance.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raybev.config import CLASSES,DEFAULT_CALIBRATION,ModelConfig
from raybev.rosa_dataset import RosaCombinedDataset,training_boxes_sha256
from raybev.heading_supervision import HeadingSupervision
from raybev.training import calibration_sha256


def rebind(source,revised,cache_path,calibration=DEFAULT_CALIBRATION):
    if len(source)!=len(revised):raise ValueError('Review changed TRAIN frame count')
    if source.box_frame!=revised.box_frame or source.z_reference!=revised.z_reference:
        raise ValueError('Review changed coordinate contract')
    args=SimpleNamespace(heading_teacher=str(cache_path),heading_teacher_weight=.2,track_heading_weight=0.,
                         calibration=str(calibration),lidar_calibration=str(calibration))
    # Validate original cache identity against the ORIGINAL TRAIN labels first.
    HeadingSupervision(source,source.cfg,args)
    doc=json.loads(Path(cache_path).read_text());result=deepcopy(doc);frames={}
    remapped=dropped=0
    for old,new,old_boxes,new_boxes in zip(source.rows,revised.rows,source.boxes,revised.boxes):
        if old['metainfo'].get('temporal_split')!='train' or new['metainfo'].get('temporal_split')!='train':
            raise ValueError('TRAIN-only rebind required')
        if (str(old['token'])!=str(new['token']) or
                str(old['metainfo']['sequence_id'])!=str(new['metainfo']['sequence_id']) or
                not np.array_equal(old['annos']['gt_boxes'],new['annos']['gt_boxes']) or
                not np.array_equal(old['annos']['gt_ids'],new['annos']['gt_ids'])):
            raise ValueError('Review changed tokens/geometry/track identity')
        records=[]
        for item in doc['frames'][str(old['token'])]:
            gt=np.asarray(item['gt'],dtype=float)
            if not np.isclose(old_boxes.numpy(),gt,atol=1e-4,rtol=0).all(1).any():
                raise ValueError('Teacher target does not match original GT')
            candidates=np.flatnonzero(np.isclose(new_boxes[:,:7].numpy(),gt[:7],atol=1e-4,rtol=0).all(1))
            if not len(candidates):
                ignored=revised.ignore_boxes[len(frames)][:,:7].numpy()
                if not np.isclose(ignored,gt[:7],atol=1e-4,rtol=0).all(1).any():
                    raise ValueError('Teacher target disappeared without an explicit ignore annotation')
                dropped+=1;continue
            if len(candidates)!=1:raise ValueError('Ambiguous geometry-only teacher remap')
            target=new_boxes[candidates[0]].tolist();record=deepcopy(item)
            remapped+=int(int(target[7])!=int(gt[7]));record['gt']=target;records.append(record)
        frames[str(old['token'])]=records
    result.update(frames=frames,training_boxes_sha256=training_boxes_sha256(revised),
                  annotation_sha256=calibration_sha256(revised.annotation_path),training_class_counts=revised.gt_counts,
                  matched_class_counts={name:sum(int(r['gt'][7])==ci for records in frames.values() for r in records)
                                        for ci,name in enumerate(CLASSES)},
                  reviewed_rebind=dict(source_teacher_sha256=calibration_sha256(cache_path),
                      source_training_boxes_sha256=training_boxes_sha256(source),
                      remapped_class_records=remapped,dropped_ignored_records=dropped,
                      geometry_and_teacher_angles_unchanged=True))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--source-ann',default='rosa/fisheye3dod_infos_train.pkl')
    p.add_argument('--reviewed-ann',required=True)
    p.add_argument('--source-teacher',default='teacher/raybev_heading_teacher_train.json')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new output; refusing teacher overwrite')
    options=dict(root=a.root,cfg=ModelConfig(),calibration=a.calibration)
    source=RosaCombinedDataset(ann_file=a.source_ann,**options)
    revised=RosaCombinedDataset(ann_file=str(Path(a.reviewed_ann).resolve()),**options)
    cache=rebind(source,revised,a.source_teacher,a.calibration)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(cache,allow_nan=False)+'\n')
    print('Rebound TRAIN yaw teacher:',cache['reviewed_rebind'],'->',a.output)


if __name__=='__main__':main()
