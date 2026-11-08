#!/usr/bin/env python3
"""Extend a combined dataset into a NEW root, preserving benchmark split tokens.

Hardlink only referenced immutable images; republish independent annotations and
training-only anchors. Both ROSA and Fisheye3DOD views share the exact same PKLs.
Never overwrite the base dataset or an existing destination.

This script is the user's existing extension step. It requires their ORIGINAL
local build_improved_detzero_rosa_dataset.py,
append_improved_detzero_rosa_dataset.py and ocamcamera modules. Their source
was not uploaded, so it cannot be executed in the generated ZIP by itself.
Use with scripts/prepare_combined_rosa.py for native + 800x640 collection.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
from collections import Counter,defaultdict

import mmengine
import numpy as np
from ocamcamera import OcamCamera
from build_improved_detzero_rosa_dataset import CAMERAS,materialize,sha256,build_rosa_anchor
from append_improved_detzero_rosa_dataset import load_sequence,assign_chronological_split,publish_pickle,publish_view


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--add-sequence',nargs=2,action='append',required=True,metavar=('NAME','SYNCHRONIZED_DATASET'))
    p.add_argument('--sequence-boundaries',nargs=3,action='append',default=[],metavar=('NAME','TRAIN_FRAMES','VAL_START_FRAME'),
                   help='Chronological source-frame boundaries; e.g. 25_09 413 433')
    p.add_argument('--storage',choices=('hardlink','copy','symlink'),default='hardlink')
    p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--purge-seconds',type=float,default=5.)
    p.add_argument('--max-scene-gap-seconds',type=float,default=4.)
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args();base=args.base.resolve();output=args.output.resolve()
    if output.exists():raise FileExistsError('Choose a new output; existing datasets are protected: '+str(output))
    if not 0<args.val_fraction<1 or args.purge_seconds<0 or args.max_scene_gap_seconds<=0:raise ValueError('Invalid split options')
    summary=json.loads((base/'summary.json').read_text());source=base/'annotations/fisheye3dod_infos_all.pkl'
    base_sha=sha256(source)
    if summary['annotation_sha256']['all']!=base_sha:raise ValueError('Base annotation checksum differs')
    info=mmengine.load(source);samples=copy.deepcopy(info['data_list'])
    membership={str(s['token']):s['metainfo']['temporal_split'] for s in samples}
    known={str(s['metainfo']['sequence_id']) for s in samples}
    additions=[(name,Path(path).resolve()) for name,path in args.add_sequence]
    names=[name for name,_ in additions]
    if len(set(names))!=len(names) or set(names)&known:raise ValueError('Duplicate/existing added sequence name')
    import re
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',name) for name in names):raise ValueError('Unsafe sequence name')
    boundaries={}
    for name,end,start in args.sequence_boundaries:
        if name not in names or name in boundaries:raise ValueError('Unknown/duplicate sequence boundary')
        boundaries[name]=(int(end),int(start))
    calibration=base/'calib_results.txt'
    if not calibration.is_file():calibration=Path(summary['ocam_calibration'])
    if sha256(calibration)!=summary['ocam_calibration_sha256']:raise ValueError('OCam calibration differs')
    ocam=OcamCamera(filename=str(calibration),fov=220.)
    next_id=max((int(v) for s in samples for v in s['annos']['gt_ids']),default=-1)+1
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=None if args.dry_run else Path(tempfile.mkdtemp(prefix='.'+output.name+'.building-',dir=output.parent))
    destination=output if staging is None else staging
    try:
        new_entries=[]
        for name,sequence in additions:
            new,entry,next_id=load_sequence(name,sequence,destination,ocam,args.storage,args.dry_run,next_id,False)
            ordered=sorted(new,key=lambda s:int(s['metainfo']['acquisition_timestamp_ns']))
            if name in boundaries:
                end,start=boundaries[name]
                if not 0<end<=start<len(new):raise ValueError('Invalid source-frame split boundaries')
                if [s['metainfo']['source_frame_id'] for s in ordered]!=list(range(len(new))):
                    raise ValueError('Source frame order differs from acquisition order')
                for i,s in enumerate(ordered):membership[str(s['token'])]='train' if i<end else 'gap' if i<start else 'val'
            else:
                assign_chronological_split(new,args.val_fraction,args.purge_seconds,args.max_scene_gap_seconds)
                membership.update({str(s['token']):s['metainfo']['temporal_split'] for s in new})
            samples.extend(new);new_entries.append(entry)
        if len(membership)!=len(samples):raise ValueError('Duplicate tokens across bags')
        train,val,purged=assign_chronological_split(samples,args.val_fraction,args.purge_seconds,args.max_scene_gap_seconds,membership)
        old_tokens={str(s['token']) for s in info['data_list']}
        for split,rows in [('train',train),('val',val)]:
            original=mmengine.load(base/'annotations'/('fisheye3dod_infos_'+split+'.pkl'))['data_list']
            if {str(s['token']) for s in rows if str(s['token']) in old_tokens}!={str(s['token']) for s in original}:
                raise ValueError('Existing benchmark membership changed')
        counts=Counter(str(n) for s in samples for n in s['annos']['gt_names'])
        split_counts={part:Counter(str(n) for s in rows for n in s['annos']['gt_names']) for part,rows in [('train',train),('val',val)]}
        result=copy.deepcopy(summary)
        for key in ('annotation_sha256','cleanup'):result.pop(key,None)
        entries=result['sequences']+new_entries
        for entry in entries:
            buckets=defaultdict(list)
            for s in samples:
                if s['metainfo']['sequence_id']==entry['name']:buckets[s['metainfo']['temporal_split']].append(int(s['metainfo']['acquisition_timestamp_ns']))
            entry['temporal_split']={part:dict(frames=len(ts),first_acquisition_ns=min(ts),last_acquisition_ns=max(ts)) for part,ts in buckets.items()}
        source_counts=Counter()
        for entry in entries:source_counts.update(entry.get('source_counts',{}))
        result.update(output=str(output),dry_run=args.dry_run,storage=args.storage,frames=len(samples),train_frames=len(train),
            val_frames=len(val),purged_frames=purged,empty_frames=sum(not len(s['annos']['gt_names']) for s in samples),
            boxes=sum(counts.values()),class_counts=dict(counts),split_class_counts={k:dict(v) for k,v in split_counts.items()},
            instances=len({int(v) for s in samples for v in s['annos']['gt_ids']}),source_counts=dict(source_counts),sequences=entries,
            base_dataset=str(base),base_annotation_sha256=base_sha,ocam_calibration=str(output/'calib_results.txt'),
            split=dict(method='per-sequence chronological tail holdout',clock='lidar_sensor_timestamp_ns',
                fraction_basis='existing benchmark tokens preserved; explicit new boundaries or elapsed acquisition time',
                default_new_val_fraction=args.val_fraction,default_new_purge_seconds=args.purge_seconds,
                sequence_frame_boundaries={k:dict(train_frames=v[0],val_start_frame=v[1]) for k,v in boundaries.items()},
                max_scene_gap_seconds=args.max_scene_gap_seconds,scene_links_cross_split=False),
            format_views_byte_identical=True)
        print(json.dumps({k:result[k] for k in ('output','frames','train_frames','val_frames','purged_frames','boxes','class_counts','split_class_counts')},indent=2),flush=True)
        if args.dry_run:return
        for s in info['data_list']:
            for camera in CAMERAS:
                relative=Path(s['cam_info']['cam_fisheye'][camera]['cam_path'])
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('Base image path is not portable')
                materialize(base/relative,destination/relative,args.storage,False)
        shutil.copy2(calibration,destination/'calib_results.txt')
        anchor,training_boxes=build_rosa_anchor(train,300,3407)
        anchor_path=destination/'rosa/anchors/kmeans300_seed3407.npy';anchor_path.parent.mkdir(parents=True,exist_ok=True);np.save(anchor_path,anchor)
        result['rosa_anchor']=dict(path='rosa/anchors/kmeans300_seed3407.npy',shape=list(anchor.shape),seed=3407,
            training_boxes=training_boxes,sha256=sha256(anchor_path),update='refit from new training split only',
            split_note='No validation or purge boxes used')
        shutil.copy2(anchor_path,destination/'improved_detzero_kmeans300_seed3407.npy')
        metadata=copy.deepcopy(info['metainfo']);metadata['version']='improved-detzero-four-bag-preserved-holdout-v1'
        paths={}
        for split,rows in [('all',samples),('train',train),('val',val)]:
            path=destination/'annotations'/('fisheye3dod_infos_'+split+'.pkl');publish_pickle(path,dict(metainfo=metadata,data_list=rows));paths[split]=path
            for view in ('rosa','fisheye3dod','ImageSets-improved-detzero'):publish_view(path,destination/view/path.name)
        result['annotation_sha256']={k:sha256(path) for k,path in paths.items()}
        (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        if sha256(source)!=base_sha:raise RuntimeError('Base annotations changed during build')
        os.rename(destination,output)
        print('Published '+str(output),flush=True)
    except Exception:
        if staging is not None:print('Incomplete build retained for inspection: '+str(staging),flush=True)
        raise

if __name__=='__main__':main()
