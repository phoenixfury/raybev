#!/usr/bin/env python3
"""Read-only, source-index audit of a desktop annotation release and its YOLO evidence."""
import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path
import pickle
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT));sys.path.insert(0,str(Path(__file__).resolve().parent))
from review_heavy_annotations import sha, verify_export, valid_track_id
from raybev.config import CLASSES,DEFAULT_CALIBRATION
from raybev.rosa_dataset import load_annotations,RosaCombinedDataset
from raybev.sparse_config import SparseConfig
from raybev.sparse_extent import VerifiedExtents


def audit(root,release,source,calibration):
    release=Path(release).resolve();source=Path(source).resolve();root=Path(root).resolve()
    manifest=json.loads((release/'review_manifest.json').read_text())
    saved=json.loads((release/'review_decisions.json').read_text())
    if saved['identity']!=manifest['source_identity']:raise ValueError('Decision/source identity mismatch')
    verification=verify_export(root,release,calibration)
    track_edits={}
    if 'track_id_revisions' in manifest:
        path=release/'track_id_revisions.json'
        if sha(path)!=manifest['track_id_revisions']['sha256']:raise ValueError('Track revision hash mismatch')
        for edit in json.loads(path.read_text())['edits']:
            if edit['id'] in track_edits:raise ValueError('Duplicate track revision')
            track_edits[edit['id']]=edit
    track_seen=set()
    decisions=saved['decisions'];seen=set();transitions=Counter();removed=[];geometry_flags=[]
    queue=[json.loads(line) for line in (release/'review_queue.jsonl').read_text().splitlines() if line.strip()]
    queued={r['id']:r for r in queue};queue_seen=set()
    if len(queued)!=len(queue) or not set(decisions).issubset(queued):raise ValueError('Incomplete/duplicate review queue')
    datasets={};split_counts={};track_labels=defaultdict(set)
    for split in ('train','val'):
        filename=f'fisheye3dod_infos_{split}.pkl';before=source/filename;after=release/filename
        if sha(before)!=saved['identity']['annotations'][split]:raise ValueError('Wrong source PKL: '+split)
        original=load_annotations(before);revised=load_annotations(after)
        if original.keys()!=revised.keys():raise ValueError('Document fields differ')
        for name in original:
            if name!='data_list' and pickle.dumps(original[name],4)!=pickle.dumps(revised[name],4):
                raise ValueError('Document metadata changed: '+name)
        if len(original['data_list'])!=len(revised['data_list']):raise ValueError('Frame removal detected')
        counts=Counter();removed_count=0
        for old,new in zip(original['data_list'],revised['data_list']):
            if old.keys()!=new.keys() or old['annos'].keys()!=new['annos'].keys():raise ValueError('Frame fields differ')
            for field in old:
                if field!='annos' and pickle.dumps(old[field],4)!=pickle.dumps(new[field],4):
                    raise ValueError('Frame metadata changed: '+field)
            token=str(old['token']);ann=old['annos'];kept=[];expected_names=[];expected_mesh=[];expected_boxes=[]
            for index,name in enumerate(ann['gt_names']):
                key=f'{split}:{token}:{index}';choice=decisions.get(key);label=str(name).lower()
                if key in queued:
                    rec=queued[key];queue_seen.add(key)
                    if (rec['split']!=split or str(rec['token'])!=token or rec['index']!=index or
                            rec['sequence']!=str(old['metainfo']['sequence_id']) or
                            rec['track_id']!=str(ann['gt_ids'][index]) or rec['original_class']!=label):
                        raise ValueError('Review queue/source identity mismatch: '+key)
                    np.testing.assert_array_equal(rec['box_CLIDAR'],ann['gt_boxes'][index,:7])
                if choice:
                    seen.add(key)
                    if choice['label'] not in CLASSES or choice['status'] not in ('reviewed','uncertain','removed'):
                        raise ValueError('Invalid decision: '+key)
                if choice and choice['status']=='removed':
                    removed_count+=1
                    removed.append(dict(id=key,split=split,sequence=str(old['metainfo']['sequence_id']),
                                        track_id=str(ann['gt_ids'][index]),class_name=label,box=ann['gt_boxes'][index].tolist()))
                    continue
                kept.append(index);new_label=choice['label'] if choice else label
                expected_names.append(new_label.title() if new_label!='pedestrian' else 'Pedestrian')
                expected_mesh.append('detzero.'+new_label if label!=new_label else str(ann['gt_mesh'][index]))
                box=np.asarray(ann['gt_boxes'][index]).copy()
                if choice and 'dimensions_lwh' in choice:box[3:6]=choice['dimensions_lwh']
                expected_boxes.append(box);counts[new_label]+=1
                if label!=new_label:transitions[f'{split}:{label}->{new_label}']+=1
                if not np.isfinite(box).all() or (box[3:6]<=0).any():raise ValueError('Invalid cuboid: '+key)
                sequence=str(old['metainfo']['sequence_id']);track=str(ann['gt_ids'][index])
                if key in track_edits:
                    edit=track_edits[key];track_seen.add(key)
                    if edit['old_track_id']!=track or edit['sequence']!=sequence or not edit.get('evidence'):
                        raise ValueError('Track revision/source identity mismatch')
                    track=str(edit['new_track_id'])
                if valid_track_id(track):track_labels[sequence,track].add(new_label)
                l,w,h=map(float,box[3:6]);suspicious=[]
                if new_label=='car' and (l>6.5 or w>2.6 or h>2.8):suspicious.append('Car with unusually large retained dimensions')
                if new_label=='van' and (l>8.5 or w>2.8 or h>3.8):suspicious.append('Van with heavy-vehicle-sized retained dimensions')
                if new_label=='pedestrian' and (l>2. or w>1.5 or h>2.5):suspicious.append('Pedestrian with vehicle-sized retained dimensions')
                if new_label=='cyclist' and (l>3.5 or w>1.5 or h>2.6):suspicious.append('Cyclist with unusually large retained dimensions')
                if new_label in ('truck','bus') and (l<4.5 or w<1.5 or h<1.7):suspicious.append('Heavy class with unusually small retained dimensions')
                if l>18:suspicious.append('Long/tram-like cuboid: directed heading needs separate audit')
                if suspicious:geometry_flags.append(dict(id=key,sequence=sequence,track_id=track,
                    class_name=new_label,LWH=[l,w,h],reason=suspicious,manually_reviewed=bool(choice)))
            for field,values in ann.items():
                actual=new['annos'][field]
                if field=='gt_names':np.testing.assert_array_equal(actual,expected_names)
                elif field=='gt_mesh':np.testing.assert_array_equal(actual,expected_mesh)
                elif field=='gt_boxes':np.testing.assert_array_equal(actual,np.asarray(expected_boxes,dtype=values.dtype).reshape(-1,values.shape[1]))
                elif field=='gt_ids':
                    expected=values.copy()
                    for index in kept:
                        edit=track_edits.get(f'{split}:{token}:{index}')
                        if edit:expected[index]=edit['new_track_id']
                    np.testing.assert_array_equal(actual,expected[kept])
                elif isinstance(values,np.ndarray) and values.ndim and len(values)==len(ann['gt_boxes']):
                    np.testing.assert_array_equal(actual,values[kept])
                elif isinstance(values,(list,tuple)) and len(values)==len(ann['gt_boxes']):
                    if list(actual)!=[values[i] for i in kept]:raise ValueError('Retained field changed: '+field)
                elif pickle.dumps(actual,4)!=pickle.dumps(values,4):raise ValueError('Annotation metadata changed')
        split_counts[split]=dict(frames=len(revised['data_list']),objects=dict(counts),removed=removed_count)
        datasets[split]=RosaCombinedDataset(root,after,cfg=SparseConfig(architecture_version=5),calibration=calibration)
    if seen!=set(decisions):raise ValueError('Unknown/unapplied decisions')
    if track_seen!=set(track_edits):raise ValueError('Unknown/unapplied track revisions')
    if queue_seen!=set(queued):raise ValueError('Unknown queue annotation')
    for rec in queue:
        choice=decisions.get(rec['id'])
        if choice is not None and rec['decision']!=choice:raise ValueError('Queue/decision mismatch')
    old=RosaCombinedDataset(root,source/'fisheye3dod_infos_train.pkl',cfg=SparseConfig(architecture_version=5),calibration=calibration)
    original_ext=source/'train_yolo_extents_v1.json'
    if sha(original_ext)!=saved['identity']['extents']:raise ValueError('Wrong source extent JSON')
    VerifiedExtents(original_ext,old,calibration)
    prior=json.loads(original_ext.read_text());current=json.loads((release/'train_yolo_extents_v1.json').read_text())
    expected_frames=copy.deepcopy(prior['frames']);extent_removed=extent_changed=extent_geometry_dropped=0
    by_token={str(row['token']):(boxes.numpy(),indices) for row,boxes,indices in zip(old.rows,old.boxes,old.box_source_indices)}
    for token,records in prior['frames'].items():
        expected=[];boxes,indices=by_token[token]
        for record in records:
            matches=np.flatnonzero(np.isclose(boxes,np.asarray(record['gt'],np.float32),atol=1e-4,rtol=0).all(-1))
            if len(matches)!=1:raise ValueError('Ambiguous source extent: '+token)
            choice=decisions.get(f'train:{token}:{int(indices[matches[0]])}')
            if choice and choice['status']=='removed':extent_removed+=1;continue
            if choice and 'dimensions_lwh' in choice and not np.allclose(choice['dimensions_lwh'],record['gt'][3:6],rtol=0,atol=1e-4):
                extent_geometry_dropped+=1;continue
            revised=copy.deepcopy(record)
            if choice:
                ci=CLASSES.index(choice['label']);extent_changed+=int(ci!=record['gt'][7]);revised['gt'][7]=float(ci)
            expected.append(revised)
        expected_frames[token]=expected
    if current['frames']!=expected_frames:raise ValueError('YOLO records differ beyond approved removals/class edits')
    seen_tracks={(str(row['metainfo']['sequence_id']),str(row['annos']['gt_ids'][raw]))
                 for row,indices in zip(datasets['train'].rows,datasets['train'].box_source_indices) for raw in indices
                 if valid_track_id(row['annos']['gt_ids'][raw])}
    overlap={name:Counter() for name in CLASSES}
    for row,boxes,indices in zip(datasets['val'].rows,datasets['val'].boxes,datasets['val'].box_source_indices):
        for box,raw in zip(boxes,indices):
            track=str(row['annos']['gt_ids'][raw]);key=(str(row['metainfo']['sequence_id']),track)
            overlap[CLASSES[int(box[7])]]['seen' if key in seen_tracks else 'unseen' if valid_track_id(track) else 'unknown_id']+=1
    conflicts=[dict(sequence=key[0],track_id=key[1],classes=sorted(labels)) for key,labels in track_labels.items() if len(labels)>1]
    result=dict(status='passed',verification=verification,reviewed=len(decisions),
        decision_statuses=dict(Counter(d['status'] for d in decisions.values())),transitions=dict(transitions),
        splits=split_counts,removed_annotations=removed,track_class_conflicts=conflicts,track_id_edits=len(track_edits),
        geometry_flags=geometry_flags,seen_unseen_val={k:dict(v) for k,v in overlap.items()},
        yolo=dict(records=sum(map(len,current['frames'].values())),changed=extent_changed,
                  removed=extent_removed,geometry_dropped=extent_geometry_dropped,pixel_evidence_preserved=True),
        credible_geometry=dict(approved_boxes=0,endpoint_training_allowed=False,
            reason='Class confirmation does not certify cuboid centers/dimensions/yaw. No geometry approval ledger supplied.'),
        limitation='Exhaustive source/decision/export consistency, not independent visual certification of every user label.')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True);p.add_argument('--release',type=Path,required=True)
    p.add_argument('--source',type=Path,default=PROJECT/'datasets/combined_bags_camera_ray_heavy_review_v4')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=audit(a.root,a.release,a.source,a.calibration)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'annotation_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    summary={key:report[key] for key in ('status','reviewed','decision_statuses','splits','transitions','yolo','seen_unseen_val','credible_geometry')}
    summary['track_class_conflicts']=len(report['track_class_conflicts']);summary['geometry_flags']=len(report['geometry_flags'])
    summary['geometry_flags_by_class']=dict(Counter(row['class_name'] for row in report['geometry_flags']))
    warnings='\n'.join('- '+json.dumps(row) for row in report['track_class_conflicts'])
    (a.output/'REPORT.md').write_text('# Manual v5 annotation audit\n\n'+report['limitation']+
        '\n\nAll edits/removals and all review queue source identities checked; originals untouched.\n\n'
        '## Remaining flags\n\n'+warnings+'\n\nGeometry flags are not automatic class corrections. '
        'Class review alone does not certify cuboid association/geometry. Inspect `annotation_audit.json` '
        'and the final-index `followup_review.jsonl`. The controlled launcher is guarded. '
        'See `scripts/README_v10_controlled.md` for commands and experiment details.\n\n```json\n'+json.dumps(summary,indent=2)+'\n```\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
