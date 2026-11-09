#!/usr/bin/env python3
"""Apply an approved track-class review to NEW annotation files only.

No image copies, geometry changes, train/val reassignment or source overwrites.
Without --write this is a dry run. Annotation paths remain relative to --root.
"""
import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from raybev.config import DEFAULT_CALIBRATION,CLASSES
from raybev.rosa_dataset import load_annotations,names_to_ids,boxes_to_baselink,read_T_base_clidar


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gt_signature(document):
    """Portable label identity; ignore pickle protocol and relocated image paths."""
    digest=hashlib.sha256();tokens=set()
    for row in sorted(document['data_list'],key=lambda r:str(r['token'])):
        token=str(row['token'])
        if token in tokens:raise ValueError('Duplicate annotation token')
        tokens.add(token);meta=row['metainfo'];ann=row['annos']
        value=[token,str(meta['sequence_id']),meta['temporal_split'],
               np.asarray(ann['gt_boxes'],dtype=float).tolist(),names_to_ids(ann['gt_names']).tolist(),
               list(map(str,ann['gt_ids']))]
        if 'gt_ignore' in ann:value.append(np.asarray(ann['gt_ignore'],dtype=bool).tolist())
        digest.update(json.dumps(value,separators=(',',':'),allow_nan=False).encode());digest.update(b'\n')
    return digest.hexdigest()


def revise_document(document,rules,split):
    result=copy.deepcopy(document);changes=[];counts=Counter()
    lookup={}
    for ri,rule in enumerate(rules):
        key=(str(rule['sequence_id']),str(rule['track_id']))
        if key in lookup:raise ValueError('Duplicate track review rule')
        if names_to_ids([rule['from']])[0]==names_to_ids([rule['to']])[0]:raise ValueError('Review does not change class')
        lookup[key]=(ri,rule)
    for row in result['data_list']:
        meta=row['metainfo']
        if meta.get('temporal_split')!=split:raise ValueError('Annotation split mismatch')
        ann=row['annos'];names=list(map(str,ann['gt_names']))
        if len(ann['gt_ids'])!=len(names):raise ValueError('Track/class length mismatch')
        touched=False
        for i,uid in enumerate(ann['gt_ids']):
            key=(str(meta['sequence_id']),str(uid))
            if key not in lookup:continue
            ri,rule=lookup[key]
            if names_to_ids([names[i]])[0]!=names_to_ids([rule['from']])[0]:raise ValueError(f'Review old class mismatch: {key}')
            changes.append(dict(token=str(row['token']),sequence_id=key[0],track_id=key[1],box_index=i,
                                old_name=names[i],new_name=rule['to'],box=list(map(float,np.asarray(ann['gt_boxes'])[i,:7]))))
            names[i]=rule['to'];counts[ri]+=1;touched=True
        if touched:
            # Never assign Truck into a fixed-width <U3 Bus array: it truncates!
            ann['gt_names']=np.asarray(names)
            if 'gt_mesh' in ann:ann['gt_mesh']=np.asarray(['detzero.'+n.lower() for n in names])
    for ri,rule in enumerate(rules):
        expected=rule.get('expected_raw_changes',{}).get(split)
        if expected is None or counts[ri]!=expected:
            raise ValueError(f'Review count mismatch {split} {rule["sequence_id"]}/{rule["track_id"]}: {counts[ri]} != {expected}')
    return result,changes


def counts(document,T):
    raw=Counter();roi=Counter();trainable=Counter();ignored=0
    for row in document['data_list']:
        ann=row['annos'];ids=names_to_ids(ann['gt_names'])
        boxes=boxes_to_baselink(np.asarray(ann['gt_boxes'])[:,:7],'CLIDAR',T)
        valid=np.isfinite(boxes).all(1)&(boxes[:,3:6]>0).all(1)
        keep=valid&(boxes[:,0]>=-48)&(boxes[:,0]<48)&(boxes[:,1]>=-48)&(boxes[:,1]<48)
        raw.update(CLASSES[int(i)] for i in ids)
        roi.update(CLASSES[int(i)] for i in ids[keep])
        ignore=np.asarray(ann.get('gt_ignore',np.zeros(len(ids),dtype=bool)),dtype=bool)
        trainable.update(CLASSES[int(i)] for i in ids[keep&~ignore]);ignored+=int((keep&ignore).sum())
    return dict(raw=dict(raw),in_range=dict(roi),trainable_in_range=dict(trainable),ignored_in_range=ignored)


def ignore_uncertain_tracks(document,rules,split):
    """Preserve uncertain raw boxes; exclude TRAIN targets, never validation GT."""
    counts=Counter();records=[]
    lookup={(str(r['sequence_id']),str(r['track_id'])):r for r in rules}
    if len(lookup)!=len(rules):raise ValueError('Duplicate ignore-track rule')
    for row in document['data_list']:
        ann=row['annos'];flags=np.asarray(ann.get('gt_ignore',np.zeros(len(ann['gt_ids']),dtype=bool)),dtype=bool).copy()
        for i,uid in enumerate(ann['gt_ids']):
            key=(str(row['metainfo']['sequence_id']),str(uid))
            if key not in lookup:continue
            if split!='train':raise ValueError('Ignore review may not alter validation GT')
            if flags[i]:raise ValueError('Ignore review overlaps an existing ignore flag')
            flags[i]=True;counts[key]+=1
            records.append(dict(token=str(row['token']),track_id=key[1],sequence_id=key[0],box_index=i))
        if flags.any():ann['gt_ignore']=flags
    if split=='train':
        for key,rule in lookup.items():
            if counts[key]!=rule['expected_train_boxes']:raise ValueError(f'Ignore-track count mismatch: {key}')
    return records


def create_revision(root,rules_path,output,calibration=DEFAULT_CALIBRATION,write=False):
    root=Path(root);output=Path(output);review=json.loads(Path(rules_path).read_text())
    if review.get('type')!='raybev_track_class_review_v1' or review.get('approved') is not True:
        raise ValueError('An explicitly approved review is required')
    if output.exists():raise FileExistsError('Revision output already exists; refusing overwrite')
    rules=review['rules'];T=read_T_base_clidar(calibration)
    keep={(str(r['sequence_id']),str(r['track_id'])) for r in review.get('keep_bus_tracks',[])}
    if any((str(r['sequence_id']),str(r['track_id'])) in keep for r in rules):
        raise ValueError('Review attempts to change an explicitly protected Bus/Tram track')
    report=dict(type='raybev_annotation_revision_v1',rules_sha256=sha(rules_path),calibration_sha256=sha(calibration),
                note='Reviewed classes and optional TRAIN ignore flags; original geometry, images, tokens, IDs and splits unchanged.',splits={})
    payloads={}
    for split in ('train','val'):
        source=root/('rosa/fisheye3dod_infos_'+split+'.pkl')
        original=load_annotations(source);revised,changes=revise_document(original,rules,split)
        ignored=ignore_uncertain_tracks(revised,review.get('ignore_tracks',[]),split)
        # Make accidental box/token mutation a hard failure before publishing.
        for a,b in zip(original['data_list'],revised['data_list']):
            if a['token']!=b['token'] or not np.array_equal(a['annos']['gt_boxes'],b['annos']['gt_boxes']):
                raise ValueError('Class review changed geometry or sample identity')
        before=counts(original,T);after=counts(revised,T)
        report['splits'][split]=dict(source=str(source),source_sha256=sha(source),raw_changes=len(changes),
                                    source_semantic_sha256=gt_signature(original),
                                    before=before,after=after,changes=changes,ignored_boxes=ignored)
        if changes or ignored:
            revised.setdefault('metainfo',{})['label_review_sha256']=report['rules_sha256']
        payloads[split]=(source,revised,changes or ignored)
    if write:
        output.mkdir(parents=True)
        for split,(source,revised,changes) in payloads.items():
            destination=output/(split+'.pkl')
            if not changes:shutil.copyfile(source,destination)  # unchanged TRAIN stays byte-identical for teacher checks
            else:
                with destination.open('wb') as fh:pickle.dump(revised,fh,protocol=4)
            report['splits'][split]['output_sha256']=sha(destination)
        shutil.copyfile(rules_path,output/'rules.json')
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def verify_revision(root,output,calibration=DEFAULT_CALIBRATION):
    root=Path(root);output=Path(output);report=json.loads((output/'manifest.json').read_text())
    if report['calibration_sha256']!=sha(calibration) or report['rules_sha256']!=sha(output/'rules.json'):
        raise ValueError('Revision calibration/review mismatch')
    for split,info in report['splits'].items():
        source=root/('rosa/fisheye3dod_infos_'+split+'.pkl')
        if sha(source)!=info['source_sha256'] and gt_signature(load_annotations(source))!=info.get('source_semantic_sha256'):
            raise ValueError(f'Revision {split} source labels differ')
        if sha(output/(split+'.pkl'))!=info['output_sha256']:raise ValueError(f'Revision {split} output differs')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rules',type=Path,default=Path(__file__).resolve().parents[1]/'datasets/heavy_review_v1_rules.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--write',action='store_true')
    p.add_argument('--verify',action='store_true',help='Verify existing revision against current source labels, allowing pickle/path relocation only')
    a=p.parse_args()
    if a.write and a.verify:p.error('--write and --verify are mutually exclusive')
    report=verify_revision(a.root,a.output,a.calibration) if a.verify else create_revision(a.root,a.rules,a.output,a.calibration,a.write)
    print('VERIFIED' if a.verify else 'WROTE' if a.write else 'DRY RUN',a.output)
    for split,r in report['splits'].items():
        print(split,'raw class changes',r['raw_changes'],'ROI counts',r['before']['in_range'],'->',r['after']['in_range'])
        if r['after'].get('ignored_in_range',0):
            print(split,'trainable ROI counts',r['after']['trainable_in_range'],
                  'ignored raw/ROI boxes',len(r.get('ignored_boxes',[])),r['after']['ignored_in_range'])


if __name__=='__main__':main()
