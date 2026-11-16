#!/usr/bin/env python3
"""Preview remaining v6 issues, then apply an explicit evidence-backed v7 plan.

Sources stay immutable. Default dimensions are prescribed priors, not fitted GT.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from review_heavy_annotations import (ReviewSession, PROJECT, CLASSES, DEFAULT_CALIBRATION,
                                     fit_preview, overlay_font, sha, atomic_json)
from raybev.rosa_dataset import load_annotations


def split_mixed_track_ids(directory, session, implicated):
    """Separate conflicting class/identity groups; never force a majority class.

    New IDs are conservative review segments, not proof of physical identity.
    No geometry, image association, or class is changed by this operation.
    """
    groups=defaultdict(list);max_ids=defaultdict(int);source_lookup={}
    source_counts={(split,str(row['token'])):len(row['annos']['gt_boxes'])
                   for split,doc in session.documents.items() for row in doc['data_list']}
    for rec in session.records:
        source_lookup[(rec['split'],rec['token'],rec['index'])]=rec
        try:max_ids[rec['sequence']]=max(max_ids[rec['sequence']],int(rec['track_id']))
        except ValueError:pass
    docs={s:load_annotations(directory/f'fisheye3dod_infos_{s}.pkl') for s in ('train','val')}
    for split,doc in docs.items():
        for row in doc['data_list']:
            token=str(row['token']);kept=[i for i in range(source_counts[(split,token)])
                if session.decisions.get(f'{split}:{token}:{i}',{}).get('status')!='removed']
            for index,raw in enumerate(kept):
                rec=source_lookup[(split,token,raw)];key=(rec['sequence'],rec['track_id'])
                if key in implicated:groups[key].append((rec,row,index,str(row['annos']['gt_names'][index]).lower()))
    edits=[]
    for (bag,old_id),items in sorted(groups.items()):
        labels=sorted({item[3] for item in items})
        if len(labels)<2:continue
        canonical=min(items,key=lambda item:item[0]['token'])[3]
        replacements={}
        for label in labels:
            if label!=canonical:
                max_ids[bag]+=1;replacements[label]=max_ids[bag]
        for rec,row,index,label in items:
            if label not in replacements:continue
            new_id=replacements[label];values=row['annos']['gt_ids']
            if np.asarray(values).dtype.kind not in 'iu':raise ValueError('Track segmentation requires integer IDs')
            if new_id>np.iinfo(values.dtype).max:raise ValueError('New track ID overflows source dtype')
            values[index]=new_id
            edits.append(dict(id=rec['id'],sequence=bag,old_track_id=old_id,new_track_id=new_id,
                evidence='Four-camera review: different vehicle classes/ambiguous association share this ID. '+
                'Conservative class-consistent review segment; not a certified physical track. '+
                f'Contact sheet: {bag}_{old_id}.jpg'))
    atomic_json(directory/'track_id_revisions.json',dict(type='raybev_track_identity_revision_v1',edits=edits,
        limitation='Review segmentation only. Seen/unseen metrics must use this frozen release consistently.'))
    manifest=json.loads((directory/'review_manifest.json').read_text())
    for split,doc in docs.items():
        path=directory/f'fisheye3dod_infos_{split}.pkl'
        with path.open('wb') as stream:pickle.dump(doc,stream,protocol=4)
        manifest['splits'][split]['output_sha256']=sha(path)
    manifest['track_id_revisions']=dict(records=len(edits),sha256=sha(directory/'track_id_revisions.json'))
    manifest['limitations'].append('Conflicting IDs split into conservative class-consistent review segments; physical identity is not certified.')
    atomic_json(directory/'review_manifest.json',manifest)
    return edits


def reviewed_plan(session, geometry, issues):
    """Explicit decisions after inspection of the saved four-camera contact sheets."""
    car_to_truck={('09_16_09_58_56','2350'),('09_16_09_58_56','2376'),
        ('09_24_08_52_50','5813'),('09_24_08_52_50','5920'),
        ('09_24_08_52_50','5928'),('11_09_08_01_03','634')}
    removals={
        'val:1787825893246948695:12','val:1787825894500342959:11',
        'train:1790240194797370933:10','train:1790240196174029662:17','train:1790240196564527903:13',
        'train:1790240383493262827:13','train:1790240384674492384:11',
        'train:1790240393834864045:15','val:1790763780382908014:5'}
    # Keys above are source-v5 audit keys: use the corresponding final-v6 keys below.
    raw_to_final={row['id']:key for key,row in geometry.items()}
    removals={raw_to_final.get(key,key) for key in removals}
    # The final 6118 Truck frame is in the mixed-track queue, not the geometry list.
    for key in issues:
        rec=session.by_id[key]
        if rec['sequence']=='09_24_08_52_50' and rec['track_id']=='6118' and rec['original_class']=='truck':
            removals.add(key)
    edits=[]
    for key in sorted(issues):
        rec=session.by_id[key];track=(rec['sequence'],rec['track_id']);label=rec['original_class']
        evidence=f"Four-camera contact sheet {rec['sequence']}_{rec['track_id']}.jpg. "
        edit=dict(id=key,label=label,status='reviewed',evidence=evidence+'Retain existing per-frame class; no majority-vote propagation.')
        if key in removals:
            edit.update(status='removed',evidence=evidence+'Exclude: projected target overlaps different foreground/background vehicles; association/metric cuboid is not credible enough to relabel or resize.')
        elif key in geometry:
            if label=='van':
                edit.update(dimensions_lwh=[5.46,2.14,2.26],evidence=evidence+'Panel van retained. User-authorized default LWH: rounded median of 1,517 plausible TRAIN Vans; prescribed prior, not a fitted cuboid.')
            elif track==('08_27_10_12_18','5158') or track==('08_27_10_12_18','5566'):
                edit.update(label='van',dimensions_lwh=[5.46,2.14,2.26],evidence=evidence+'Integrated high-roof panel-van body, not a passenger Car. Apply the TRAIN-derived Van default; center/yaw not certified.')
            elif track in car_to_truck or track==('09_24_08_52_50','5794'):
                edit.update(label='truck',evidence=evidence+'Visible separate cab and cargo/box/tow body identify a Truck, not a passenger Car. Retain source LWH; no guessed truck geometry.')
            elif track==('09_30_10_10_54','5010'):
                edit.update(dimensions_lwh=[4.8,1.9,1.75],evidence=evidence+'Passenger SUV. Replace prescribed 3 m height with a conservative SUV size prior; not metric geometry certification.')
            elif track==('11_06_10_07_06','202'):
                edit.update(dimensions_lwh=[4.8,1.9,1.85],evidence=evidence+'Passenger SUV. Replace prescribed 3 m height with an SUV size prior; center/yaw not certified.')
            else:raise ValueError('No visually reviewed action for geometry flag: '+key)
        edits.append(edit)
    return dict(source_identity=session.identity,van_default_lwh=[5.46,2.14,2.26],
        van_default_policy='Flagged/reclassified Vans only; rounded coordinate-wise median of 1,517 plausible TRAIN Vans. Unflagged Van geometry stays unchanged.',
        limitation='Image-only review plus prescribed dimensions, not LiDAR-fitted geometry. No endpoint approval.',edits=edits)


def final_issue_ids(session, audit, source):
    """Audit keys address v5 indices; remap through v6 removals before editing."""
    saved = json.loads((source/'review_decisions.json').read_text())['decisions']
    prior = source.parent/'combined_bags_camera_ray_manual_review_v5'
    mapping = {}
    for split in ('train','val'):
        old = load_annotations(prior/f'fisheye3dod_infos_{split}.pkl')
        new = session.documents[split]
        for before, after in zip(old['data_list'],new['data_list']):
            token = str(before['token'])
            if token != str(after['token']):raise ValueError('Source frame mismatch')
            kept = [i for i in range(len(before['annos']['gt_boxes']))
                    if saved.get(f'{split}:{token}:{i}',{}).get('status') != 'removed']
            if len(kept) != len(after['annos']['gt_boxes']):raise ValueError('Source removal mismatch')
            for index, raw in enumerate(kept):mapping[f'{split}:{token}:{raw}'] = f'{split}:{token}:{index}'
    geometry = {mapping[row['id']]:row for row in audit['geometry_flags']}
    conflicts = {(row['sequence'],row['track_id']) for row in audit['track_class_conflicts']}
    issues = set(geometry)
    for track in conflicts:issues.update(session.by_track[track])
    return geometry, conflicts, issues


def previews(session, geometry, conflicts, output):
    groups = defaultdict(list)
    for rec in session.records:
        key = (rec['sequence'],rec['track_id'])
        if rec['id'] in geometry or key in conflicts:groups[key].append(rec)
    output.mkdir(parents=True,exist_ok=True)
    for (bag,track),records in sorted(groups.items()):
        records.sort(key=lambda r:r['token'])
        selected = set(range(len(records)))
        if len(records)>9:
            selected = {0,len(records)-1}
            for i in range(1,len(records)):
                if records[i]['original_class']!=records[i-1]['original_class']:
                    selected.update(range(max(0,i-1),min(len(records),i+2)))
        chosen = [records[i] for i in sorted(selected)]
        sheet = Image.new('RGB',(1800,360*len(chosen)),(25,25,25))
        draw = ImageDraw.Draw(sheet);font = overlay_font(22)
        for ri,rec in enumerate(chosen):
            top = ri*360
            draw.text((8,top+2),f"{bag}/{track} | {rec['id']} | {rec['original_class']} | LWH {np.round(rec['box_CLIDAR'][3:6],2)}",font=font,fill='white')
            for ci,(name,image,_) in enumerate(session.render(rec['id'],context=.45)):
                small = fit_preview(image,444,310)
                sheet.paste(small,(ci*450+(450-small.width)//2,top+45+(310-small.height)//2))
                draw.text((ci*450+5,top+26),name,font=font,fill='yellow')
        sheet.save(output/f'{bag}_{track}.jpg',quality=93)
        print('Preview:',bag,track,len(chosen),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=PROJECT/'datasets/combined_bags_camera_ray_manual_review_v6')
    parser.add_argument('--audit',type=Path,default=PROJECT/'analysis/v10_manual_review_v6/annotation_audit.json')
    parser.add_argument('--analysis',type=Path,default=PROJECT/'analysis/v10_manual_review_v7')
    parser.add_argument('--output',type=Path,default=PROJECT/'datasets/combined_bags_camera_ray_manual_review_v7')
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--write-reviewed-plan',action='store_true',help='Write the explicit contact-sheet review decisions; do not export')
    parser.add_argument('--previews-only',action='store_true')
    args = parser.parse_args();source=args.source.resolve();analysis=args.analysis.resolve()
    if not args.previews_only and not args.write_reviewed_plan and args.plan is None:parser.error('Supply an explicit --plan or --previews-only')
    state=analysis/'decisions.json'
    session = ReviewSession(args.root,source/'fisheye3dod_infos_train.pkl',source/'fisheye3dod_infos_val.pkl',
        source/'train_yolo_extents_v1.json',DEFAULT_CALIBRATION,state,classes=CLASSES)
    geometry,conflicts,issues=final_issue_ids(session,json.loads(args.audit.read_text()),source)
    if args.previews_only:
        previews(session,geometry,conflicts,analysis/'previews')
        return
    if args.write_reviewed_plan:
        path=analysis/'correction_plan.json'
        if path.exists():raise FileExistsError('Plan already exists: '+str(path))
        atomic_json(path,reviewed_plan(session,geometry,issues));print(path)
        return
    if state.exists():raise FileExistsError('Refusing to reuse a previous v7 build state: '+str(state))
    plan=json.loads(args.plan.read_text())
    if plan['source_identity']!=session.identity:raise ValueError('Correction plan belongs to different source files')
    edits=plan['edits'];seen=set()
    for edit in edits:
        key=edit['id']
        if key in seen or key not in issues:raise ValueError('Duplicate/unflagged correction: '+key)
        seen.add(key)
        if not edit.get('evidence'):raise ValueError('Every correction needs evidence/provenance')
        session.validate_decision(key,dict(label=edit['label'],status=edit['status'],note=edit['evidence'],
            **({'dimensions_lwh':edit['dimensions_lwh']} if 'dimensions_lwh' in edit else {})))
    if seen != issues:raise ValueError('Plan must explicitly account for all remaining issue objects')
    for edit in edits:
        if edit['status']=='removed':session.remove(edit['id'],edit['evidence'])
        else:session.apply(edit['id'],edit['label'],uncertain=edit['status']=='uncertain',
            dimensions=edit.get('dimensions_lwh'),note=edit['evidence'])
    if args.output.exists():raise FileExistsError('Output already exists: '+str(args.output))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    # Verify the complete staged release before publishing the final directory.
    with tempfile.TemporaryDirectory(prefix='raybev-v7-build-',dir=args.output.parent) as temporary:
        staged=Path(temporary)/'release'
        session.export(staged)
        implicated=conflicts|{(r['sequence'],r['track_id']) for r in geometry.values()}
        split_mixed_track_ids(staged,session,implicated)
        from audit_manual_release import audit
        checked=audit(args.root,staged,source,DEFAULT_CALIBRATION)
        if checked['track_class_conflicts'] or checked['geometry_flags']:
            raise ValueError('Unresolved track/geometry flags in staged v7; see the correction plan')
        report=json.loads((staged/'review_manifest.json').read_text())
        report['verification']=checked['verification'];atomic_json(staged/'review_manifest.json',report)
        atomic_json(analysis/'annotation_audit.json',checked)
        os.rename(staged,args.output)
    atomic_json(analysis/'build_report.json',dict(source_identity=session.identity,plan_sha256=sha(args.plan),
        export=report,limitation='Image review and default dimensions are not independent LiDAR geometry certification.'))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
