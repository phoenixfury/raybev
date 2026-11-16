#!/usr/bin/env python3
"""Freeze reviewed inputs, audit flags and TRAIN-only initialization for V10."""
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT),str(Path(__file__).resolve().parent)]
from audit_manual_release import audit
from review_heavy_annotations import sha
from raybev.config import DEFAULT_CALIBRATION
from raybev.rosa_dataset import RosaCombinedDataset,training_boxes_sha256
from raybev.sparse_config import SparseConfig
from raybev.sparse_training import fit_sparse_anchors
from raybev.sparse_diagnostics import raw_query_counts,merge_raw_counts
from raybev.training import calibration_sha256


def write_followup(report,source,release,output):
    """Address flagged objects by final V5 indices, not pre-removal indices."""
    from raybev.rosa_dataset import load_annotations
    conflicts={(row['sequence'],row['track_id']) for row in report['track_class_conflicts']}
    geometry_keys={row['id'] for row in report['geometry_flags']}
    saved=json.loads((release/'review_decisions.json').read_text())['decisions'];followup=[]
    for split in ('train','val'):
        original=load_annotations(source/f'fisheye3dod_infos_{split}.pkl')
        revised=load_annotations(release/f'fisheye3dod_infos_{split}.pkl')
        for old,row in zip(original['data_list'],revised['data_list']):
            token=str(row['token']);source_indices=[i for i in range(len(old['annos']['gt_boxes']))
                if saved.get(f'{split}:{token}:{i}',{}).get('status')!='removed']
            for index,raw in enumerate(source_indices):
                sequence=str(row['metainfo']['sequence_id']);track=str(row['annos']['gt_ids'][index]);reasons=[]
                if (sequence,track) in conflicts:reasons.append('Mixed classes within one retained track: confirm identity/class; do not auto-majority-vote.')
                if f'{split}:{token}:{raw}' in geometry_keys:reasons.append('Class/dimension mismatch: review image/LiDAR association and geometry, not class alone.')
                if reasons:followup.append(dict(split=split,token=token,index=index,sequence=sequence,track_id=track,
                    status='unresolved',reason=' '.join(reasons)))
    (output/'followup_review.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in followup))
    return len(followup)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--release',type=Path,default=PROJECT/'datasets/combined_bags_camera_ray_manual_review_v5')
    p.add_argument('--source',type=Path,default=PROJECT/'datasets/combined_bags_camera_ray_heavy_review_v4')
    p.add_argument('--output',type=Path,default=PROJECT/'analysis/v10_manual_review_v5')
    p.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    p.add_argument('--seed',type=int,default=3407)
    p.add_argument('--tram-tracks',type=Path,help='Inherit an explicit physical-tram registry when the immediate source has no original review ledger')
    a=p.parse_args();a.release=a.release.resolve();a.root=a.root.resolve();a.output.mkdir(parents=True,exist_ok=True)
    report=audit(a.root,a.release,a.source,a.calibration)
    (a.output/'annotation_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    cfg=SparseConfig(architecture_version=5,backbone='yolo26m',queries=225,channels=96,ffn_channels=192,
        stages=4,max_detections=225,stride4=True,observation_conditioning=True,proposal_queries=64,
        class_positive_weights=(1.,)*6,teacher_weight=0.,facing_cosine_prior=0.,proposal_cell_targets=True)
    train=RosaCombinedDataset(a.root,a.release/'fisheye3dod_infos_train.pkl',cfg=cfg,calibration=a.calibration)
    val=RosaCombinedDataset(a.root,a.release/'fisheye3dod_infos_val.pkl',cfg=cfg,calibration=a.calibration)
    initialization={}
    for arm in ('control','shapes'):
        print('Fitting TRAIN-only anchor hypotheses:',arm,flush=True)
        setting=replace(cfg,shape_balanced_anchors=arm=='shapes',shape_balanced_proposals=arm=='shapes')
        fitted,anchors=fit_sparse_anchors(train,setting,a.seed)
        torch.save(dict(cfg=fitted.to_dict(),anchors=anchors,seed=a.seed),a.output/f'{arm}_initialization.pt')
        groups=np.digitize(anchors[:,3].numpy(),[2.5,5.5,8.5,14.]);prefix=cfg.queries-cfg.proposal_queries
        coverage={}
        for split,ds in (('train',train),('val',val)):
            records=[{'learned_anchors_only':raw_query_counts(anchors[:prefix].numpy(),boxes.numpy())} for boxes in ds.boxes]
            coverage[split]=merge_raw_counts(records)
        initialization[arm]=dict(learned_shape_bins=np.bincount(groups[:prefix],minlength=5).tolist(),
            proposal_size_shape_bins=np.bincount(groups[prefix:],minlength=5).tolist(),coverage=coverage)
    (a.output/'initialization_audit.json').write_text(json.dumps(dict(seed=a.seed,
        shape_bin_length_edges_m=[2.5,5.5,8.5,14.],arms=initialization,
        limitation='Static learned-anchor XY coverage only; image-scored proposal and refinement coverage is logged during training.'),indent=2)+'\n')
    ledger=a.source/'annotation_review.jsonl';tracks=set()
    if a.tram_tracks is not None:
        registry=json.loads(a.tram_tracks.read_text())
        if registry.get('type')!='raybev_tram_tracks_v1':raise ValueError('Invalid inherited tram registry')
        tracks={(str(row['sequence']),str(row['track_id'])) for row in registry['tracks']}
        ledger=a.tram_tracks
    elif ledger.exists():
        for line in ledger.read_text().splitlines():
            entry=json.loads(line)
            if entry.get('physical_class')=='tram':tracks.add((str(entry['sequence']),str(entry['track_id'])))
    (a.output/'tram_tracks.json').write_text(json.dumps(dict(type='raybev_tram_tracks_v1',
        source=str(ledger),tracks=[dict(sequence=bag,track_id=uid) for bag,uid in sorted(tracks)]),indent=2)+'\n')
    template=a.output/'geometry_approvals_TEMPLATE.json'
    if not template.exists():template.write_text(json.dumps(dict(type='raybev_geometry_approvals_v1',
        training_boxes_sha256=training_boxes_sha256(train),calibration_sha256=calibration_sha256(a.calibration),
        instructions='Only add independently geometry-reviewed TRAIN boxes as token, gt=[x,y,z,L,W,H,yaw,class_id], geometry_reviewed=true, evidence. Class review is insufficient.',
        approvals=[]),indent=2)+'\n')
    files=['fisheye3dod_infos_train.pkl','fisheye3dod_infos_val.pkl','train_yolo_extents_v1.json','review_decisions.json','review_manifest.json','review_queue.jsonl']
    if (a.release/'track_id_revisions.json').exists():files.append('track_id_revisions.json')
    contract=dict(release=str(a.release),files={name:sha(a.release/name) for name in files},
        calibration_sha256=sha(a.calibration),seed=a.seed,track_class_conflicts=len(report['track_class_conflicts']),
        geometry_flags=len(report['geometry_flags']),endpoint_ready=False)
    contract['geometry_flags_by_class']=dict(Counter(row['class_name'] for row in report['geometry_flags']))
    (a.output/'input_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    write_followup(report,a.source,a.release,a.output)
    print(json.dumps(contract,indent=2))


if __name__=='__main__':main()
