"""TRAIN-only teacher contracts, world-pose track references, sparse gradients."""
import importlib.util
import json
import math
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from raybev.config import ModelConfig,DEFAULT_CALIBRATION
from raybev.rosa_dataset import RosaCombinedDataset,training_boxes_sha256
from raybev.detection import make_targets
from raybev.heading_supervision import HeadingSupervision,track_references
from raybev.training import check_resume,calibration_sha256


def prepare_module():
    path=Path(__file__).parents[1]/'scripts/prepare_heading_teacher.py'
    spec=importlib.util.spec_from_file_location('prepare_heading_teacher_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.fixture
def scene(tmp_path):
    rows=[]
    for i in range(2):
        ego=i*.4;c,s=math.cos(ego),math.sin(ego)
        T=np.eye(4);T[:2,:2]=[[c,-s],[s,c]]
        rows.append(dict(token=str(i),annos=dict(gt_boxes=np.array([[4.,2.,0.,4.5,2.,1.5,.3-ego]]),
            gt_names=['Car'],gt_ids=np.array([7])),metainfo=dict(temporal_split='train',sequence_id='bag',
            acquisition_timestamp_ns=1_000_000_000+i*500_000_000,T_world_CLIDAR=T)))
    annotation=tmp_path/'train.pkl'
    with annotation.open('wb') as f:pickle.dump({'data_list':rows},f)
    cfg=ModelConfig().downscale(5)
    dataset=RosaCombinedDataset(tmp_path,annotation,calibration=DEFAULT_CALIBRATION,cfg=cfg)
    checkpoint=tmp_path/'rosa.pth';checkpoint.write_bytes(b'test fixture only')
    document=dict(box_origin='gravity_center',checkpoint=str(checkpoint),frames=[
        dict(token=str(i),boxes_3d=row['annos']['gt_boxes'].tolist(),names=['Car'],scores=[.9]) for i,row in enumerate(rows)])
    source=tmp_path/'predictions.json';source.write_text(json.dumps(document))
    cache=prepare_module().build_cache(dataset,document,checkpoint,DEFAULT_CALIBRATION,DEFAULT_CALIBRATION,source)
    cachefile=tmp_path/'teacher.json';cachefile.write_text(json.dumps(cache))
    args=SimpleNamespace(heading_teacher=str(cachefile),heading_teacher_weight=.2,track_heading_weight=.1,
        calibration=DEFAULT_CALIBRATION,lidar_calibration=DEFAULT_CALIBRATION)
    return dataset,cfg,args,document,checkpoint,source,cache


def test_training_teacher_matches_and_sparse_losses_backpropagate(scene):
    ds,cfg,args,*_=scene
    supervision=HeadingSupervision(ds,cfg,args)
    assert sum(map(len,supervision.teacher.values()))==2
    assert sum(map(len,supervision.track.values()))==2
    output=[torch.zeros(2,c,cfg.bev_height,cfg.bev_width,requires_grad=True) for c in (6,8,1,2)]
    targets=make_targets(ds.boxes,cfg,return_class_ids=True)
    loss,parts=supervision.loss(output,targets,['0','1']);loss.backward()
    assert loss>0 and parts['teacher_centers']==parts['track_centers']==2
    assert output[1].grad.abs().sum()>0 and output[3].grad.abs().sum()>0
    assert torch.isfinite(output[1].grad).all() and torch.isfinite(output[3].grad).all()


def test_track_reference_rotates_world_heading_back_into_current_baselink(scene):
    ds,cfg,args,*_=scene
    refs=track_references(ds,DEFAULT_CALIBRATION)
    for row,box in zip(ds.rows,ds.boxes):
        yaw=refs[row['token']][0]['yaw'];expected=float(box[0,6])
        assert abs(math.atan2(math.sin(yaw-expected),math.cos(yaw-expected)))<.002


def test_reviewed_teacher_rebind_remaps_only_classes_drops_ignored_and_preserves_yaws(scene,tmp_path):
    from copy import deepcopy
    ds,cfg,args,_,_,_,original=scene
    path=Path(__file__).parents[1]/'scripts/rebind_heading_teacher.py'
    spec=importlib.util.spec_from_file_location('rebind_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    doc=deepcopy(ds.annotation);doc['data_list'][0]['annos']['gt_ignore']=[True]
    doc['data_list'][1]['annos']['gt_names']=['Truck']
    ann=tmp_path/'reviewed.pkl'
    with ann.open('wb') as f:pickle.dump(doc,f)
    revised=RosaCombinedDataset(tmp_path,ann,calibration=DEFAULT_CALIBRATION,cfg=cfg)
    assert revised.gt_counts['car']==0 and revised.gt_counts['truck']==1 and len(revised.ignore_boxes[0])==1
    cache=module.rebind(ds,revised,args.heading_teacher)
    assert cache['frames']['0']==[] and cache['frames']['1'][0]['gt'][7]==3
    assert cache['frames']['1'][0]['yaw']==original['frames']['1'][0]['yaw']
    assert cache['teacher_checkpoint_sha256']==original['teacher_checkpoint_sha256']
    assert cache['predictions_sha256']==original['predictions_sha256']
    assert cache['reviewed_rebind']['remapped_class_records']==cache['reviewed_rebind']['dropped_ignored_records']==1
    rebound=tmp_path/'rebound.json';rebound.write_text(json.dumps(cache));args.heading_teacher=str(rebound)
    circular=replace(cfg,facing_loss='circular_bins',heading_bins=8)
    supervisor=HeadingSupervision(revised,circular,args)
    output=[torch.zeros(2,c,cfg.bev_height,cfg.bev_width,requires_grad=True) for c in (6,8,1,8)]
    targets=make_targets(revised.boxes,circular,return_class_ids=True)
    loss,parts=supervisor.loss(output,targets,['0','1']);loss.backward()
    assert parts['teacher_centers']==1 and output[3].grad.abs().sum()>0 and torch.isfinite(output[3].grad).all()
    # Rebinding cannot be used as a way to bless a changed box or token.
    revised.rows[1]['annos']['gt_boxes'][0,0]+=.1
    with pytest.raises(ValueError,match='geometry'):
        module.rebind(ds,revised,Path(scene[2].heading_teacher).parent/'teacher.json')


@pytest.mark.parametrize('field,value',[
    ('split','val'),('training_boxes_sha256','wrong'),('camera_calibration_sha256','wrong'),
    ('lidar_calibration_sha256','wrong'),('box_frame','BASELINK'),('z_reference','bottom')])
def test_teacher_contract_rejects_split_calibration_and_annotation_changes(scene,field,value):
    ds,cfg,args,_,_,_,cache=scene
    cache[field]=value;Path(args.heading_teacher).write_text(json.dumps(cache))
    with pytest.raises(ValueError,match='contract mismatch'):HeadingSupervision(ds,cfg,args)


def test_teacher_validation_tokens_cannot_be_used_as_training_cache(scene):
    ds,cfg,args,document,checkpoint,source,_=scene
    document['frames'][0]['token']='validation_only'
    with pytest.raises(ValueError,match='TRAIN split exactly'):
        prepare_module().build_cache(ds,document,checkpoint,DEFAULT_CALIBRATION,DEFAULT_CALIBRATION,source)


def test_portable_teacher_survives_relocated_paths_and_pickle_serialization(scene,tmp_path):
    ds,cfg,args,*_=scene
    payload=dict(ds.annotation,source_root='/different/server/path',data_list=list(reversed(ds.rows)))
    moved=tmp_path/'relocated.pkl'
    with moved.open('wb') as f:pickle.dump(payload,f,protocol=2)
    relocated=RosaCombinedDataset(tmp_path,moved,calibration=DEFAULT_CALIBRATION,cfg=cfg)
    assert calibration_sha256(moved)!=calibration_sha256(ds.annotation_path)
    assert training_boxes_sha256(relocated)==training_boxes_sha256(ds)
    supervisor=HeadingSupervision(relocated,cfg,args)
    assert sum(map(len,supervisor.teacher.values()))==2


def test_portable_teacher_rejects_changed_gt_even_when_original_pickle_hash_matches(scene):
    ds,cfg,args,*_=scene
    ds.boxes[0][0,0]+=.1
    with pytest.raises(ValueError,match='contract mismatch'):HeadingSupervision(ds,cfg,args)


def test_historical_teacher_without_portable_signature_keeps_strict_file_hash(scene):
    ds,cfg,args,_,_,_,cache=scene
    cache.pop('training_boxes_sha256');Path(args.heading_teacher).write_text(json.dumps(cache))
    assert sum(map(len,HeadingSupervision(ds,cfg,args).teacher.values()))==2
    cache['annotation_sha256']='wrong';Path(args.heading_teacher).write_text(json.dumps(cache))
    with pytest.raises(ValueError,match='contract mismatch'):HeadingSupervision(ds,cfg,args)


def test_reversed_teacher_heading_is_not_copied(scene):
    ds,cfg,args,document,checkpoint,source,_=scene
    document['frames'][0]['boxes_3d'][0][6]+=math.pi
    cache=prepare_module().build_cache(ds,document,checkpoint,DEFAULT_CALIBRATION,DEFAULT_CALIBRATION,source)
    assert cache['frames']['0']==[] and len(cache['frames']['1'])==1


def test_any_validation_row_forbids_teacher_or_track_supervision(scene):
    ds,cfg,args,*_=scene;ds.rows[0]['metainfo']['temporal_split']='val'
    with pytest.raises(ValueError,match='TRAIN rows only'):HeadingSupervision(ds,cfg,args)


def test_resume_rejects_teacher_content_change(scene):
    ds,cfg,args,*_=scene
    state=dict(cfg=vars(cfg),heading_teacher_sha256='old')
    with pytest.raises(ValueError,match='heading teacher differs'):check_resume(state,cfg,args)


def test_sparse_owner_collisions_do_not_distill_another_objects_heading(scene):
    ds,cfg,args,*_=scene
    # Shift the cache GT within a cell without changing the real regression owner.
    data=json.loads(Path(args.heading_teacher).read_text())
    data['frames']['0'][0]['gt'][0]+=.01
    Path(args.heading_teacher).write_text(json.dumps(data))
    s=HeadingSupervision(ds,cfg,args)
    assert s.teacher['0']==[] and len(s.teacher['1'])==1


def test_disabled_teacher_and_empty_tracks_preserve_finite_empty_backward(scene):
    ds,cfg,args,*_=scene;args.heading_teacher=None;args.track_heading_weight=0.
    s=HeadingSupervision(ds,cfg,args)
    output=[torch.zeros(1,c,8,8,requires_grad=True) for c in (6,8,1,2)]
    cfg=replace(cfg,bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.)
    loss,parts=s.loss(output,make_targets([torch.empty(0,8)],cfg,return_class_ids=True),['none'])
    loss.backward()
    assert loss==0 and output[1].grad is not None and output[3].grad is not None


def test_teacher_inference_resolves_paths_before_changing_subprocess_directory(scene,monkeypatch,tmp_path):
    import sys
    ds,cfg,args,document,checkpoint,source,_=scene
    module=prepare_module();native=tmp_path/'native';native.mkdir()
    config=tmp_path/'rosa.py';config.write_text('# fixture')
    output=tmp_path/'generated.json'
    monkeypatch.chdir(tmp_path)
    def infer(command,check,cwd):
        for flag in ('--config','--checkpoint','--data-root','--output-dir'):
            assert Path(command[command.index(flag)+1]).is_absolute()
        assert command[command.index('--split')+1]=='train'
        folder=Path(command[command.index('--output-dir')+1])/checkpoint.stem
        folder.mkdir(parents=True);(folder/'predictions.json').write_text(json.dumps(document))
    monkeypatch.setattr(module.subprocess,'run',infer)
    monkeypatch.setattr(sys,'argv',['prepare','--root','.', '--train-ann','train.pkl',
        '--teacher-checkpoint','rosa.pth','--rosa-config','rosa.py','--rosa-data-root','native',
        '--output',str(output)])
    module.main()
    cache=json.loads(output.read_text())
    assert cache['matched_class_counts']['car']==2 and cache['split']=='train'
