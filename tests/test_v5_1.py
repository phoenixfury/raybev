"""Circular facing modes, safe ignore targets, and strict v5 -> v5.1 migration."""
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from raybev import ModelConfig,RayBEVLite
from raybev.config import DEFAULT_CALIBRATION,CLASSES
from raybev.detection import decode_heading,facing_vector_loss,make_targets,detection_loss,metric_to_cell
from raybev.diagnostics import center_statistics
from raybev.rosa_dataset import parse_box_annotations
from raybev.training import initialize_recovery,calibration_sha256


def config():
    torch.set_num_threads(2)
    return ModelConfig(facing_loss='circular_bins',heading_bins=8).downscale(5)


def test_config_keeps_legacy_defaults_and_requires_eight_circular_channels():
    assert ModelConfig().facing_loss=='unit_vector' and ModelConfig().heading_bins==2
    assert config().validate()
    for cfg in (replace(config(),heading_bins=2),replace(config(),heading_encoding='binned_residual'),
                replace(config(),facing_loss='unit_vector')):
        with pytest.raises(ValueError,match='heading bins'):cfg.validate()


def test_perfect_axis_and_facing_bins_recover_full_circle_including_seam():
    cfg=config();yaw=torch.linspace(-math.pi,math.pi,1001)
    axis=torch.stack(((2*yaw).sin(),(2*yaw).cos()),-1)
    bins=(torch.remainder(yaw,2*math.pi)*8/(2*math.pi)).round().long()%8
    logits=torch.full((len(yaw),8),-12.).scatter_(1,bins[:,None],12.)
    actual,confidence=decode_heading(axis,logits,cfg)
    error=torch.atan2((actual-yaw).sin(),(actual-yaw).cos()).abs()
    assert error.max()<1e-5 and confidence.min()>.999


def test_circular_soft_targets_are_continuous_and_gradients_finite():
    cfg=config();yaw=torch.tensor([-1e-6,1e-6,-math.pi+1e-6,math.pi-1e-6])
    target=torch.stack((yaw.sin(),yaw.cos()),-1)
    logits=torch.arange(8.).expand(4,-1).clone().requires_grad_()
    loss=facing_vector_loss(logits,target,cfg)
    torch.testing.assert_close(loss[0],loss[1],atol=2e-5,rtol=0)
    torch.testing.assert_close(loss[2],loss[3],atol=2e-5,rtol=0)
    loss.sum().backward();assert torch.isfinite(logits.grad).all()
    yaw=torch.tensor([.3*(2*math.pi/8)]);target=torch.stack((yaw.sin(),yaw.cos()),-1)
    logp=logits[:1].detach().log_softmax(-1)
    torch.testing.assert_close(facing_vector_loss(logits[:1],target,cfg),-.7*logp[:,0]-.3*logp[:,1])


def test_opposite_modes_do_not_average_to_zero_facing_vector():
    cfg=config();axis=torch.tensor([[0.,1.]])
    logits=torch.tensor([[8.,-12.,-12.,-12.,8.,-12.,-12.,-12.]])
    angle,confidence=decode_heading(axis,logits,cfg)
    assert torch.isfinite(angle).all() and abs(float(confidence)-.5)<1e-5
    logits[:,4]+=1
    angle,_=decode_heading(axis,logits,cfg)
    assert abs(abs(float(angle))-math.pi)<1e-5


def test_heading_confidence_describes_selected_mode_not_opposite_hemisphere():
    cfg=config();axis=torch.tensor([[0.,1.]])
    probabilities=torch.tensor([[.2,.2,1e-6,1e-6,.4,1e-6,1e-6,.2]])
    probabilities/=probabilities.sum(-1,keepdim=True)
    angle,confidence=decode_heading(axis,probabilities.log(),cfg)
    assert abs(abs(float(angle))-math.pi)<1e-5 and .39<float(confidence)<.41


def output(cfg):
    return [torch.zeros(1,c,cfg.bev_height,cfg.bev_width,requires_grad=True) for c in (6,8,1,8,1)]


def test_ignore_regions_are_neither_foreground_nor_background_and_keep_known_positive():
    cfg=config();known=[4.,2.,1.5,7.,2.5,3.,.1,3.];unknown=[-6.,-4.,1.5,8.,3.,3.,.3,-1.]
    targets=make_targets([torch.tensor([known,unknown])],cfg,return_class_ids=True)
    assert len(targets)==5 and int(targets[2].sum())==1
    r,c=metric_to_cell(unknown[0],unknown[1],cfg);r,c=round(r),round(c)
    assert not targets[4][0,0,r,c]
    pred=output(cfg);loss,_=detection_loss(pred,targets,cfg=cfg);loss.backward()
    assert pred[4].grad[0,0,r,c]==0 and pred[4].grad[0,0,0,0]>0
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in pred)
    assert center_statistics(pred,targets,cfg)[:,0].sum()==1
    overlap=make_targets([torch.tensor([known,[*known[:7],-1.]])],cfg,return_class_ids=True)
    pred=output(cfg);loss,_=detection_loss(pred,overlap,cfg=cfg);loss.backward()
    r,c=metric_to_cell(known[0],known[1],cfg)
    assert pred[4].grad[0,0,round(r),round(c)]<0  # positive still supervised


def test_ignore_footprint_can_intersect_roi_when_center_is_outside():
    cfg=config();targets=make_targets([torch.tensor([[49.,0.,1.,8.,3.,3.,0.,-1.]])],cfg,return_class_ids=True)
    assert not targets[4][0,0,0].all() and not targets[2].any()
    with pytest.raises(ValueError,match='return_class_ids'):make_targets([torch.tensor([[0.,0.,1.,8.,3.,3.,0.,-1.]])],cfg)


def test_parse_ignored_annotations_preserves_geometry_but_forbids_validation_ignores():
    sample=dict(token='x',metainfo=dict(temporal_split='train'),annos=dict(
        gt_names=['Truck','Bus'],gt_boxes=np.array([[1.,0.,1.,7.,3.,3.,0.],[2.,0.,1.,12.,3.,3.,0.]]),gt_ignore=[False,True]))
    kept=parse_box_annotations(sample,'BASELINK');ignored=parse_box_annotations(sample,'BASELINK',only_ignored=True)
    assert kept.shape==ignored.shape==(1,8) and kept[0,7]==3 and ignored[0,7]==5
    sample['metainfo']['temporal_split']='val'
    with pytest.raises(ValueError,match='TRAIN-only'):parse_box_annotations(sample,'BASELINK')


def test_recovery_resets_only_circular_projection_and_changed_size_priors(tmp_path):
    cfg=config();old_cfg=replace(cfg,facing_loss='unit_vector',heading_bins=2)
    old=RayBEVLite(old_cfg,DEFAULT_CALIBRATION);new=RayBEVLite(cfg,DEFAULT_CALIBRATION)
    path=tmp_path/'v5.pt';h=calibration_sha256(DEFAULT_CALIBRATION)
    torch.save(dict(cfg=vars(old_cfg),state_dict=old.state_dict(),classes=CLASSES,box_frame='CLIDAR',epoch=3,
                    calibration_sha256=h,lidar_calibration_sha256=h),path)
    args=SimpleNamespace(init_from=str(path),resume=None,box_frame='CLIDAR',z_reference='center',
                         calibration=DEFAULT_CALIBRATION,lidar_calibration=DEFAULT_CALIBRATION)
    report=initialize_recovery(new,cfg,args)
    assert set(report['reset_parameters'])=={'bev_detector.heading.1.weight','bev_detector.heading.1.bias'}
    for key,value in old.state_dict().items():
        if key not in report['reset_parameters']:torch.testing.assert_close(new.state_dict()[key],value,atol=0,rtol=0)
    assert new.bev_detector.heading[1].out_channels==8


def test_ignore_rules_cannot_remove_validation_gt():
    path=Path(__file__).parents[1]/'scripts/prepare_label_revision.py'
    spec=importlib.util.spec_from_file_location('review_v51',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    doc=dict(data_list=[dict(token='x',metainfo=dict(sequence_id='bag'),annos=dict(gt_ids=[7]))])
    rules=[dict(sequence_id='bag',track_id='7',expected_train_boxes=1)]
    with pytest.raises(ValueError,match='validation'):module.ignore_uncertain_tracks(doc,rules,'val')
    records=module.ignore_uncertain_tracks(doc,rules,'train')
    assert len(records)==1 and doc['data_list'][0]['annos']['gt_ignore'][0]
