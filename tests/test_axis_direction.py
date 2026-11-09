"""Continuous axes, facing decisions, backbone compatibility and heading reports."""
import math
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from torchvision.models import resnet18
from raybev.config import ModelConfig
from raybev.model import ResNet18Encoder,RayBEVLite
from raybev.detection import make_targets,detection_loss,decode_heading,decode_detections
from raybev.metrics import heading_diagnostics
from raybev.training import heading_selection_score

CAL=Path(__file__).parents[1]/'configs/calibration_baselink.json'


@pytest.mark.parametrize('degrees',[-180,-179,-136,-134,-91,-90,-89,-46,-44,0,44,46,89,90,91,134,136,179,180])
def test_axis_facing_roundtrip_without_quadrant_or_canonical_axis_jumps(degrees):
    cfg=ModelConfig();yaw=torch.tensor([math.radians(degrees)])
    axis=torch.stack(((2*yaw).sin(),(2*yaw).cos()),dim=1)
    facing=4*torch.stack((yaw.sin(),yaw.cos()),dim=1)
    angle,conf=decode_heading(axis,facing,cfg)
    error=torch.atan2((angle-yaw).sin(),(angle-yaw).cos()).abs()
    assert error.max()<1e-5 and conf.min()>.99


def test_axis_target_is_continuous_across_old_bin_boundary_and_reversals():
    yaw=torch.deg2rad(torch.tensor([44.,46.,224.]))
    target=torch.stack(((2*yaw).sin(),(2*yaw).cos()),1)
    assert torch.linalg.vector_norm(target[0]-target[1])<.08
    torch.testing.assert_close(target[0],target[2],atol=1e-6,rtol=1e-6)


def test_facing_classifier_corrects_exact_180_reversal_without_changing_axis():
    cfg=replace(ModelConfig(),bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.,classification_mode='heatmap')
    gt=torch.tensor([[1.5,1.5,1.,10.,3.,3.5,math.pi,5.]])
    outputs=[torch.zeros(1,c,8,8) for c in (6,8,1,2)]
    outputs[1][:,7]=1.;outputs[3][:,1]=4.
    outputs=[x.requires_grad_() for x in outputs]
    loss,parts=detection_loss(outputs,make_targets([gt],cfg,return_class_ids=True),cfg=cfg)
    loss.backward()
    assert outputs[3].grad[0,1,2,2]>1.
    assert outputs[1].grad[0,7,2,2].abs()<1e-5
    assert all(torch.isfinite(v.grad).all() for v in outputs)


def test_axis_heading_size_decode_contract_keeps_nine_public_box_columns():
    cfg=replace(ModelConfig(),bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.,classification_mode='heatmap')
    gt=torch.tensor([[1.5,1.5,1.,10.,3.,3.5,2.4,5.]])
    heat,reg,mask,_=make_targets([gt],cfg,return_class_ids=True)
    yaw=gt[0,6];reg[0,6:8,2,2]=torch.stack(((2*yaw).sin(),(2*yaw).cos()))
    facing=torch.zeros(1,2,8,8);facing[0,:,2,2]=4*torch.stack((yaw.sin(),yaw.cos()))
    logits=torch.full_like(heat,-20.);logits[heat==1]=20.
    det=decode_detections((logits,reg,torch.full((1,1,8,8),20.),facing),cfg)[0]
    assert det.shape==(1,9)
    torch.testing.assert_close(det[0,:8],gt[0],atol=1e-5,rtol=1e-5)


def test_legacy_mobile_checkpoint_preserves_old_backbone_and_four_bin_semantics():
    cfg=ModelConfig.from_checkpoint({'cfg':dict(architecture_version=2,heading_bins=4)})
    assert cfg.backbone=='mobilenet_v2' and cfg.heading_encoding=='binned_residual'
    model=RayBEVLite(cfg.downscale(5),CAL)
    assert not isinstance(model.image_encoder,ResNet18Encoder)
    assert model.bev_detector.heading[-1].out_channels==4


def test_resnet18_offline_weights_and_pyramid_shapes(tmp_path):
    cfg=ModelConfig().downscale(5);net=resnet18(weights=None)
    with torch.no_grad():net.conv1.weight.fill_(.123)
    path=tmp_path/'resnet18.pt';torch.save(net.state_dict(),path)
    enc=ResNet18Encoder(cfg,pretrained=True,backbone_weights=path).eval()
    torch.testing.assert_close(enc.stage8[0].weight,net.conv1.weight)
    with torch.no_grad():f,q=enc(torch.rand(1,4,3,128,160),torch.zeros(4,8,16,20))
    assert f.shape==(1,4,32,16,20) and q.shape==(1,4,4,16,20)


def test_heading_reports_score_distance_sequence_and_null_missing_matches():
    gt=[np.array([[5,0,1,4,2,2,0,0.],[30,0,1,4,2,2,0,0.]])]
    pred=[np.array([[5,0,1,4,2,2,0,0,.9],[30,0,1,4,2,2,math.pi,0,.05]])]
    report=heading_diagnostics(pred,gt,['bag'])
    assert report['by_score']['0.01']['car']['AOE_deg']==pytest.approx(90)
    assert report['by_score']['0.3']['car']['AOE_deg']==pytest.approx(0)
    assert report['by_score']['0.3']['car']['recall']==.5
    assert report['by_score']['0.3']['bus']['AOE'] is None
    assert report['by_distance']['0.01']['car']['20_35m']['flip_rate']==1
    assert 'bag' in report['by_sequence']
    assert math.isinf(heading_selection_score(report))  # two examples cannot win a checkpoint
