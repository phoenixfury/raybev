"""Objective gradients, ignored-region protection, and real medium feature taps."""
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from raybev.config import DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_diagnostics import heavy_confusion_diagnostics
from raybev.sparse_loss import SparseCriterion,background_quality_loss,matched_class_loss
from raybev.sparse_model import YOLOROSALite,YOLOPyramid
from raybev.yolo26 import build_backbone


def config(**changes):
    torch.set_num_threads(2)
    return replace(SparseConfig(architecture_version=5,image_width=160,image_height=128,
        channels=32,ffn_channels=64,queries=12,max_detections=12,stages=2,
        stride4=True,observation_conditioning=True,proposal_queries=4),**changes).validate()


def test_old_sparse_config_defaults_and_medium_contract():
    cfg=config();old=cfg.to_dict()
    for key in ('matched_class_weight','quality_background_weight','heading_hard_weight',
                'facing_binary_weight','facing_label_smoothing'):
        old.pop(key)
    assert SparseConfig.from_checkpoint(dict(architecture=cfg.architecture,cfg=old))==cfg
    assert config(backbone='yolo26m').backbone=='yolo26m'
    with pytest.raises(ValueError):config(matched_class_weight=-1)
    with pytest.raises(ValueError):config(facing_label_smoothing=1)


@pytest.mark.parametrize('label',[3,5])
def test_matched_class_competition_suppresses_car_for_truck_and_bus(label):
    logits=torch.tensor([[4.,-2.,-2.,0.,0.,0.]],requires_grad=True)
    loss=matched_class_loss(logits,torch.tensor([label]),config(),1.)
    loss.backward()
    assert logits.grad[0,0]>0 and logits.grad[0,label]<0
    assert torch.isfinite(logits.grad).all()


def test_background_quality_excludes_matched_ignored_and_near_object_duplicates():
    logits=torch.zeros(4,requires_grad=True)
    single=dict(boxes=torch.tensor([[5.,0.,0.],[5.5,0.,0.],[30.,0.,0.],[40.,0.,0.]]),quality_logits=logits)
    known=torch.tensor([[5.,0.,0.,4.,2.,2.,0.,0.]])
    loss=background_quality_loss(single,known,torch.tensor([0]),torch.tensor([True,True,False,True]))
    loss.backward()
    assert logits.grad[:3].eq(0).all() and logits.grad[3]>0
    zero=background_quality_loss(single,known,torch.tensor([0]),torch.zeros(4,dtype=torch.bool))
    assert zero==0 and torch.isfinite(zero)


def heading_stage(angle):
    logits=torch.full((1,8),-4.)
    logits[0,round(angle/(math.pi/4))%8]=4.
    return dict(boxes=torch.tensor([[5.,0.,1.,4.,2.,2.,angle]],requires_grad=True),
        axis=torch.tensor([[0.,1.]],requires_grad=True),facing=logits.requires_grad_(),
        class_logits=torch.zeros(1,6,requires_grad=True),quality_logits=torch.zeros(1,requires_grad=True))


def test_heading_hard_weight_increases_flip_penalty_and_has_correct_gradient():
    gt=torch.tensor([[5.,0.,1.,4.,2.,2.,0.,0.]])
    pi=torch.tensor([0]);baseline=SparseCriterion(config())
    criterion=SparseCriterion(config(heading_hard_weight=2.,facing_label_smoothing=.05))
    flipped=heading_stage(math.pi);correct=heading_stage(0.)
    hard=criterion._regression(flipped,pi,gt,1.)
    old=baseline._regression(flipped,pi,gt,1.)
    good=criterion._regression(correct,pi,gt,1.)
    assert hard['facing_binary']>old['facing_binary']>good['facing_binary']
    (hard['facing_binary']+hard['facing']).backward()
    assert flipped['facing'].grad[0,0]<0 and flipped['facing'].grad[0,4]>0
    assert torch.isfinite(flipped['facing'].grad).all()


@pytest.mark.parametrize('backbone',['yolo26n','yolo26m'])
def test_combined_objective_has_finite_gradients_and_strict_reload(backbone):
    cfg=config(backbone=backbone,matched_class_weight=1.,quality_background_weight=.25,
               heading_hard_weight=2.,facing_binary_weight=1.,facing_label_smoothing=.05)
    model=YOLOROSALite(cfg).train()
    truth=[torch.tensor([[5.,1.,1.,8.,2.5,3.,.1,3.],[-8.,2.,1.,11.,3.,3.5,3.,5.],
                         [3.,-5.,1.,.7,.7,1.8,1.,1.]]),torch.empty(0,8)]
    images=torch.randn(2,4,3,128,160)
    output=model(images,truth);loss,parts=SparseCriterion(cfg)(output,truth)
    assert torch.isfinite(loss) and parts['matched_class']>0 and parts['quality_background']>0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored=YOLOROSALite(SparseConfig.from_checkpoint(dict(architecture=cfg.architecture,cfg=cfg.to_dict())))
    restored.load_state_dict(model.state_dict(),strict=True)
    model.eval();restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(images)['stages'][-1]['boxes'],restored(images)['stages'][-1]['boxes'])


def test_heavy_confusion_reports_car_sized_bus_without_double_claiming_prediction():
    truth=[np.array([[5.,0.,1.,10.,3.,4.,0.,5.],[5.2,0.,1.,4.,2.,2.,0.,0.]])]
    pred=[np.array([[5.,0.,1.,4.,2.,2.,0.,0.,.9]])]
    audit=heavy_confusion_diagnostics(pred,truth)
    assert audit['confusion']['bus']['car']==1
    np.testing.assert_allclose(audit['dimensions']['bus->car']['median_predicted_over_gt_LWH'],[.4,2/3,.5])


def test_real_medium_pretraining_matches_all_four_official_taps():
    path=Path(__file__).resolve().parents[3]/'checkpoints/yolo/yolo26m.pt'
    if not path.is_file():pytest.skip('Local medium COCO checkpoint unavailable')
    state=torch.load(path,map_location='cpu',weights_only=False)
    official=(state.get('ema') or state['model']).float().eval()
    encoder=YOLOPyramid(config(backbone='yolo26m'),pretrained=True,backbone_weights=path).eval()
    actual=[]
    handles=[module.register_forward_hook(lambda module,args,out:actual.append(out.detach()))
             for module in (encoder.stage8[2],encoder.stage8[4],encoder.stage16,encoder.stage32)]
    rgb=torch.rand(4,3,128,160)
    with torch.no_grad():
        encoder.forward_single((rgb-encoder.mean)/encoder.std)
        x=rgb
        for i,module in enumerate(official.model[:10]):
            x=module(x)
            if i in (2,4,6,9):torch.testing.assert_close(actual.pop(0),x,atol=3e-5,rtol=3e-5)
    for handle in handles:handle.remove()
    assert encoder.backbone_profile['feature_channels']==[256,512,512,512]
    with pytest.raises(ValueError,match='YOLO26n'):build_backbone(backbone_weights=path)
