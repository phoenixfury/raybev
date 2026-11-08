"""Difficulty weighting, exact target ownership, and score compatibility."""
from dataclasses import replace
import math

import pytest
import torch

from raybev.config import ModelConfig
from raybev.detection import make_targets,detection_loss,decode_detections,detection_scores


def small_config():
    return replace(ModelConfig(),bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,
                   bev_y_max=4.,bev_resolution=1.).validate()


def outputs(cfg):
    return [torch.zeros(1,c,cfg.bev_height,cfg.bev_width,requires_grad=True) for c in (6,8,1)]


@pytest.mark.parametrize('class_id,weight',[(0,1.),(1,2.),(2,4.),(3,4.),(4,4.),(5,4.)])
def test_hard_class_foreground_box_and_quality_gradients_increase(class_id,weight):
    cfg=small_config()
    gt=torch.tensor([[1.5,1.5,1.,4.,2.,2.,0.,class_id]])
    targets=make_targets([gt],cfg,return_class_ids=True)
    baseline=replace(cfg,class_positive_weights=(1.,)*6,class_regression_weights=(1.,)*6)
    weighted=outputs(cfg);unweighted=outputs(cfg)
    loss,_=detection_loss(weighted,targets,cfg=cfg)
    base_loss,_=detection_loss(unweighted,targets,cfg=baseline)
    loss.backward();base_loss.backward()
    for branch,channel in [(0,class_id),(1,3),(2,0)]:
        expected=unweighted[branch].grad[0,channel,2,2]*weight
        assert expected.abs()>0
        torch.testing.assert_close(weighted[branch].grad[0,channel,2,2],expected)
    # Dense negatives must NOT be upweighted along with a difficult class.
    torch.testing.assert_close(weighted[0].grad[0,class_id,0,0],unweighted[0].grad[0,class_id,0,0])


def test_regression_weight_uses_closest_box_owner_not_heatmap_argmax():
    cfg=small_config()
    # Both heatmap peaks occupy (2,2); Bus is closer to the cell midpoint.
    gt=torch.tensor([[1.2,1.5,1.,4.,2.,2.,0.,0.],
                     [1.5,1.5,1.,10.,2.8,3.5,0.,5.]])
    targets=make_targets([gt],cfg,return_class_ids=True)
    heat,reg,mask,owners=targets
    assert heat[0,0,2,2]==heat[0,5,2,2]==1
    assert owners[0,0,2,2]==5 and mask.sum()==1
    assert reg[0,3,2,2]==pytest.approx(math.log(10))
    assert (owners[~mask]==-1).all()
    pred=outputs(cfg);plain=outputs(cfg)
    loss,_=detection_loss(pred,targets,cfg=cfg)
    base_loss,_=detection_loss(plain,targets,cfg=replace(cfg,class_regression_weights=(1.,)*6))
    loss.backward();base_loss.backward()
    torch.testing.assert_close(pred[1].grad[0,3,2,2],4*plain[1].grad[0,3,2,2])
    with pytest.raises(ValueError,match='return_class_ids=True'):
        detection_loss(outputs(cfg),targets[:3],cfg=cfg)


@pytest.mark.parametrize('with_owners',[False,True])
def test_empty_batches_keep_all_heads_in_the_graph(with_owners):
    cfg=small_config();pred=outputs(cfg)
    targets=make_targets([torch.empty(0,8)],cfg,return_class_ids=with_owners)
    loss,parts=detection_loss(pred,targets,cfg=cfg)
    loss.backward()
    assert torch.isfinite(loss) and parts['regression']==parts['quality']==0
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in pred)


def score_outputs(cfg):
    h,w=cfg.bev_height,cfg.bev_width
    heat=torch.full((1,6,h,w),-20.)
    reg=torch.zeros(1,8,h,w);reg[:,7]=1
    quality=torch.zeros(1,1,h,w)
    return heat,reg,quality


def logit(value):
    return math.log(value/(1-value))


def test_sqrt_quality_penalty_is_bounded_and_not_a_class_score_boost():
    cfg=small_config();pred=score_outputs(cfg)
    pred[0][0,5,2,2]=logit(.4);pred[2][0,0,2,2]=logit(.25)
    decoded=decode_detections(pred,cfg,score_threshold=.15)[0]
    assert len(decoded)==1 and decoded[0,8]==pytest.approx(.2)
    assert decoded[0,8]<=.4
    old=replace(cfg,quality_score_power=1.,score_peak_mode='heatmap')
    assert len(decode_detections(pred,old,score_threshold=.15)[0])==0
    # Loss weights never multiply inference confidence.
    unweighted=replace(cfg,class_positive_weights=(1.,)*6,class_regression_weights=(1.,)*6)
    torch.testing.assert_close(detection_scores(pred,cfg),detection_scores(pred,unweighted))


def test_peaks_are_selected_after_quality_fusion():
    cfg=small_config();pred=score_outputs(cfg)
    pred[0][0,3,2,2]=logit(.9);pred[2][0,0,2,2]=logit(.01)
    pred[0][0,3,2,3]=logit(.8);pred[2][0,0,2,3]=logit(.81)
    decoded=decode_detections(pred,cfg,score_threshold=.1)[0]
    assert len(decoded)==1
    assert decoded[0,1]==pytest.approx(.5) and decoded[0,8]==pytest.approx(.72)
    stale_peaks=replace(cfg,score_peak_mode='heatmap')
    assert len(decode_detections(pred,stale_peaks,score_threshold=.1)[0])==0


@pytest.mark.parametrize('branch',[(0,0),(1,3),(2,0)])
def test_invalid_candidate_cannot_consume_topk(branch):
    cfg=small_config();pred=score_outputs(cfg)
    pred[0][0,0,2,2]=logit(.99)
    pred[0][0,0,6,6]=logit(.6)
    pred[branch[0]][0,branch[1],2,2]=float('nan')
    decoded=decode_detections(pred,cfg,score_threshold=.1,topk=1)[0]
    assert len(decoded)==1 and torch.isfinite(decoded).all()
    assert decoded[0,0]==pytest.approx(-2.5)


def test_score_fusion_and_output_decode_use_fp32_with_half_inputs():
    cfg=small_config();pred=tuple(t.half() for t in score_outputs(cfg))
    pred[0][0,0,2,2]=0
    scores=detection_scores(pred,cfg)
    assert scores.dtype==torch.float32
    assert scores[0,0,2,2]==pytest.approx(.5*math.sqrt(.5),abs=1e-7)
    decoded=decode_detections(pred,cfg,score_threshold=.3)[0]
    assert decoded.dtype==torch.float32 and torch.isfinite(decoded).all()


def test_zero_threshold_does_not_select_invalid_zeroed_scores():
    cfg=small_config();pred=score_outputs(cfg)
    pred[1].fill_(float('nan'))
    assert len(decode_detections(pred,cfg,score_threshold=0)[0])==0


def test_quality_can_be_disabled_without_using_invalid_quality_logits():
    cfg=replace(small_config(),quality_score_power=0.)
    pred=score_outputs(cfg);pred[0][0,0,2,2]=logit(.4)
    pred[2].fill_(float('nan'))
    with_quality=detection_scores(pred,cfg)
    torch.testing.assert_close(with_quality,detection_scores(pred[:2],cfg))
    assert with_quality[0,0,2,2]==pytest.approx(.4)


def test_checkpoint_score_rule_and_loss_weights_round_trip_with_legacy_defaults():
    cfg=small_config()
    assert ModelConfig.from_checkpoint({'cfg':vars(cfg)})==cfg
    original=dict(vars(cfg),class_positive_weights=(1.,1.5,3.,2.,3.,2.))
    for key in ('class_regression_weights','quality_score_power','score_peak_mode'):original.pop(key)
    legacy=ModelConfig.from_checkpoint({'cfg':original})
    assert legacy.class_regression_weights==(1.,)*6
    assert legacy.quality_score_power==1. and legacy.score_peak_mode=='heatmap'
    assert legacy.class_positive_weights==(1.,1.5,3.,2.,3.,2.)
    pred=score_outputs(cfg);pred[0][0,5,2,2]=logit(.4);pred[2][0,0,2,2]=logit(.25)
    assert detection_scores(pred,legacy)[0,5,2,2]==pytest.approx(.1)


@pytest.mark.parametrize('changes',[
    {'class_regression_weights':(1.,)*5},
    {'class_regression_weights':(1.,1.,1.,0.,1.,1.)},
    {'class_positive_weights':(1.,1.,1.,float('nan'),1.,1.)},
    {'quality_score_power':-1.},{'quality_score_power':float('nan')},
    {'score_peak_mode':'invalid'}])
def test_invalid_loss_and_score_controls_are_rejected(changes):
    with pytest.raises(ValueError):replace(small_config(),**changes).validate()
