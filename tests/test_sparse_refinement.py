"""V6.1 heading recovery and metre-scale refinement regression tests."""
from dataclasses import replace
import importlib
import math
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from raybev.config import DEFAULT_CALIBRATION
from raybev.detection import facing_vector_loss
from raybev.sparse_config import SparseConfig
from raybev.sparse_loss import SparseCriterion
from raybev.sparse_model import (OpenCVProjector, SparseStage, YOLOROSALite,
    box_points, class_balanced_dn_indices)


def config(version=4):
    torch.set_num_threads(2)
    return SparseConfig(architecture_version=version, image_width=160,
        image_height=128, channels=32, ffn_channels=64, queries=12,
        stages=4, max_detections=12).validate()


def inputs(cfg):
    query=torch.zeros(1,2,cfg.channels)
    boxes=torch.tensor([[[10.,2.,1.,8.,3.,3.,0.],
                         [40.,2.,1.,8.,3.,3.,0.]]])
    features=tuple(torch.zeros(1,4,cfg.channels,cfg.image_height//stride,
        cfg.image_width//stride) for stride in (8,16,32))
    return query,boxes,features,OpenCVProjector(cfg,DEFAULT_CALIBRATION)


def constant_head(head,value):
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.as_tensor(value,device=head.bias.device,dtype=head.bias.dtype))


def test_refined_config_keeps_legacy_v3_and_rejects_unknown_versions():
    refined=config()
    legacy=config(3)
    assert refined.is_refined and not legacy.is_refined
    for cfg in (refined,legacy):
        restored=SparseConfig.from_checkpoint(dict(architecture=cfg.architecture,
            cfg=cfg.to_dict()))
        assert restored==cfg
    with pytest.raises(ValueError,match='contract'):
        replace(refined,architecture_version=5).validate()


def test_raw_axis_has_gradient_at_exact_90_degree_physical_error_and_recovers():
    cfg=config();stage=SparseStage(cfg,stage_index=0).eval()
    args=inputs(cfg)
    for parameter in stage.parameters():parameter.requires_grad_(False)
    stage.axis.bias.requires_grad_(True)
    constant_head(stage.axis,[0.,1.])
    target=torch.tensor([0.,-1.])  # yaw pi/2, not a 180-degree facing flip
    _,output=stage(*args)
    first=F.smooth_l1_loss(output['axis'],target.expand_as(output['axis']))
    first.backward()
    assert torch.isfinite(stage.axis.bias.grad).all()
    assert stage.axis.bias.grad[1].abs()>.1
    optimizer=torch.optim.SGD([stage.axis.bias],lr=.3)
    for _ in range(30):
        optimizer.zero_grad()
        _,output=stage(*args)
        loss=F.smooth_l1_loss(output['axis'],target.expand_as(output['axis']))
        loss.backward();optimizer.step()
    with torch.no_grad():
        _,output=stage(*args)
        final=F.smooth_l1_loss(output['axis'],target.expand_as(output['axis']))
        axis_yaw=.5*torch.atan2(output['axis'][...,0],output['axis'][...,1])
        error=.5*torch.atan2((2*(axis_yaw-math.pi/2)).sin(),
            (2*(axis_yaw-math.pi/2)).cos()).abs()
    assert final<first.detach()*.02
    assert error.max()<.02


def test_zero_raw_axis_falls_back_before_atan2_and_backpropagates_finitely():
    cfg=config();stage=SparseStage(cfg,stage_index=0).eval()
    query,boxes,features,projector=inputs(cfg)
    boxes[...,6]=.4
    constant_head(stage.axis,[0.,0.])
    _,output=stage(query,boxes,features,projector)
    assert torch.count_nonzero(output['axis'])==0
    error=torch.atan2((output['boxes'][...,6]-.4).sin(),
        (output['boxes'][...,6]-.4).cos()).abs()
    assert error.max()<1e-5
    loss=output['boxes'].square().mean()+output['axis'].sum()
    loss.backward()
    assert stage.axis.bias.grad is not None
    assert torch.isfinite(stage.axis.bias.grad).all()
    for parameter in stage.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_legacy_v3_keeps_unit_axis_and_spherical_center_decoding():
    cfg=config(3);stage=SparseStage(cfg,stage_index=3).eval()
    query,boxes,features,projector=inputs(cfg)
    boxes[...,6]=.4
    center_delta=torch.tensor([.02,-.01,.03])
    constant_head(stage.center,center_delta)
    constant_head(stage.axis,[1.2,1.6])
    _,output=stage(query,boxes,features,projector)
    old=2*boxes[...,6]
    expected_axis=torch.stack((old.sin()*.8+old.cos()*.6,
        old.cos()*.8-old.sin()*.6),-1)
    torch.testing.assert_close(output['axis'],expected_axis,atol=1e-6,rtol=1e-6)
    direction=F.normalize(boxes[...,:3],dim=-1)
    tangent=F.normalize(torch.stack((-direction[...,1],direction[...,0],
        torch.zeros_like(direction[...,0])),-1),dim=-1)
    vertical=torch.cross(direction,tangent,dim=-1)
    expected_direction=F.normalize(direction+.5*center_delta[0].tanh()*tangent+
        .5*center_delta[1].tanh()*vertical,dim=-1)
    expected_distance=(boxes[...,:3].norm(dim=-1,keepdim=True).log()+
        .7*center_delta[2].tanh()).exp()
    torch.testing.assert_close(output['boxes'][...,:3],expected_direction*
        expected_distance,atol=1e-5,rtol=1e-6)
    expected_yaw=boxes[...,6]+.5*math.atan2(.6,.8)
    torch.testing.assert_close(output['boxes'][...,6],expected_yaw,atol=1e-6,rtol=1e-6)
    refined=SparseStage(config(),stage_index=3).eval()
    constant_head(refined.axis,[1.2,1.6])
    _,raw=refined(query,boxes,features,projector)
    torch.testing.assert_close(raw['axis'],expected_axis*2,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('stage_index,limits',[(2,[2.,2.,1.]),(3,[1.,1.,.5])])
def test_final_metric_residual_is_range_independent_and_has_bounded_limits(stage_index,limits):
    cfg=config();stage=SparseStage(cfg,stage_index=stage_index).eval()
    query,boxes,features,projector=inputs(cfg)
    delta=torch.tensor([.01,-.02,.03])
    constant_head(stage.center,delta)
    _,output=stage(query,boxes,features,projector)
    actual=output['boxes'][...,:3]-boxes[...,:3]
    expected=delta.tanh()*torch.tensor(limits)
    torch.testing.assert_close(actual,expected.expand_as(actual),atol=3e-6,rtol=1e-4)
    torch.testing.assert_close(actual[:,0],actual[:,1],atol=3e-6,rtol=1e-4)
    constant_head(stage.center,[100.,-100.,100.])
    _,output=stage(query,boxes,features,projector)
    actual=output['boxes'][...,:3]-boxes[...,:3]
    expected=torch.tensor(limits)*torch.tensor([1.,-1.,1.])
    torch.testing.assert_close(actual,expected.expand_as(actual),atol=3e-6,rtol=1e-6)


def test_final_metric_center_can_fit_submeter_offsets_near_and_far():
    cfg=config();stage=SparseStage(cfg,stage_index=3).eval()
    query,boxes,features,projector=inputs(cfg)
    for parameter in stage.parameters():parameter.requires_grad_(False)
    stage.center.bias.requires_grad_(True)
    target=boxes[...,:3]+torch.tensor([.17,-.23,.08])
    optimizer=torch.optim.Adam([stage.center.bias],lr=.05)
    first=None
    for _ in range(90):
        optimizer.zero_grad()
        _,output=stage(query,boxes,features,projector)
        loss=F.smooth_l1_loss(output['boxes'][...,:3],target,beta=.2)
        if first is None:first=loss.detach()
        loss.backward();optimizer.step()
    with torch.no_grad():
        _,output=stage(query,boxes,features,projector)
        error=(output['boxes'][...,:3]-target).norm(dim=-1)
    assert error.max()<.01
    assert loss.detach()<first*.01


def test_refined_facing_can_recover_an_exact_180_degree_flip():
    cfg=config();stage=SparseStage(cfg,stage_index=3).eval()
    args=inputs(cfg)
    for parameter in stage.parameters():parameter.requires_grad_(False)
    bias=stage.facing[-1].bias
    bias.requires_grad_(True)
    optimizer=torch.optim.SGD([bias],lr=.5)
    target=torch.tensor([0.,-1.])
    for _ in range(40):
        optimizer.zero_grad();_,output=stage(*args)
        loss=facing_vector_loss(output['facing'],target.expand(1,2,2),cfg).mean()
        loss.backward();optimizer.step()
    with torch.no_grad():
        _,output=stage(*args)
        error=torch.atan2((output['boxes'][...,6]-math.pi).sin(),
            (output['boxes'][...,6]-math.pi).cos()).abs()
    assert error.max()<1e-5


@pytest.mark.parametrize('limit',[6,8,100])
def test_class_balanced_denoising_includes_rare_classes_and_never_ignored_rows(limit):
    cfg=config()
    labels=[0]*10+[1,2,3,4,5,-1]
    gt=torch.zeros(len(labels),8);gt[:,3:6]=1;gt[:,7]=torch.tensor(labels)
    selected=class_balanced_dn_indices(gt,limit,cfg.class_positive_weights)
    assert len(selected)==min(limit,len(labels)-1)
    assert len(selected.unique())==len(selected)
    assert set(gt[selected,7].tolist())==set(range(6))
    assert not len(class_balanced_dn_indices(gt[-1:],limit,cfg.class_positive_weights))


def test_thirteen_support_points_scale_rotate_and_differentiate_learned_offsets():
    boxes=torch.tensor([[[10.,20.,1.,8.,4.,2.,math.pi/2]]])
    offsets=torch.tensor([[[[.25,.2,.1],[-.1,.3,-.2],[.4,-.2,.3],
        [.1,.1,-.1],[-.2,-.3,.2],[.3,-.1,.4]]]],requires_grad=True)
    points=box_points(boxes,offsets)
    assert points.shape==(1,1,13,3)
    torch.testing.assert_close(points[...,:7,:],box_points(boxes))
    x,y,z=offsets.unbind(-1)
    expected=torch.stack((10-4*y,20+8*x,1+2*z),-1)
    torch.testing.assert_close(points[...,7:,:],expected,atol=2e-6,rtol=1e-6)
    points[...,7:,:].square().sum().backward()
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()
    assert (offsets.grad.abs().sum(-1)>0).all()


@pytest.mark.parametrize('kind',['foreground','empty','ignored'])
def test_refined_model_all_parameters_have_finite_backward_for_every_target_kind(kind):
    torch.manual_seed(491)
    cfg=config();model=YOLOROSALite(cfg).train()
    target=torch.tensor([[10.,2.,1.,8.,3.,3.,.4,3.]])
    if kind=='empty':target=torch.empty(0,8)
    elif kind=='ignored':target[:,-1]=-1
    output=model(torch.randn(1,4,3,128,160),[target])
    loss,parts=SparseCriterion(cfg)(output,[target])
    assert torch.isfinite(loss)
    if kind!='foreground':
        assert parts['center']==0 and parts['facing']==0 and parts['denoising']==0
    loss.backward()
    missing=[name for name,parameter in model.named_parameters() if parameter.grad is None]
    nonfinite=[name for name,parameter in model.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()]
    assert not missing and not nonfinite


def test_refined_denoising_queries_cannot_change_matching_queries():
    torch.manual_seed(492)
    cfg=replace(config(),dn_queries=4);model=YOLOROSALite(cfg).train()
    target=torch.tensor([[10.,2.,1.,8.,3.,3.,.4,3.],
        [20.,-4.,1.,12.,3.,4.,-2.,5.]])
    features=model.image_encoder(torch.randn(1,4,3,128,160))
    plain=model.forward_features(features)
    denoised=model.forward_features(features,[target])
    for first,second in zip(plain['stages'],denoised['stages']):
        for key in ('boxes','axis','facing','class_logits','quality_logits'):
            torch.testing.assert_close(first[key],second[key][:,:cfg.queries],
                atol=2e-5,rtol=1e-5)


def test_refined_quality_target_depends_on_center_not_predicted_size():
    gt=torch.tensor([[10.,2.,1.,8.,3.,3.,.4,3.]])
    def quality(version,scale):
        boxes=gt[:,:7].clone();boxes[:,0]+=.3;boxes[:,3:6]*=scale
        stage=dict(boxes=boxes,axis=torch.tensor([[0.,1.]]),
            facing=torch.zeros(1,8),class_logits=torch.zeros(1,6),
            quality_logits=torch.full((1,),2.))
        return SparseCriterion(config(version))._regression(stage,
            torch.tensor([0]),gt,1.)['quality']
    torch.testing.assert_close(quality(4,1.),quality(4,.4))
    assert not torch.isclose(quality(3,1.),quality(3,.4))


def test_refined_class_and_size_contexts_are_independent_and_sizes_ignore_class_winner():
    torch.manual_seed(493)
    cfg=config();stage=SparseStage(cfg,stage_index=3).eval();args=inputs(cfg)
    with torch.no_grad():stage.size.weight.fill_(.02)
    _,baseline=stage(*args)
    with torch.no_grad():stage.class_context[-1].bias.add_(1.)
    _,semantic=stage(*args)
    torch.testing.assert_close(baseline['boxes'],semantic['boxes'])
    assert not torch.allclose(baseline['class_logits'],semantic['class_logits'])
    with torch.no_grad():stage.size_context[-1].bias.add_(1.)
    _,geometric=stage(*args)
    assert not torch.allclose(semantic['boxes'][...,3:6],geometric['boxes'][...,3:6])
    torch.testing.assert_close(semantic['class_logits'],geometric['class_logits'])
    for winner in (0,3,5):
        bias=[-8.]*6;bias[winner]=8.
        constant_head(stage.classifier,bias)
        _,output=stage(*args)
        assert (output['class_logits'].argmax(-1)==winner).all()
        torch.testing.assert_close(output['boxes'],geometric['boxes'])


def test_refined_onnx_encoder_and_thirteen_point_head_have_cpu_parity(tmp_path,monkeypatch):
    pytest.importorskip('onnx');pytest.importorskip('onnxruntime')
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    export=importlib.import_module('export_yolo_rosa_sparse')
    model=YOLOROSALite(config()).eval();output=tmp_path/'onnx'
    metadata=export.export_model(model,output)
    assert metadata['support_points']==model.cfg.num_support_points==13
    assert metadata['cfg']['architecture_version']==4
    report=export.verify_export(model,output)
    assert set(report)=={'encoder_four_camera','encoder_single_camera','sparse_head'}
    assert report['sparse_head']['output_shapes'][0]==[1,12,7]
    assert not model.training and not model.image_encoder.training
