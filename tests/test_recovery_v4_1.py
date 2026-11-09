"""Fixed BEV normalization, class-independent size, and bounded heavy ranking."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from raybev.config import CLASSES,DEFAULT_CALIBRATION,ModelConfig
from raybev.model import BEVDetector,RayBEVLite,FixedStatsBatchNorm2d
from raybev.detection import (make_targets,detection_loss,decode_detections,decode_sizes,
                              heavy_vehicle_classification)
from raybev.training import fit_training_config,initialize_recovery,calibration_sha256


def config():
    return replace(ModelConfig().downscale(5),bev_x_min=-4.,bev_x_max=4.,
                   bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.,classification_mode='heatmap',
                   foreground_label_smoothing=0.,center_classification_weight=.1,
                   heavy_classification_weight=.2,size_encoding='shared_relative')


def boxes(ci=5,x=1.5):
    return torch.tensor([[x,1.5,1.5,11.,2.8,3.5,.2,ci]])


def test_bev_train_eval_and_batch_composition_are_identical_with_trainable_affines():
    torch.manual_seed(8);cfg=config();model=BEVDetector(cfg)
    x=torch.randn(1,cfg.bev_input_channels,8,8)
    buffers={k:v.clone() for k,v in model.named_buffers()}
    model.train();train=model(x)
    # A different/crowded scene cannot change the first example's prediction.
    batch=model(torch.cat((x,torch.randn_like(x)*4+7)))
    model.eval();evaluation=model(x)
    for a,b,c in zip(train,evaluation,batch):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        torch.testing.assert_close(a,c[:1],atol=2e-6,rtol=1e-5)
    for k,v in model.named_buffers():torch.testing.assert_close(v,buffers[k],atol=0,rtol=0)
    sum(t.sum() for t in train).backward()
    fixed=[m for m in model.modules() if isinstance(m,FixedStatsBatchNorm2d)]
    assert fixed and all(m.weight.requires_grad and m.weight.grad is not None for m in fixed)
    assert all(m.bias.grad is not None for m in fixed)


def test_size_is_physical_even_if_bus_is_misclassified_as_car():
    cfg=config();heat,reg,mask,_=make_targets([boxes()],cfg,return_class_ids=True)
    yaw=torch.tensor(.2);reg[0,6:8,2,2]=torch.stack(((2*yaw).sin(),(2*yaw).cos()))
    facing=torch.zeros(1,2,8,8);facing[0,:,2,2]=torch.stack((yaw.sin(),yaw.cos()))
    for ci in [0,3,5]:
        logits=torch.full_like(heat,-20.);logits[0,ci,2,2]=20.
        det=decode_detections((logits,reg,torch.ones(1,1,8,8)*20,facing),cfg)[0]
        assert int(det[0,7])==ci
        torch.testing.assert_close(det[0,3:6],boxes()[0,3:6],rtol=1e-5,atol=1e-5)


@pytest.mark.parametrize('ci,wrong',[(5,0),(3,0),(5,3),(3,5),(0,5),(4,3)])
def test_heavy_ranking_corrects_confusions_and_has_light_vehicle_negatives(ci,wrong):
    logits=torch.zeros(1,6);logits[0,wrong]=4.;logits.requires_grad_()
    loss=heavy_vehicle_classification(logits,torch.tensor([ci]),torch.ones(1),torch.tensor(1.))
    loss.backward()
    assert logits.grad[0,ci]<-.9 and logits.grad[0,wrong]>.9


def test_heavy_ranking_is_not_applied_to_pedestrians_or_empty_batches():
    for classes in [torch.tensor([1,2]),torch.empty(0,dtype=torch.long)]:
        logits=torch.zeros(len(classes),6,requires_grad=True)
        loss=heavy_vehicle_classification(logits,classes,torch.ones(len(classes)),torch.tensor(1.))
        loss.backward()
        assert loss==0 and torch.isfinite(logits.grad).all() and (logits.grad==0).all()


def test_fixed_mean_loss_and_gradients_match_microbatch_accumulation():
    cfg=replace(config(),mean_training_objects=3.)
    labels=[boxes(0),torch.cat((boxes(3),boxes(5,x=-1.5))),torch.empty(0,8)]
    torch.manual_seed(5)
    full=[torch.randn(3,c,8,8,requires_grad=True) for c in [6,8,1,2]]
    loss,_=detection_loss(full,make_targets(labels,cfg,return_class_ids=True),cfg=cfg)
    loss.backward();micro=[t.detach().clone().requires_grad_() for t in full]
    losses=[]
    for i,gt in enumerate(labels):
        part,_=detection_loss([p[i:i+1] for p in micro],make_targets([gt],cfg,return_class_ids=True),cfg=cfg)
        losses.append(part.detach());(part/len(labels)).backward()
    torch.testing.assert_close(loss.detach(),torch.stack(losses).mean())
    for a,b in zip(full,micro):torch.testing.assert_close(a.grad,b.grad,atol=2e-6,rtol=1e-5)


def test_shared_reference_and_mean_are_fitted_only_from_train_and_ignore_outside_boxes():
    cfg=config();gt=torch.cat((boxes(0),boxes(5,x=-1.5)));gt[0,3:6]=torch.tensor([4.,2.,1.5])
    fitted=fit_training_config(cfg,[gt,torch.empty(0,8),boxes(x=100.)])
    assert fitted.mean_training_objects==pytest.approx(2/3)
    assert fitted.size_reference==pytest.approx((7.5,2.4,2.5))
    target=make_targets([gt],fitted,return_class_ids=True)
    decoded=decode_sizes(target[1][0,3:6,2,2],torch.tensor(0),fitted)
    torch.testing.assert_close(decoded,gt[0,3:6])


def recovery_fixture(tmp_path):
    cfg=fit_training_config(config(),[boxes(0),boxes(5)])
    old=replace(cfg,bev_normalization='batchnorm',loss_normalization='batch_objects',
                heavy_classification_weight=0.,size_encoding='class_relative')
    model=RayBEVLite(old,DEFAULT_CALIBRATION)
    with torch.no_grad():model.bev_detector.size[-1].bias.fill_(.7)
    state=dict(cfg=vars(old),state_dict=model.state_dict(),classes=CLASSES,epoch=20,
               box_frame='CLIDAR',calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
               lidar_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
               backbone_initialization='RESNET18_IMAGENET1K_V1')
    path=tmp_path/'source.pt';torch.save(state,path)
    args=SimpleNamespace(init_from=str(path),resume=None,box_frame='CLIDAR',
                         calibration=DEFAULT_CALIBRATION,lidar_calibration=DEFAULT_CALIBRATION,z_reference='center')
    return cfg,model,state,path,args


def test_recovery_loads_exact_backbone_and_heading_but_resets_size_projection(tmp_path):
    cfg,source,state,path,args=recovery_fixture(tmp_path)
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION)
    report=initialize_recovery(model,cfg,args)
    assert report['epoch']==20 and len(report['reset_parameters'])==2
    for k,v in model.state_dict().items():
        if k in report['reset_parameters']:assert (v==0).all()
        else:torch.testing.assert_close(v,state['state_dict'][k],atol=0,rtol=0)
    assert model.image_encoder.initialization=='RESNET18_IMAGENET1K_V1'
    assert any(isinstance(m,FixedStatsBatchNorm2d) for m in model.bev_detector.modules())


@pytest.mark.parametrize('problem',['geometry','calibration','classes','resume'])
def test_recovery_rejects_incompatible_sources(tmp_path,problem):
    cfg,source,state,path,args=recovery_fixture(tmp_path)
    if problem=='geometry':state['cfg']=dict(state['cfg'],bev_resolution=.5)
    elif problem=='calibration':state['calibration_sha256']='wrong'
    elif problem=='classes':state['classes']=tuple(reversed(CLASSES))
    else:args.resume=str(path)
    torch.save(state,path)
    with pytest.raises(ValueError):initialize_recovery(RayBEVLite(cfg,DEFAULT_CALIBRATION),cfg,args)


def test_old_checkpoints_keep_legacy_bn_size_and_loss_interpretations():
    old=ModelConfig.from_checkpoint({'cfg':dict(architecture_version=2,backbone='resnet18',
        heading_encoding='axis_direction',heading_bins=2,size_encoding='class_relative')})
    assert old.bev_normalization=='batchnorm' and old.loss_normalization=='batch_objects'
    assert old.heavy_classification_weight==0 and old.size_encoding=='class_relative'
    assert ModelConfig.from_checkpoint({'cfg':vars(ModelConfig())})==ModelConfig()


@pytest.mark.parametrize('changes',[{'bev_normalization':'groupnorm'},{'loss_normalization':'unknown'},
    {'mean_training_objects':0.},{'mean_training_objects':float('nan')},
    {'size_reference':(0.,2.,3.)},{'heavy_classification_weight':-1.}])
def test_invalid_recovery_controls_fail(changes):
    with pytest.raises(ValueError):replace(config(),**changes).validate()
