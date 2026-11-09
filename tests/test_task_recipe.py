"""Heading reversal, class-relative sizes and heavy-vehicle supervision."""
from dataclasses import replace
import math
import numpy as np
import pytest
import torch

from raybev.config import ModelConfig
from raybev.detection import (make_targets,detection_loss,decode_detections,
                              heading_targets,decode_heading,detection_scores)
from raybev.training import fit_training_config,balanced_training_view,check_resume
from raybev.metrics import evaluate_detections
from raybev.model import BEVDetector


def config():
    return replace(ModelConfig(),backbone='mobilenet_v2',heading_encoding='binned_residual',heading_bins=4,
        bev_normalization='batchnorm',size_encoding='class_relative',loss_normalization='batch_objects',heavy_classification_weight=0.,
        classification_mode='heatmap',foreground_label_smoothing=0.,center_classification_weight=.1,
        bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.,
        class_size_priors=((4.5,1.9,1.6),(.7,.7,1.7),(1.8,.7,1.6),(6.,2.5,2.8),(5.,2.,2.),(10.,2.8,3.5))).validate()


def predictions(cfg,requires_grad=True):
    return [torch.zeros(1,c,8,8,requires_grad=requires_grad) for c in (6,8,1,cfg.heading_bins)]


def box(cfg,ci=5,yaw=0.,x=1.5,y=1.5):
    return [x,y,1.,*cfg.class_size_priors[ci],yaw,ci]


def test_coarse_direction_has_a_corrective_gradient_for_exact_front_back_reversal():
    cfg=config();gt=torch.tensor([box(cfg,yaw=math.pi)])
    target=make_targets([gt],cfg,return_class_ids=True)
    pred=predictions(cfg,False)
    pred[1][:,7]=1  # zero residual is correct for both bin 0 and bin 2
    pred[3][:,0]=5  # wrong coarse direction
    pred=[p.requires_grad_() for p in pred]
    loss,parts=detection_loss(pred,target,cfg=cfg);loss.backward()
    assert pred[3].grad[0,2,2,2]<-.1
    assert pred[3].grad[0,0,2,2]>.1
    assert parts['heading_bin']>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pred)


def test_raw_sincos_also_removes_the_legacy_antipodal_dead_zone():
    cfg=replace(config(),heading_bins=0)
    target=make_targets([torch.tensor([box(cfg,ci=0,yaw=math.pi)])],cfg,return_class_ids=True)
    gradients=[]
    for normalize in [True,False]:
        pred=predictions(config(),False)[:3];pred[1][:,7]=1
        pred=[p.requires_grad_() for p in pred]
        loss,_=detection_loss(pred,target,cfg=replace(cfg,normalize_yaw_loss=normalize))
        loss.backward();gradients.append(float(pred[1].grad[0,7,2,2]))
    assert abs(gradients[0])<1e-5 and gradients[1]>.5


@pytest.mark.parametrize('yaw',[-math.pi,-math.pi+1e-5,-3*math.pi/4,-math.pi/4-1e-5,
    -math.pi/4+1e-5,0.,math.pi/4-1e-5,math.pi/4+1e-5,math.pi/2,math.pi-1e-5])
@pytest.mark.parametrize('ci',[3,5])
def test_heading_and_nonprior_truck_bus_dimensions_round_trip(yaw,ci):
    cfg=config();gt=torch.tensor([box(cfg,ci,yaw)])
    gt[:,3:6]*=torch.tensor([1.3,.9,1.1])
    target=make_targets([gt],cfg,return_class_ids=True)
    heat,reg,mask,owners=target
    logits=torch.full_like(heat,-20.);logits[heat==1]=20
    bins,residual=heading_targets(gt[:,6],cfg)
    reg=reg.clone();reg[0,6:8,2,2]=residual[0]
    heading=torch.full((1,4,8,8),-20.);heading[0,bins[0],2,2]=20.
    pred=decode_detections((logits,reg,torch.full((1,1,8,8),20.),heading),cfg,score_threshold=.5)[0]
    assert pred.shape==(1,9)
    torch.testing.assert_close(pred[0,:6],gt[0,:6],atol=2e-5,rtol=1e-5)
    error=math.atan2(math.sin(float(pred[0,6])-yaw),math.cos(float(pred[0,6])-yaw))
    assert abs(error)<1e-5 and pred[0,7]==ci


def test_size_medians_are_fitted_from_supplied_training_labels_only():
    cfg=config();train=torch.tensor([box(cfg,ci=5) for _ in range(3)])
    train[:,3]=torch.tensor([8.,10.,12.]);original=train.clone()
    outside=torch.tensor([box(cfg,ci=5,x=100.)]);outside[:,3]=200.
    fitted=fit_training_config(cfg,[train,outside])
    assert fitted.training_class_counts==(0,0,0,0,0,3)
    assert fitted.class_size_priors[5]==pytest.approx((10.,2.8,3.5))
    assert fitted.class_size_priors[0]==cfg.class_size_priors[0]
    target=make_targets([torch.tensor([box(fitted)])],fitted)
    torch.testing.assert_close(target[1][0,3:6,2,2],torch.zeros(3))
    torch.testing.assert_close(train,original)


def test_historical_macro_recipe_preserves_its_original_gradient_contract():
    cfg=replace(config(),class_balanced_losses=True,center_classification_weight=.5);gradients=[]
    for cars in (1,30):
        gt=[box(cfg,x=3.5,y=3.5)]
        for i in range(1,cars+1):gt.append(box(cfg,ci=0,x=3.5-i//8,y=3.5-i%8))
        target=make_targets([torch.tensor(gt)],cfg,return_class_ids=True)
        pred=predictions(cfg,False);pred[1][:,3]=.3
        pred=[p.requires_grad_() for p in pred]
        loss,_=detection_loss(pred,target,cfg=cfg);loss.backward()
        gradients.append(torch.stack([pred[0].grad[0,5,0,0],pred[1].grad[0,3,0,0],pred[3].grad[0,0,0,0]]))
    torch.testing.assert_close(gradients[0],gradients[1])


def test_auxiliary_class_loss_explicitly_pushes_bus_above_car_at_bus_center():
    cfg=config();target=make_targets([torch.tensor([box(cfg)])],cfg,return_class_ids=True)
    gradients=[]
    for weight in (0.,.5):
        pred=predictions(cfg,False);pred[0][0,0,2,2]=4.;pred[0][0,5,2,2]=-2.
        pred=[p.requires_grad_() for p in pred]
        loss,_=detection_loss(pred,target,cfg=replace(cfg,center_classification_weight=weight))
        loss.backward();gradients.append(pred[0].grad[0,:,2,2])
    assert gradients[1][0]>gradients[0][0]  # gradient descent lowers the wrong Car logit
    assert gradients[1][5]<gradients[0][5]  # raises the correct Bus logit


def test_object_sampling_repeats_bus_frames_even_when_frame_presence_exceeds_five_percent():
    cfg=replace(config(),sampling_mode='object_frequency')
    class Dataset:
        boxes=[torch.tensor([box(cfg,ci=0) for _ in range(10)]+([box(cfg)] if i<10 else [])) for i in range(90)]+[torch.empty(0,8)]
        def __len__(self):return len(self.boxes)
    dataset=Dataset()
    old=balanced_training_view(dataset,42,replace(cfg,sampling_mode='frame_presence')).indices
    new=balanced_training_view(dataset,42,cfg).indices
    assert old.count(0)==1 and new.count(0)==3
    assert set(new)==set(range(91)) and new.count(90)==1
    assert new==balanced_training_view(dataset,42,cfg).indices


def test_stable_defaults_do_not_stack_macro_loss_and_object_frequency_repeats():
    cfg=ModelConfig()
    assert not cfg.class_balanced_losses
    assert cfg.center_classification_weight==pytest.approx(1.)
    assert cfg.sampling_mode=='frame_presence'
    assert cfg.class_positive_weights[3]==cfg.class_positive_weights[5]==4.
    assert cfg.class_regression_weights[3]==cfg.class_regression_weights[5]==4.
    assert cfg.heading_bins==2 and cfg.heading_encoding=='axis_direction' and cfg.size_encoding=='class_relative'
    assert cfg.classification_mode=='objectness_softmax' and cfg.heavy_classification_weight==0
    assert cfg.bev_normalization=='frozen_batchnorm' and cfg.loss_normalization=='fixed_train_mean'
    assert cfg.backbone=='resnet18'


def test_failed_v3_recipe_is_loadable_but_cannot_resume_with_corrected_defaults():
    cfg=ModelConfig()
    failed=replace(cfg,class_balanced_losses=True,center_classification_weight=.5,
                   sampling_mode='object_frequency')
    state={'cfg':vars(failed)}
    assert ModelConfig.from_checkpoint(state)==failed
    with pytest.raises(ValueError,match='Resume config differs'):
        check_resume(state,cfg,None)


def test_global_focal_uses_one_object_denominator_for_positives_and_negatives():
    cfg=replace(config(),center_classification_weight=0.)
    gt=torch.tensor([box(cfg,ci=0,x=2.5,y=2.5),box(cfg,ci=0,x=-2.5,y=-2.5),box(cfg)])
    target=make_targets([gt],cfg,return_class_ids=True)
    pred=predictions(cfg)
    _,parts=detection_loss(pred,target,cfg=cfg)
    pos=target[0]>=1-1e-5
    class_weights=torch.tensor(cfg.class_positive_weights)[None,:,None,None]
    expected=math.log(2)*.25*((pos*class_weights).sum()+((1-target[0]).pow(4)*~pos).sum())/pos.sum()
    torch.testing.assert_close(parts['heatmap'],expected)


def test_car_bias_is_not_suppressed_by_stacked_rare_class_balancing():
    # Realistic early-training prior: many cars, a few heavy vehicles, and
    # slightly higher truck/bus probabilities across a dense 384x384 map.
    cfg=replace(ModelConfig(),classification_mode='heatmap',foreground_label_smoothing=0.,center_classification_weight=.1);boxes=[]
    for i,ci in enumerate([0]*17+[1]*3+[3]*2+[5]):
        boxes.append([30.-(i//6)*10.,30.-(i%6)*10.,1.,*config().class_size_priors[ci],0.,ci])
    target=make_targets([torch.tensor(boxes)],cfg,return_class_ids=True)
    probabilities=torch.tensor([.025,.03,.01,.08,.02,.06])
    gradients=[]
    for recipe in (cfg,replace(cfg,class_balanced_losses=True,center_classification_weight=.5)):
        heat=torch.logit(probabilities)[None,:,None,None].expand(1,6,384,384).clone().requires_grad_()
        pred=[heat]+[torch.zeros(1,c,384,384,requires_grad=True) for c in (8,1,cfg.heading_bins)]
        loss,_=detection_loss(pred,target,cfg=recipe);loss.backward()
        gradients.append(float(heat.grad[:,0].sum()))
        for ci in (3,5):
            centers=target[0][:,ci]==1
            assert (heat.grad[:,ci][centers]<0).all()  # heavy positives still learn
    assert gradients[0]<0,gradients  # descent increases the corrected car bias
    assert gradients[1]>0,gradients  # reproduces the failed macro recipe


def test_sparse_shared_classifier_learns_cars_trucks_and_buses_with_background():
    # A learned classification/decode smoke test, not an accuracy claim for the
    # camera backbone: geometry is oracle and input features identify classes.
    torch.manual_seed(17)
    cfg=replace(config(),bev_x_min=-16.,bev_x_max=16.,bev_y_min=-16.,bev_y_max=16.)
    boxes=[box(cfg,ci=0,x=x,y=y) for x in (-12.5,-7.5,-2.5,2.5)
           for y in (-12.5,-7.5,-2.5,2.5,7.5)]
    boxes.extend([box(cfg,ci=3,x=12.5,y=-9.5),box(cfg,ci=5,x=12.5,y=6.5)])
    gt=torch.tensor(boxes);targets=make_targets([gt],cfg,return_class_ids=True)
    heat,reg,mask,owners=targets
    features=torch.zeros(1,7,32,32);features[:,6]=1
    for ci in (0,3,5):
        centers=(owners[:,0]==ci)
        features[:,ci][centers]=1.;features[:,6][centers]=0.
    classifier=torch.nn.Conv2d(7,6,1)
    torch.nn.init.zeros_(classifier.weight)
    torch.nn.init.constant_(classifier.bias,math.log(.01/.99))
    optimizer=torch.optim.Adam(classifier.parameters(),lr=.12)
    heading=torch.full((1,4,32,32),-8.);heading[:,0]=8.
    quality=torch.full((1,1,32,32),8.)
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss,_=detection_loss((classifier(features),reg,quality,heading),targets,cfg=cfg)
        loss.backward();optimizer.step()
    with torch.no_grad():
        output=(classifier(features),reg,quality,heading)
        decoded=decode_detections(output,cfg,score_threshold=.3,apply_nms=False)[0]
    assert len(decoded)==len(gt)
    metrics=evaluate_detections([decoded.numpy()],[gt.numpy()])
    for name in ('car','truck','bus'):
        assert metrics['class_results'][name]['AP']>.99


def test_one_class_per_cell_prevents_car_and_bus_priors_decoding_the_same_shared_box():
    cfg=config();pred=predictions(cfg,False)
    pred[0].fill_(-20.);pred[0][0,0,2,2]=4.;pred[0][0,5,2,2]=5.
    pred[1][:,7]=1.;pred[2].fill_(20.)
    detection=decode_detections(pred,cfg,score_threshold=.5)[0]
    assert detection.shape==(1,9) and detection[0,7]==5
    torch.testing.assert_close(detection[0,3:6],torch.tensor(cfg.class_size_priors[5]))


def test_heading_confidence_is_separate_and_invalid_heading_does_not_consume_topk():
    cfg=config();vectors=torch.tensor([[0.,1.],[0.,1.]])
    angles,confidence=decode_heading(vectors,torch.tensor([[10.,-10.,-10.,-10.],[0.,0.,0.,0.]]),cfg)
    assert confidence[0]>.99 and confidence[1]==pytest.approx(.25)
    pred=predictions(cfg,False);pred[0].fill_(-20.)
    pred[0][0,0,2,2]=5.;pred[0][0,0,6,6]=4.;pred[1][:,7]=1.
    scores=detection_scores(pred,cfg)
    pred[3][:,0]=10.
    torch.testing.assert_close(detection_scores(pred,cfg),scores)
    pred[3][0,0,2,2]=float('nan')
    detection=decode_detections(pred,cfg,score_threshold=.1,topk=1)[0]
    assert len(detection)==1 and detection[0,0]==pytest.approx(-2.5)


def test_heading_metrics_expose_reversals_that_center_distance_ap_cannot():
    gt=np.array([[0,0,1,10,2.8,3.5,0,5.]])
    pred=np.array([[0,0,1,10,2.8,3.5,math.pi,5,.9]])
    result=evaluate_detections([pred],[gt])['class_results']['bus']
    assert result['AP']==pytest.approx(1.) and result['AOE']==pytest.approx(math.pi)
    assert result['axis_AOE']==pytest.approx(0.) and result['heading_flip_rate']==1.


def test_legacy_checkpoint_defaults_keep_original_head_and_absolute_size_contract():
    cfg=ModelConfig.from_checkpoint({'cfg':{'architecture_version':2}})
    assert cfg.heading_bins==0 and cfg.normalize_yaw_loss
    assert cfg.size_encoding=='absolute' and not cfg.single_label
    assert not cfg.class_balanced_losses and cfg.center_classification_weight==0


def test_new_size_head_starts_at_the_class_prior_and_has_a_four_bin_output():
    cfg=config();model=BEVDetector(cfg).eval()
    with torch.no_grad():output=model(torch.rand(1,cfg.bev_input_channels,8,8))
    assert len(output)==4 and output[3].shape==(1,4,8,8)
    torch.testing.assert_close(output[1][:,3:6],torch.zeros(1,3,8,8))
    torch.testing.assert_close(output[1][:,6],torch.zeros(1,8,8))
    torch.testing.assert_close(output[1][:,7],torch.ones(1,8,8))


def test_balanced_uniform_weights_still_require_exact_owner_for_colliding_classes():
    cfg=replace(config(),class_regression_weights=(1.,)*6)
    gt=torch.tensor([box(cfg,ci=0,x=1.2),box(cfg,ci=5)])
    target=make_targets([gt],cfg,return_class_ids=True)
    with pytest.raises(ValueError,match='return_class_ids=True'):
        detection_loss(predictions(cfg),target[:3],cfg=cfg)
    loss,parts=detection_loss(predictions(cfg),target,cfg=cfg)
    assert torch.isfinite(loss) and parts['center_classification']==0
