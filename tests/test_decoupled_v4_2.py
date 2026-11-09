"""Separate center objectness from foreground semantics; retain old checkpoints."""
from dataclasses import replace
from types import SimpleNamespace
import math

import pytest
import torch

from raybev.config import ModelConfig,CLASSES,DEFAULT_CALIBRATION
from raybev.model import RayBEVLite,BEVDetector
from raybev.detection import make_targets,detection_loss,detection_scores,decode_detections
from raybev.training import initialize_recovery,fit_training_config,calibration_sha256
from raybev.rosa_dataset import photometric_augment


def config():
    return replace(ModelConfig().downscale(5),bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,
                   bev_y_max=4.,bev_resolution=1.,mean_training_objects=2.,
                   class_size_priors=((4.5,1.9,1.6),(.7,.7,1.75),(2.,.7,1.5),
                                      (6.,2.4,2.6),(5.,2.,2.),(11.,2.9,3.5)))


def prediction(cfg):
    return [torch.zeros(1,c,8,8,requires_grad=True) for c in (6,8,1,2,1)]


def boxes(ci=5):
    return torch.tensor([[1.5,1.5,1.5,11.,2.9,3.5,0.,ci]])


def test_background_only_trains_objectness_not_foreground_class_biases():
    cfg=config();out=prediction(cfg)
    loss,parts=detection_loss(out,make_targets([boxes()],cfg,return_class_ids=True),cfg=cfg)
    loss.backward()
    assert out[0].grad[0,5,2,2]<0 and out[0].grad[0,0,2,2]>0
    assert (out[0].grad[:,:,0,0]==0).all()
    assert out[4].grad[0,0,2,2]<0 and out[4].grad[0,0,0,0]>0
    assert parts['objectness']>0 and parts['center_classification']>0


def test_background_class_logits_cannot_create_a_dense_rare_class_penalty():
    cfg=config();target=make_targets([boxes()],cfg,return_class_ids=True)
    a=prediction(cfg);b=[p.detach().clone() for p in a]
    b[0].fill_(100.);b[0][:,:,2,2]=0.
    la,pa=detection_loss(a,target,cfg=cfg);lb,pb=detection_loss(b,target,cfg=cfg)
    torch.testing.assert_close(la,lb)
    torch.testing.assert_close(pa['objectness'],pb['objectness'])


def test_center_detection_is_independent_of_vehicle_semantic_identity():
    cfg=config();gradients=[]
    for ci in (3,5):
        p=prediction(cfg);loss,_=detection_loss(p,make_targets([boxes(ci)],cfg,return_class_ids=True),cfg=cfg)
        loss.backward();gradients.append(p[4].grad)
    torch.testing.assert_close(*gradients)


def test_empty_frames_keep_every_output_in_backward_without_semantic_background_training():
    cfg=config();out=prediction(cfg)
    loss,_=detection_loss(out,make_targets([torch.empty(0,8)],cfg,return_class_ids=True),cfg=cfg)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in out)
    assert (out[0].grad==0).all() and out[4].grad.abs().sum()>0


def test_score_is_objectness_times_conditional_class_times_quality_not_loss_weights():
    cfg=replace(config(),single_label=False);out=[p.detach() for p in prediction(cfg)]
    probabilities=torch.tensor([.6,.1,.05,.1,.05,.1])
    out[4].fill_(-20.)
    out[0][0,:,2,2]=probabilities.log();out[4][0,0,2,2]=math.log(.8/.2)
    out[2][0,0,2,2]=math.log(.81/.19)
    score=detection_scores(out,cfg)
    torch.testing.assert_close(score[0,:,2,2],probabilities*.8*.9)
    torch.testing.assert_close(score,detection_scores(out,replace(cfg,class_positive_weights=(1.,)*6)))


@pytest.mark.parametrize('bad',['objectness','all_classes','one_class'])
def test_invalid_logits_do_not_poison_other_candidates(bad):
    cfg=config();out=[p.detach() for p in prediction(cfg)]
    out[4].fill_(-20.);out[4][0,0,2,2]=20.
    if bad=='objectness':out[4][0,0,2,2]=float('nan')
    elif bad=='all_classes':out[0][0,:,2,2]=float('nan')
    else:out[0][0,5,2,2]=float('nan')
    score=detection_scores(out,cfg)
    assert torch.isfinite(score).all()
    if bad!='one_class':assert score[0,:,2,2].sum()==0
    else:assert score[0,5,2,2]==0 and score[0,:5,2,2].sum()>0


def test_zero_size_residual_restores_heavy_vehicle_priors_and_box_contract():
    cfg=config();out=[p.detach() for p in prediction(cfg)]
    out[0].fill_(-20.);out[0][0,5,2,2]=20.
    out[4].fill_(-20.);out[4][0,0,2,2]=20.
    out[1][:,7]=1.;out[3][:,1]=1.
    det=decode_detections(out,cfg,score_threshold=.3)[0]
    assert det.shape==(1,9) and int(det[0,7])==5
    torch.testing.assert_close(det[0,3:6],torch.tensor(cfg.class_size_priors[5]))


def test_v4_1_migration_copies_car_proposal_head_preserves_other_weights_and_resets_size(tmp_path):
    cfg=fit_training_config(config(),[boxes(0),boxes(3),boxes(5)])
    old=replace(cfg,classification_mode='heatmap',foreground_label_smoothing=0.,
                center_classification_weight=.1,heavy_classification_weight=.2,size_encoding='shared_relative')
    source=RayBEVLite(old,DEFAULT_CALIBRATION)
    with torch.no_grad():source.bev_detector.heatmap[-1].bias.copy_(torch.arange(6.))
    checkpoint=tmp_path/'v4_1.pt';old_values=vars(old).copy()
    for key in ['classification_mode','foreground_label_smoothing']:old_values.pop(key)
    torch.save(dict(cfg=old_values,state_dict=source.state_dict(),classes=CLASSES,box_frame='CLIDAR',epoch=9,
                    calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
                    lidar_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION)),checkpoint)
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION)
    args=SimpleNamespace(init_from=str(checkpoint),resume=None,calibration=DEFAULT_CALIBRATION,
                         lidar_calibration=DEFAULT_CALIBRATION,box_frame='CLIDAR',z_reference='center')
    report=initialize_recovery(model,cfg,args)
    assert report['initialized_parameters'] and len(report['reset_parameters'])==2
    for key,value in model.state_dict().items():
        if key in report['reset_parameters']:assert (value==0).all()
        elif key in report['initialized_parameters']:
            original=key.replace('objectness.','heatmap.');expected=source.state_dict()[original]
            if '.objectness.1.' in key:expected=expected[0:1]
            torch.testing.assert_close(value,expected)
        else:torch.testing.assert_close(value,source.state_dict()[key])
    with torch.no_grad():assert len(model.eval()(torch.rand(1,4,3,128,160)))==5


def test_v4_1_remains_strictly_loadable_with_old_score_and_size_semantics():
    saved=vars(replace(config(),classification_mode='heatmap',size_encoding='shared_relative',
                       foreground_label_smoothing=0.,center_classification_weight=.1)).copy()
    for key in ['classification_mode','foreground_label_smoothing']:saved.pop(key)
    old=ModelConfig.from_checkpoint({'cfg':saved})
    assert old.classification_mode=='heatmap' and old.size_encoding=='shared_relative'
    assert old.foreground_label_smoothing==0
    assert RayBEVLite(old,DEFAULT_CALIBRATION).bev_detector.objectness is None


def test_missing_new_objectness_output_fails_closed():
    cfg=config();out=prediction(cfg)[:4]
    with pytest.raises(ValueError,match='objectness'):
        detection_loss(out,make_targets([boxes()],cfg,return_class_ids=True),cfg=cfg)
    with pytest.raises(ValueError,match='objectness'):detection_scores(out,cfg)


def test_training_photometry_preserves_pixel_order_shape_and_original_images():
    images=torch.linspace(.1,.9,32).view(1,1,1,32).expand(4,3,8,32).clone();original=images.clone()
    torch.manual_seed(23);a=photometric_augment(images)
    torch.manual_seed(23);b=photometric_augment(images)
    torch.testing.assert_close(a,b);torch.testing.assert_close(images,original)
    assert a.shape==images.shape and torch.isfinite(a).all() and a.min()>=0 and a.max()<=1
    assert (a[:,:,:,1:]>=a[:,:,:,:-1]).all()  # no flips/crops/warps of the calibration
    assert not torch.allclose(a,images)


def test_decoupling_only_adds_one_small_convolutional_head():
    cfg=config();new=BEVDetector(cfg);old=BEVDetector(replace(cfg,classification_mode='heatmap'))
    assert sum(p.numel() for p in new.parameters())-sum(p.numel() for p in old.parameters())==1473


def test_foreground_class_weights_do_not_depend_on_regression_weight_controls():
    cfg=config();target=make_targets([boxes()],cfg,return_class_ids=True)
    gradients=[]
    for classification_weights in [(1.,)*6,cfg.class_positive_weights]:
        out=prediction(cfg)
        loss,_=detection_loss(out,target,cfg=replace(cfg,class_positive_weights=classification_weights,class_regression_weights=(1.,)*6))
        loss.backward();gradients.append(out[0].grad)
    torch.testing.assert_close(gradients[1],4*gradients[0])
