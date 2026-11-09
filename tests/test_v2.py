"""Physical geometry, head semantics, and deployment/training contracts for v2."""
from dataclasses import replace
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torchvision.models import mobilenet_v2
from raybev import ModelConfig,RayBEVLite
from raybev.geometry import load_cameras,project_fisheye,build_image_geometry
from raybev.model import MobileNetEncoder
from raybev.detection import (make_targets,detection_loss,decode_detections,
                               gaussian_radius,rotated_bev_nms,metric_to_cell)
from raybev.training import balanced_training_view,build_optimizer,build_scheduler

CAL=Path(__file__).parents[1]/'configs/calibration_baselink.json'
torch.set_num_threads(2)


def test_opencv_projection_matches_cv2_for_all_four_cameras():
    cv2=pytest.importorskip('cv2')
    xyz=np.array([[0.,0.,1.],[1.,.4,2.],[-2.,.3,1.],[.2,-3.,1.]])
    for cam in load_cameras(CAL):
        base=xyz@cam.T_base_cam[:3,:3].T+cam.T_base_cam[:3,3]
        uv,_,_=project_fisheye(base,cam)
        reference,_=cv2.fisheye.projectPoints(xyz[None],np.zeros(3),np.zeros(3),cam.intrinsic,cam.distortion)
        np.testing.assert_allclose(uv,reference[0],atol=1e-7)


def test_image_ray_geometry_reprojects_to_feature_centers():
    cfg=ModelConfig().downscale(5)
    cameras=load_cameras(CAL)
    cues=build_image_geometry(cameras,cfg)
    assert cues.shape==(4,8,cfg.image_height//8,cfg.image_width//8)
    for ci,cam in enumerate(cameras):
        geometry=cues[ci]
        valid=np.linalg.norm(geometry[:3],axis=0)>.9
        row,col=np.argwhere(valid)[len(np.argwhere(valid))//2]
        ray=geometry[:3,row,col].astype(float)
        point=cam.T_base_cam[:3,:3]@ray*10+cam.T_base_cam[:3,3]
        uv,visible,_=project_fisheye(point[None],cam,cfg.theta_max_degrees)
        assert visible[0]
        # Aspect-preserving 1920x1536 -> 160x128, matching half-pixel resize.
        target=np.array([(col+.5)*8-.5,(row+.5)*8-.5])
        np.testing.assert_allclose((uv[0]+.5)*cfg.image_width/cam.width-.5,target,atol=1e-4)


def test_large_vehicle_gaussian_and_far_range_round_trip():
    cfg=replace(ModelConfig(),heading_bins=0,size_encoding='absolute',classification_mode='heatmap')
    assert gaussian_radius(12/.25,2.5/.25)>gaussian_radius(4/.25,1.8/.25)
    boxes=torch.tensor([[45.,-43.,1.7,12.,2.5,3.4,2.2,5.]])
    targets=make_targets([boxes],cfg)
    heat,reg,mask=targets
    assert mask.sum()==1
    logits=torch.full_like(heat,-20.)
    logits[heat==1]=20.
    result=decode_detections((logits,reg),cfg)[0]
    torch.testing.assert_close(result[0,:7],boxes[0,:7],atol=1e-4,rtol=1e-5)


def test_rotated_nms_preserves_crossed_vehicles_and_different_classes():
    boxes=torch.tensor([[0,0,1,12,2,3,0,5,.9],
                        [.05,0,1,12,2,3,0,5,.8],
                        [0,0,1,12,2,3,math.pi/2,5,.7],
                        [0,0,1,12,2,3,0,3,.6]],dtype=torch.float32)
    assert rotated_bev_nms(boxes,.25).tolist()==[0,2,3]
    assert rotated_bev_nms(torch.empty(0,9)).numel()==0


def test_quality_controls_ranking_and_losses_have_finite_empty_gradients():
    cfg=replace(ModelConfig().downscale(5),heading_bins=0,classification_mode='heatmap')
    h,w=cfg.bev_height,cfg.bev_width
    heat=torch.full((1,6,h,w),-10.)
    heat[0,0,10,10]=heat[0,0,20,20]=5.
    reg=torch.zeros(1,8,h,w)
    quality=torch.zeros(1,1,h,w)
    quality[0,0,10,10]=-5
    quality[0,0,20,20]=5
    detections=decode_detections((heat,reg,quality),cfg,score_threshold=.1)[0]
    assert len(detections)==1
    assert detections[0,0]==pytest.approx(cfg.bev_x_max-20.5*cfg.bev_resolution)
    output=[v.requires_grad_() for v in (heat,reg,quality)]
    loss,parts=detection_loss(output,make_targets([torch.empty(0,8)],cfg),cfg=cfg)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in output)


@pytest.mark.parametrize('empty',[False,True])
def test_all_model_parameters_participate_in_backward(empty):
    cfg=ModelConfig().downscale(5)
    model=RayBEVLite(cfg,CAL)
    gt=torch.empty(0,8) if empty else torch.tensor([[42.,2.,1.5,12.,2.5,3.4,.5,5.]])
    output=model(torch.rand(1,4,3,cfg.image_height,cfg.image_width))
    loss,_=detection_loss(output,make_targets([gt],cfg),cfg=cfg)
    loss.backward()
    missing=[name for name,p in model.named_parameters() if p.grad is None]
    assert not missing,missing
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.transport.position_cues.shape[-2:]==(cfg.bev_height,cfg.bev_width)
    assert cfg.bev_input_channels==40


def test_local_backbone_weights_load_strictly_without_network(tmp_path):
    net=mobilenet_v2(weights=None)
    with torch.no_grad():net.features[0][0].weight.fill_(.123)
    path=tmp_path/'mobilenet.pth'
    torch.save(net.state_dict(),path)
    encoder=MobileNetEncoder(ModelConfig(),pretrained=True,backbone_weights=path)
    torch.testing.assert_close(encoder.stage8[0][0].weight,net.features[0][0].weight)
    with pytest.raises(ValueError,match='mobile_width'):
        MobileNetEncoder(replace(ModelConfig(),mobile_width=.75),pretrained=True)


def test_config_rejects_legacy_checkpoints_and_invalid_range():
    with pytest.raises(ValueError,match='predates'):
        ModelConfig.from_checkpoint({'cfg':{'bev_x_min':-30}})
    cfg=ModelConfig()
    assert ModelConfig.from_checkpoint({'cfg':vars(cfg)})==cfg
    with pytest.raises(ValueError):replace(cfg,bev_x_max=-50).validate()


def test_sparse_grid_heatmap_prior():
    model=RayBEVLite(ModelConfig().downscale(5),CAL)
    torch.testing.assert_close(model.bev_detector.heatmap[-1].bias.sigmoid(),torch.full((6,),.01))


def test_balancing_is_bounded_reproducible_and_retains_empty_frames():
    class Dataset:
        boxes=[torch.tensor([[0,0,1,4,2,2,0,0.]]) for _ in range(98)]+[
            torch.tensor([[0,0,1,8,2,3,0,4.]]),torch.empty(0,8)]
        def __len__(self):return len(self.boxes)
    ds=Dataset()
    a=balanced_training_view(ds,42).indices
    assert a==balanced_training_view(ds,42).indices
    assert set(a)==set(range(100))
    assert 1<a.count(98)<=3 and a.count(99)==1


def test_optimizer_warmup_and_saved_scheduler_resume():
    model=RayBEVLite(ModelConfig().downscale(5),CAL)
    optimizer=build_optimizer(model,2e-4,.001)
    scheduler=build_scheduler(optimizer,20,4)
    rates=[]
    for _ in range(10):
        rates.append(optimizer.param_groups[0]['lr'])
        optimizer.step();scheduler.step()
    assert rates[0]<rates[3]<rates[4] and rates[4]>rates[-1]
    state=scheduler.state_dict()
    restored=build_scheduler(optimizer,20,4)
    restored.load_state_dict(state)
    assert restored.last_epoch==scheduler.last_epoch
