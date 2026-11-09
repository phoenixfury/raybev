"""Independent geometry/loss identities, backward compatibility and migration."""
from dataclasses import replace
from types import SimpleNamespace
import math
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CLASSES
from raybev.geometry import load_cameras,build_projection_lut,project_fisheye,letterbox_params
from raybev.model import RayBEVLite,FisheyeRayTransport,MobileNetEncoder,ResNet18Encoder
from raybev.detection import facing_vector_loss,decode_heading,make_targets,decode_detections
from raybev.diagnostics import center_statistics,summarize_centers
from raybev.metrics import evaluate_detections
from raybev.training import initialize_recovery,calibration_sha256,accumulation_weight


def config():
    return replace(ModelConfig().downscale(5),bev_x_min=-4.,bev_x_max=4.,bev_y_min=-4.,bev_y_max=4.,bev_resolution=1.)


def test_old_checkpoints_keep_historical_lattice_pool_and_facing_objective():
    saved=vars(config()).copy()
    for key in ('feature_pixel_origin','height_fusion','facing_loss'):saved.pop(key)
    old=ModelConfig.from_checkpoint({'cfg':saved})
    assert old.feature_pixel_origin==3.5 and old.height_fusion=='pooled' and old.facing_loss=='contrast'
    assert old.bev_input_channels==40 and config().bev_input_channels==136


@pytest.mark.parametrize('backbone',['resnet18','mobilenet_v2'])
def test_actual_backbone_stride8_sampling_lattice_is_zero_origin(backbone):
    cfg=replace(config(),backbone=backbone)
    encoder=ResNet18Encoder(cfg) if backbone=='resnet18' else MobileNetEncoder(cfg)
    if backbone=='resnet18':
        # Main branch; downsample shortcuts share this same lattice.
        layers=[encoder.stage8[0],encoder.stage8[3],encoder.stage8[5][0].conv1]
    else:
        layers=[m for m in encoder.stage8.modules() if isinstance(m,torch.nn.Conv2d)]
    origin=0.;jump=1.
    for layer in layers:
        kernel=layer.kernel_size if isinstance(layer.kernel_size,int) else layer.kernel_size[0]
        padding=layer.padding if isinstance(layer.padding,int) else layer.padding[0]
        stride=layer.stride if isinstance(layer.stride,int) else layer.stride[0]
        origin+=((kernel-1)/2-padding)*jump
        jump*=stride
    assert jump==8 and origin==cfg.feature_pixel_origin==0


def test_lut_samples_coordinate_ramp_at_projected_input_pixels_not_resize_bins():
    cfg=config();cameras=load_cameras(DEFAULT_CALIBRATION)
    grid,_,valid=build_projection_lut(cameras,cfg)
    fh,fw=cfg.image_height//8,cfg.image_width//8
    yy,xx=torch.meshgrid(torch.arange(fh)*8,torch.arange(fw)*8,indexing='ij')
    ramps=torch.stack((xx,yy)).float()[None]
    checked=0
    for k,ci,row,col in np.argwhere(valid):
        x=cfg.bev_x_max-(row+.5)*cfg.bev_resolution
        y=cfg.bev_y_max-(col+.5)*cfg.bev_resolution
        uv,_,_=project_fisheye([[x,y,cfg.heights[k]]],cameras[ci],cfg.theta_max_degrees)
        cam=cameras[ci];nw,nh,left,top=letterbox_params(cam.width,cam.height,cfg.image_width,cfg.image_height)
        target=(uv[0]+.5)*np.array([nw/cam.width,nh/cam.height])-.5+[left,top]
        if not (0<=target[0]<=(fw-1)*8 and 0<=target[1]<=(fh-1)*8):continue
        actual=F.grid_sample(ramps,torch.tensor(grid[k,ci,row,col]).view(1,1,1,2),align_corners=False)[0,:,0,0]
        np.testing.assert_allclose(actual.numpy(),target,atol=2e-5)
        checked+=1
        if checked==32:break
    assert checked==32


def test_legacy_lut_is_exact_old_half_pixel_mapping():
    cfg=replace(config(),feature_pixel_origin=3.5)
    cams=load_cameras(DEFAULT_CALIBRATION);grid,_,valid=build_projection_lut(cams,cfg)
    k,ci,row,col=np.argwhere(valid)[0];cam=cams[ci]
    xyz=[[cfg.bev_x_max-(row+.5)*cfg.bev_resolution,cfg.bev_y_max-(col+.5)*cfg.bev_resolution,cfg.heights[k]]]
    uv,_,_=project_fisheye(xyz,cam,cfg.theta_max_degrees)
    nw,nh,left,top=letterbox_params(cam.width,cam.height,cfg.image_width,cfg.image_height)
    expected=2*((uv[0]+.5)*[nw/cam.width,nh/cam.height]+[left,top])/[cfg.image_width,cfg.image_height]-1
    np.testing.assert_array_equal(grid[k,ci,row,col],expected.astype(np.float32))


def test_height_planes_use_common_denominator_preserve_pooled_sum_and_invalid_zeros():
    cfg=config();transport=FisheyeRayTransport(cfg,DEFAULT_CALIBRATION)
    transport.sample_grid.zero_();transport.geometric_prior.fill_(1.)
    feature=torch.arange(1.,5.).view(1,4,1,1,1).expand(1,4,32,16,20)
    probabilities=torch.tensor([.2,.4,.6,.8])
    logits=torch.logit(probabilities).view(1,1,4,1,1).expand(1,4,4,16,20).clone().requires_grad_()
    actual=transport(feature,logits)
    expected=2.5*probabilities/probabilities.sum()
    torch.testing.assert_close(actual[0,:128,0,0].reshape(4,32),expected[:,None].expand(4,32))
    transport.cfg=replace(cfg,height_fusion='pooled');pooled=transport(feature,logits)
    torch.testing.assert_close(actual[:,:128].reshape(1,4,32,8,8).sum(1),pooled[:,:32])
    torch.testing.assert_close(actual[:,128:],pooled[:,32:])
    actual[:,:32].sum().backward()
    assert logits.grad.abs().sum()>0 and torch.isfinite(logits.grad).all()
    transport.geometric_prior.zero_()
    invalid=transport(feature,logits)
    assert (invalid[:,:37]==0).all() and torch.isfinite(invalid).all()


def test_height_expansion_warm_start_preserves_first_convolution_exactly(tmp_path):
    old=replace(config(),height_fusion='pooled',feature_pixel_origin=3.5,facing_loss='contrast')
    cfg=replace(old,height_fusion='concat')
    torch.manual_seed(7);source=RayBEVLite(old,DEFAULT_CALIBRATION).eval()
    path=tmp_path/'v4_2.pt'
    torch.save(dict(cfg=vars(old),state_dict=source.state_dict(),classes=CLASSES,epoch=9,box_frame='CLIDAR',
                    calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
                    lidar_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION)),path)
    target=RayBEVLite(cfg,DEFAULT_CALIBRATION).eval()
    args=SimpleNamespace(init_from=str(path),resume=None,box_frame='CLIDAR',calibration=DEFAULT_CALIBRATION,
                         lidar_calibration=DEFAULT_CALIBRATION,z_reference='center')
    report=initialize_recovery(target,cfg,args)
    assert report['initialized_parameters']==['bev_detector.fine.0.weight'] and not report['reset_parameters']
    features=torch.randn(1,4,32,16,20);logits=torch.randn(1,4,4,16,20)
    a=source.transport(features,logits);b=target.transport(features,logits)
    torch.testing.assert_close(source.bev_detector.fine(a),target.bev_detector.fine(b),atol=1e-6,rtol=1e-5)
    assert sum(p.numel() for p in target.parameters())-sum(p.numel() for p in source.parameters())==3072
    for key,value in target.state_dict().items():
        if key!='bev_detector.fine.0.weight':torch.testing.assert_close(value,source.state_dict()[key],rtol=0,atol=0)


def test_facing_unit_loss_closes_orthogonal_nullspace_and_has_180_degree_gradient():
    cfg=config();gt=torch.tensor([[0.,1.]])
    facing=torch.tensor([[-100.,4.]],requires_grad=True)
    axis=torch.tensor([[math.sin(math.radians(20)),math.cos(math.radians(20))]])
    yaw,confidence=decode_heading(axis,facing,cfg)
    assert abs(math.degrees(float(yaw[0]))+170)<1e-3 and confidence[0]>.99
    assert facing_vector_loss(facing,gt,replace(cfg,facing_loss='contrast'))<.001
    loss=facing_vector_loss(facing,gt,cfg).sum();loss.backward()
    assert loss>100 and facing.grad[0,0]<0 and facing.grad[0,1]>0
    opposite=torch.tensor([[0.,-1.]],requires_grad=True)
    facing_vector_loss(opposite,gt,cfg).sum().backward()
    assert opposite.grad[0,1]<0 and torch.isfinite(opposite.grad).all()


def test_scale_error_uses_union_not_enclosing_rectangular_volume():
    gt=np.array([[0,0,1,2,4,2,0,0]],dtype=float)
    pred=np.array([[0,0,1,4,2,2,0,0,.9]],dtype=float)
    m=evaluate_detections([pred],[gt])
    assert m['class_results']['car']['ASE']==pytest.approx(2/3)
    assert m['class_results']['car']['AP']==pytest.approx(1.)
    assert 'aligned_size_union' in m['metric_version']


def test_center_diagnostics_are_additive_use_GT_class_for_size_and_handle_empty():
    cfg=replace(config(),class_size_priors=((4.,2.,2.),)*5+((12.,3.,3.),))
    boxes=torch.tensor([[1.5,1.5,1.,12.,3.,3.,0.,5.]])
    targets=make_targets([boxes],cfg,return_class_ids=True)
    output=[torch.zeros(1,c,8,8) for c in (6,8,1,2,1)]
    output[1]=targets[1].clone();output[3][:,1]=1
    output[0][:,0]=4  # misclassify a bus as car
    sums=center_statistics(output,targets,cfg);summary=summarize_centers(sums*2)
    assert summary['bus']['count']==2 and summary['bus']['class_accuracy']==0
    assert summary['bus']['mean_L_ratio_GT_class']==pytest.approx(1.)
    assert summary['bus']['mean_L_ratio_predicted_class']==pytest.approx(1/3)
    assert summary['bus']['AOE_deg']==0 and summary['car']['AOE_deg'] is None
    empty=make_targets([torch.empty(0,8)],cfg,return_class_ids=True)
    assert center_statistics(output,empty,cfg).sum()==0


@pytest.mark.parametrize('samples,batch,accum',[(7,2,4),(11,2,2),(9,1,4),(3,4,2)])
def test_accumulated_sample_mean_gradient_matches_whole_window(samples,batch,accum):
    x=torch.arange(1.,samples+1)
    for start in range(0,samples,batch*accum):
        weight=torch.tensor(0.,requires_grad=True)
        for i in range(start,min(samples,start+batch*accum),batch):
            micro=x[i:i+batch]
            loss=(weight-micro).square().mean()
            (loss*accumulation_weight(i//batch,len(micro),batch,accum,samples)).backward()
        expected=-2*x[start:start+batch*accum].mean()
        torch.testing.assert_close(weight.grad,expected)


def test_per_class_candidate_limit_never_renormalizes_foreground_probabilities():
    cfg=replace(config(),single_label=False)
    output=[torch.zeros(1,c,8,8) for c in (6,8,1,2,1)]
    output[1][:,7]=1.;output[3][:,1]=1.;output[2].fill_(20.)
    output[4].fill_(-20.)
    for r,c,probabilities in [(1,1,[.6,.02,.02,.02,.04,.3]),(5,5,[.7,.06,.06,.06,.06,.06])]:
        output[0][0,:,r,c]=torch.tensor(probabilities).log()
        output[4][0,0,r,c]=math.log(.8/.2)
    det=decode_detections(output,cfg,score_threshold=.01,apply_nms=False,per_class_topk=1)[0]
    assert det[det[:,7]==5,8].item()==pytest.approx(.8*.3)
    assert det[det[:,7]==0,8].item()==pytest.approx(.8*.7)
