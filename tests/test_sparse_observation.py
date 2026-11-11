from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from raybev.config import DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,OpenCVProjector
from raybev.sparse_loss import SparseCriterion
from raybev.sparse_diagnostics import raw_query_counts,merge_raw_counts
from raybev.sparse_observation import distortion_descriptor
from raybev.sparse_extent import VerifiedExtents,extent_points
from raybev.detection import decode_heading


def test_combined_camera_aliases_keep_calibration_order():
    from raybev.rosa_dataset import camera_entry
    from raybev.config import CAMERAS
    names=['fisheye_camera_front','fisheye_camera_rear','fisheye_camera_left','fisheye_camera_right']
    cameras={name:dict(cam_path=str(i)) for i,name in enumerate(names)}
    assert [camera_entry(cameras,n)['cam_path'] for n in CAMERAS]==['0','1','2','3']
    cameras['ARC_F']=dict(cam_path='different')
    with pytest.raises(ValueError,match='Conflicting'):camera_entry(cameras,'ARC_F')


@pytest.mark.parametrize('source,destination',[((1920,1536),(800,640)),((1920,1080),(800,640)),((999,701),(800,640))])
def test_resized_intrinsics_match_pixel_transform_and_opencv(source,destination):
    import cv2
    from PIL import Image
    from raybev.geometry import resized_camera_intrinsics,letterbox_params
    from raybev.preprocessing import resize_camera_image
    k=np.array([[570.,0.,951.],[0.,568.,773.],[0.,0.,1.]])
    distortion=np.array([.05,-.01,.002,-.0001])
    knew=resized_camera_intrinsics(k,*source,*destination)
    points=np.array([[[1.,2.,5.],[-2.,1.,4.],[.1,-.2,8.]]])
    raw=cv2.fisheye.projectPoints(points,np.zeros(3),np.zeros(3),k,distortion)[0]
    projected=cv2.fisheye.projectPoints(points,np.zeros(3),np.zeros(3),knew,distortion)[0]
    w,h,left,top=letterbox_params(*source,*destination)
    expected=(raw+.5)*np.array([w/source[0],h/source[1]])-.5+np.array([left,top])
    np.testing.assert_allclose(projected,expected,atol=1e-10)
    image=resize_camera_image(Image.new('RGB',source,(100,100,100)),*destination)
    assert image.size==destination
    if top:assert image.getpixel((0,0))==(0,0,0)


def test_network_intrinsics_are_used_once_in_sparse_projection():
    from raybev.geometry import load_cameras,project_fisheye,letterbox_params
    cfg=config();projector=OpenCVProjector(cfg,DEFAULT_CALIBRATION)
    points=torch.tensor([[[[12.,2.,1.],[5.,-8.,2.],[-8.,3.,1.]]]])
    uv,_,_=projector(points)
    for i,camera in enumerate(load_cameras(DEFAULT_CALIBRATION)):
        raw,_,_=project_fisheye(points.numpy().reshape(-1,3),camera)
        w,h,left,top=letterbox_params(camera.width,camera.height,cfg.image_width,cfg.image_height)
        expected=(raw+.5)*np.array([w/camera.width,h/camera.height])-.5+[left,top]
        # FP32 camera transforms versus NumPy FP64, including backward rays.
        np.testing.assert_allclose(uv[0,0,:,i].numpy(),expected,atol=1e-4)


def config(**kw):
    torch.set_num_threads(2)
    return replace(SparseConfig(architecture_version=5,image_width=160,image_height=128,
        channels=32,ffn_channels=64,queries=12,max_detections=12,proposal_queries=4,
        proposal_grid=4,stride4=True,observation_conditioning=True,facing_cosine_prior=0.),**kw).validate()


@pytest.mark.parametrize('empty',[False,True])
def test_observation_training_all_parameters_finite_and_used(empty):
    cfg=config();model=YOLOROSALite(cfg)
    targets=[torch.empty(0,8) if empty else torch.tensor([[10.,2.,1.,8.,3.,3.,.4,3.]])]
    output=model(torch.randn(1,4,3,128,160),targets)
    assert output['initial_boxes'].shape==(1,12,7)
    loss,parts=SparseCriterion(cfg)(output,targets);loss.backward()
    assert torch.isfinite(loss) and parts['proposal']>=0
    for name,p in model.named_parameters():
        assert p.grad is not None,name
        assert torch.isfinite(p.grad).all(),name
    model.eval()
    features=model.image_encoder(torch.randn(1,4,3,128,160))
    assert [x.shape[-1] for x in features]==[40,20,10,5]
    with torch.no_grad():out=model.forward_features(features)
    assert out['stages'][-1]['boxes'].shape==(1,12,7)
    assert not out['dn_indices']


def test_max_cardinality_recall_and_range_denominators():
    # Distance-only assignment prefers 0 + 1.1, losing a valid pair at 1m;
    # cardinality matching chooses .6 + .6 instead.
    candidates=np.array([[0.,0.],[.6,0.]])
    gt=np.array([[0.,0.,1.,4.,2.,2.,0.,3.],[-.5,0.,1.,8.,2.,3.,0.,5.]])
    counts=raw_query_counts(candidates,gt)
    assert counts['truck/all/1m']==[1,1] and counts['bus/all/1m']==[1,1]
    assert counts['car/all/1m']==[0,0]
    merged=merge_raw_counts([{'initial':counts}]*2)
    assert merged['initial']['bus/all/1m']['total']==2
    assert merged['initial']['car/all/1m']['recall'] is None


def test_facing_decision_uses_supervised_hemisphere_probability():
    cfg=config();axis=torch.tensor([[0.,1.]])
    logits=torch.tensor([[1.,1.,-10.,1.5,1.5,1.5,-10.,1.]])
    yaw,confidence=decode_heading(axis,logits,cfg)
    bins=torch.arange(8)*math.pi/4
    p=(logits.softmax(-1)*(bins.cos()>=0)).sum(-1)
    assert torch.allclose(confidence,torch.maximum(p,1-p))
    assert (torch.cos(yaw)<0).item()==(p<.5).item()


def test_lens_scale_at_optical_axis_and_invalid_observations():
    cfg=config();projector=OpenCVProjector(cfg,DEFAULT_CALIBRATION)
    projector.intrinsics.copy_(torch.eye(3)[None].repeat(4,1,1))
    projector.intrinsics[:,:2,:2]*=100
    projector.resize_scale.fill_(1);projector.distortion.zero_()
    cues=torch.zeros(1,1,1,4,6);cues[...,2]=1;cues[...,5]=math.log(10)/math.log(cfg.max_range)
    visible=torch.ones(1,1,1,4,dtype=torch.bool)
    boxes=torch.tensor([[[10.,0.,1.,4.,2.,2.,0.]]])
    descriptor=distortion_descriptor(cues,visible,boxes,projector,cfg.feature_strides)
    torch.testing.assert_close(descriptor[...,:2],torch.full_like(descriptor[...,:2],math.log(100)))
    assert torch.isfinite(descriptor).all()
    assert not distortion_descriptor(cues,~visible,boxes,projector,cfg.feature_strides).any()


def test_extent_loss_visibility_mask_and_yaw_size_gradients():
    cfg=config();projector=OpenCVProjector(cfg,DEFAULT_CALIBRATION)
    gt=torch.tensor([[12.,1.,1.5,8.,2.5,3.,.3,5.]])
    uv,valid,_=projector(extent_points(gt[None,:,:7]))
    cameras=valid[0,0].all(0).nonzero().flatten()
    assert len(cameras)
    camera=int(cameras[0]);bbox=torch.cat((uv.amin(2),uv.amax(2)),-1)[0,0,camera]
    extents=object.__new__(VerifiedExtents)
    extents.frames={'frame':[dict(gt=gt[0].tolist(),camera=camera,bbox=bbox.tolist(),edge_valid=[True]*4)]}
    pred=gt[:,:7].clone();pred[:,3]*=.6;pred[:,6]+=.15;pred.requires_grad_()
    loss=extents.loss(pred,gt,'frame',projector);loss.backward()
    assert loss>0 and pred.grad[0,3].abs()>0 and pred.grad[0,6].abs()>0
    assert torch.isfinite(pred.grad).all()
    assert extents.loss(pred,gt,None,projector)==0
    extents.frames['frame'][0]['edge_valid']=[False]*4
    assert extents.loss(pred,gt,'frame',projector)==0


def test_extent_training_requires_annotations():
    with pytest.raises(ValueError,match='verified'):
        SparseCriterion(config(extent_weight=1.))
