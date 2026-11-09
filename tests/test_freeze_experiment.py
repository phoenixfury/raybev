from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CLASSES
from raybev.model import RayBEVLite
from raybev.training import configure_image_training,build_optimizer,check_resume,calibration_sha256
from raybev.detection import make_targets,detection_loss


@pytest.mark.parametrize('backbone',['resnet18','mobilenet_v2'])
def test_early_freeze_preserves_bn_buffers_and_only_optimizes_trainable_parameters(backbone):
    cfg=replace(ModelConfig().downscale(5),backbone=backbone)
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION).train();configure_image_training(model,'early')
    assert not model.image_encoder.stage8.training and not model.image_encoder.stage16.training
    assert model.image_encoder.stage32.training
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith(('image_encoder.stage8.','image_encoder.stage16.'))}
    optimizer=build_optimizer(model,1e-4,.001)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    model.eval();model.train();configure_image_training(model,'early')  # real epoch transitions
    output=model(torch.randn(1,4,3,128,160))
    target=make_targets([torch.tensor([[4.,1.,1.,10.,3.,3.,.3,5.]])],cfg,return_class_ids=True)
    loss,_=detection_loss(output,target,cfg=cfg);loss.backward();optimizer.step()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    for k,v in model.state_dict().items():
        if k in frozen:torch.testing.assert_close(v,frozen[k],rtol=0,atol=0)


def test_resume_rejects_changed_freeze_policy_before_optimizer_restore():
    cfg=ModelConfig();h=calibration_sha256(DEFAULT_CALIBRATION)
    state=dict(cfg=vars(cfg),calibration_sha256=h,lidar_calibration_sha256=h,box_frame='CLIDAR',recipe={})
    args=SimpleNamespace(calibration=DEFAULT_CALIBRATION,lidar_calibration=None,box_frame='CLIDAR',freeze_image_stages='early')
    with pytest.raises(ValueError,match='freeze policy'):check_resume(state,cfg,args)


def test_resume_rejects_edited_annotations(tmp_path):
    cfg=ModelConfig();h=calibration_sha256(DEFAULT_CALIBRATION)
    file=tmp_path/'val.pkl';file.write_bytes(b'new GT')
    state=dict(cfg=vars(cfg),calibration_sha256=h,lidar_calibration_sha256=h,box_frame='CLIDAR',recipe={},annotation_sha256={'val':'old hash'})
    args=SimpleNamespace(calibration=DEFAULT_CALIBRATION,lidar_calibration=None,box_frame='CLIDAR',root=str(tmp_path),val_ann='val.pkl')
    with pytest.raises(ValueError,match='annotation content'):check_resume(state,cfg,args)


@pytest.mark.parametrize('mode',['none','all'])
def test_freeze_modes_have_explicit_parameter_and_bn_policies(mode):
    model=RayBEVLite(ModelConfig().downscale(5),DEFAULT_CALIBRATION).train()
    configure_image_training(model,mode)
    for name in ('stage8','stage16','stage32'):
        stage=getattr(model.image_encoder,name)
        assert stage.training==(mode=='none')
        assert all(p.requires_grad==(mode=='none') for p in stage.parameters())
    assert all(p.requires_grad for p in model.bev_detector.parameters())
