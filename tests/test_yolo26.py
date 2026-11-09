"""YOLO26n pretrained identity, input contract, checkpoint and trainability checks."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse

import pytest
import torch

pytest.importorskip('ultralytics')
from raybev import ModelConfig,RayBEVLite
from raybev.config import DEFAULT_CALIBRATION
from raybev.model import YOLO26nEncoder
from raybev.preprocessing import normalize_imagenet
from raybev.detection import detection_loss,make_targets
from raybev.training import add_training_options,build_optimizer,configure_image_training,initialize_recovery
from raybev.yolo26 import DEFAULT_WEIGHTS,WEIGHTS_SHA256,build_backbone,sha256,ensure_official_weights


@pytest.fixture
def cfg():
    torch.set_num_threads(2)
    return ModelConfig(backbone='yolo26n').downscale(5)


@pytest.fixture
def official():
    if not DEFAULT_WEIGHTS.is_file():pytest.skip('Copy/download official weights for the offline integration checks')
    assert sha256(DEFAULT_WEIGHTS)==WEIGHTS_SHA256
    checkpoint=torch.load(DEFAULT_WEIGHTS,map_location='cpu',weights_only=False)
    return (checkpoint.get('ema') or checkpoint['model']).float().eval()


def test_pretrained_stages_and_normalization_equal_official_yolo(cfg,official):
    encoder=YOLO26nEncoder(cfg,pretrained=True).eval()
    rgb=torch.rand(1,4,3,cfg.image_height,cfg.image_width)
    raw=rgb.reshape(4,3,cfg.image_height,cfg.image_width)
    normalized=normalize_imagenet(rgb).reshape_as(raw)
    captured=[]
    def capture(module,inputs,output):captured.append(output.detach().clone())
    handles=[stage.register_forward_hook(capture) for stage in (encoder.stage8,encoder.stage16,encoder.stage32)]
    with torch.no_grad():
        features,confidence=encoder.forward_single(normalized,torch.zeros(4,8,cfg.image_height//8,cfg.image_width//8))
        x=raw
        for i,layer in enumerate(official.model[:10]):
            x=layer(x)
            if i in (4,6,9):
                actual=captured.pop(0)
                torch.testing.assert_close(actual,x,atol=3e-5,rtol=3e-5)
    for handle in handles:handle.remove()
    assert features.shape==(4,32,16,20) and confidence.shape==(4,4,16,20)
    assert 'sha256='+WEIGHTS_SHA256 in encoder.initialization
    # Every retained parameter/buffer has its pretrained value, not only conv1.
    for actual,expected in zip((encoder.stage8,encoder.stage16,encoder.stage32),
                               (official.model[:5],official.model[5:7],official.model[7:10])):
        for a,e in zip(actual.state_dict().values(),expected.state_dict().values()):
            torch.testing.assert_close(a,e,rtol=0,atol=0)
    assert all(m.eps==.001 and m.momentum==.03 for stage in
               (encoder.stage8,encoder.stage16,encoder.stage32) for m in stage.modules() if isinstance(m,torch.nn.BatchNorm2d))
    assert all(p.requires_grad for p in encoder.parameters())


def test_image_fpn_and_geometry_contract(cfg):
    encoder=YOLO26nEncoder(cfg).eval()
    assert [encoder.lat8[0].in_channels,encoder.lat16[0].in_channels,encoder.lat32[0].in_channels]==[128,128,256]
    assert not {type(m).__name__ for m in encoder.modules()} & {'C2PSA','PSABlock','Attention','Detect'}
    assert encoder.backbone_profile['retained_layers']==list(range(10))
    jump=1;origin=0
    for layer in (encoder.stage8[0].conv,encoder.stage8[1].conv,encoder.stage8[3].conv):
        origin+=((layer.kernel_size[0]-1)/2-layer.padding[0])*jump
        jump*=layer.stride[0]
    assert origin==cfg.feature_pixel_origin==0 and jump==8
    with torch.no_grad():
        features,confidence=encoder(torch.randn(2,4,3,128,160),torch.randn(4,8,16,20))
    assert features.shape==(2,4,32,16,20) and confidence.shape==(2,4,4,16,20)
    assert (cfg.bev_x_min,cfg.bev_x_max)==(-48,48) and cfg.bev_input_channels==136


@pytest.mark.parametrize('mode',['none','early','all'])
def test_finite_heavy_class_backward_freeze_and_optimizer(cfg,mode):
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION).train()
    configure_image_training(model,mode)
    frozen={k:v.clone() for k,v in model.state_dict().items() if k.startswith('image_encoder.stage') and
            not getattr(model.image_encoder,k.split('.')[1]).training}
    optimizer=build_optimizer(model,1e-4,.001,.5)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    for group in optimizer.param_groups:
        expected=5e-5 if any(id(p)==id(q) for stage in (model.image_encoder.stage8,model.image_encoder.stage16,model.image_encoder.stage32)
                            for p in stage.parameters() for q in group['params']) else 1e-4
        assert group['lr']==expected
    images=normalize_imagenet(torch.rand(1,4,3,128,160))
    output=model(images)
    targets=make_targets([torch.tensor([[4.,1.,1.5,10.,2.8,3.5,.6,5.],[-6.,-2.,1.5,8.,2.7,3.,-2.,3.]])],cfg,return_class_ids=True)
    loss,_=detection_loss(output,targets,cfg=cfg)
    assert torch.isfinite(loss)
    loss.backward();optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    for key,value in frozen.items():torch.testing.assert_close(model.state_dict()[key],value,rtol=0,atol=0)


def test_raybev_checkpoint_reload_does_not_need_original_yolo_weights(cfg,tmp_path,monkeypatch):
    import raybev.yolo26 as yolo
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION).eval()
    checkpoint=tmp_path/'raybev.pt'
    torch.save(dict(cfg=vars(cfg),state_dict=model.state_dict()),checkpoint)
    def forbidden(*args,**kwargs):raise AssertionError('Checkpoint reload must not download pretrained weights')
    monkeypatch.setattr(yolo,'ensure_official_weights',forbidden)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    restored_cfg=ModelConfig.from_checkpoint(state)
    restored=RayBEVLite(restored_cfg,DEFAULT_CALIBRATION).eval()
    restored.load_state_dict(state['state_dict'],strict=True)
    images=normalize_imagenet(torch.rand(1,4,3,128,160))
    with torch.no_grad():
        for a,e in zip(restored(images),model(images)):torch.testing.assert_close(a,e,rtol=0,atol=0)


@pytest.mark.parametrize('field,value',[('scale','s'),('reg_max',16),('nc',6),('backbone',[]),
                                      ('head',[[-1,1,'Segment',[80,32,256]]])])
def test_reject_wrong_official_model_not_silent_partial_load(official,tmp_path,field,value):
    official.yaml[field]=value
    path=tmp_path/'not_yolo26n.pt';torch.save({'model':official},path)
    with pytest.raises(ValueError,match='Expected unfused official COCO YOLO26n'):
        build_backbone(backbone_weights=path)


def test_reject_missing_or_incomplete_pretrained_weights(official,tmp_path):
    with pytest.raises(FileNotFoundError):build_backbone(backbone_weights=tmp_path/'missing.pt')
    official.model[0].conv=torch.nn.Conv2d(3,8,3,2,1,bias=False)
    path=tmp_path/'incomplete.pt';torch.save({'model':official},path)
    with pytest.raises(RuntimeError,match='size mismatch'):build_backbone(backbone_weights=path)


def test_reject_modified_official_cache(tmp_path):
    path=tmp_path/'yolo26n.pt';path.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='SHA256 mismatch'):ensure_official_weights(path)


def test_no_random_fallback_after_download_failure(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise OSError('offline')
    monkeypatch.setattr(torch.hub,'download_url_to_file',fail)
    with pytest.raises(RuntimeError,match='no random fallback'):ensure_official_weights(tmp_path/'yolo26n.pt')
    assert not list(tmp_path.iterdir())


def test_cli_and_explicit_no_cross_backbone_recovery(cfg,tmp_path):
    parser=argparse.ArgumentParser();add_training_options(parser)
    assert parser.parse_args(['--backbone','yolo26n']).backbone=='yolo26n'
    checkpoint=tmp_path/'r18.pt';torch.save({'cfg':vars(replace(cfg,backbone='resnet18'))},checkpoint)
    with pytest.raises(ValueError,match='NEW pretrained run'):
        initialize_recovery(None,cfg,SimpleNamespace(init_from=str(checkpoint),resume=None))
    with pytest.raises(ValueError,match='feature_pixel_origin=0'):
        replace(cfg,feature_pixel_origin=3.5).validate()
