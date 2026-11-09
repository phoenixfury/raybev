"""Static learned graph export, operator boundary, and ONNX Runtime parity."""
from pathlib import Path
import numpy as np
import pytest
import torch
from raybev import ModelConfig,RayBEVLite


@pytest.mark.parametrize('backbone',['resnet18','yolo26n'])
def test_exported_subgraphs_match_torch(tmp_path,backbone):
    onnx=pytest.importorskip('onnx')
    ort=pytest.importorskip('onnxruntime')
    torch.set_num_threads(2)
    if backbone=='yolo26n':pytest.importorskip('ultralytics')
    cfg=ModelConfig(backbone=backbone).downscale(5)
    model=RayBEVLite(cfg,Path(__file__).parents[1]/'configs/calibration_baselink.json').eval()
    class RegisteredEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=model.image_encoder
        def forward(self,x,g):return self.encoder.forward_single(x,g)
    image=torch.rand(1,3,cfg.image_height,cfg.image_width)
    geometry=model.transport.image_geometry[:1]
    bev=torch.rand(1,cfg.bev_input_channels,cfg.bev_height,cfg.bev_width)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    for name,module,inputs,input_names in [
            ('image',RegisteredEncoder().eval(),(image,geometry),['image','geometry']),
            ('bev',model.bev_detector,(bev,),['bev'])]:
        path=str(tmp_path/(name+'.onnx'))
        torch.onnx.export(module,inputs,path,opset_version=17,input_names=input_names)
        graph=onnx.load(path);onnx.checker.check_model(graph)
        operators={n.op_type for n in graph.graph.node}
        assert not operators.intersection({'GridSample','LayerNormalization','Softmax','Attention'})
        if name=='image' and backbone=='yolo26n':
            assert not operators.intersection({'Shape','Gather','Div','MatMul'})
            assert 'Split' in operators
        if name=='bev':assert not operators.intersection({'ReduceMean','ReduceSum','InstanceNormalization','BatchNormalization'})
        session=ort.InferenceSession(path,options,providers=['CPUExecutionProvider'])
        actual=session.run(None,{n:v.numpy() for n,v in zip(input_names,inputs)})
        with torch.no_grad():expected=module(*inputs)
        for a,e in zip(actual,expected):np.testing.assert_allclose(a,e.numpy(),atol=1e-5,rtol=1e-4)


@pytest.mark.parametrize('backbone',['resnet18','yolo26n'])
def test_recipe4_export_metadata_records_axis_and_facing_contract(tmp_path,monkeypatch,backbone):
    pytest.importorskip('onnx')
    import importlib.util
    import json
    import sys
    torch.set_num_threads(2)
    if backbone=='yolo26n':pytest.importorskip('ultralytics')
    cfg=ModelConfig(backbone=backbone).downscale(5)
    calibration=Path(__file__).parents[1]/'configs/calibration_baselink.json'
    model=RayBEVLite(cfg,calibration).eval()
    checkpoint=tmp_path/'test.pt'
    training_profile=dict(model.image_encoder.backbone_profile,ultralytics_version='8.4.182') if backbone=='yolo26n' else None
    torch.save({'cfg':vars(cfg),'state_dict':model.state_dict(),
                'backbone_profile':training_profile},checkpoint)
    source=Path(__file__).parents[1]/'scripts/export_onnx.py'
    spec=importlib.util.spec_from_file_location('test_export_script',source)
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    output=tmp_path/'export'
    monkeypatch.setattr(sys,'argv',['export','--checkpoint',str(checkpoint),
                                  '--calibration',str(calibration),'--output',str(output)])
    export.main()
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['backbone']==backbone and metadata['heading_encoding']=='axis_direction'
    if backbone=='yolo26n':
        import ultralytics
        assert metadata['backbone_profile']['ultralytics_version']==ultralytics.__version__
        assert metadata['training_backbone_profile']['ultralytics_version']=='8.4.182'
        assert metadata['backbone_profile']['feature_layers']==[4,6,9]
        assert metadata['backbone_profile']['self_attention'] is False
        assert '[0,1]' in metadata['backbone_profile']['encoder_input']
    assert metadata['bev_normalization']=='frozen_batchnorm'
    assert metadata['height_fusion']=='concat' and metadata['feature_pixel_origin']==0
    assert metadata['facing_loss']=='unit_vector'
    assert metadata['bev_input_channels']==136
    assert metadata['size_encoding']=='class_relative'
    assert metadata['classification_mode']=='objectness_softmax'
    assert 'sigmoid(objectness_logits)' in metadata['quality_score']
    assert metadata['output_names'][-1]=='objectness_logits'
    assert metadata['regression_semantics'][-2:]==['sin_2yaw','cos_2yaw']
    assert '0.5*atan2' in metadata['heading_decode']
    import onnx
    graph=onnx.load(str(output/'bev_detector.onnx'))
    outputs={o.name:o for o in graph.graph.output}
    assert outputs['facing_vector'].type.tensor_type.shape.dim[1].dim_value==2
    assert outputs['objectness_logits'].type.tensor_type.shape.dim[1].dim_value==1
