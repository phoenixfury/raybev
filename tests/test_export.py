"""Static learned graph export, operator boundary, and ONNX Runtime parity."""
from pathlib import Path
import numpy as np
import pytest
import torch
from raybev import ModelConfig,RayBEVLite


def test_exported_subgraphs_match_torch(tmp_path):
    onnx=pytest.importorskip('onnx')
    ort=pytest.importorskip('onnxruntime')
    torch.set_num_threads(2)
    cfg=ModelConfig().downscale(5)
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
        if name=='bev':assert not operators.intersection({'ReduceMean','ReduceSum','InstanceNormalization','BatchNormalization'})
        session=ort.InferenceSession(path,options,providers=['CPUExecutionProvider'])
        actual=session.run(None,{n:v.numpy() for n,v in zip(input_names,inputs)})
        with torch.no_grad():expected=module(*inputs)
        for a,e in zip(actual,expected):np.testing.assert_allclose(a,e.numpy(),atol=1e-5,rtol=1e-4)


def test_recipe4_export_metadata_records_axis_and_facing_contract(tmp_path,monkeypatch):
    pytest.importorskip('onnx')
    import importlib.util
    import json
    import sys
    torch.set_num_threads(2)
    cfg=ModelConfig().downscale(5)
    calibration=Path(__file__).parents[1]/'configs/calibration_baselink.json'
    model=RayBEVLite(cfg,calibration).eval()
    checkpoint=tmp_path/'test.pt'
    torch.save({'cfg':vars(cfg),'state_dict':model.state_dict()},checkpoint)
    source=Path(__file__).parents[1]/'scripts/export_onnx.py'
    spec=importlib.util.spec_from_file_location('test_export_script',source)
    export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
    output=tmp_path/'export'
    monkeypatch.setattr(sys,'argv',['export','--checkpoint',str(checkpoint),
                                  '--calibration',str(calibration),'--output',str(output)])
    export.main()
    metadata=json.loads((output/'metadata.json').read_text())
    assert metadata['backbone']=='resnet18' and metadata['heading_encoding']=='axis_direction'
    assert metadata['bev_normalization']=='frozen_batchnorm'
    assert metadata['size_encoding']=='shared_relative'
    assert 'independent of predicted class' in metadata['size_decode']
    assert metadata['regression_semantics'][-2:]==['sin_2yaw','cos_2yaw']
    assert '0.5*atan2' in metadata['heading_decode']
    import onnx
    graph=onnx.load(str(output/'bev_detector.onnx'))
    assert graph.graph.output[-1].name=='facing_vector'
    assert graph.graph.output[-1].type.tensor_type.shape.dim[1].dim_value==2
