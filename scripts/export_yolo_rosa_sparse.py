#!/usr/bin/env python3
"""Static single-camera encoder + portable dynamic-sampling sparse head.

The head is a CPU/C7x reference export, NOT certified fully TIDL offloaded.
Calibration is baked into the head; camera order/normalization are explicit.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev.config import CAMERAS,CLASSES,DEFAULT_CALIBRATION
from infer_yolo_rosa_sparse import load_model


def export_model(model,output):
    import onnx
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new sparse export directory')
    output.mkdir(parents=True);cfg=model.cfg;model.cpu().eval()
    image=torch.zeros(1,3,cfg.image_height,cfg.image_width)
    torch.onnx.export(model.image_encoder,(image[:,None].expand(-1,4,-1,-1,-1),),str(output/'encoder_four_camera.onnx'),
        input_names=['images'],output_names=['p3','p4','p5'],opset_version=17)
    class SingleCamera(torch.nn.Module):
        def __init__(self,encoder):super().__init__();self.encoder=encoder
        def forward(self,image):return self.encoder.forward_single(image)
    torch.onnx.export(SingleCamera(model.image_encoder).eval(),(image,),str(output/'encoder_single_camera.onnx'),
        input_names=['image'],output_names=['p3','p4','p5'],opset_version=17)
    class Head(torch.nn.Module):
        def __init__(self,network):super().__init__();self.network=network
        def forward(self,p3,p4,p5):
            final=self.network.forward_features((p3,p4,p5))['stages'][-1]
            return tuple(final[k] for k in ('boxes','class_logits','quality_logits','axis','facing','supported'))
    features=tuple(torch.zeros(1,4,cfg.channels,cfg.image_height//s,cfg.image_width//s) for s in (8,16,32))
    names=['boxes','class_logits','quality_logits','axis','facing_bin_logits','supported']
    torch.onnx.export(Head(model).eval(),features,str(output/'sparse_head.onnx'),input_names=['p3','p4','p5'],output_names=names,opset_version=17)
    operators={}
    for name in ('encoder_four_camera','encoder_single_camera','sparse_head'):
        graph=onnx.load(str(output/(name+'.onnx')));onnx.checker.check_model(graph)
        operators[name]=sorted(set(node.op_type for node in graph.graph.node))
    metadata=dict(architecture=cfg.architecture,cfg=cfg.to_dict(),camera_order=list(CAMERAS),classes=list(CLASSES),
        image_input='RGB ImageNet-normalized NCHW; YOLO internally restores RGB [0,1]',
        mean=[.485,.456,.406],std=[.229,.224,.225],feature_strides=[8,16,32],
        feature_pixel_origin=0,align_corners=False,coordinate_frame='BASELINK',box_origin='gravity_center',
        camera_model='OpenCV fisheye, four coefficients, calibrated FOV including rays beyond 90 degrees',
        queries=cfg.queries,stages=cfg.stages,support_points=cfg.num_support_points,learned_support_points=cfg.num_learned_points,head_outputs=names,
        size_decode='class-independent positive LWH; never select a prior using predicted class',
        score=f'max(sigmoid(class_logits))*sigmoid(quality_logits) ** {cfg.quality_score_power}',
        postprocess='Host: mask unsupported/nonfinite/out-of-range candidates; optional class-wise rotated NMS; max 100',
        heading='continuous doubled-angle axis; strongest of eight circular facing modes selects front/rear',
        heading_axis_output='raw doubled-axis vector (normalize for direction only)' if cfg.is_refined else 'unit doubled-axis vector',
        center_refinement='early spherical capture, final two stages bounded metric XYZ residuals' if cfg.is_refined else 'spherical direction/log-range residuals',
        feature_fusion='zero-initialized residual P5->P4->P3 semantic fusion' if cfg.is_refined else 'independent pyramid projections',
        projection='Dynamic per-stage box support points. Calibration constants are baked into the exported head.',
        training_only_excluded=['Hungarian matching','denoising queries','teacher','track-balanced sampling'],
        operators=operators,deployment_status='Reference ONNX only. Exact-SDK TIDL import, INT8 accuracy, memory and board 5-FPS verification required.')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return metadata


def verify_export(model,output):
    """CPU FP32 parity at the checkpoint's exact native shapes, including yaw wrap."""
    import numpy as np
    import onnxruntime as ort
    output=Path(output);cfg=model.cfg;model.cpu().eval()
    images=torch.randn(1,4,3,cfg.image_height,cfg.image_width)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    with torch.inference_mode():
        features=model.image_encoder(images);head=model.forward_features(features)['stages'][-1]
    report={}
    for name,inputs,expected in [
        ('encoder_four_camera',{'images':images.numpy()},features),
        ('encoder_single_camera',{'image':images[:,0].numpy()},[f[:,0] for f in features]),
        ('sparse_head',{n:f.numpy() for n,f in zip(('p3','p4','p5'),features)},
         [head[k] for k in ('boxes','class_logits','quality_logits','axis','facing','supported')])]:
        session=ort.InferenceSession(str(output/(name+'.onnx')),options,providers=['CPUExecutionProvider'])
        actual=session.run(None,inputs);errors=[]
        for i,(value,reference) in enumerate(zip(actual,expected)):
            reference=reference.numpy()
            if value.dtype==np.bool_:
                np.testing.assert_array_equal(value,reference);errors.append(0.)
            elif name=='sparse_head' and i==0:
                np.testing.assert_allclose(value[...,:6],reference[...,:6],atol=2e-3,rtol=1e-3)
                yaw=np.arctan2(np.sin(value[...,6]-reference[...,6]),np.cos(value[...,6]-reference[...,6]))
                assert np.abs(yaw).max()<1e-3
                errors.append(float(max(np.abs(value[...,:6]-reference[...,:6]).max(),np.abs(yaw).max())))
            else:
                np.testing.assert_allclose(value,reference,atol=3e-4,rtol=1e-3)
                errors.append(float(np.abs(value-reference).max()))
        report[name]=dict(max_absolute_errors=errors,output_shapes=[list(v.shape) for v in actual])
    (output/'parity.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verify',action='store_true',help='Requires onnxruntime; check all exported graphs at native shapes')
    a=p.parse_args();torch.set_num_threads(2)
    model,state=load_model(a.checkpoint,a.calibration)
    metadata=export_model(model,a.output)
    metadata['calibration_sha256']=state['calibration_sha256'];metadata['backbone_initialization']=state['backbone_initialization']
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if a.verify:verify_export(model,a.output)
    print('Exported encoder + portable sparse head:',a.output,flush=True)


if __name__=='__main__':main()
