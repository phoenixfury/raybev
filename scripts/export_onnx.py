#!/usr/bin/env python3
"""Export TIDL-friendly learned subgraphs ONLY; geometry remains a CPU/C7x LUT stage.

Requires `pip install onnx`. Never claims whole-model TIDL compatibility.
"""
import argparse
from pathlib import Path
import json
import sys
import hashlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev import ModelConfig,RayBEVLite,CLASSES


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--calibration',default='configs/calibration_baselink.json')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='exported')
    args=p.parse_args()
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_checkpoint(state)
    calibration_hash=hashlib.sha256(Path(args.calibration).read_bytes()).hexdigest()
    if state.get('calibration_sha256',calibration_hash)!=calibration_hash:
        raise ValueError('Export calibration does not match checkpoint calibration')
    model=RayBEVLite(cfg,args.calibration).eval()
    if isinstance(state, dict) and 'classes' in state and tuple(state['classes']) != CLASSES:
        raise ValueError(f'Checkpoint class mapping {state["classes"]} does not match {CLASSES}; retrain six-class heatmap head')
    model.load_state_dict(state.get('state_dict',state),strict=True)  # old 10-regression-channel checkpoints need retraining
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    # Export a conventional 4D NCHW single-camera encoder for TIDL;
    # run it four times with shared weights at deployment.
    class PerCameraEncoder(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, x, geometry):
            return self.encoder.forward_single(x,geometry)

    torch.onnx.export(PerCameraEncoder(model.image_encoder),
                      (torch.zeros(1,3,cfg.image_height,cfg.image_width),model.transport.image_geometry[0:1]),
                      str(out/'image_encoder.onnx'),input_names=['image','geometry'],
                      output_names=['feature_map','confidence_logits'],opset_version=17)
    cin=cfg.bev_input_channels
    axis_heading=cfg.heading_bins and cfg.heading_encoding=='axis_direction'
    decoupled=cfg.classification_mode=='objectness_softmax'
    output_names=['class_logits' if decoupled else 'heatmap_logits','regression','quality_logits']+(['facing_vector' if axis_heading else 'heading_logits'] if cfg.heading_bins else [])
    if decoupled:output_names.append('objectness_logits')
    torch.onnx.export(model.bev_detector,torch.zeros(1,cin,cfg.bev_height,cfg.bev_width),
                      str(out/'bev_detector.onnx'),input_names=['bev_features'],
                      output_names=output_names,opset_version=17)
    np.savez_compressed(out/'geometry_lut.npz',
                        sample_grid=model.transport.sample_grid.numpy(),
                        geometric_prior=model.transport.geometric_prior.numpy(),
                        image_geometry=model.transport.image_geometry.numpy(),
                        position_cues=model.transport.position_cues.numpy())
    import onnx
    operator_sets={}
    for name in ('image_encoder','bev_detector'):
        graph=onnx.load(str(out/(name+'.onnx')))
        onnx.checker.check_model(graph)
        operator_sets[name]=sorted(set(node.op_type for node in graph.graph.node))
    (out/'metadata.json').write_text(json.dumps({'image_size_hw': [cfg.image_height,cfg.image_width],
             'architecture_version':2,'cfg':vars(cfg),'calibration_sha256':calibration_hash,
             'camera_model':'opencv_fisheye','camera_order':['ARC_F','ARC_R','ARC_SL','ARC_SR'],
             'image_geometry_shape':[1,8,cfg.image_height//8,cfg.image_width//8],
             'feature_shape':[1,cfg.feature_channels,cfg.image_height//8,cfg.image_width//8],
             'confidence_shape':[1,len(cfg.heights),cfg.image_height//8,cfg.image_width//8],
             'feature_stride':8,'feature_pixel_origin':cfg.feature_pixel_origin,
             'feature_coordinate_math':'u_net=(u_raw+0.5)*resize_scale-0.5+padding; u_feature=(u_net-feature_pixel_origin)/8; grid_x=2*(u_feature+0.5)/feature_width-1',
             'sample_align_corners':False,'height_planes_m':list(cfg.heights),
             'bev_input_channels':cin,
             'height_fusion':cfg.height_fusion,
             'height_normalization':'common sum of confidence*geometric_prior over ALL cameras and heights, clamped to 1e-4',
             'bev_layout':['height_concatenated_features' if cfg.height_fusion=='concat' else 'height_pooled_features','per_height_coverage','normalized_mean_height','x_normalized','y_normalized','range_normalized'],
             'classification_mode':cfg.classification_mode,
             'foreground_class_softmax_location':'host postprocess, NOT learned TIDL graph' if decoupled else None,
             'output_names':output_names,
             'quality_score':(('sigmoid(objectness_logits) * softmax(class_logits, class_axis) * ' if decoupled else 'sigmoid(heatmap_logits) * ')+f'sigmoid(quality_logits) ** {cfg.quality_score_power:g}'),
             'quality_score_power':cfg.quality_score_power,'score_peak_mode':cfg.score_peak_mode,
             'score_peak_order':'fuse_then_peak' if cfg.score_peak_mode=='fused' else 'heatmap_peak_then_fuse',
             'score_dtype':'float32',
             'single_label_per_cell':cfg.single_label,
             'backbone':cfg.backbone,'heading_encoding':cfg.heading_encoding,
             'bev_normalization':cfg.bev_normalization,
             'heading_bins':cfg.heading_bins,
             'facing_loss':cfg.facing_loss,
             'heading_decode':('axis=0.5*atan2(sin_2yaw,cos_2yaw); yaw=wrap(axis + pi if dot(facing_vector,[sin(axis),cos(axis)])<0 else axis)' if axis_heading else
                 'wrap(argmax(heading_logits) * 2*pi/heading_bins + atan2(sin_residual,cos_residual))' if cfg.heading_bins else 'atan2(sin_yaw,cos_yaw)'),
             'heading_confidence':('sigmoid(2*abs(dot(facing_vector,[sin(axis),cos(axis)]))); uncalibrated, separate from object score' if axis_heading else
                 'max(softmax(heading_logits)); separate from object score' if cfg.heading_bins else 'not available'),
             'training_only_components_excluded':['ROSA teacher','track IDs','world poses','heading distillation','track consistency'],
             'size_encoding':cfg.size_encoding,'class_size_priors_LWH_m':cfg.class_size_priors,
             'size_reference_LWH_m':cfg.size_reference,
             'size_decode':('exp(clamp(size_residual + log(class_prior[predicted_class]), -3, 5))' if cfg.size_encoding=='class_relative' else
                 'exp(clamp(size_residual + log(shared_size_reference), -3, 5)); independent of predicted class' if cfg.size_encoding=='shared_relative' else 'exp(clamp(log_size, -3, 5))'),
             'invalid_candidate_handling':'mask nonfinite class, used quality, and regression values before peaks/topk; exclude zero scores',
             'score_interpretation':'ranking confidence, not a calibrated detection probability; no class-loss weights applied',
             'normalization_mean':[.485,.456,.406],'normalization_std':[.229,.224,.225],
             'image_channels':'RGB','onnx_operators':operator_sets,
             'deployment_status':'ONNX exported; requires target SDK TIDL compilation, INT8 validation, and on-device profiling',
             'bev_size_hw': [cfg.bev_height,cfg.bev_width],
             'classes': list(CLASSES), 'class_id_to_name': {str(i): s for i,s in enumerate(CLASSES)},
             'regression_channels': 8,
             'regression_semantics': ['offset_row','offset_col','z']+(
                 ['log_length_relative_to_class','log_width_relative_to_class','log_height_relative_to_class'] if cfg.size_encoding=='class_relative' else
                 ['log_length_relative_to_shared_reference','log_width_relative_to_shared_reference','log_height_relative_to_shared_reference'] if cfg.size_encoding=='shared_relative' else ['log_length','log_width','log_height'])+(
                 ['sin_2yaw','cos_2yaw'] if axis_heading else ['sin_heading_residual','cos_heading_residual'] if cfg.heading_bins else ['sin_yaw','cos_yaw']),
             'decoded_detection_semantics': ['x','y','z','length','width','height','yaw','class_id','confidence'],
             'bev_x_range_m': [cfg.bev_x_min, cfg.bev_x_max],
             'bev_y_range_m': [cfg.bev_y_min, cfg.bev_y_max],
             'bev_cell_size_m': cfg.bev_resolution}, indent=2)+'\n')
    print('Exported learned modules only; run TIDL compiler and deploy custom LUT transform.')

if __name__=='__main__':main()
