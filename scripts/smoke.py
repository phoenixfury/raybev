#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev import ModelConfig, RayBEVLite, CLASSES
from raybev.detection import make_targets,detection_loss,decode_detections

cfg=ModelConfig().downscale(5)
model=RayBEVLite(cfg,str(Path(__file__).resolve().parents[1]/'configs/calibration_baselink.json'))
model.eval()
torch.set_num_threads(min(4,torch.get_num_threads()))
with torch.no_grad():
    out=model(torch.zeros((1,4,3,cfg.image_height,cfg.image_width)))
print('Image',cfg.image_width,cfg.image_height, 'BEV',cfg.bev_height,cfg.bev_width)
print('Classes',list(enumerate(CLASSES)))
print('Output shapes',*[tuple(v.shape) for v in out])
print('Parameters',sum(p.numel() for p in model.parameters()))
boxes=[torch.tensor([[5.,2.,0.5,4.2,1.8,1.6,0.3,0.]])]
targets=make_targets(boxes,cfg)
loss,parts=detection_loss(out,targets,cfg=cfg)
print('Finite loss',loss.item(),{k:v.item() for k,v in parts.items()})
print('Predicted boxes (untrained)',len(decode_detections(out,cfg)[0]))
