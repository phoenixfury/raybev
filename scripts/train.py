#!/usr/bin/env python3
"""Minimal supervised trainer for preconverted four-camera NPZ samples.

Each NPZ sample contains images [4,3,H,W] (float32 normalized or raw uint8)
and boxes [N,8] in CLIDAR by default: x,y,z,l,w,h,yaw,class_id.
Use --box-frame BASELINK only for labels already converted to BASELINK.
Legacy [N,10] annotations including vx,vy are accepted but the velocities
are discarded before target generation.
Images must be in ARC_F, ARC_R, ARC_SL, ARC_SR order.
Raw images are converted into [0,1] then resized to 800x640 and
normalized by ImageNet statistics. Pre-normalized images are not accepted.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from raybev import ModelConfig, RayBEVLite, CLASSES
from raybev.config import DEFAULT_BOX_FRAME,DEFAULT_CALIBRATION
from raybev.rosa_dataset import boxes_to_baselink,read_T_base_clidar
from raybev.preprocessing import letterbox_images, normalize_imagenet
from raybev.detection import make_targets,detection_loss,in_bev_range
from raybev.training import add_training_options,build_optimizer,build_scheduler,calibration_sha256,fit_training_config,initialize_recovery


class NPZDataset(Dataset):
    def __init__(self, folder, cfg, box_frame=DEFAULT_BOX_FRAME, calibration=None, z_reference='center'):
        self.samples=sorted(Path(folder).glob('*.npz'))
        if not self.samples:
            raise FileNotFoundError(f'No .npz samples in {folder}')
        self.cfg=cfg
        self.box_frame=box_frame.upper()
        self.T_base_clidar=read_T_base_clidar(calibration) if self.box_frame=='CLIDAR' else None
        self.z_reference=z_reference
    def __len__(self):return len(self.samples)
    def read_boxes(self,index):
        with np.load(self.samples[index]) as data:
            boxes=torch.from_numpy(np.array(data['boxes'],dtype=np.float32))
        if boxes.ndim!=2 or boxes.shape[1] not in (8,10):
            raise ValueError('Expected boxes [N,8] or legacy [N,10]')
        if boxes.shape[1] == 10:
            # Drop former vx, vy columns, retain the original class ID.
            boxes = boxes[:, [0,1,2,3,4,5,6,9]]
        boxes[:,:7]=torch.as_tensor(boxes_to_baselink(boxes[:,:7].numpy(),self.box_frame,
                                                    self.T_base_clidar,self.z_reference),dtype=boxes.dtype)
        return boxes[in_bev_range(boxes,self.cfg)&torch.isfinite(boxes).all(dim=1)&(boxes[:,3:6]>0).all(dim=1)]
    def __getitem__(self,index):
        with np.load(self.samples[index]) as data:ims=torch.from_numpy(np.array(data['images'])).float()
        if ims.ndim!=4 or ims.shape[:2]!=(4,3):raise ValueError('Expected images [4,3,H,W]')
        boxes=self.read_boxes(index)
        if ims.max()>1.5:ims/=255.
        ims=letterbox_images(ims[None],self.cfg.image_width,self.cfg.image_height)
        return normalize_imagenet(ims)[0],boxes


def build_parser():
    a=argparse.ArgumentParser()
    a.add_argument('--data',required=True)
    a.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    a.add_argument('--box-frame',choices=['BASELINK','CLIDAR'],default=DEFAULT_BOX_FRAME)
    a.add_argument('--lidar-calibration',default=None,help='Defaults to --calibration')
    a.add_argument('--z-reference',choices=['center','bottom'],default='center')
    a.add_argument('--epochs',type=int,default=10)
    a.add_argument('--batch-size',type=int,default=2)
    a.add_argument('--lr',type=float,default=2e-4)
    a.add_argument('--checkpoint',default='raybev_v4_3_r18_geometry.pt')
    a.add_argument('--pretrained',action='store_true',help='Retained for compatibility; pretrained is now the default')
    add_training_options(a)
    return a


def main():
    args=build_parser().parse_args()
    args.lidar_calibration=args.lidar_calibration or args.calibration
    cfg=ModelConfig(backbone=args.backbone,bev_normalization=args.bev_normalization,height_fusion=args.height_fusion,
                    heavy_classification_weight=args.heavy_classification_weight).validate()
    if args.heading_teacher:raise ValueError('NPZ inputs lack tokens/track poses; use the combined trainer for ROSA distillation')
    print(f'Backbone={cfg.backbone}; heading={cfg.heading_encoding}; NPZ teacher/track supervision unavailable',flush=True)
    print('Foreground/box class weights:',dict(zip(CLASSES,zip(cfg.class_positive_weights,cfg.class_regression_weights))),flush=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    dataset=NPZDataset(args.data,cfg,args.box_frame,args.lidar_calibration,args.z_reference)
    cfg=fit_training_config(cfg,(dataset.read_boxes(i) for i in range(len(dataset))))
    print(f'TRAIN mean objects/frame={cfg.mean_training_objects:.6f}; shared size reference={cfg.size_reference}',flush=True)
    dataset.cfg=cfg
    print('TRAIN-only class size priors:',dict(zip(CLASSES,cfg.class_size_priors)),flush=True)
    model=RayBEVLite(cfg,args.calibration,pretrained=not args.no_pretrained and not args.init_from,
                     backbone_weights=args.backbone_weights if not args.init_from else None).to(device)
    recovery=initialize_recovery(model,cfg,args)
    loader=DataLoader(dataset,batch_size=args.batch_size,
                      shuffle=True,collate_fn=lambda examples:([x[0] for x in examples],
                                                                 [x[1] for x in examples]))
    from raybev.training import configure_image_training
    configure_image_training(model,args.freeze_image_stages)
    optimizer=build_optimizer(model,args.lr,.001,args.backbone_lr_mult)
    scheduler=build_scheduler(optimizer,len(loader)*args.epochs,args.warmup_steps)
    for epoch in range(args.epochs):
        model.train()
        configure_image_training(model,args.freeze_image_stages)
        running=0.
        for image_list,box_list in loader:
            images=torch.stack(image_list).to(device)
            targets=make_targets(box_list,cfg,device=device,return_class_ids=True)
            optimizer.zero_grad(set_to_none=True)
            outputs=model(images)
            loss,parts=detection_loss(outputs,targets,cfg=cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.)
            optimizer.step()
            scheduler.step()
            running+=loss.item()
        print(f'epoch={epoch+1} loss={running/len(loader):.4f}')
        torch.save({'state_dict':model.state_dict(),'cfg':vars(cfg),
                    'epoch':epoch+1,'classes':list(CLASSES),'recipe':vars(args),
                    'box_frame':args.box_frame,
                    'lidar_calibration_sha256':calibration_sha256(args.lidar_calibration) if args.box_frame=='CLIDAR' else None,
                    'backbone_initialization':model.image_encoder.initialization,'warm_start':recovery,
                    'calibration_sha256':calibration_sha256(args.calibration)},args.checkpoint)

if __name__=='__main__':main()
