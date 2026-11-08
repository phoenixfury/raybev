"""Shared v2 training recipe; inference has no augmentation/sampling dependency."""
import math
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Subset


def add_training_options(parser):
    parser.add_argument('--no-pretrained',action='store_true',help='Explicit random image initialization (ablation only)')
    parser.add_argument('--backbone-weights',help='Local TorchVision MobileNetV2-1.0 state_dict for offline training')
    parser.add_argument('--backbone-lr-mult',type=float,default=.5)
    parser.add_argument('--warmup-steps',type=int,default=500,help='Optimizer steps, not microbatches')
    parser.add_argument('--no-balance',action='store_true')
    parser.add_argument('--no-augmentation',action='store_true')


def balanced_training_view(dataset,seed=3407):
    """Bounded repeats based on class presence, keeping all background frames.

    Fractional repeats are rounded once with a seeded generator, at most three
    entries per original frame. DistributedSampler then shards these entries.
    """
    present=torch.zeros(len(dataset),6,dtype=torch.bool)
    for i,boxes in enumerate(dataset.boxes):
        if len(boxes):present[i,boxes[:,-1].long().unique()]=True
    frequency=present.float().mean(dim=0).clamp(min=1/max(1,len(dataset)))
    class_repeat=torch.sqrt(.05/frequency).clamp(min=1,max=3)
    repeats=torch.where(present,class_repeat[None],torch.ones_like(present,dtype=torch.float32)).max(dim=1).values
    generator=torch.Generator().manual_seed(seed)
    counts=repeats.floor().long()+(torch.rand(len(dataset),generator=generator)<repeats.frac()).long()
    indices=torch.repeat_interleave(torch.arange(len(dataset)),counts).tolist()
    return Subset(dataset,indices)


def build_optimizer(model,lr,weight_decay,backbone_lr_mult=.5):
    if lr<=0 or weight_decay<0 or backbone_lr_mult<=0:
        raise ValueError('Invalid optimizer controls')
    groups={}
    for name,param in model.named_parameters():
        # Only the pretrained MobileNet stages use the reduced learning rate.
        backbone=name.startswith(('image_encoder.stage8.','image_encoder.stage16.','image_encoder.stage32.'))
        decay=0. if param.ndim==1 or name.endswith('.bias') else weight_decay
        key=(backbone,decay)
        groups.setdefault(key,[]).append(param)
    return torch.optim.AdamW([dict(params=params,lr=lr*(backbone_lr_mult if back else 1.),weight_decay=decay)
                              for (back,decay),params in groups.items()],lr=lr)


def build_scheduler(optimizer,total_steps,warmup_steps):
    if total_steps<1 or warmup_steps<0:
        raise ValueError('Invalid scheduler controls')
    warmup=min(warmup_steps,max(0,total_steps-1))
    def factor(step):
        if warmup and step<warmup:return (1/3)+(2/3)*step/warmup
        progress=min(1.,max(0.,(step-warmup)/max(1,total_steps-warmup)))
        return .02+.98*.5*(1+math.cos(math.pi*progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)


def check_resume(state,cfg,args):
    from .config import ModelConfig
    restored=ModelConfig.from_checkpoint(state)
    if restored!=cfg:
        raise ValueError('Resume config differs from checkpoint')
    if state.get('calibration_sha256')!=calibration_sha256(args.calibration):
        raise ValueError('Resume camera calibration differs from checkpoint; start a fresh run without --resume')
    if state.get('box_frame')!=args.box_frame:
        raise ValueError('Resume label box frame differs from checkpoint; start a fresh run without --resume')
    if args.box_frame=='CLIDAR':
        lidar_path=args.lidar_calibration or args.calibration
        if state.get('lidar_calibration_sha256')!=calibration_sha256(lidar_path):
            raise ValueError('Resume CLIDAR calibration differs or has no saved hash; start a fresh run without --resume')
    recipe=state.get('recipe',{})
    for key in ('epochs','warmup_steps','batch_size','accum_steps','no_balance',
                'no_augmentation','backbone_lr_mult','lr','weight_decay','z_reference',
                'seed','root','train_ann','val_ann'):
        if key in recipe and recipe[key]!=getattr(args,key,None):
            raise ValueError(f'Resume {key} differs from checkpoint recipe; use the original value')


def calibration_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
