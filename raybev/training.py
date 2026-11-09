"""Shared v2 training recipe; inference has no augmentation/sampling dependency."""
import math
import hashlib
from dataclasses import replace
from pathlib import Path
import torch
from torch.utils.data import Subset
from .config import CLASSES,ModelConfig
from .detection import in_bev_range


def add_training_options(parser):
    parser.add_argument('--backbone',choices=['resnet18','mobilenet_v2'],default='resnet18')
    parser.add_argument('--bev-normalization',choices=['batchnorm','frozen_batchnorm'],default='frozen_batchnorm')
    parser.add_argument('--heavy-classification-weight',type=float,default=0.)
    parser.add_argument('--init-from',help='Trusted local checkpoint: weights-only recovery, new optimizer/schedule/output; NOT --resume')
    parser.add_argument('--heading-teacher',help='Verified TRAIN-only ROSA heading cache from prepare_heading_teacher.py')
    parser.add_argument('--heading-teacher-weight',type=float,default=.2)
    parser.add_argument('--track-heading-weight',type=float,default=.1)
    parser.add_argument('--no-pretrained',action='store_true',help='Explicit random image initialization (ablation only)')
    parser.add_argument('--backbone-weights',help='Matching TorchVision ResNet-18 or MobileNetV2 state_dict for offline training')
    parser.add_argument('--backbone-lr-mult',type=float,default=.5)
    parser.add_argument('--warmup-steps',type=int,default=500,help='Optimizer steps, not microbatches')
    parser.add_argument('--no-balance',action='store_true')
    parser.add_argument('--no-augmentation',action='store_true')


def fit_training_config(cfg,batch_boxes):
    """Fit size references and mean object count ONLY from training labels.

    Never call this on validation/test data. Missing training classes retain
    the explicitly configured fallback prior and are reported in the counts.
    """
    grouped=[[] for _ in CLASSES];frames=0
    for boxes in batch_boxes:
        frames+=1
        boxes=torch.as_tensor(boxes,dtype=torch.float32).detach().cpu()
        valid=in_bev_range(boxes,cfg)&torch.isfinite(boxes).all(dim=1)&(boxes[:,3:6]>0).all(dim=1)
        boxes=boxes[valid]
        if len(boxes) and ((boxes[:,7]<0)|(boxes[:,7]>=len(CLASSES))|(boxes[:,7]!=boxes[:,7].long())).any():
            raise ValueError('Invalid training class IDs')
        for ci in range(len(CLASSES)):
            selected=boxes[boxes[:,7]==ci,3:6]
            if len(selected):grouped[ci].append(selected)
    counts=[];priors=[]
    for ci,items in enumerate(grouped):
        values=torch.cat(items) if items else torch.empty(0,3)
        counts.append(len(values))
        priors.append(tuple(torch.quantile(values,.5,dim=0).tolist()) if len(values) else tuple(cfg.class_size_priors[ci]))
    if not sum(counts):raise ValueError('No valid training objects to fit the task recipe')
    all_sizes=torch.cat([torch.cat(items) for items in grouped if items])
    return replace(cfg,training_class_counts=tuple(counts),mean_training_objects=sum(counts)/frames,
                   size_reference=tuple(torch.quantile(all_sizes,.5,dim=0).tolist()),
                   class_size_priors=tuple(priors) if cfg.size_encoding in ('class_relative','shared_relative') else cfg.class_size_priors).validate()


def initialize_recovery(model,cfg,args):
    """Strict weight migration; never restores optimizer, epoch or best metrics."""
    source=getattr(args,'init_from',None)
    if not source:return None
    if getattr(args,'resume',None):raise ValueError('--init-from and --resume are mutually exclusive')
    state=torch.load(source,map_location='cpu',weights_only=False)
    old=ModelConfig.from_checkpoint(state)
    allowed={'bev_normalization','heavy_classification_weight','loss_normalization',
             'mean_training_objects','size_encoding','size_reference','class_size_priors',
             'classification_mode','foreground_label_smoothing','center_classification_weight'}
    differences={k for k,v in vars(cfg).items() if v!=getattr(old,k)}
    if differences-allowed:raise ValueError(f'Recovery architecture/data config mismatch: {sorted(differences-allowed)}')
    if (tuple(state.get('classes',()))!=CLASSES or state.get('box_frame')!=args.box_frame or
            state.get('calibration_sha256')!=calibration_sha256(args.calibration) or
            state.get('recipe',{}).get('z_reference',args.z_reference)!=args.z_reference):
        raise ValueError('Recovery class/frame/calibration contract mismatch')
    if args.box_frame=='CLIDAR' and state.get('lidar_calibration_sha256')!=calibration_sha256(args.lidar_calibration or args.calibration):
        raise ValueError('Recovery CLIDAR calibration mismatch')
    weights=dict(state['state_dict']);current=model.state_dict()
    added=[]
    if old.classification_mode=='heatmap' and cfg.classification_mode=='objectness_softmax':
        for key in current:
            if key.startswith('bev_detector.objectness.'):
                original=key.replace('bev_detector.objectness.','bev_detector.heatmap.',1)
                value=weights[original]
                if key in ('bev_detector.objectness.1.weight','bev_detector.objectness.1.bias'):
                    value=value[0:1]  # warm-start generic centers from the learned Car detector
                weights[key]=value.clone();added.append(key)
    if set(weights)!=set(current):raise ValueError('Recovery state keys differ; refusing partial model load')
    reset=[]
    if (old.size_encoding!=cfg.size_encoding or
            (cfg.size_encoding=='shared_relative' and old.size_reference!=cfg.size_reference) or
            (cfg.size_encoding=='class_relative' and old.class_size_priors!=cfg.class_size_priors)):
        # Switching shared/class priors isn't one constant across examples.
        # Retain size features, but relearn the final residual projection.
        reset=['bev_detector.size.1.weight','bev_detector.size.1.bias']
        if cfg.size_encoding not in ('shared_relative','class_relative'):raise ValueError('Unsupported recovery size migration')
        for key in reset:weights[key]=torch.zeros_like(current[key])
    model.load_state_dict(weights,strict=True)
    model.image_encoder.initialization=state.get('backbone_initialization','recovery_checkpoint')
    report=dict(checkpoint=str(source),sha256=calibration_sha256(source),epoch=state.get('epoch'),
                changed_config=sorted(differences),reset_parameters=reset,initialized_parameters=added)
    print('Weights-only recovery:',report,flush=True)
    return report


def balanced_training_view(dataset,seed=3407,cfg=None):
    """Bounded repeats based on class presence, keeping all background frames.

    Fractional repeats are rounded once with a seeded generator, at most three
    entries per original frame. DistributedSampler then shards these entries.
    """
    cfg=cfg or ModelConfig()
    present=torch.zeros(len(dataset),len(CLASSES),dtype=torch.bool)
    objects=torch.zeros(len(CLASSES))
    for i,boxes in enumerate(dataset.boxes):
        if len(boxes):
            labels=boxes[:,-1].long()
            present[i,labels.unique()]=True
            objects+=torch.bincount(labels,minlength=len(CLASSES))
    if cfg.sampling_mode=='object_frequency':
        frequency=objects/objects.sum().clamp(min=1)
        class_repeat=torch.sqrt(.1/frequency.clamp(min=1/max(1,float(objects.sum())))).clamp(min=1,max=3)
    else:
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
        # Only the pretrained image stages use the reduced learning rate.
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
    teacher=getattr(args,'heading_teacher',None)
    if state.get('heading_teacher_sha256')!=(calibration_sha256(teacher) if teacher else None):
        raise ValueError('Resume heading teacher differs from checkpoint')
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
                'seed','root','train_ann','val_ann','heading_teacher_weight','track_heading_weight','backbone'):
        if key in recipe and recipe[key]!=getattr(args,key,None):
            raise ValueError(f'Resume {key} differs from checkpoint recipe; use the original value')


def calibration_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heading_selection_score(diagnostics):
    """High-confidence car heading checkpoint, guarded against tiny recall."""
    car=diagnostics['by_score']['0.3']['car']
    if car['matched_count']<50 or car['recall']<.1 or car['AOE'] is None:return float('inf')
    return car['AOE']
