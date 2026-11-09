"""TRAIN-only anchor fitting and explicit class -> track -> frame sampling."""
import math
import hashlib
import json
from collections import defaultdict
from dataclasses import replace
import numpy as np
import torch
from torch.utils.data import Sampler
from .config import CLASSES
from .detection import in_bev_range
from .rosa_dataset import boxes_to_baselink,names_to_ids


def sparse_annotation_sha256(dataset):
    """Portable resume identity includes ignored footprints and track sampling IDs.

    The older yaw-cache signature intentionally covers only known GT. That is
    insufficient for resuming a sparse run whose background masks and sampler
    also depend on ignored boxes and track identity. Image paths remain portable.
    """
    digest=hashlib.sha256()
    for i in sorted(range(len(dataset.rows)),key=lambda i:str(dataset.rows[i]['token'])):
        row=dataset.rows[i];ann=row['annos']
        value=[str(row['token']),str(row['metainfo'].get('sequence_id','')),
               np.asarray(ann.get('gt_ids',np.arange(len(ann['gt_boxes'])))).tolist(),
               dataset.boxes[i].tolist(),dataset.ignore_boxes[i].tolist()]
        digest.update(json.dumps(value,separators=(',',':'),allow_nan=False).encode());digest.update(b'\n')
    return digest.hexdigest()


def fit_sparse_anchors(dataset,cfg,seed=3407,iterations=25):
    if any(r['metainfo'].get('temporal_split')!='train' for r in dataset.rows):raise ValueError('Sparse anchors must use TRAIN only')
    arrays=[b.numpy() for b in dataset.boxes if len(b)]
    if not arrays:raise ValueError('No valid TRAIN boxes for sparse anchors')
    boxes=np.concatenate(arrays);counts=np.bincount(boxes[:,7].astype(int),minlength=len(CLASSES))
    valid=np.linalg.norm(boxes[:,:3],axis=1)>=cfg.min_range
    values=boxes[valid];labels=values[:,7].astype(int)
    if not len(values):raise ValueError('No TRAIN anchor centers beyond minimum range')
    weights=1/np.sqrt(np.maximum(counts[labels],1));weights/=weights.sum()
    rng=np.random.RandomState(seed);xy=values[:,:2].astype(np.float64)
    selected=[rng.choice(len(xy),p=weights)];distance=((xy-xy[selected[0]])**2).sum(1)
    for _ in range(1,cfg.queries):
        probability=weights*distance
        index=rng.choice(len(xy),p=probability/probability.sum()) if probability.sum()>1e-12 else rng.choice(len(xy),p=weights)
        selected.append(index);distance=np.minimum(distance,((xy-xy[index])**2).sum(1))
    centers=xy[selected].copy()
    for _ in range(iterations):
        membership=np.concatenate([((chunk[:,None]-centers[None])**2).sum(2).argmin(1) for chunk in np.array_split(xy,max(1,math.ceil(len(xy)/4096)))])
        for j in range(cfg.queries):
            keep=membership==j
            if keep.any():centers[j]=np.average(xy[keep],axis=0,weights=weights[keep])
    anchors=[]
    for center in centers:
        # Choose a real TRAIN box nearest this class-balanced spatial cluster;
        # don't average opposite headings or impose a predicted-class size prior.
        index=((xy-center)**2).sum(1).argmin();anchors.append(values[index,:7])
    fitted=replace(cfg,mean_training_objects=float(counts.sum()/len(dataset)),training_class_counts=tuple(counts.tolist())).validate()
    return fitted,torch.tensor(np.asarray(anchors),dtype=torch.float32)


class TrackBalancedSampler(Sampler):
    """30% uniform frames (including empties), 70% class/track-balanced TRAIN.

    Rare classes are explicitly sampled, unlike the dense 5%-presence recipe.
    Shared global draws are deterministically sharded without rank collisions in
    draw positions (frames may repeat intentionally). Long tracks don't dominate
    the class-balanced component. No validation labels enter sampling.
    """
    def __init__(self,dataset,seed=3407,rank=0,world_size=1):
        self.dataset=dataset;self.seed=seed;self.rank=rank;self.world=world_size;self.epoch=0
        if world_size<1 or not 0<=rank<world_size:raise ValueError('Invalid sparse sampler rank')
        self.groups=[defaultdict(list) for _ in CLASSES]
        for i,row in enumerate(dataset.rows):
            if row['metainfo'].get('temporal_split')!='train':raise ValueError('Track balancing is TRAIN-only')
            ann=row['annos'];raw=ann['gt_boxes'];classes=names_to_ids(ann['gt_names'])
            boxes=boxes_to_baselink(raw,dataset.box_frame,dataset.T_base_clidar,dataset.z_reference)
            keep=in_bev_range(boxes,dataset.cfg)&np.isfinite(boxes).all(1)&(boxes[:,3:6]>0).all(1)
            keep&=~np.asarray(ann.get('gt_ignore',np.zeros(len(boxes),bool)))
            ids=ann.get('gt_ids',np.arange(len(boxes)))
            for ci,uid,valid in zip(classes,ids,keep):
                if valid:self.groups[ci][(str(row['metainfo']['sequence_id']),str(uid))].append(i)
        self.groups=[list(group.values()) for group in self.groups]
        self.active=[ci for ci,group in enumerate(self.groups) if group and ci!=0]

    def set_epoch(self,epoch):self.epoch=epoch
    def __len__(self):return math.ceil(len(self.dataset)/self.world)
    def __iter__(self):
        rng=np.random.RandomState(self.seed+self.epoch);indices=[]
        for _ in range(len(self)*self.world):
            if not self.active or rng.rand()<.3:index=int(rng.randint(len(self.dataset)))
            else:
                ci=int(rng.choice(self.active));tracks=self.groups[ci]
                frames=tracks[int(rng.randint(len(tracks)))];index=frames[int(rng.randint(len(frames)))]
            indices.append(index)
        return iter(indices[self.rank::self.world])
