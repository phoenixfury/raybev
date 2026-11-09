"""Training-only, sparse ROSA response distillation and track-heading references.

No teacher, poses, track memory or extra image forward is needed at inference.
Track references are circular means of neighboring TRAIN labels in world axes,
not student-to-student temporal feature learning or a temporal inference model.
"""
import json
import math
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F
from .config import CLASSES
from .detection import metric_to_cell,in_bev_range,loss_denominator,facing_vector_loss
from .rosa_dataset import boxes_to_baselink,names_to_ids,read_T_base_clidar,training_boxes_sha256
from .training import calibration_sha256


def track_references(dataset,calibration):
    """Neighbor-label smoothing only; no validation rows, no motion extrapolation."""
    tracks=defaultdict(list)
    base_from_lidar=read_T_base_clidar(calibration)
    for row,boxes in zip(dataset.rows,dataset.boxes):
        meta=row['metainfo']
        if meta.get('temporal_split')!='train':raise ValueError('Heading supervision accepts TRAIN rows only')
        if 'T_world_CLIDAR' not in meta or 'gt_ids' not in row['annos']:continue
        raw=row['annos']['gt_boxes']
        base=boxes_to_baselink(raw,dataset.box_frame,dataset.T_base_clidar,dataset.z_reference)
        classes=names_to_ids(row['annos']['gt_names'])
        keep=np.isfinite(base).all(1)&(base[:,3:6]>0).all(1)
        keep&=~np.asarray(row['annos'].get('gt_ignore',np.zeros(len(base),dtype=bool)),dtype=bool)
        ids=np.asarray(row['annos']['gt_ids'])
        if len(ids)!=len(base):raise ValueError('Heading track ID/box length mismatch')
        R=(np.asarray(meta['T_world_CLIDAR'])@np.linalg.inv(base_from_lidar))[:3,:3]
        timestamp=int(meta['acquisition_timestamp_ns'])*1e-9
        token=str(row['token'])
        for b,ci,uid,valid in zip(base,classes,ids,keep):
            if not valid or ci==1 or not bool(in_bev_range(torch.tensor(b[None]),dataset.cfg)[0]):continue
            direction=R@np.array([math.cos(b[6]),math.sin(b[6]),0.])
            yaw=math.atan2(direction[1],direction[0])
            gt=np.r_[b,ci].tolist()
            tracks[(meta['sequence_id'],str(uid),int(ci))].append((timestamp,token,gt,yaw,R))
    result=defaultdict(list)
    for observations in tracks.values():
        observations.sort(key=lambda x:x[0])
        for i,(time,token,gt,yaw,R) in enumerate(observations):
            neighbors=[]
            for j in (i-1,i+1):
                if not 0<=j<len(observations):continue
                t,_,_,angle,_=observations[j]
                delta=abs(math.atan2(math.sin(angle-yaw),math.cos(angle-yaw)))
                if 0<abs(t-time)<=1.5 and delta<=math.radians(20):neighbors.append(angle)
            if not neighbors:continue
            world=np.array([np.mean(np.cos(neighbors)),np.mean(np.sin(neighbors)),0.])
            local=R.T@world
            result[token].append({'gt':gt,'yaw':math.atan2(local[1],local[0]),'weight':1.})
    return dict(result)


class HeadingSupervision:
    def __init__(self,dataset,cfg,args):
        self.cfg=cfg
        self.teacher_weight=args.heading_teacher_weight if args.heading_teacher else 0.
        self.track_weight=args.track_heading_weight
        if any(not math.isfinite(w) or w<0 for w in (args.heading_teacher_weight,self.track_weight)):
            raise ValueError('Heading auxiliary weights must be finite and nonnegative')
        if cfg.heading_encoding!='axis_direction' and (self.teacher_weight or self.track_weight):
            raise ValueError('Heading auxiliary losses require axis_direction encoding')
        if any(r['metainfo'].get('temporal_split')!='train' for r in dataset.rows):
            raise ValueError('Heading supervision accepts TRAIN rows only')
        self.teacher={};self.teacher_sha256=None
        if args.heading_teacher:
            data=json.loads(open(args.heading_teacher).read())
            tokens={str(r['token']) for r in dataset.rows}
            # New caches survive pickle/path rewriting on the server, but every
            # training token and GT box must still match. Old caches stay strict.
            semantic_hash=data.get('training_boxes_sha256')
            annotation_ok=(semantic_hash==training_boxes_sha256(dataset) if semantic_hash is not None else
                           data.get('annotation_sha256')==calibration_sha256(dataset.annotation_path))
            if (data.get('type')!='raybev_heading_teacher' or data.get('split')!='train' or
                    not annotation_ok or
                    data.get('camera_calibration_sha256')!=calibration_sha256(args.calibration) or
                    data.get('lidar_calibration_sha256')!=calibration_sha256(args.lidar_calibration or args.calibration) or
                    data.get('box_frame')!=dataset.box_frame or data.get('z_reference')!=dataset.z_reference or
                    set(data.get('frames',{}))!=tokens):
                raise ValueError('Heading teacher split/token/calibration/annotation contract mismatch')
            if not data.get('teacher_checkpoint_sha256') or not data.get('predictions_sha256'):
                raise ValueError('Heading teacher provenance hashes missing')
            self.teacher=data['frames'];self.teacher_sha256=calibration_sha256(args.heading_teacher)
        self.track=track_references(dataset,args.lidar_calibration or args.calibration) if self.track_weight else {}
        self.teacher=self._sparse(self.teacher,dataset)
        self.track=self._sparse(self.track,dataset)
        print(f'Heading supervision: teacher_centers={sum(map(len,self.teacher.values()))}, '
              f'teacher_weight={self.teacher_weight}; track_centers={sum(map(len,self.track.values()))}, '
              f'track_weight={self.track_weight}',flush=True)

    def _sparse(self,records,dataset):
        result={}
        for sample,boxes in zip(dataset.rows,dataset.boxes):
            token=str(sample['token']);owners={}
            for gt in boxes.tolist():
                r,c=metric_to_cell(gt[0],gt[1],self.cfg)
                row,col=min(self.cfg.bev_height-1,math.floor(r+.5)),min(self.cfg.bev_width-1,math.floor(c+.5))
                distance=(r-row)**2+(c-col)**2
                if (row,col) not in owners or distance<owners[row,col][0]:owners[row,col]=(distance,gt)
            kept=[]
            for item in records.get(token,[]):
                gt=item['gt'];r,c=metric_to_cell(gt[0],gt[1],self.cfg)
                row,col=min(self.cfg.bev_height-1,math.floor(r+.5)),min(self.cfg.bev_width-1,math.floor(c+.5))
                if not 0<=row<self.cfg.bev_height or not 0<=col<self.cfg.bev_width:continue
                if (row,col) not in owners or not np.allclose(gt,owners[row,col][1],atol=1e-4):continue
                weight=float(item['weight']);yaw=float(item['yaw'])
                if not math.isfinite(yaw) or not math.isfinite(weight) or not 0<=weight<=1:
                    raise ValueError('Invalid sparse heading teacher target')
                kept.append((row,col,yaw,weight))
            result[token]=kept
        return result

    def _loss(self,records,tokens,output,targets):
        indices=[];angles=[];weights=[]
        for bi,token in enumerate(tokens):
            for row,col,yaw,weight in records.get(str(token),[]):
                indices.append((bi,row,col));angles.append(yaw);weights.append(weight)
        zero=output[1].float().sum()*0.+output[3].float().sum()*0.
        if not angles:return zero,0
        indices=torch.tensor(indices,device=zero.device,dtype=torch.long)
        b,r,c=indices.unbind(dim=1)
        axis=output[1].float().permute(0,2,3,1)[b,r,c][:,6:8]
        facing=output[3].float().permute(0,2,3,1)[b,r,c]
        yaw=zero.new_tensor(angles);w=zero.new_tensor(weights)
        axis_target=torch.stack(((2*yaw).sin(),(2*yaw).cos()),dim=1)
        axis_loss=F.smooth_l1_loss(axis,axis_target,reduction='none').sum(1)
        direction_loss=facing_vector_loss(facing,torch.stack((yaw.sin(),yaw.cos()),dim=1),self.cfg)
        # Bounded sparse auxiliaries; do not reintroduce inverse-class/match counts.
        return ((axis_loss+direction_loss)*w).sum()/loss_denominator(targets[2],self.cfg),len(angles)

    def loss(self,output,targets,tokens):
        teacher,nt=self._loss(self.teacher,tokens,output,targets)
        track,nr=self._loss(self.track,tokens,output,targets)
        return self.teacher_weight*teacher+self.track_weight*track,{
            'heading_teacher':teacher.detach(),'heading_track':track.detach(),
            'teacher_centers':nt,'track_centers':nr}
