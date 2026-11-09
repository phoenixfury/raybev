"""Sparse Hungarian supervision, ignored-region protection, and TRAIN-only yaw cache."""
import json
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .config import CLASSES
from .detection import facing_vector_loss
from .rosa_dataset import training_boxes_sha256
from .training import calibration_sha256


def validate_targets(targets,cfg):
    if targets.ndim!=2 or targets.shape[1]!=8 or not torch.isfinite(targets).all():raise ValueError('Expected finite [N,8] sparse GT')
    if (targets[:,3:6]<=0).any():raise ValueError('Sparse GT dimensions must be positive')
    labels=targets[:,7]
    if ((labels!=labels.long())|(labels<-1)|(labels>=cfg.num_classes)).any():raise ValueError('Invalid sparse GT class')
    if int((labels>=0).sum())>cfg.queries:raise ValueError('More GT objects than matching queries; increase sparse query count')


def ignored_queries(boxes,ignored):
    """A conservative rotated 3D-box footprint, expanded by 0.25 m in XY.

    Only UNMATCHED query classification is masked. Known matched positives in
    these regions remain supervised. Does not modify validation GT/inference.
    """
    if not len(ignored):return torch.zeros(len(boxes),device=boxes.device,dtype=torch.bool)
    delta=boxes[:,None,:2]-ignored[None,:,:2]
    c,s=ignored[:,6].cos(),ignored[:,6].sin()
    x=delta[...,0]*c+delta[...,1]*s;y=-delta[...,0]*s+delta[...,1]*c
    return ((x.abs()<=ignored[None,:,3]/2+.25)&(y.abs()<=ignored[None,:,4]/2+.25)).any(1)


@torch.no_grad()
def match_queries(stage,gt,cfg):
    if not len(gt):return torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)
    boxes=stage['boxes'][:cfg.queries].float()
    p=stage['class_logits'][:cfg.queries].float().sigmoid().clamp(1e-6,1-1e-6)
    # Focal classification cost, meter-space centre cost and relative size.
    neg=.75*p.square()*(-(1-p).log());pos=.25*(1-p).square()*(-p.log())
    cls=(pos-neg)[:,gt[:,7].long()]
    center=torch.cdist(boxes[:,:3],gt[:,:3],p=1)
    size=torch.cdist(boxes[:,3:6].clamp_min(.05).log(),gt[:,3:6].log(),p=1)
    cost=cls+center*.25+size*.5
    if not torch.isfinite(cost).all():raise FloatingPointError('Nonfinite sparse matching cost')
    prediction,target=linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(prediction),torch.as_tensor(target)


class SparseYawTeacher:
    """Same audited yaw-cache contract, without dense BEV/cell-owner assumptions."""
    def __init__(self,path,dataset,calibration,lidar_calibration=None):
        doc=json.loads(open(path).read());self.sha256=calibration_sha256(path)
        if any(r['metainfo'].get('temporal_split')!='train' for r in dataset.rows):raise ValueError('Sparse teacher is TRAIN-only')
        if (doc.get('type')!='raybev_heading_teacher' or doc.get('split')!='train' or
                doc.get('training_boxes_sha256')!=training_boxes_sha256(dataset) or
                doc.get('camera_calibration_sha256')!=calibration_sha256(calibration) or
                doc.get('lidar_calibration_sha256')!=calibration_sha256(lidar_calibration or calibration) or
                doc.get('box_frame')!=dataset.box_frame or doc.get('z_reference')!=dataset.z_reference or
                set(doc.get('frames',{}))!={str(r['token']) for r in dataset.rows} or
                not doc.get('teacher_checkpoint_sha256') or not doc.get('predictions_sha256')):
            raise ValueError('Sparse heading teacher calibration/TRAIN annotation/provenance mismatch')
        self.frames=doc['frames']
        for row,gt in zip(dataset.rows,dataset.boxes):
            for record in self.frames[str(row['token'])]:
                value=np.asarray(record['gt'])
                if (value.shape!=(8,) or not np.isclose(gt.numpy(),value,atol=1e-4,rtol=0).all(1).any() or
                        not math.isfinite(record['yaw']) or not 0<=record['weight']<=1):
                    raise ValueError('Invalid sparse teacher GT/angle/weight')

    def targets(self,token,gt):
        yaw=gt[:,6].clone();weight=gt[:,6].new_zeros(len(gt))
        for record in self.frames.get(str(token),[]):
            value=gt.new_tensor(record['gt']);mask=torch.isclose(gt,value,atol=1e-4,rtol=0).all(1)
            yaw[mask]=record['yaw'];weight[mask]=record['weight']
        return yaw,weight


class SparseCriterion(nn.Module):
    def __init__(self,cfg,teacher=None):
        super().__init__();self.cfg=cfg;self.teacher=teacher

    def _regression(self,stage,pi,gt,denominator,token=None):
        # This zero keeps all task projections in the DDP graph on empty ranks.
        zero=sum(stage[k].float().sum()*0 for k in ('boxes','axis','facing','class_logits','quality_logits'))
        result={name:zero for name in ('center','size','axis','facing','quality','teacher')}
        if not len(gt):return result
        pred=stage['boxes'][pi].float()
        weights=gt.new_tensor(self.cfg.class_positive_weights)[gt[:,7].long()]
        def average(value):return (value*weights).sum()/denominator
        result['center']=average(F.smooth_l1_loss(pred[:,:3],gt[:,:3],beta=self.cfg.center_loss_beta,reduction='none').sum(-1))
        size_error=(pred[:,3:6].clamp_min(.05).log()-gt[:,3:6].log())
        size_loss=size_error.abs() if self.cfg.is_refined else F.smooth_l1_loss(size_error,torch.zeros_like(size_error),reduction='none')
        result['size']=average(size_loss.sum(-1))
        axis_target=torch.stack(((2*gt[:,6]).sin(),(2*gt[:,6]).cos()),-1)
        result['axis']=average(F.smooth_l1_loss(stage['axis'][pi].float(),axis_target,reduction='none').sum(-1))
        direction=torch.stack((gt[:,6].sin(),gt[:,6].cos()),-1)
        result['facing']=average(facing_vector_loss(stage['facing'][pi].float(),direction,self.cfg))
        distance=(pred[:,:2]-gt[:,:2]).norm(dim=-1)
        quality=torch.exp(-distance if self.cfg.is_refined else -distance-size_error.abs().mean(-1)).detach()
        result['quality']=average(F.binary_cross_entropy_with_logits(stage['quality_logits'][pi].float(),quality,reduction='none'))
        if self.teacher is not None and token is not None:
            yaw,confidence=self.teacher.targets(token,gt)
            axis=torch.stack(((2*yaw).sin(),(2*yaw).cos()),-1)
            direction=torch.stack((yaw.sin(),yaw.cos()),-1)
            auxiliary=F.smooth_l1_loss(stage['axis'][pi].float(),axis,reduction='none').sum(-1)
            auxiliary+=facing_vector_loss(stage['facing'][pi].float(),direction,self.cfg)
            result['teacher']=(auxiliary*confidence).sum()/denominator
        return result

    def forward(self,output,targets,tokens=None):
        cfg=self.cfg;b=len(targets);q=cfg.queries
        if not b:raise ValueError('Empty sparse minibatch')
        for gt in targets:validate_targets(gt,cfg)
        denominator=b*cfg.mean_training_objects
        totals={name:output['stages'][0]['boxes'].sum()*0 for name in ('classification','center','size','axis','facing','quality','teacher','denoising')}
        matches=0
        for stage in output['stages']:
            zero=sum(stage[k].float().sum()*0 for k in ('boxes','axis','facing','class_logits','quality_logits'))
            current={k:zero.clone() for k in totals}
            for bi,target in enumerate(targets):
                target=target.to(stage['boxes'].device);known=target[target[:,7]>=0];ignored=target[target[:,7]<0]
                single={k:v[bi] for k,v in stage.items()}
                pi,ti=match_queries(single,known,cfg);pi=pi.to(target.device);ti=ti.to(target.device)
                logits=single['class_logits'][:q].float();truth=torch.zeros_like(logits)
                truth[pi,known[ti,7].long()]=1
                keep=~ignored_queries(single['boxes'][:q].detach(),ignored);keep[pi]=True
                p=logits.sigmoid();pt=torch.where(truth.bool(),p,1-p)
                weights=torch.where(truth.bool(),logits.new_tensor(cfg.class_positive_weights)[None]*.25,.75)
                focal=F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*(1-pt).square()*weights
                current['classification']+=focal.mul(keep[:,None]).sum()/denominator
                reg=self._regression(single,pi,known[ti],denominator,None if tokens is None else tokens[bi])
                for key,value in reg.items():current[key]+=value
                if len(output['dn_indices']):
                    ids=output['dn_indices'][bi].to(target.device);dn=torch.arange(len(ids),device=target.device)+q
                    dn_truth=torch.zeros_like(single['class_logits'][dn].float())
                    dn_truth[torch.arange(len(ids),device=target.device),target[ids,7].long()]=1
                    dn_class=F.binary_cross_entropy_with_logits(single['class_logits'][dn].float(),dn_truth,reduction='none').sum()/denominator
                    dn_reg=self._regression(single,dn,target[ids],denominator)
                    current['denoising']+=dn_class+2*dn_reg['center']+dn_reg['size']+dn_reg['axis']+dn_reg['facing']+.5*dn_reg['quality']
                matches+=len(pi)
            for key in totals:totals[key]+=current[key]/len(output['stages'])
        loss=(totals['classification']+2*totals['center']+totals['size']+totals['axis']+
              totals['facing']+.5*totals['quality']+cfg.teacher_weight*totals['teacher']+cfg.dn_weight*totals['denoising'])
        return loss,{**{k:v.detach() for k,v in totals.items()},'matches':matches//len(output['stages'])}
