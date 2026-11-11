"""Score/class agnostic maximum-cardinality XY coverage and facing reliability."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import CLASSES


def raw_query_counts(candidates,truth):
    """Each query can cover at most one GT, regardless of class or confidence."""
    truth=np.asarray(truth);truth=truth[truth[:,7]>=0]
    candidates=np.asarray(candidates);candidates=candidates[np.isfinite(candidates).all(-1)]
    distance=np.linalg.norm(candidates[:,None,:2]-truth[None,:,:2],axis=-1)
    radius=np.linalg.norm(truth[:,:2],axis=-1);result={}
    for threshold in (.5,1.,2.,4.):
        hit=np.zeros(len(truth),bool)
        if len(candidates) and len(truth):
            # Cardinality first, distance second. Ordinary distance-only
            # Hungarian matching can lose valid pairs at a tight threshold.
            cost=(distance>=threshold)*(len(truth)+1)+np.minimum(distance/threshold,1)
            pi,ti=linear_sum_assignment(cost);hit[ti[distance[pi,ti]<threshold]]=True
        for ci,name in enumerate(CLASSES):
            for label,region in [('all',radius>=0),('0_20m',radius<20),('20_35m',(radius>=20)&(radius<35)),('35m_plus',radius>=35)]:
                keep=(truth[:,7]==ci)&region
                result[f'{name}/{label}/{threshold:g}m']=[int((hit&keep).sum()),int(keep.sum())]
    return result


def merge_raw_counts(records):
    merged={}
    for record in records:
        for stage,counts in record.items():
            target=merged.setdefault(stage,{})
            for key,(hit,total) in counts.items():
                pair=target.setdefault(key,[0,0]);pair[0]+=hit;pair[1]+=total
    return {stage:{key:dict(matched=n,total=d,recall=n/d if d else None) for key,(n,d) in counts.items()}
            for stage,counts in merged.items()}


def output_raw_counts(output,targets,cfg):
    stages={'initial':output['initial_boxes'].detach().float().cpu().numpy()}
    stages.update({f'stage_{i+1}':s['boxes'][:,:cfg.queries].detach().float().cpu().numpy() for i,s in enumerate(output['stages'])})
    return [{name:raw_query_counts(value[bi],gt.cpu().numpy()) for name,value in stages.items()} for bi,gt in enumerate(targets)]


def facing_samples(output,targets,cfg):
    """Pre-NMS, class-correct, >=.1 score, one-to-one XY<2m matches."""
    import torch
    from .detection import decode_heading,in_bev_range
    final=output['stages'][-1];q=cfg.queries
    probability=final['class_logits'][:,:q].detach().float().sigmoid()
    score,labels=probability.max(-1)
    score*=final['quality_logits'][:,:q].detach().float().sigmoid().pow(cfg.quality_score_power)
    if 'facing_confidence' in final:confidence=final['facing_confidence'][:,:q].detach()
    else:_,confidence=decode_heading(final['axis'][:,:q].detach(),final['facing'][:,:q].detach(),cfg)
    result=[]
    for bi,gt in enumerate(targets):
        boxes=final['boxes'][bi,:q].detach().float()
        valid=final['supported'][bi,:q]&(score[bi]>=.1)&in_bev_range(boxes,cfg)&torch.isfinite(boxes).all(-1)
        boxes=boxes.cpu().numpy();truth=gt.cpu().numpy();valid=valid.cpu().numpy();label=labels[bi].cpu().numpy()
        conf=confidence[bi].float().cpu().numpy()
        for ci in range(len(CLASSES)):
            pi=np.flatnonzero(valid&(label==ci));ti=np.flatnonzero(truth[:,7]==ci)
            result.append(dict(class_id=ci,gt=len(ti),samples=[]))
            if not len(pi) or not len(ti):continue
            distance=np.linalg.norm(boxes[pi,None,:2]-truth[None,ti,:2],axis=-1)
            a,b=linear_sum_assignment((distance>=2)*(len(ti)+1)+np.minimum(distance/2,1))
            good=distance[a,b]<2;a=pi[a[good]];b=ti[b[good]]
            error=np.abs(np.arctan2(np.sin(boxes[a,6]-truth[b,6]),np.cos(boxes[a,6]-truth[b,6])))
            result[-1]['samples']=[[float(conf[j]),float(e<np.pi/2),float(min(e,np.pi-e)),float(e)] for j,e in zip(a,error)]
    return result


def summarize_facing(records):
    result={}
    for ci,name in enumerate(CLASSES):
        selected=[r for r in records if r['class_id']==ci];total=sum(r['gt'] for r in selected)
        samples=np.asarray([s for r in selected for s in r['samples']]).reshape(-1,4)
        n=len(samples);row=dict(gt_count=total,matched_count=n,coverage=n/total if total else None,
            brier=None,ece=None,unsigned_axis_AOE_deg=None,directed_AOE_deg=None,flip_rate=None)
        if n:
            conf,correct,axis,directed=samples.T;ece=0.
            for i in range(10):
                keep=(conf>=i/10)&((conf<(i+1)/10) if i<9 else (conf<=1))
                if keep.any():ece+=keep.mean()*abs(conf[keep].mean()-correct[keep].mean())
            row.update(brier=float(((conf-correct)**2).mean()),ece=float(ece),
                unsigned_axis_AOE_deg=float(np.degrees(axis.mean())),directed_AOE_deg=float(np.degrees(directed.mean())),
                flip_rate=float(1-correct.mean()))
        result[name]=row
    return dict(protocol='pre-NMS class-correct XY<2m, score>=0.1, maximum-cardinality matching',by_class=result)
