"""Opt-in controlled heavy-object experiments; no class-conditioned size lookup."""
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .config import CLASSES
from .rosa_dataset import training_boxes_sha256
from .training import calibration_sha256


def shape_hypotheses(boxes,count):
    """Geometry-only bins, 60% compact coverage, independent of semantic labels."""
    values=np.asarray(boxes);length=values[:,3]
    membership=np.digitize(length,[2.5,5.5,8.5,14.])
    budgets=np.array([.08,.60,.12,.12,.08])
    active=np.array([(membership==i).any() for i in range(5)])
    budgets=budgets*active;budgets/=budgets.sum()
    quotas=np.floor(count*budgets).astype(int)
    for i in np.argsort(-(count*budgets-quotas),kind='stable')[:count-quotas.sum()]:quotas[i]+=1
    # Interleave the groups, including the proposal tail, rather than sorting
    # all long hypotheses into a single spatial sector or contiguous slots.
    sizes=[];used=np.zeros(5,int)
    medians={i:np.exp(np.median(np.log(values[membership==i,3:6]),axis=0)) for i in np.flatnonzero(active)}
    for slot in range(count):
        remaining=quotas-used
        priorities=np.where(remaining>0,(slot+1)*budgets-used,-np.inf)
        group=int(priorities.argmax());sizes.append(medians[group]);used[group]+=1
    return np.asarray(sizes,dtype=np.float32)


def foreground_weights(labels,cfg):
    if not cfg.foreground_balance or not len(labels):return torch.ones_like(labels,dtype=torch.float32)
    counts=torch.tensor(cfg.training_class_counts,device=labels.device,dtype=torch.float32).clamp_min(1)
    weights=(counts.max()/counts[labels]).sqrt().clamp_max(4.)
    return weights/weights.mean().clamp_min(1e-8)


def semantic_loss(logits,labels,cfg):
    if not len(labels):return logits.sum()*0
    return (F.cross_entropy(logits.float(),labels,reduction='none')*foreground_weights(labels,cfg)).mean()


def objectness_loss(logits,matched,keep):
    """Separate positive-object and background means; no semantic class weights."""
    truth=torch.zeros_like(logits);truth[matched]=1
    losses=F.binary_cross_entropy_with_logits(logits.float(),truth,reduction='none')
    positive=truth.bool();negative=keep&~positive
    return .5*((losses*positive).sum()/positive.sum().clamp_min(1)+
               (losses*negative).sum()/negative.sum().clamp_min(1))


class GeometryApprovals:
    """Explicit TRAIN-only cuboid approvals, not inferred from class review."""
    def __init__(self,path,dataset,calibration):
        self.sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()
        document=json.loads(Path(path).read_text())
        if (document.get('type')!='raybev_geometry_approvals_v1' or
                document.get('training_boxes_sha256')!=training_boxes_sha256(dataset) or
                document.get('calibration_sha256')!=calibration_sha256(calibration)):
            raise ValueError('Geometry approval dataset/calibration identity mismatch')
        available={str(row['token']):boxes.numpy() for row,boxes in zip(dataset.rows,dataset.boxes)}
        self.boxes={}
        for entry in document.get('approvals',[]):
            token=str(entry['token']);gt=np.asarray(entry['gt'],np.float32)
            if entry.get('geometry_reviewed') is not True or not isinstance(entry.get('evidence'),str) or not entry['evidence'].strip():
                raise ValueError('Cuboid approval requires geometry review and evidence')
            if gt.shape!=(8,) or token not in available or not np.isclose(available[token],gt,atol=1e-4,rtol=0).all(-1).any():
                raise ValueError('Geometry approval is not a current TRAIN cuboid')
            if int(gt[7]) not in (3,4,5):raise ValueError('Endpoint approvals must identify credible heavy cuboids')
            self.boxes.setdefault(token,[]).append(gt.tolist())
        if not self.boxes:raise ValueError('No explicitly approved credible TRAIN cuboids; endpoint arm is blocked')

    def mask(self,token,gt):
        approved=gt.new_tensor(self.boxes.get(str(token),[])).reshape(-1,8)
        return torch.isclose(gt[:,None],approved[None],atol=1e-4,rtol=0).all(-1).any(-1)


def summarize_supervision(stats):
    fields=('gt','matched','supported','argmax_correct','within_2m','correct_within_2m','semantic_weight','semantic_nll_sum')
    values=stats.detach().cpu().tolist();result={}
    for stage,rows in enumerate(values,1):
        result[f'stage_{stage}']={}
        for name,row in zip(CLASSES,rows):
            item=dict(zip(fields,row));den=item['matched']
            item['matched_accuracy']=item['argmax_correct']/den if den else None
            item['supported_fraction']=item['supported']/den if den else None
            item['matched_semantic_nll']=item['semantic_nll_sum']/den if den else None
            result[f'stage_{stage}'][name]=item
    return result


def geometry_stage_confusion(output,targets,cfg):
    """Score/class-agnostic XY matching, separate from training assignment."""
    from scipy.optimize import linear_sum_assignment
    counts=np.zeros((cfg.stages,cfg.num_classes,cfg.num_classes+1),dtype=np.int64)
    for si,stage in enumerate(output['stages']):
        boxes=stage['boxes'][:,:cfg.queries].detach().float().cpu().numpy()
        labels=stage['class_logits'][:,:cfg.queries].detach().float().argmax(-1).cpu().numpy()
        supported=stage['supported'][:,:cfg.queries].detach().cpu().numpy()
        for bi,target in enumerate(targets):
            gt=target.detach().cpu().numpy();gt=gt[gt[:,7]>=0]
            valid=np.flatnonzero(supported[bi]&np.isfinite(boxes[bi]).all(-1));hit={}
            if len(valid) and len(gt):
                distance=np.linalg.norm(boxes[bi,valid,None,:2]-gt[None,:,:2],axis=-1)
                pi,ti=linear_sum_assignment((distance>=2)*(len(gt)+1)+np.minimum(distance/2,1))
                hit={int(j):int(labels[bi,valid[i]]) for i,j in zip(pi,ti) if distance[i,j]<2}
            for i,g in enumerate(gt):counts[si,int(g[7]),hit.get(i,cfg.num_classes)]+=1
    return counts


def summarize_stage_confusion(values):
    values=values.detach().cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
    result={}
    for si,rows in enumerate(values,1):
        result[f'stage_{si}']={name:dict(gt=int(row.sum()),geometry_covered=int(row[:-1].sum()),
            correct=int(row[ci]),classification_accuracy_given_coverage=float(row[ci]/row[:-1].sum()) if row[:-1].sum() else None,
            confusion=dict(zip((*CLASSES,'no_supported_2m_match'),map(int,row)))) for ci,(name,row) in enumerate(zip(CLASSES,rows))}
    return result


def fixed_class_recall(predictions,truths,ci,recall):
    from .metrics import _match_class
    gt={i:t[t[:,7]==ci,:7] for i,t in enumerate(truths)}
    n=sum(map(len,gt.values()))
    if not n:return dict(status='no_gt')
    candidates=[(i,p[:7],float(p[8])) for i,ps in enumerate(predictions) for p in ps if p[7]==ci]
    candidates.sort(key=lambda item:-item[2])
    tp,fp,_=_match_class(candidates,gt,2.)
    tp=np.cumsum(tp);fp=np.cumsum(fp)
    for i,candidate in enumerate(candidates):
        if tp[i]/n>=recall and (i+1==len(candidates) or candidate[2]!=candidates[i+1][2]):
            return dict(status='reached',requested_recall=recall,recall=float(tp[i]/n),threshold=candidate[2],
                        fp=int(fp[i]),fp_per_frame=float(fp[i]/len(truths)))
    return dict(status='unreachable',requested_recall=recall,max_recall=float(tp[-1]/n) if len(tp) else 0.)


def evaluation_slices(predictions,truths,train,val,tram_tracks=()):
    """Match globally before assigning GT to seen/unseen or tram slices."""
    from .sparse_audit import false_positive_audit
    def valid_id(value):
        value=str(value).strip().lower()
        return value not in ('','none','nan','unknown','-1') and not value.startswith('-')
    seen={(str(row['metainfo']['sequence_id']),str(row['annos']['gt_ids'][raw]))
          for row,indices in zip(train.rows,train.box_source_indices) for raw in indices if valid_id(row['annos']['gt_ids'][raw])}
    trams={(str(item['sequence']),str(item['track_id'])) for item in tram_tracks}
    groups={};per_track={};tram=[];tram_gt=0
    for ps,truth,row,indices in zip(predictions,truths,val.rows,val.box_source_indices):
        gt=truth[truth[:,7]>=0];used=set();hit={}
        for prediction in sorted((p for p in ps if p[8]>=.1),key=lambda p:-p[8]):
            pool=[i for i,g in enumerate(gt) if i not in used and g[7]==prediction[7]]
            if not pool:continue
            index=min(pool,key=lambda i:np.linalg.norm(prediction[:2]-gt[i,:2]))
            distance=float(np.linalg.norm(prediction[:2]-gt[index,:2]))
            if distance>=2:continue
            used.add(index)
            directed=abs(float(np.arctan2(np.sin(prediction[6]-gt[index,6]),np.cos(prediction[6]-gt[index,6]))))
            hit[index]=(distance,float(np.degrees(directed)),float(np.degrees(min(directed,np.pi-directed))))
        for i,(box,raw) in enumerate(zip(gt,indices)):
            key=(str(row['metainfo']['sequence_id']),str(row['annos']['gt_ids'][raw]))
            name=CLASSES[int(box[7])];group='seen' if key in seen else 'unseen' if valid_id(key[1]) else 'unknown_identity'
            item=groups.setdefault(name+'/'+group,dict(gt=0,matches=[],tracks=set()));item['gt']+=1;item['tracks'].add(key)
            if i in hit:item['matches'].append(hit[i])
            if int(box[7]) in (3,4,5):
                track_item=per_track.setdefault((name,group,*key),dict(gt=0,matches=[]));track_item['gt']+=1
                if i in hit:track_item['matches'].append(hit[i])
            if key in trams and name=='bus':
                tram_gt+=1
                if i in hit:tram.append(hit[i])
    def summarize(n,matches):
        return dict(gt=n,tp=len(matches),recall=len(matches)/n if n else None,
            xy_error_m=float(np.mean([v[0] for v in matches])) if matches else None,
            directed_AOE_deg=float(np.mean([v[1] for v in matches])) if matches else None,
            unsigned_axis_AOE_deg=float(np.mean([v[2] for v in matches])) if matches else None,
            flip_rate=float(np.mean([v[1]>90 for v in matches])) if matches else None)
    fixed={name:{str(r):fixed_class_recall(predictions,truths,ci,r) for r in (.25,.5,.75)} for ci,name in enumerate(CLASSES)}
    car_points={}
    for r in (.5,.75):
        point=fixed['car'][str(r)].copy()
        if point['status']=='reached':
            point['all_class_false_positives']=false_positive_audit(predictions,truths,point['threshold'])[0]
        car_points[str(r)]=point
    return dict(protocol='Post-NMS candidates decoded at >=0.01; exact score ties retained. Class-aware XY<2m; slices at score>=0.1.',
        fixed_recall=fixed,at_fixed_car_recall=car_points,
        seen_unseen={name:dict(track_count=len(item['tracks']),**summarize(item['gt'],item['matches'])) for name,item in groups.items()},
        heavy_per_track=[dict(class_name=name,overlap=group,sequence=bag,track_id=uid,**summarize(item['gt'],item['matches']))
                         for (name,group,bag,uid),item in sorted(per_track.items())],
        tram_heading=dict(status='provided' if trams else 'no_verified_tram_registry',**summarize(tram_gt,tram)))
