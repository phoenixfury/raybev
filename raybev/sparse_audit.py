"""Read-only diagnostics. Benchmark AP/GT are never rewritten by an audit."""
from collections import Counter,defaultdict
import numpy as np
from .config import CLASSES

HEAVY=(3,4,5)


def raw_snapshot(output,cfg):
    final=output['stages'][-1]
    keys=('boxes','class_logits','quality_logits','supported','axis','facing')
    if 'objectness_logits' in final:keys+=('objectness_logits',)
    return {k:final[k][:,:cfg.queries].detach().cpu() if k=='supported' else
            final[k][:,:cfg.queries].detach().float().cpu() for k in keys}


def audit_queries(output,cfg,targets,matched_outputs=None,score_threshold=.1,**decode_options):
    """Audit candidate availability per GT; only final evaluator recall is 1-to-1."""
    from .sparse_model import sparse_candidates,decode_sparse
    final=output['stages'][-1]
    boxes,labels,scores,support,finite,in_range,prob=sparse_candidates(
        final,cfg,decode_options.get('quality_power'),decode_options.get('class_bias'))
    decoded=matched_outputs if matched_outputs is not None else decode_sparse(
        output,cfg,.01,return_indices=True,**decode_options)
    arrays=[t.detach().cpu().numpy() for t in (boxes,labels,scores,support,finite,in_range,prob)]
    thresholds=decode_options.get('class_thresholds')
    result=[]
    for ground,(_,survived),box,label,score,visible,ok,region,p in zip(targets,decoded,*arrays):
        gt=ground.detach().cpu().numpy();gt=gt[gt[:,7]>=0]
        keep=np.zeros(len(box),bool);keep[survived.detach().cpu().numpy()]=True
        dist=np.linalg.norm(box[:,None,:2]-gt[None,:,:2],axis=-1)
        valid=ok&region&visible
        score_floor=np.maximum(score_threshold,np.asarray(thresholds)[label]) if thresholds is not None else score_threshold
        scored=score>=score_floor
        frame=[]
        for i,g in enumerate(gt):
            c=int(g[7])
            if c not in HEAVY:continue
            near=(dist[:,i]<2.)&np.isfinite(box).all(1)
            eligible=near&valid
            correct=eligible&(label==c)
            passes=correct&scored
            if not near.any():reason='no_raw_geometry'
            elif not eligible.any():reason='only_unsupported_invalid_or_outside'
            elif not correct.any():reason='wrong_argmax'
            elif not passes.any():reason='below_score'
            elif not (passes&keep).any():reason='removed_by_nms_or_cap'
            else:reason='correct_candidate_retained'
            pool=np.flatnonzero(eligible)
            item=dict(gt_index=i,class_id=c,reason=reason)
            if len(pool):
                best=pool[p[pool,c].argmax()]
                closest=pool[dist[pool,i].argmin()]
                d=box[closest,:2]-g[:2];u=np.array([np.cos(g[6]),np.sin(g[6])])
                rad=g[:2]/max(np.linalg.norm(g[:2]),1e-6)
                delta_yaw=box[closest,6]-g[6]
                directed=abs(np.degrees(np.arctan2(np.sin(delta_yaw),np.cos(delta_yaw))))
                item.update(winner=CLASSES[label[best]],best_true_class_rank=int(1+(p[best]>p[best,c]).sum()),
                    true_class_score=float(p[best,c]),nearest_query=int(closest),distance_m=float(dist[closest,i]),
                    lwh_ratio=(box[closest,3:6]/g[3:6]).tolist(),
                    longitudinal_m=float(d@u),lateral_m=float(d@np.array([-u[1],u[0]])),
                    ego_radial_m=float(d@rad),axis_deg=float(min(directed,180-directed)),directed_deg=float(directed))
            frame.append(item)
        result.append(frame)
    return result


def summarize_query_audit(records):
    out={}
    for c in HEAVY:
        rows=[r for frame in records for r in frame if r['class_id']==c]
        measured=[r for r in rows if 'lwh_ratio' in r]
        wrong=[r for r in measured if r['reason']=='wrong_argmax']
        out[CLASSES[c]]=dict(gt=len(rows),geometry_samples=len(measured),
            reasons=dict(Counter(r['reason'] for r in rows)),
            wrong_winners=dict(Counter(r['winner'] for r in wrong)),
            best_true_class_rank=dict(Counter(r['best_true_class_rank'] for r in measured)),
            median_lwh_ratio=np.median([r['lwh_ratio'] for r in measured],axis=0).tolist() if measured else None)
        for key in ('distance_m','longitudinal_m','lateral_m','ego_radial_m','axis_deg','directed_deg'):
            out[CLASSES[c]]['median_'+key]=float(np.median([r[key] for r in measured])) if measured else None
        # Magnitude is important even when positive/negative errors cancel.
        for key in ('longitudinal_m','lateral_m','ego_radial_m'):
            out[CLASSES[c]]['mean_abs_'+key]=float(np.mean([abs(r[key]) for r in measured])) if measured else None
    return out


def false_positive_audit(predictions,truths,score_threshold=.1):
    """Reproduce class-aware 2 m TP assignment, then partition unmatched rows."""
    from .metrics import _match_class
    if len(predictions)!=len(truths):raise ValueError('Audit frame mismatch')
    keys=('duplicate','wrong_class','mislocalized','ignored_region','background_or_unlabeled')
    by_class={name:Counter({k:0 for k in keys}) for name in CLASSES}
    per_frame=[]
    for ps,gt in zip(predictions,truths):
        gt=np.asarray(gt).reshape(-1,8)
        ignore=gt[gt[:,7]<0];gt=gt[gt[:,7]>=0]
        ps=np.asarray(ps).reshape(-1,9);ps=ps[ps[:,8]>=score_threshold]
        errors=[]
        for ci,name in enumerate(CLASSES):
            subset=ps[ps[:,7]==ci];subset=subset[np.argsort(-subset[:,8],kind='stable')]
            g=gt[gt[:,7]==ci]
            tp,_,_=_match_class([(0,r[:7],float(r[8])) for r in subset],{0:g[:,:7]} if len(g) else {},2.)
            for box,hit in zip(subset,tp):
                if hit:continue
                d=np.linalg.norm(gt[:,:2]-box[:2],axis=-1)
                same=gt[:,7]==ci
                if (same&(d<2.)).any():reason='duplicate'
                elif ((~same)&(d<2.)).any():reason='wrong_class'
                elif (same&(d<4.)).any():reason='mislocalized'
                else:
                    delta=box[:2]-ignore[:,:2];co=np.cos(ignore[:,6]);si=np.sin(ignore[:,6])
                    local=np.stack((delta[:,0]*co+delta[:,1]*si,-delta[:,0]*si+delta[:,1]*co),-1)
                    reason='ignored_region' if (np.abs(local)<=ignore[:,3:5]/2+.25).all(1).any() else 'background_or_unlabeled'
                by_class[name][reason]+=1
                errors.append(dict(class_name=name,reason=reason,score=float(box[8]),box=box[:7].tolist()))
        per_frame.append(errors)
    n=len(truths)
    return dict(score_threshold=score_threshold,by_class={k:dict(v) for k,v in by_class.items()},
                car_fp_per_frame=sum(by_class['car'].values())/n if n else None),per_frame


def fixed_car_recall(predictions,truths,target_recall):
    """Exact score operating point; score ties stay together, no interpolation."""
    from .metrics import _match_class
    if not 0<target_recall<=1:raise ValueError('Target Car recall must be in (0,1]')
    gt={};pred=[]
    for i,(ps,ts) in enumerate(zip(predictions,truths)):
        gt[i]=ts[ts[:,7]==0,:7]
        if not len(gt[i]):del gt[i]
        pred.extend((i,row[:7],float(row[8])) for row in ps if row[7]==0)
    n=sum(map(len,gt.values()))
    if not n:return dict(status='no_car_ground_truth')
    pred.sort(key=lambda x:-x[2])
    tp,fp,_=_match_class(pred,gt,2.);tp=np.cumsum(tp);fp=np.cumsum(fp)
    for i,x in enumerate(pred):
        if tp[i]/n>=target_recall and (i+1==len(pred) or x[2]!=pred[i+1][2]):
            return dict(status='reached',threshold=x[2],recall=float(tp[i]/n),
                fp=int(fp[i]),fp_per_frame=float(fp[i]/len(truths)),precision=float(tp[i]/(i+1)))
    return dict(status='unreachable',max_recall=float(tp[-1]/n) if len(tp) else 0.)


def summarize_tracks(records,tracks):
    grouped=defaultdict(list)
    for frame,keys in zip(records,tracks):
        for row in frame:
            key=keys[row['gt_index']]
            if key is not None:grouped[(CLASSES[row['class_id']],*key)].append(row)
    return [dict(class_name=name,bag=bag,track_id=uid,**summarize_query_audit([rows])[name])
            for (name,bag,uid),rows in sorted(grouped.items())]
