"""Transparent validation evaluator for the 6-class RayBEV-Lite dataset.

Center-distance AP follows nuScenes-style 0.5/1/2/4 metre matching and
precision interpolation. mATE/mASE/mAOE are simple matched-pair diagnostics;
not exact devkit cumulative-mean TP metrics. Not official ROSA FDS or NDS.

Absent validation classes are omitted from the mAP mean (reported as null).
Prediction of an absent class does NOT count as another class's TP.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import math
import numpy as np
from .config import CLASSES
DISTANCES=(0.5,1.0,2.0,4.0)


def _ap_from_flags(tp,fp,n_gt):
    if n_gt==0:return None
    if not tp:return 0.0
    tp=np.cumsum(np.asarray(tp,dtype=float))
    fp=np.cumsum(np.asarray(fp,dtype=float))
    recall=tp/n_gt
    precision=tp/np.maximum(tp+fp,1e-12)
    # Integrate official-style 101-point precision-recall curve with 10% gates.
    xp=np.concatenate(([0.0],recall))
    yp=np.concatenate(([precision[0]],precision))
    interpolation=np.interp(np.linspace(0,1,101),xp,yp,right=0.0)
    # same min_recall / min_precision ideas as nuScenes, not a devkit replacement
    return float(np.maximum(0.0,interpolation[11:]-0.1).mean()/0.9)


def _match_class(pred_list,gt_list,distance_threshold):
    """Each prediction (frame,box,score) matched to closest unmatched GT."""
    matched=defaultdict(set);tp=[];fp=[];pairs=[]
    for frame,box,score in sorted(pred_list,key=lambda r:-r[2]):
        gt=gt_list.get(frame)
        if gt is None or len(gt)==0:
            tp.append(0);fp.append(1);continue
        xy=np.asarray(box[:2],float)
        distances=np.linalg.norm(gt[:,:2]-xy,axis=1)
        for idx in matched[frame]:distances[idx]=np.inf
        idx=int(np.argmin(distances))
        if distances[idx]<distance_threshold:
            matched[frame].add(idx);tp.append(1);fp.append(0)
            pairs.append((box,gt[idx],float(distances[idx])))
        else:tp.append(0);fp.append(1)
    return tp,fp,pairs


def _yaw_err(a,b):return abs(math.atan2(math.sin(float(a)-float(b)),math.cos(float(a)-float(b))))


def evaluate_detections(predictions,ground_truth,score_threshold=.3):
    """predictions: list [M,9] (xyz,lwh,yaw,class,score), GT [N,8].

    Returns per-class AP, mean AP over nonempty GT classes and diagnostics.
    NDS is None because the model has neither velocity nor attributes.
    """
    if len(predictions)!=len(ground_truth):raise ValueError('GT/prediction frame count differs')
    gts_by_class=[defaultdict(list) for _ in CLASSES]
    predictions_by_class=[[] for _ in CLASSES]
    counts=Counter();fps=Counter();n_gt=Counter()
    for frame,(ps,gs) in enumerate(zip(predictions,ground_truth)):
        ps=np.asarray(ps,dtype=float).reshape(-1,9)
        gs=np.asarray(gs,dtype=float).reshape(-1,8)
        for b in gs:
            cls=int(b[7]);
            if not 0<=cls<len(CLASSES):raise ValueError(f'Out-of-range GT class {cls}')
            gts_by_class[cls][frame].append(b[:7]);n_gt[cls]+=1
        for b in ps:
            cls=int(b[7]);
            if not 0<=cls<len(CLASSES):raise ValueError(f'Out-of-range predicted class {cls}')
            predictions_by_class[cls].append((frame,b[:7],float(b[8])))
            if b[8]>=score_threshold:counts[cls]+=1
    result={};active=[];all_aps=[];errors={'ATE':[],'ASE':[],'AOE':[]};total_TP=total_FP=total_FN=0
    for cls,name in enumerate(CLASSES):
        g={frame:np.asarray(arr,dtype=float).reshape(-1,7) for frame,arr in gts_by_class[cls].items()}
        n=n_gt[cls]
        pr=predictions_by_class[cls]
        if n==0:
            result[name]={'gt_count':0,'prediction_count':len(pr),'AP':None,
                          'AP_by_distance':None,'ATE':None,'ASE':None,'AOE':None,
                          'precision':None,'recall':None,'status':'excluded_no_validation_ground_truth'}
            continue
        active.append(name)
        thresholds={};pairs2=[]
        for dist in DISTANCES:
            tp,fp,pairs=_match_class(pr,g,dist)
            thresholds[str(dist)]=_ap_from_flags(tp,fp,n)
            if dist==2.0:pairs2=pairs
        ap=float(np.mean(list(thresholds.values())));all_aps.append(ap)
        # Fixed-score operational precision/recall at 2 m; threshold used for counts.
        pr_threshold=[r for r in pr if r[2]>=score_threshold]
        tps,fps_flags,_=_match_class(pr_threshold,g,2.0)
        true_pos=sum(tps);false_pos=sum(fps_flags);miss=max(0,n-true_pos)
        total_TP+=true_pos;total_FP+=false_pos;total_FN+=miss
        ate=[];ase=[];aoe=[]
        for pred,gt,d in pairs2:
            ate.append(d)
            # Size error equivalent to 1-aligned 3D IoU after identical centers and yaw.
            a=np.maximum(pred[3:6],1e-5);b=np.maximum(gt[3:6],1e-5)
            ase.append(1.0-float(np.prod(np.minimum(a,b))/np.prod(np.maximum(a,b))))
            aoe.append(_yaw_err(pred[6],gt[6]))
        # Missing matches are represented as error 1, not artificially perfect.
        e={'ATE':float(np.mean(ate)) if ate else 1.0,
           'ASE':float(np.mean(ase)) if ase else 1.0,
           'AOE':float(np.mean(aoe)) if aoe else 1.0}
        for key,val in e.items():errors[key].append(val)
        result[name]={'gt_count':n,'prediction_count':len(pr),'AP':ap,
                      'AP_by_distance':thresholds,'ATE':e['ATE'],'ASE':e['ASE'],'AOE':e['AOE'],
                      'precision':true_pos/max(1,true_pos+false_pos),
                      'recall':true_pos/max(1,n),'TP':true_pos,'FP':false_pos,'FN':miss,
                      'status':'evaluated'}
    if not active:raise ValueError('No validation ground-truth objects for any class')
    mean_ap=float(np.mean(all_aps))
    means={f'm{k}':float(np.mean(v)) for k,v in errors.items()}
    # NOT official NDS; the real nuScenes NDS also requires mAVE and mAAE.
    # Always explicitly record NDS=None, never silently boost missing terms.
    geometry_score=(5*mean_ap+sum(max(0.0,1-min(1,means[f'm{k}'])) for k in ('ATE','ASE','AOE')))/8
    lower_bound=(5*mean_ap+sum(max(0.0,1-min(1,means[f'm{k}'])) for k in ('ATE','ASE','AOE')))/10
    return {'active_classes':active,'excluded_classes':[n for n in CLASSES if n not in active],
            'mAP':mean_ap,'NDS':None,'NDS_status':'unavailable: no velocity/attribute outputs and not the official nuScenes dataset',
            'geometry_score_3term_nonstandard':geometry_score,
            'NDS_missing_velocity_attribute_zero_score_nonstandard':lower_bound,
            **means,'precision_at_score_threshold':total_TP/max(1,total_TP+total_FP),
            'recall_at_score_threshold':total_TP/max(1,total_TP+total_FN),
            'score_threshold':score_threshold,'AP_distance_thresholds':list(DISTANCES),
            'class_results':result}
