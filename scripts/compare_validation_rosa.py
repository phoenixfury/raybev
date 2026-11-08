#!/usr/bin/env python3
"""Reproducible validation comparison using native OpenCV images and verified ROSA cache.

Inference preserves the checkpoint's actual box-frame contract. Nothing is retrained
or corrected retrospectively. All metrics below share GT, range and class order.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from raybev import CLASSES,ModelConfig,RayBEVLite
from raybev.detection import decode_detections,detection_scores,decode_sizes,decode_heading,in_bev_range,metric_to_cell
from raybev.geometry import load_cameras,project_fisheye
from raybev.metrics import evaluate_detections,_match_class
from raybev.rosa_dataset import RosaCombinedDataset,collate_samples,load_annotations,names_to_ids,boxes_to_baselink
from raybev.training import balanced_training_view

CAMERA_KEYS={'ARC_F':'fisheye_camera_front','ARC_R':'fisheye_camera_rear',
             'ARC_SL':'fisheye_camera_left','ARC_SR':'fisheye_camera_right'}
CAMERA_ORDER=('ARC_F','ARC_SL','ARC_SR','ARC_R')
COLORS=((255,220,0),(0,255,120),(255,0,210),(0,165,255),(255,100,40),(40,50,255))
EDGES=((0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=ModelConfig.from_checkpoint(state)
    if sha(args.calibration)!=state['calibration_sha256']:
        raise ValueError('Checkpoint camera calibration hash mismatch')
    if tuple(state['classes'])!=CLASSES:raise ValueError('Class order mismatch')
    lidar_calibration=getattr(args,'lidar_calibration',None) or args.calibration
    if state['box_frame']=='CLIDAR' and sha(lidar_calibration)!=state.get('lidar_calibration_sha256'):
        raise ValueError('Checkpoint CLIDAR calibration hash mismatch')
    dataset=RosaCombinedDataset(args.root,'rosa/fisheye3dod_infos_val.pkl',cfg=cfg,
        box_frame=state['box_frame'],calibration=lidar_calibration,z_reference=state['recipe']['z_reference'])
    if any(r['metainfo']['temporal_split']!='val' for r in dataset.rows):
        raise ValueError('Comparison must contain validation frames only')
    plan=json.loads((args.rosa_analysis/'plan.json').read_text())
    folder=args.rosa_analysis/'new_best6'
    done=json.loads((folder/'inference_complete.json').read_text())
    if (sha(folder/'predictions.json')!=done['predictions_sha256'] or
        sha(plan['union'])!=done['union_sha256'] or
        done['checkpoint_sha256']!=plan['models']['new_best6']['checkpoint_sha256']):
        raise ValueError('ROSA cache provenance failed')
    rosa=json.loads((folder/'predictions.json').read_text())
    if rosa['box_origin']!='gravity_center' or rosa['score_threshold']>.01:
        raise ValueError('ROSA cache origin/score floor incompatible')
    old_rows={str(r['token']):r for r in load_annotations(plan['union'])['data_list']}
    rosa_by_token={str(r['token']):r for r in rosa['frames']}
    tokens=[str(r['token']) for r in dataset.rows]
    if len(tokens)!=len(set(tokens)):raise ValueError('Duplicate validation tokens')
    for row in dataset.rows:
        token=str(row['token'])
        if token not in old_rows or token not in rosa_by_token:
            raise ValueError(f'No matched ROSA validation frame {token}')
        if not np.array_equal(row['annos']['gt_boxes'],old_rows[token]['annos']['gt_boxes']):
            raise ValueError(f'GT mismatch for token {token}')
    contract={'ray_checkpoint':str(args.checkpoint),'ray_checkpoint_sha256':sha(args.checkpoint),
        'ray_epoch':state['epoch'],'ray_recipe':state['recipe'],'cfg':vars(cfg),
        'annotation_sha256':sha(args.root/'rosa/fisheye3dod_infos_val.pkl'),
        'calibration_sha256':sha(args.calibration),
        'lidar_calibration_sha256':sha(lidar_calibration) if dataset.T_base_clidar is not None else None,
        'rosa':plan['models']['new_best6'],
        'rosa_predictions_sha256':done['predictions_sha256'],'frames':len(tokens),
        'tokens':tokens,'ground_truth_equal_to_ROSA_cache':True,
        'range_m':'X,Y centers in [-48,48); same GT and class order for both',
        'metric':'raybev.metrics diagnostic evaluator for BOTH models; not official ROSA FDS/NDS',
        'decoder_note':'Native Ray NMS=.25, ROSA cached NMS=.5; each capped at 150 boxes and floor .01 for primary comparison',
        'box_frame_note':('Corrected CLIDAR-label run: GT, ROSA predictions and cloud transformed once to BASELINK.'
            if dataset.T_base_clidar is not None else
            'Legacy recipe says BASELINK; original GT/ROSA/cloud coordinates originate in CLIDAR. Preserves as-trained numerical coordinates.'),
        'video_note':'All validation frames, scene/bag cuts explicitly labeled; native OpenCV overlays use recorded LiDAR-to-camera transforms; no OCam lens projection.'}
    if (args.output/'manifest.json').exists():
        prior=json.loads((args.output/'manifest.json').read_text())
        for k in ('ray_checkpoint_sha256','annotation_sha256','calibration_sha256','lidar_calibration_sha256','rosa_predictions_sha256'):
            if prior.get(k)!=contract[k]:raise ValueError('Output belongs to different inputs; choose a new output')
    dump(args.output/'manifest.json',contract)
    return state,cfg,dataset,old_rows,rosa_by_token


def as_rosa_array(frame,cfg,T_base_clidar=None):
    boxes=np.asarray(frame['boxes_3d'],dtype=np.float32).reshape(-1,9)[:,:7]
    if T_base_clidar is not None:boxes=boxes_to_baselink(boxes,'CLIDAR',T_base_clidar)
    labels=names_to_ids(frame['names'])
    scores=np.asarray(frame['scores'],dtype=np.float32)
    out=np.column_stack((boxes,labels,scores)).astype(np.float32)
    out=out[np.isfinite(out).all(axis=1)&in_bev_range(out,cfg)&(scores>=.01)]
    return out[np.argsort(-out[:,8],kind='stable')][:cfg.max_detections]


def infer(args,state,cfg,dataset):
    destination=args.output/'ray_predictions.json'
    if destination.exists():
        print('Using completed inference cache',destination,flush=True);return
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(2)
    model=RayBEVLite(cfg,args.calibration).to(device).eval()
    model.load_state_dict(state['state_dict'])
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,
        collate_fn=collate_samples,pin_memory=device.type=='cuda')
    variants={k:[] for k in ('trained','heat_only','floor_001','classwise_50')}
    diagnostics=[];candidate_counts=np.zeros(6,dtype=np.int64)
    started=time.monotonic();position=0
    with torch.inference_mode():
        for images,boxes,tokens in loader:
            amp=torch.autocast('cuda') if device.type=='cuda' else nullcontext()
            with amp:output=model(images.to(device,non_blocking=device.type=='cuda'))
            output=tuple(v.float() for v in output)
            decoded=decode_detections(output,cfg,score_threshold=.01,topk=300)
            raw=decode_detections(output,replace(cfg,quality_score_power=0.),score_threshold=.01,topk=300)
            low=decode_detections(output,cfg,score_threshold=.001,topk=300)
            heat=output[0].sigmoid();quality=output[2].sigmoid()
            ranked=detection_scores(output,cfg)
            ids=ranked.flatten(2).topk(50,dim=2).indices
            allowed=torch.zeros_like(heat,dtype=torch.bool).flatten(2).scatter_(2,ids,True).reshape_as(heat)
            class_output=(torch.where(allowed,output[0],output[0].new_full((),-30.)),*output[1:])
            per_class=decode_detections(class_output,cfg,score_threshold=.001,topk=300)
            neighborhood=F.max_pool2d(heat,5,1,2)
            for bi,(gt,token) in enumerate(zip(boxes,tokens)):
                for name,values in zip(variants,(decoded,raw,low,per_class)):
                    variants[name].append({'token':token,'detections':values[bi].cpu().tolist()})
                _,global_ids=ranked[bi].flatten().topk(300)
                candidate_counts+=torch.bincount(global_ids//(cfg.bev_height*cfg.bev_width),minlength=6).cpu().numpy()
                if len(gt):
                    rows=((cfg.bev_x_max-gt[:,0])/cfg.bev_resolution).floor().long().clamp(0,cfg.bev_height-1)
                    cols=((cfg.bev_y_max-gt[:,1])/cfg.bev_resolution).floor().long().clamp(0,cfg.bev_width-1)
                    classes=gt[:,7].long()
                    rr,cc=rows.to(device),cols.to(device)
                    probs=heat[bi,:,rr,cc].T.cpu().numpy()
                    regs=output[1][bi,:,rr,cc].T.cpu().numpy()
                    oracle_sizes=decode_sizes(torch.from_numpy(regs[:,3:6]),classes,cfg).numpy()
                    bin_logits=output[3][bi,:,rr,cc].T.cpu() if cfg.heading_bins else None
                    oracle_yaw,_=decode_heading(torch.from_numpy(regs[:,6:8]),bin_logits,cfg)
                    qs=quality[bi,0,rr,cc].cpu().numpy()
                    near=neighborhood[bi,:,rr,cc].T.cpu().numpy()
                    for j,box in enumerate(gt.numpy()):
                        reg=regs[j]
                        oracle=[cfg.bev_x_max-(float(rows[j])+.5+reg[0])*cfg.bev_resolution,
                            cfg.bev_y_max-(float(cols[j])+.5+reg[1])*cfg.bev_resolution,
                            float(reg[2]),*oracle_sizes[j].tolist(),float(oracle_yaw[j])]
                        diagnostics.append({'token':token,'gt':box.tolist(),'class_probabilities':probs[j].tolist(),
                            'heat_5x5_peak':float(near[j,int(classes[j])]),'quality':float(qs[j]),'oracle_box':oracle})
                position+=1
            if position==len(dataset) or position%20<args.batch_size:
                print(f'Inference {position}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    dump(destination,{'variants':variants,'gt_diagnostics':diagnostics,
        'pre_nms_top300_class_counts':candidate_counts.tolist(),'device':str(device),
        'seconds':time.monotonic()-started,'precision':'autocast FP16 on CUDA; FP32 decoding'})


def arrays(document,name):
    return [np.asarray(r['detections'],dtype=np.float32).reshape(-1,9) for r in document['variants'][name]]


def matched(predictions,ground_truth,threshold):
    predictions_by_class=[[] for _ in CLASSES];truths=[defaultdict(list) for _ in CLASSES]
    for frame,(ps,gs) in enumerate(zip(predictions,ground_truth)):
        for g in gs:truths[int(g[7])][frame].append(g[:7])
        for p in ps:
            if p[8]>=threshold:predictions_by_class[int(p[7])].append((frame,p[:7],float(p[8])))
    results={};all_tp=all_fp=all_gt=0
    for ci,name in enumerate(CLASSES):
        gt={f:np.asarray(x) for f,x in truths[ci].items()}
        tp,fp,pairs=_match_class(predictions_by_class[ci],gt,2.)
        n=sum(len(x) for x in gt.values());t,f=sum(tp),sum(fp)
        errors=[]
        for pred,truth,d in pairs:
            yaw=abs(math.atan2(math.sin(pred[6]-truth[6]),math.cos(pred[6]-truth[6])))
            axis=min(yaw,abs(math.pi-yaw))
            errors.append([d,abs(float(pred[2]-truth[2])),yaw*180/math.pi,axis*180/math.pi,
                           *abs(pred[3:6]-truth[3:6]).tolist()])
        results[name]={'GT':n,'TP':t,'FP':f,'FN':n-t,'precision':t/max(1,t+f),'recall':t/max(1,n),
            'matched_error_medians_XY_Z_yawdeg_axisdeg_L_W_H':np.median(errors,axis=0).tolist() if errors else None}
        all_tp+=t;all_fp+=f;all_gt+=n
    precision=all_tp/max(1,all_tp+all_fp);recall=all_tp/max(1,all_gt)
    return {'TP':all_tp,'FP':all_fp,'FN':all_gt-all_tp,'precision':precision,'recall':recall,
        'F1':2*precision*recall/max(1e-12,precision+recall),'classes':results}


def confusion(predictions,ground_truth,threshold=.1):
    counts=Counter();misses=Counter()
    for ps,gs in zip(predictions,ground_truth):
        used=set()
        for p in sorted((p for p in ps if p[8]>=threshold),key=lambda p:-p[8]):
            choices=[(np.linalg.norm(p[:2]-g[:2]),j) for j,g in enumerate(gs) if j not in used]
            if choices:
                d,j=min(choices)
                if d<2:
                    used.add(j);counts[CLASSES[int(gs[j,7])]+' -> '+CLASSES[int(p[7])]]+=1
        misses.update(CLASSES[int(g[7])] for j,g in enumerate(gs) if j not in used)
    return {'class_agnostic_spatial_matches':dict(counts),'unmatched_GT':dict(misses),'threshold':threshold}


def analyze(args,cfg,dataset,rosa):
    document=json.loads((args.output/'ray_predictions.json').read_text())
    gt=[b.numpy() for b in dataset.boxes]
    predictions={k:arrays(document,k) for k in document['variants']}
    predictions['ROSA']=[as_rosa_array(rosa[str(r['token'])],cfg,dataset.T_base_clidar) for r in dataset.rows]
    results={}
    for name,ps in predictions.items():
        print('Evaluating',name,flush=True)
        results[name]={'metrics':evaluate_detections(ps,gt,score_threshold=.1)}
        if name in ('trained','ROSA'):
            results[name]['operating']={str(t):matched(ps,gt,t) for t in (.03,.05,.1,.2,.3)}
            results[name]['confusion']=confusion(ps,gt)
            results[name]['per_bag']={}
            for bag in sorted(set(r['metainfo']['sequence_id'] for r in dataset.rows)):
                ids=[i for i,r in enumerate(dataset.rows) if r['metainfo']['sequence_id']==bag]
                results[name]['per_bag'][bag]=evaluate_detections([ps[i] for i in ids],[gt[i] for i in ids],score_threshold=.1)
    diagnostics={}
    for ci,name in enumerate(CLASSES):
        rows=[r for r in document['gt_diagnostics'] if int(r['gt'][7])==ci]
        if not rows:continue
        truth=np.asarray([r['gt'] for r in rows]);oracle=np.asarray([r['oracle_box'] for r in rows])
        probabilities=np.asarray([r['class_probabilities'] for r in rows])
        yaw=np.abs(np.arctan2(np.sin(oracle[:,6]-truth[:,6]),np.cos(oracle[:,6]-truth[:,6])))
        diagnostics[name]={'GT':len(rows),'median_exact_center_heat':float(np.median(probabilities[:,ci])),
            'median_5x5_peak_heat':float(np.median([r['heat_5x5_peak'] for r in rows])),
            'median_quality':float(np.median([r['quality'] for r in rows])),
            'oracle_center_class_argmax_counts':dict(Counter(CLASSES[i] for i in probabilities.argmax(axis=1))),
            'oracle_xy_error_median':float(np.median(np.linalg.norm(oracle[:,:2]-truth[:,:2],axis=1))),
            'oracle_Z_error_median':float(np.median(abs(oracle[:,2]-truth[:,2]))),
            'oracle_yaw_error_deg_median':float(np.median(yaw)*180/math.pi),
            'GT_LWH_medians':np.median(truth[:,3:6],axis=0).tolist(),
            'oracle_predicted_LWH_medians':np.median(oracle[:,3:6],axis=0).tolist()}
    train=RosaCombinedDataset(args.root,'rosa/fisheye3dod_infos_train.pkl',cfg=cfg,
        box_frame=dataset.box_frame,calibration=getattr(args,'lidar_calibration',None) or args.calibration,
        z_reference=dataset.z_reference)
    view=balanced_training_view(train,cfg=cfg)
    frame_counts=Counter();box_counts=Counter()
    for b in train.boxes:
        frame_counts.update(CLASSES[i] for i in b[:,7].long().unique().tolist())
        box_counts.update(CLASSES[i] for i in b[:,7].long().tolist())
    weighted_box_counts=Counter()
    for i in view.indices:weighted_box_counts.update(CLASSES[j] for j in train.boxes[i][:,7].long().tolist())
    report={'models':results,'GT_center_diagnostics':diagnostics,
        'candidate_top300_counts':dict(zip(CLASSES,document['pre_nms_top300_class_counts'])),
        'training_sampling':{'original_frames':len(train),'balanced_entries':len(view),
            'frame_presence_counts':dict(frame_counts),'original_boxes':dict(box_counts),'balanced_boxes':dict(weighted_box_counts)},
        'validation_frames':len(dataset),'validation_boxes':sum(len(x) for x in gt),
        'note':'All models use the same lightweight evaluator, same GT/range/active classes. Native decoders differ; ablations are diagnostic only.'}
    dump(args.output/'comparison.json',report)
    return report


def corners(box):
    local=np.array([[sx*box[3]/2,sy*box[4]/2,sz*box[5]/2] for sz in (-1,1) for sy in (-1,1) for sx in (-1,1)])
    c,s=math.cos(box[6]),math.sin(box[6])
    return local@np.array([[c,s,0],[-s,c,0],[0,0,1]])+box[:3]


def dashed(image,a,b,color,thickness=1):
    a,b=np.asarray(a,float),np.asarray(b,float)
    for t in np.arange(0,1,.2):
        cv2.line(image,tuple(np.rint(a+(b-a)*t).astype(int)),
                 tuple(np.rint(a+(b-a)*min(t+.1,1)).astype(int)),color,thickness,cv2.LINE_AA)


def text(image,value,point,size=.5,color=(235,235,235),thickness=1):
    cv2.putText(image,str(value),point,cv2.FONT_HERSHEY_SIMPLEX,size,color,thickness,cv2.LINE_AA)


def draw_camera(image,camera,gt,ps):
    h,w=image.shape[:2]
    for g,box in [(True,b) for b in gt]+[(False,b) for b in ps]:
        uv,visible,_=project_fisheye(corners(box),camera)
        uv=(uv+.5)*np.array([w/camera.width,h/camera.height])-.5
        color=(210,210,210) if g else COLORS[int(box[7])]
        for a,b in EDGES:
            if not (visible[a] and visible[b]) or np.linalg.norm(uv[a]-uv[b])>w*.65:continue
            p1,p2=tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int))
            if g:dashed(image,p1,p2,color)
            else:cv2.line(image,p1,p2,color,2,cv2.LINE_AA)
        if not g and visible.any():
            anchor=tuple(np.rint(uv[np.flatnonzero(visible)[0]]).astype(int))
            text(image,f'{CLASSES[int(box[7])]} {box[8]:.2f}',anchor,.4,color)
    return image


def cloud_path(root,row,summary):
    source=Path(summary['bag_stats'][str(row['metainfo']['sequence_id'])]['source'])
    return source/'lidar_pcd'/Path(row['pc_info']['pts_path']).name


def cloud(path):
    data=path.read_bytes();marker=b'DATA binary\n';offset=data.find(marker)
    if offset>=0:data=data[offset+len(marker):]
    points=np.frombuffer(data,dtype='<f4').reshape(-1,4)
    return points[np.isfinite(points).all(axis=1)]


def bev_base(points,size=640):
    image=np.full((size,size,3),8,np.uint8)
    scale=(size-32)/96
    for radius in (12,24,36,48):cv2.circle(image,(size//2,size//2),int(radius*scale),(40,40,40),1,cv2.LINE_AA)
    cv2.line(image,(size//2,0),(size//2,size),(40,40,40),1)
    cv2.line(image,(0,size//2),(size,size//2),(40,40,40),1)
    keep=(abs(points[:,0])<48)&(abs(points[:,1])<48)&(points[:,2]>-3)&(points[:,2]<4)
    p=points[keep][::3]
    xy=np.rint(np.column_stack((size/2-p[:,1]*scale,size/2-p[:,0]*scale))).astype(int)
    gray=np.clip((p[:,2]+2.5)*50+25,30,190).astype(np.uint8)
    image[xy[:,1],xy[:,0]]=gray[:,None]
    text(image,'+X forward',(size//2+5,18),.45)
    text(image,'+Y left',(8,size//2-8),.45)
    cv2.drawMarker(image,(size//2,size//2),(0,255,140),cv2.MARKER_TRIANGLE_UP,15,2)
    return image


def draw_bev(base,gt,ps):
    image=base.copy();size=image.shape[0];scale=(size-32)/96
    def pixel(xy):return tuple(np.rint([size/2-xy[1]*scale,size/2-xy[0]*scale]).astype(int))
    for g,box in [(True,b) for b in gt]+[(False,b) for b in ps]:
        xyz=corners(box);polygon=xyz[[0,1,3,2,0],:2]
        color=(210,210,210) if g else COLORS[int(box[7])]
        for a,b in zip(polygon[:-1],polygon[1:]):
            if g:dashed(image,pixel(a),pixel(b),color)
            else:cv2.line(image,pixel(a),pixel(b),color,2,cv2.LINE_AA)
        head=box[:2]+np.array([math.cos(box[6]),math.sin(box[6])])*box[3]/2
        cv2.line(image,pixel(box[:2]),pixel(head),color,1 if g else 2,cv2.LINE_AA)
    return image


def frame_counts(ps,gt):
    op=matched([ps],[gt],0.)
    return f"TP {op['TP']}   FP {op['FP']}   FN {op['FN']}  (2m class-aware)"


def video_writer(path,fps):
    return subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','2560x900','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)


def render(args,cfg,dataset,old_rows,rosa,report):
    cv2.setNumThreads(2)
    document=json.loads((args.output/'ray_predictions.json').read_text())
    ray=arrays(document,'trained')
    rs=[as_rosa_array(rosa[str(r['token'])],cfg,dataset.T_base_clidar) for r in dataset.rows]
    summary=json.loads((args.root/'summary.json').read_text())
    cameras={c.name:c for c in load_cameras(args.calibration)}
    destination=args.output/'validation_raybev_vs_rosa.mp4'
    if destination.exists():raise FileExistsError('Completed video exists; choose a new output or inspect it')
    temporary=args.output/'validation_raybev_vs_rosa.partial.mp4'
    process=video_writer(temporary,args.fps)
    manifest=json.loads((args.output/'manifest.json').read_text())
    previous_scene=None;rendered=0;previews=set();started=time.monotonic()
    try:
        for i,row in enumerate(dataset.rows):
            gt=dataset.boxes[i].numpy();token=str(row['token']);meta=row['metainfo']
            source=old_rows[token]
            points=cloud(cloud_path(args.root,row,summary))
            if dataset.T_base_clidar is not None:
                points=points.copy();T=dataset.T_base_clidar
                points[:,:3]=points[:,:3]@T[:3,:3].T+T[:3,3]
            base=bev_base(points)
            images={}
            projection={}
            for name in CAMERA_ORDER:
                image=cv2.imread(str(args.root/row['cam_info']['cam_fisheye'][name]['cam_path']))
                if image is None:raise FileNotFoundError(token+'/'+name)
                images[name]=image
                # Draw BOTH models and GT using the same native OpenCV lens and
                # recorded rig extrinsics, in the same frame as the boxes/cloud.
                T_camera_output=np.asarray(source['cam_info']['cam_fisheye'][CAMERA_KEYS[name]]['lidar2cam'])
                if dataset.T_base_clidar is not None:T_camera_output=T_camera_output@np.linalg.inv(dataset.T_base_clidar)
                projection[name]=replace(cameras[name],T_base_cam=np.linalg.inv(T_camera_output))
            canvas=np.full((900,2560,3),15,np.uint8)
            scene=str(meta['scene_name']);cut=scene!=previous_scene
            text(canvas,f"VALIDATION ONLY | {i+1}/{len(dataset)} | bag {meta['sequence_id']} | frame {meta['source_frame_id']} | {scene}"+(' | SCENE CUT' if cut else ''),(18,28),.7)
            previous_scene=scene
            for side,(name,ps,title) in enumerate((('trained',ray[i],f"RayBEV-Lite v2 epoch {manifest['ray_epoch']}"),('ROSA',rs[i],'ROSA four-bag epoch 6'))):
                ps=ps[ps[:,8]>=args.score_threshold]
                x=side*1280
                text(canvas,title,(x+20,68),.85,thickness=2)
                metric_label=(f"Same-evaluator mAP {report['models'][name]['metrics']['mAP']:.4f}"
                              if report is not None else 'Matched validation frames')
                text(canvas,f'{metric_label} | display score >= {args.score_threshold:.2f}',(x+20,100),.55)
                canvas[140:780,x:x+640]=draw_bev(base,gt,ps)
                for ci,cam in enumerate(CAMERA_ORDER):
                    image=draw_camera(images[cam].copy(),projection[cam],gt,ps)
                    image=cv2.resize(image,(320,256),interpolation=cv2.INTER_AREA)
                    left=x+640+(ci%2)*320;top=150+(ci//2)*276
                    canvas[top:top+256,left:left+320]=image
                    text(canvas,cam,(left+8,top+20),.55,thickness=2)
                text(canvas,frame_counts(ps,gt),(x+20,815),.65)
                text(canvas,'BEV: annotation/LiDAR axes, +/-48m | white dashed = GT',(x+20,847),.5)
                for ci,c in enumerate(CLASSES):text(canvas,c,(x+15+ci*170,880),.5,COLORS[ci],2)
            cv2.line(canvas,(1280,40),(1280,900),(100,100,100),2)
            process.stdin.write(canvas.tobytes());rendered+=1
            bag=str(meta['sequence_id'])
            if bag not in previews:
                cv2.imwrite(str(args.output/f'preview_{bag}.jpg'),canvas);previews.add(bag)
            if i%25==0:print(f'Render {i+1}/{len(dataset)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally:
        process.stdin.close();code=process.wait()
    if code or rendered!=len(dataset):raise RuntimeError(f'Incomplete video: ffmpeg={code}, frames={rendered}')
    temporary.replace(destination)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(destination)],text=True))['streams'][0]
    if int(probe['nb_read_frames'])!=len(dataset):raise ValueError('Video frame count mismatch')
    manifest.update(video=str(destination),video_validation=probe,display_score_threshold=args.score_threshold,
                    playback_fps=args.fps,video_sha256=sha(destination))
    dump(args.output/'manifest.json',manifest)
    print('Completed video',destination,probe,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--lidar-calibration',type=Path,default=None,help='Defaults to --calibration')
    parser.add_argument('--rosa-analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('infer','analyze','render','all'),default='all')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--fps',type=float,default=2.)
    parser.add_argument('--score-threshold',type=float,default=.1)
    args=parser.parse_args()
    state,cfg,dataset,old,rosa=prepare(args)
    if args.stage in ('all','infer'):infer(args,state,cfg,dataset)
    if args.stage in ('all','analyze'):report=analyze(args,cfg,dataset,rosa)
    else:report=None
    if args.stage in ('all','render'):
        if report is None and (args.output/'comparison.json').exists():
            report=json.loads((args.output/'comparison.json').read_text())
        render(args,cfg,dataset,old,rosa,report)


if __name__=='__main__':main()
