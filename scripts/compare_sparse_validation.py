#!/usr/bin/env python3
"""Read-only sparse/ROSA comparison on identical combined validation tokens."""
import argparse
from collections import Counter,defaultdict
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import pickle
import sys
import time
PROJECT=Path(__file__).resolve().parents[1];ROOT=PROJECT.parents[1]
sys.path[:0]=[str(PROJECT),str(ROOT),str(ROOT/'projects/ROSA'),str(ROOT/'projects/ROSA/tools'),
             str(ROOT/'projects/Fisheye3DOD'),str(ROOT/'tools/custom_data')]
os.environ.setdefault('FISHEYE3DOD_OFFICIAL_ROOT',str(ROOT/'projects/Fisheye3DOD'))
import numpy as np
for name,value in [('float',float),('int',int),('bool',bool)]:
    if name not in np.__dict__:setattr(np,name,value)
import cv2
import torch
from raybev.config import DEFAULT_CALIBRATION,CLASSES,CAMERAS
from raybev.sparse_config import SparseConfig
from raybev.rosa_dataset import RosaCombinedDataset,camera_entry,load_annotations,parse_box_annotations
from raybev.detection import in_bev_range
from raybev.sparse_training import sparse_annotation_sha256
from raybev.geometry import load_cameras
from raybev.metrics import evaluate_detections,heading_diagnostics
from compare_validation_rosa import (sha,dump,as_rosa_array,matched,confusion,cloud,bev_base,
    draw_bev,draw_camera,text,frame_counts,video_writer,CAMERA_ORDER,COLORS)


def inputs(a):
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=False);cfg=SparseConfig.from_checkpoint(state)
    ds=RosaCombinedDataset(a.root,a.root/'annotations/fisheye3dod_infos_val.pkl',cfg=cfg,
        calibration=a.calibration,box_frame=state['box_frame'],z_reference=state['z_reference'],strict_size=False)
    if state['semantic_annotation_sha256']['val']!=sparse_annotation_sha256(ds):raise ValueError('VAL mismatch')
    if state['calibration_sha256']!=sha(a.calibration):raise ValueError('Calibration mismatch')
    if any(r['metainfo']['temporal_split']!='val' for r in ds.rows):raise ValueError('VAL only')
    return state,cfg,ds


def prepare(a):
    from ocamcamera import OcamCamera
    from prepare_fisheye3dod_inference import make_remap
    state,cfg,ds=inputs(a);a.output.mkdir(parents=True,exist_ok=True)
    target=a.output/'rosa_inputs';target.mkdir(exist_ok=True)
    complete=target/'complete.json'
    identity=dict(annotation_sha256=sha(ds.annotation_path),calibration_sha256=sha(a.calibration),
        ocam_sha256=sha(ROOT/'data/Fisheye3DODdataset/calib_results.txt'))
    if complete.exists():
        if json.loads(complete.read_text())!=identity:raise ValueError('Different prepared inputs')
        return
    ocam=OcamCamera(filename=str(ROOT/'data/Fisheye3DODdataset/calib_results.txt'),fov=220.)
    raw=json.loads(Path(a.calibration).read_text())
    remaps={name:make_remap(ocam,raw['intrinsics'][name]) for name in CAMERAS}
    cv2.setNumThreads(1)
    def frame(row):
        row=copy.deepcopy(row)
        for name in CAMERAS:
            info=camera_entry(row['cam_info']['cam_fisheye'],name)
            source=a.root/info['cam_path'];image=cv2.imread(str(source))
            intr=raw['intrinsics'][name]
            if image is None or image.shape[:2]!=(intr['height'],intr['width']):raise ValueError('Native source size '+str(source))
            warped=cv2.remap(image,*remaps[name],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
            destination=target/'images'/name/(str(row['token'])+'.jpg');destination.parent.mkdir(parents=True,exist_ok=True)
            if not cv2.imwrite(str(destination),warped,[cv2.IMWRITE_JPEG_QUALITY,95]):raise IOError(destination)
            info['cam_path']=str(destination.relative_to(target));info['cam2img']=copy.deepcopy(ocam);info['image_size']=(800,800)
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,row in enumerate(pool.map(frame,ds.rows)):
            rows.append(row)
            if i%100==0:print('ROSA calibrated input preparation',i+1,'/',len(ds),flush=True)
    union=target/'validation.pkl'
    with union.open('wb') as stream:pickle.dump(dict(metainfo=ds.annotation.get('metainfo',{}),data_list=rows),stream)
    # Constructor anchors are overwritten by strict checkpoint loading.
    (target/'rosa/anchors').mkdir(parents=True,exist_ok=True)
    anchor=np.load(a.root/'rosa/anchors/kmeans300_seed3407.npy')
    np.save(target/'rosa/anchors/kmeans300_seed3407.npy',anchor)
    plan=dict(data_root=str(target),union=str(union),union_sha256=sha(union),
        models=dict(rosa_epoch6=dict(checkpoint=str(a.rosa_checkpoint),checkpoint_sha256=sha(a.rosa_checkpoint),
            config=str(ROOT/'projects/ROSA/work_dirs/combined_4bags_analysis_20261107/new_best6/resolved_inference_config.py'))))
    dump(a.output/'plan.json',plan);dump(complete,identity)


def rosa_infer(a):
    from compare_fourbag_training import infer_one
    a.model='rosa_epoch6'
    if (a.output/a.model/'inference_complete.json').exists():return
    infer_one(a)


def predictions(a,ds,cfg):
    ray=json.loads((a.output/'ray_predictions.json').read_text())
    if ray['checkpoint_sha256']!=sha(a.checkpoint):raise ValueError('Ray cache mismatch')
    path=a.output/'rosa_epoch6/predictions.json';done=json.loads((path.parent/'inference_complete.json').read_text())
    if sha(path)!=done['predictions_sha256'] or sha(a.rosa_checkpoint)!=done['checkpoint_sha256']:raise ValueError('ROSA provenance mismatch')
    rosa=json.loads(path.read_text())
    if rosa['box_origin']!='gravity_center' or rosa['score_threshold']>.01:raise ValueError('ROSA box contract')
    tokens=[str(r['token']) for r in ds.rows]
    rmap={str(r['token']):r for r in ray['frames']};smap={str(r['token']):r for r in rosa['frames']}
    if len(rmap)!=len(tokens) or set(rmap)!=set(tokens) or len(smap)!=len(tokens) or set(smap)!=set(tokens):raise ValueError('Prediction coverage mismatch')
    return dict(V7=[np.asarray(rmap[t]['detections'],np.float32).reshape(-1,9) for t in tokens],
        ROSA=[as_rosa_array(smap[t],cfg,ds.T_base_clidar) for t in tokens])


def analyze(a):
    state,cfg,ds=inputs(a);ps=predictions(a,ds,cfg);truth=[g.numpy() for g in ds.boxes]
    sequences=[str(r['metainfo']['sequence_id']) for r in ds.rows]
    old_plan=ROOT/'projects/ROSA/work_dirs/combined_4bags_analysis_20261107/plan.json'
    old_val=set(json.loads(old_plan.read_text())['subsets']['combined_validation_674'])
    subsets=dict(previous_ROSA_validation=[i for i,r in enumerate(ds.rows) if str(r['token']) in old_val],
        additional_bags=[i for i,r in enumerate(ds.rows) if str(r['token']) not in old_val])
    reports={}
    for name,pred in ps.items():
        print('Evaluating common protocol:',name,flush=True)
        reports[name]=dict(metrics=evaluate_detections(pred,truth,score_threshold=.1),
            heading=heading_diagnostics(pred,truth,sequences),confusion=confusion(pred,truth),
            operating={str(s):matched(pred,truth,s) for s in (.1,.3,.5)},per_bag={})
        for bag in sorted(set(sequences)):
            ids=[i for i,s in enumerate(sequences) if s==bag]
            reports[name]['per_bag'][bag]=evaluate_detections([pred[i] for i in ids],[truth[i] for i in ids],score_threshold=.1)
        reports[name]['subsets']={key:dict(frames=len(ids),metrics=evaluate_detections(
            [pred[i] for i in ids],[truth[i] for i in ids],score_threshold=.1)) for key,ids in subsets.items()}
    report=dict(models=reports,frames=len(ds),ray_epoch=state['epoch'],
        protocol='Identical 908 VAL tokens/GT, BASELINK gravity centre, +/-48m XY, score floor .01, cap225, native class NMS .5; same raybev.metrics evaluator.',
        caveat='V7 trained on eight bags, ROSA epoch6 on four. Different training sets and preprocessing: native OpenCV800x640 versus calibrated canonical OCam800x800. Not a controlled architecture trial.',
        ray_checkpoint_sha256=sha(a.checkpoint),rosa_checkpoint_sha256=sha(a.rosa_checkpoint),
        annotation_sha256=sha(ds.annotation_path),calibration_sha256=sha(a.calibration))
    report['matched_object_audit']=object_audit(a,ds,cfg,ps)
    dump(a.output/'comparison.json',report)


def object_audit(a,ds,cfg,predictions):
    """Track concentration and heading on the exact same class-correct objects."""
    train=load_annotations(a.root/'annotations/fisheye3dod_infos_train.pkl')['data_list']
    train_tracks={(str(r['metainfo']['sequence_id']),str(t)) for r in train for t in r['annos']['gt_ids']}
    overlap=set(map(lambda r:str(r['token']),train))&{str(r['token']) for r in ds.rows}
    if overlap:raise ValueError('TRAIN/VAL token overlap')
    groups=defaultdict(Counter);pairs={name:{} for name in predictions}
    heavy_confusions={name:defaultdict(list) for name in predictions}
    val_tracks=defaultdict(set);shared=Counter()
    for i,row in enumerate(ds.rows):
        bag=str(row['metainfo']['sequence_id']);gt=ds.boxes[i].numpy()
        full=parse_box_annotations(row,ds.box_frame,ds.T_base_clidar,ds.z_reference).numpy()
        # VAL may not contain ignored or malformed boxes: assert ID alignment.
        if len(full)!=len(row['annos']['gt_ids']):raise ValueError('Cannot align track IDs')
        ids=np.asarray(row['annos']['gt_ids'])[in_bev_range(full,cfg)]
        keys=[]
        for j,g in enumerate(gt):
            cls=CLASSES[int(g[7])];track=(bag,str(ids[j]));key=bag+'/'+cls+'/'+str(ids[j]);keys.append(key)
            groups[key]['GT']+=1;val_tracks[cls].add(track)
            if track in train_tracks:shared[cls]+=1
        for name,frames in predictions.items():
            taken=set()
            for p in sorted((p for p in frames[i] if p[8]>=.1),key=lambda p:-p[8]):
                choices=[(float(np.linalg.norm(p[:2]-g[:2])),j) for j,g in enumerate(gt) if j not in taken]
                if not choices:continue
                d,j=min(choices)
                if d>=2.:continue
                taken.add(j)
                if int(gt[j,7]) in (3,5):
                    key=CLASSES[int(gt[j,7])]+' -> '+CLASSES[int(p[7])]
                    heavy_confusions[name][key].append((p[3:6]/gt[j,3:6]).tolist())
            used=set()
            for p in sorted((p for p in frames[i] if p[8]>=.3),key=lambda p:-p[8]):
                choices=[(float(np.linalg.norm(p[:2]-g[:2])),j) for j,g in enumerate(gt) if j not in used and p[7]==g[7]]
                if not choices:continue
                d,j=min(choices)
                if d>=2.:continue
                used.add(j);groups[keys[j]][name+'_TP_score03']+=1
                yaw=abs(float(np.arctan2(np.sin(p[6]-gt[j,6]),np.cos(p[6]-gt[j,6]))))
                pairs[name][(i,j)]=dict(class_name=CLASSES[int(p[7])],xy=d,yaw=np.degrees(yaw),
                    axis=np.degrees(min(yaw,np.pi-yaw)),flip=float(yaw>np.pi/2),
                    dimensions_ratio_LWH=(p[3:6]/gt[j,3:6]).tolist())
    common=set.intersection(*(set(p) for p in pairs.values()))
    heading={}
    for cls in CLASSES:
        heading[cls]={}
        for name,rows in pairs.items():
            selected=[rows[k] for k in common if rows[k]['class_name']==cls]
            heading[cls][name]=dict(matches=len(selected),**{k:float(np.mean([r[k] for r in selected])) if selected else None
                for k in ('xy','yaw','axis','flip')})
    size={}
    for name,rows in pairs.items():
        size[name]={cls:dict(matches=sum(r['class_name']==cls for r in rows.values()),
            median_dimension_ratio_LWH=np.median([r['dimensions_ratio_LWH'] for r in rows.values() if r['class_name']==cls],axis=0).tolist()
                if any(r['class_name']==cls for r in rows.values()) else None) for cls in CLASSES}
    return dict(protocol='Class-correct greedy center<2m, score>=0.3; common heading requires both models match the identical GT.',
        train_val_token_overlap=0,train_val_track_overlap={cls:dict(unique_val_tracks=len(tracks),
            unique_val_tracks_also_in_train=len(tracks&train_tracks),val_boxes_from_train_tracks=shared[cls]) for cls,tracks in val_tracks.items()},
        common_object_heading=heading,matched_dimension_ratios=size,
        heavy_class_agnostic_dimensions_score01={name:{key:dict(matches=len(rows),
            median_dimension_ratio_LWH=np.median(rows,axis=0).tolist()) for key,rows in values.items()}
            for name,values in heavy_confusions.items()},
        heavy_tracks={k:dict(v,also_in_train=(k.split('/')[0],k.split('/')[2]) in train_tracks)
            for k,v in sorted(groups.items()) if k.split('/')[1] in ('bus','truck')})


def render(a):
    import subprocess
    state,cfg,ds=inputs(a);ps=predictions(a,ds,cfg);report=json.loads((a.output/'comparison.json').read_text())
    sources={s['name']:Path(s['dataset']) for s in json.loads((a.root/'summary.json').read_text())['sequences']}
    cameras={c.name:c for c in load_cameras(a.calibration)};cv2.setNumThreads(2)
    destination=a.output/'validation_v7_vs_rosa.mp4'
    if destination.exists():raise FileExistsError(destination)
    partial=a.output/'validation_v7_vs_rosa.partial.mp4';process=video_writer(partial,a.fps)
    count=0;seen=set();previous=None;started=time.monotonic()
    try:
        for i,row in enumerate(ds.rows):
            meta=row['metainfo'];bag=str(meta['sequence_id']);gt=ds.boxes[i].numpy()
            path=sources[bag]/'lidar_pcd'/Path(row['pc_info']['pts_path']).name
            pts=cloud(path).copy();T=ds.T_base_clidar;pts[:,:3]=pts[:,:3]@T[:3,:3].T+T[:3,3]
            base=bev_base(pts);images={}
            for cam in CAMERA_ORDER:
                image=cv2.imread(str(a.root/camera_entry(row['cam_info']['cam_fisheye'],cam)['cam_path']))
                if image is None:raise FileNotFoundError(str(row['token'])+'/'+cam)
                images[cam]=cv2.resize(image,(640,512),interpolation=cv2.INTER_AREA)
            canvas=np.full((900,2560,3),15,np.uint8);scene=str(meta['scene_name'])
            text(canvas,f"VALIDATION {i+1}/{len(ds)} | bag {bag} | source frame {meta['source_frame_id']}"+(' | SCENE CUT' if scene!=previous else ''),(18,28),.7)
            previous=scene
            titles=[f'V7 YOLO-ROSA-Lite epoch {state["epoch"]} | trained: 8 bags','ROSA VoV99 epoch 6 | trained: 4 bags']
            for side,name in enumerate(('V7','ROSA')):
                x=1280*side;pred=ps[name][i];pred=pred[pred[:,8]>=a.score_threshold]
                text(canvas,titles[side],(x+18,65),.75,thickness=2)
                text(canvas,f"Common-evaluator mAP {100*report['models'][name]['metrics']['mAP']:.2f}% | display score >= {a.score_threshold:.2f}",(x+18,98),.55)
                canvas[140:780,x:x+640]=draw_bev(base,gt,pred)
                for j,cam in enumerate(CAMERA_ORDER):
                    image=draw_camera(images[cam].copy(),cameras[cam],gt,pred)
                    image=cv2.resize(image,(320,256),interpolation=cv2.INTER_AREA)
                    left=x+640+(j%2)*320;top=150+(j//2)*276
                    canvas[top:top+256,left:left+320]=image;text(canvas,cam,(left+8,top+20),.55,thickness=2)
                text(canvas,frame_counts(pred,gt),(x+18,815),.6)
                text(canvas,'BASELINK +/-48m | white dashed: GT | colors: predictions',(x+18,845),.5)
                for j,c in enumerate(CLASSES):text(canvas,c,(x+15+j*170,880),.5,COLORS[j],2)
            cv2.line(canvas,(1280,40),(1280,900),(100,100,100),2)
            process.stdin.write(canvas.tobytes());count+=1
            if bag not in seen:cv2.imwrite(str(a.output/f'preview_{bag}.jpg'),canvas);seen.add(bag)
            if i%50==0:print('Render',i+1,'/',len(ds),'elapsed',round(time.monotonic()-started,1),flush=True)
    finally:process.stdin.close();code=process.wait()
    if code or count!=len(ds):raise RuntimeError('Incomplete video')
    partial.replace(destination)
    proof=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=width,height,nb_read_frames,r_frame_rate,duration','-of','json',str(destination)],text=True))
    if int(proof['streams'][0]['nb_read_frames'])!=len(ds):raise ValueError('Frame coverage mismatch')
    dump(a.output/'video_manifest.json',dict(video=str(destination),sha256=sha(destination),probe=proof,
        tokens=[str(r['token']) for r in ds.rows],display_score_threshold=a.score_threshold,comparison=report['protocol']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--rosa-checkpoint',type=Path,default=ROOT/'projects/ROSA/work_dirs/combined_4bags/best_fisheye_bev_mAP_epoch_6.pth')
    p.add_argument('--stage',choices=['prepare','rosa','analyze','render'],required=True)
    p.add_argument('--score-threshold',type=float,default=.3);p.add_argument('--fps',type=float,default=2.)
    a=p.parse_args();a.output=a.output.resolve();a.root=a.root.resolve();a.checkpoint=a.checkpoint.resolve();torch.set_num_threads(2)
    dict(prepare=prepare,rosa=rosa_infer,analyze=analyze,render=render)[a.stage](a)


if __name__=='__main__':main()
