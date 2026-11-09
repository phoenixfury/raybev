#!/usr/bin/env python3
"""Build a TRAIN-only heading cache from ROSA gravity-center detections.

Optionally invoke the existing ROSA inference tool on its native TRAIN dataset.
Never infer on validation or silently reuse a validation prediction cache.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CLASSES
from raybev.rosa_dataset import RosaCombinedDataset,boxes_to_baselink,names_to_ids,read_T_base_clidar,training_boxes_sha256
from raybev.training import calibration_sha256


def build_cache(dataset,document,checkpoint,calibration,lidar_calibration,predictions,
                teacher_box_frame='CLIDAR',min_score=.1,max_distance=1.,max_yaw_error=30.):
    if document.get('box_origin')!='gravity_center':raise ValueError('Teacher boxes must explicitly use gravity_center')
    if any(r['metainfo'].get('temporal_split')!='train' for r in dataset.rows):raise ValueError('TRAIN annotation required')
    tokens=[str(r['token']) for r in dataset.rows]
    frames=document['frames'];teacher_tokens=[str(r['token']) for r in frames]
    if len(set(teacher_tokens))!=len(teacher_tokens) or len(set(tokens))!=len(tokens) or set(teacher_tokens)!=set(tokens):
        raise ValueError('Teacher tokens must equal TRAIN split exactly; validation/partial caches are forbidden')
    if Path(document.get('checkpoint','')).name!=Path(checkpoint).name:
        raise ValueError('Teacher checkpoint does not match prediction provenance')
    by_token={str(r['token']):r for r in frames};result={}
    for token,gt_tensor in zip(tokens,dataset.boxes):
        frame=by_token[token];gt=gt_tensor.numpy();boxes=np.asarray(frame['boxes_3d'],dtype=float)
        if boxes.size==0:boxes=np.empty((0,7))
        if boxes.ndim!=2 or boxes.shape[1]<7:raise ValueError('Invalid teacher boxes')
        boxes=boxes_to_baselink(boxes,teacher_box_frame,read_T_base_clidar(lidar_calibration) if teacher_box_frame=='CLIDAR' else None)
        classes=names_to_ids(frame['names']);scores=np.asarray(frame['scores'],dtype=float)
        if not len(boxes)==len(classes)==len(scores):raise ValueError('Teacher output lengths differ')
        used=set();records=[]
        for i in np.argsort(-scores,kind='stable'):
            if not np.isfinite(boxes[i]).all() or not min_score<=scores[i]<=1:continue
            candidates=[j for j,g in enumerate(gt) if int(g[7])==int(classes[i]) and j not in used]
            if not candidates:continue
            j=min(candidates,key=lambda j:float(np.linalg.norm(gt[j,:2]-boxes[i,:2])))
            if np.linalg.norm(gt[j,:2]-boxes[i,:2])>max_distance:continue
            error=abs(math.atan2(math.sin(boxes[i,6]-gt[j,6]),math.cos(boxes[i,6]-gt[j,6])))
            if error>math.radians(max_yaw_error):continue  # never copy a teacher's reversed direction
            used.add(j);records.append({'gt':gt[j].tolist(),'yaw':float(boxes[i,6]),'weight':float(scores[i])})
        result[token]=records
    if not any(result.values()):raise ValueError('No reliable ROSA training matches passed the gates')
    matched_counts={name:sum(int(r['gt'][7])==ci for records in result.values() for r in records) for ci,name in enumerate(CLASSES)}
    return dict(type='raybev_heading_teacher',format_version=1,split='train',frames=result,
        matched_class_counts=matched_counts,training_class_counts=dataset.gt_counts,
        training_boxes_sha256=training_boxes_sha256(dataset),
        annotation_sha256=calibration_sha256(dataset.annotation_path),
        camera_calibration_sha256=calibration_sha256(calibration),lidar_calibration_sha256=calibration_sha256(lidar_calibration),
        teacher_checkpoint_sha256=calibration_sha256(checkpoint),predictions_sha256=calibration_sha256(predictions),
        teacher_checkpoint=str(checkpoint),box_frame=dataset.box_frame,z_reference=dataset.z_reference,
        teacher_box_frame=teacher_box_frame,min_score=min_score,max_distance=max_distance,max_yaw_error_degrees=max_yaw_error)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,help='RayBEV prepared dataset root')
    p.add_argument('--train-ann',default='rosa/fisheye3dod_infos_train.pkl')
    p.add_argument('--calibration',default=DEFAULT_CALIBRATION)
    p.add_argument('--lidar-calibration')
    p.add_argument('--predictions',type=Path)
    p.add_argument('--teacher-checkpoint',type=Path,required=True)
    p.add_argument('--teacher-box-frame',choices=['CLIDAR','BASELINK'],default='CLIDAR')
    p.add_argument('--rosa-config',type=Path,help='Required to generate predictions when --predictions is omitted')
    p.add_argument('--rosa-data-root',type=Path,help='ROSA native training dataset, not assumed to equal RayBEV resized root')
    p.add_argument('--device',default='0')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output=args.output.resolve();args.teacher_checkpoint=args.teacher_checkpoint.resolve()
    if args.predictions is not None:args.predictions=args.predictions.resolve()
    lidar=args.lidar_calibration or args.calibration
    ds=RosaCombinedDataset(args.root,args.train_ann,calibration=lidar,cfg=ModelConfig())
    if any(r['metainfo'].get('temporal_split')!='train' for r in ds.rows):
        raise ValueError('TRAIN annotation required before teacher inference')
    if args.output.exists():raise FileExistsError('Choose a new teacher output; refusing to overwrite')
    if not args.teacher_checkpoint.is_file():raise FileNotFoundError(args.teacher_checkpoint)
    if args.predictions is None:
        if args.rosa_config is None or args.rosa_data_root is None:p.error('Supply --predictions OR --rosa-config and --rosa-data-root')
        repo=Path(__file__).resolve().parents[3];folder=args.output.parent/(args.output.stem+'_rosa_inference')
        if folder.exists():raise FileExistsError('Teacher inference output exists; use its --predictions or choose a new output name')
        subprocess.run([sys.executable,str(repo/'projects/ROSA/tools/infer_fisheye_checkpoints.py'),
            '--config',str(args.rosa_config.resolve()),'--checkpoint',str(args.teacher_checkpoint),
            '--data-root',str(args.rosa_data_root.resolve()),'--split','train','--output-dir',str(folder),
            '--score-threshold','0.01','--device',args.device,'--no-lidar-visualization'],check=True,cwd=repo)
        args.predictions=folder/args.teacher_checkpoint.stem/'predictions.json'
    cache=build_cache(ds,json.loads(args.predictions.read_text()),args.teacher_checkpoint,args.calibration,lidar,args.predictions,args.teacher_box_frame)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(cache,allow_nan=False)+'\n')
    print(f'TRAIN teacher: {len(cache["frames"])} frames, {sum(map(len,cache["frames"].values()))} reliable heading matches -> {args.output}')
    print('Matched/GT class coverage:',{name:f'{n}/{cache["training_class_counts"][name]}' for name,n in cache['matched_class_counts'].items()})


if __name__=='__main__':main()
