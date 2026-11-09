#!/usr/bin/env python3
"""Read-only truck/bus track audit with OpenCV-projected image contact sheets.

Use --split train to prepare the next TRAIN curation pass. Images alone do not
prove that an occluded/poorly projected annotation is false; never auto-delete.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from raybev.config import CAMERAS,CLASSES,DEFAULT_CALIBRATION
from raybev.rosa_dataset import load_annotations,boxes_to_baselink,read_T_base_clidar,names_to_ids
from raybev.geometry import load_cameras,project_fisheye
from raybev.training import calibration_sha256
from compare_validation_rosa import corners


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--ann',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Choose a new audit directory')
    ann=a.root/(a.ann or Path('rosa/fisheye3dod_infos_'+a.split+'.pkl'))
    document=load_annotations(ann);cameras=load_cameras(a.calibration);T=read_T_base_clidar(a.calibration)
    tracks=defaultdict(lambda:dict(raw_count=0,in_range_count=0,best=None))
    for row in document['data_list']:
        if row['metainfo'].get('temporal_split')!=a.split:raise ValueError('Split mismatch')
        labels=names_to_ids(row['annos']['gt_names'])
        boxes=boxes_to_baselink(np.asarray(row['annos']['gt_boxes'])[:,:7],'CLIDAR',T)
        for box,ci,uid in zip(boxes,labels,row['annos']['gt_ids']):
            if ci not in (3,5) or not np.isfinite(box).all() or min(box[3:6])<=0:continue
            key=(str(row['metainfo']['sequence_id']),str(uid),CLASSES[int(ci)])
            track=tracks[key];track['raw_count']+=1
            inside=-48<=box[0]<48 and -48<=box[1]<48
            track['in_range_count']+=int(inside)
            if not inside:continue
            for cam_idx,cam in enumerate(cameras):
                uv,visible,_=project_fisheye(np.r_[box[None,:3],corners(box)],cam)
                if not visible[0] or visible[1:].sum()<3:continue
                uv=(uv+.5)*[800/cam.width,640/cam.height]-.5
                pts=uv[1:][visible[1:]]
                lo=np.maximum(pts.min(0),[0,0]);hi=np.minimum(pts.max(0),[799,639])
                area=float(np.prod(np.maximum(hi-lo,0)))
                if area>0 and (track['best'] is None or area>track['best'][0]):
                    track['best']=(area,row,box.copy(),cam_idx,lo,hi)
    a.output.mkdir(parents=True);tiles=[];records=[]
    for key,track in tracks.items():
        record=dict(sequence_id=key[0],track_id=key[1],label=key[2],raw_count=track['raw_count'],in_range_count=track['in_range_count'])
        records.append(record)
        if track['best'] is None:record['projection_status']='no eligible visible crop';continue
        area,row,box,cam_idx,lo,hi=track['best'];cam=CAMERAS[cam_idx]
        relative=Path(row['cam_info']['cam_fisheye'][cam]['cam_path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe image path')
        path=a.root/relative;im=cv2.imread(str(path))
        if im is None or im.shape[:2]!=(640,800):raise ValueError(f'Expected 800x640 native OpenCV image: {path}')
        size=hi-lo;lo=np.maximum(0,np.floor(lo-.18*size-10)).astype(int)
        hi=np.minimum([800,640],np.ceil(hi+.18*size+10)).astype(int)
        crop=im[lo[1]:hi[1],lo[0]:hi[0]];tile=np.full((280,360,3),25,np.uint8)
        scale=min(360/crop.shape[1],210/crop.shape[0]);small=cv2.resize(crop,None,fx=scale,fy=scale)
        y=65+(210-small.shape[0])//2;x=(360-small.shape[1])//2;tile[y:y+small.shape[0],x:x+small.shape[1]]=small
        for y,text in [(20,f'{key[0]} {key[2]} id={key[1]}'),(40,f'LWH {box[3]:.1f} {box[4]:.1f} {box[5]:.1f} {cam}'),(59,str(row['token']))]:
            cv2.putText(tile,text,(5,y),cv2.FONT_HERSHEY_SIMPLEX,.45,(235,235,235),1,cv2.LINE_AA)
        filename='_'.join(key)+'.jpg';cv2.imwrite(str(a.output/filename),tile);tiles.append(tile)
        record.update(token=str(row['token']),camera=cam,image=str(path),box_BASELINK=box.tolist(),crop_xyxy=[*lo.tolist(),*hi.tolist()],crop=filename)
    for page,start in enumerate(range(0,len(tiles),16)):
        sheet=np.full((1120,1440,3),25,np.uint8)
        for j,tile in enumerate(tiles[start:start+16]):sheet[(j//4)*280:(j//4+1)*280,(j%4)*360:(j%4+1)*360]=tile
        cv2.imwrite(str(a.output/f'contact_sheet_{page}.jpg'),sheet)
    report=dict(split=a.split,annotation_sha256=calibration_sha256(ann),calibration_sha256=calibration_sha256(a.calibration),
                note='Candidate crops and projections for MANUAL review, not visibility or correctness ground truth.',tracks=records)
    (a.output/'tracks.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Audited',len(records),'truck/bus tracks;',len(tiles),'eligible crops ->',a.output)


if __name__=='__main__':main()
