#!/usr/bin/env python3
"""Read-only native OpenCV image evidence for every Bus/Van annotation track."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from raybev.config import DEFAULT_CALIBRATION
from raybev.geometry import load_cameras,project_fisheye
from raybev.rosa_dataset import load_annotations,boxes_to_baselink,read_T_base_clidar,camera_entry
from raybev.training import calibration_sha256
from compare_validation_rosa import corners


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);cv2.setNumThreads(2)
    cameras=load_cameras(DEFAULT_CALIBRATION);T=read_T_base_clidar(DEFAULT_CALIBRATION)
    groups=defaultdict(list);rows={};hashes={};records=[]
    for split in ('train','val'):
        path=a.root/f'annotations/fisheye3dod_infos_{split}.pkl';hashes[split]=calibration_sha256(path)
        for row in load_annotations(path)['data_list']:
            token=str(row['token']);rows[token]=row
            boxes=boxes_to_baselink(np.asarray(row['annos']['gt_boxes'])[:,:7],'CLIDAR',T)
            for index,(label,uid,box) in enumerate(zip(row['annos']['gt_names'],row['annos']['gt_ids'],boxes)):
                label=str(label).lower()
                if label not in ('bus','van'):continue
                key=str(row['metainfo']['sequence_id'])+'/'+label+'/'+str(uid)
                views=[]
                for cam in cameras:
                    uv,visible,_=project_fisheye(np.r_[box[None,:3],corners(box)],cam)
                    if not visible[0] or visible[1:].sum()<2:continue
                    xy=uv[1:][visible[1:]];lo=np.maximum(xy.min(0),[0,0]);hi=np.minimum(xy.max(0),[cam.width-1,cam.height-1])
                    area=float(np.maximum(hi-lo,0).prod())
                    if area<16:continue
                    views.append(dict(camera=cam.name,area=area,bbox=[*lo.tolist(),*hi.tolist()]))
                views.sort(key=lambda v:-v['area'])
                rec=dict(key=key,token=token,split=split,index=index,box=box.tolist(),
                    in_range=bool(-48<=box[0]<48 and -48<=box[1]<48),
                    bottom_z=float(box[2]-box[5]/2),views=views)
                records.append(rec);groups[key].append(rec)
    selected=[];tracks=[]
    for number,key in enumerate(sorted(groups),1):
        group=sorted(groups[key],key=lambda r:int(r['token']))
        candidates=[r for r in group if r['views']]
        # Show largest visible observation in early/middle/late temporal thirds.
        chosen=[]
        for indices in np.array_split(np.arange(len(candidates)),min(3,len(candidates))):
            if len(indices):chosen.append(max((candidates[int(i)] for i in indices),key=lambda r:r['views'][0]['area']))
        strips=np.full((225,1200,3),22,np.uint8)
        for col,r in enumerate(chosen):
            v=r['views'][0];cam=next(c for c in cameras if c.name==v['camera']);row=rows[r['token']]
            path=a.root/camera_entry(row['cam_info']['cam_fisheye'],cam.name)['cam_path'];im=cv2.imread(str(path))
            if im is None:raise FileNotFoundError(path)
            if im.shape[:2]!=(cam.height,cam.width):raise ValueError('Native calibration/image dimensions differ')
            lo=np.array(v['bbox'][:2]);hi=np.array(v['bbox'][2:]);pad=np.maximum((hi-lo)*.3,30)
            lo=np.maximum(np.floor(lo-pad),0).astype(int);hi=np.minimum(np.ceil(hi+pad),[cam.width,cam.height]).astype(int)
            crop=im[lo[1]:hi[1],lo[0]:hi[0]].copy();scale=min(395/crop.shape[1],165/crop.shape[0])
            crop=cv2.resize(crop,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            x=col*400+(400-crop.shape[1])//2;y=57+(165-crop.shape[0])//2
            strips[y:y+crop.shape[0],x:x+crop.shape[1]]=crop
            # Small center marker identifies the annotated object without hiding its shape.
            uv,valid,_=project_fisheye(np.asarray(r['box'])[None,:3],cam)
            pt=np.rint((uv[0]-lo)*scale+[x,y]).astype(int)
            cv2.drawMarker(strips,tuple(pt),(0,255,255),cv2.MARKER_CROSS,9,1)
            title=f"{r['split']} {cam.name} LWH "+'/'.join(f'{z:.1f}' for z in r['box'][3:6])+f" bottom {r['bottom_z']:.1f}"
            cv2.putText(strips,title,(col*400+4,43),cv2.FONT_HERSHEY_SIMPLEX,.37,(240,240,240),1,cv2.LINE_AA)
            r['representative']=True
        cv2.putText(strips,f"#{number:03d} {key} | boxes={len(group)} ROI={sum(r['in_range'] for r in group)}",(4,19),cv2.FONT_HERSHEY_SIMPLEX,.53,(255,255,255),1,cv2.LINE_AA)
        if not chosen:cv2.putText(strips,'NO ELIGIBLE CAMERA VIEW',(30,130),cv2.FONT_HERSHEY_SIMPLEX,1,(0,180,255),2)
        cv2.imwrite(str(a.output/f'track_{number:03d}.jpg'),strips)
        tracks.append(dict(number=number,key=key,boxes=len(group),roi=sum(r['in_range'] for r in group),
            no_camera_view=sum(not r['views'] for r in group),representatives=[dict(token=r['token'],index=r['index'],camera=r['views'][0]['camera']) for r in chosen]))
        selected.append(strips)
        if number%25==0:print('Rendered tracks',number,'/',len(groups),flush=True)
    for start in range(0,len(selected),8):
        page=np.concatenate(selected[start:start+8],axis=0)
        cv2.imwrite(str(a.output/f'page_{start//8+1:03d}.jpg'),page)
    (a.output/'index.json').write_text(json.dumps(dict(annotation_sha256=hashes,calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
        note='All annotations indexed; visual sheets sample up to three observations per track. Center marker is GT projection, not detector evidence.',tracks=tracks,annotations=records),indent=2)+'\n')
    print('Annotations',len(records),'Tracks',len(tracks),'Pages',(len(tracks)+7)//8,flush=True)


if __name__=='__main__':main()
