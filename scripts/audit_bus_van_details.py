"""Render four-camera evidence for selected audit tracks; never edit labels."""
import json
from pathlib import Path
import cv2
import numpy as np
from audit_bus_van_images import (load_annotations, load_cameras, DEFAULT_CALIBRATION,
    camera_entry, project_fisheye, corners)

ROOT=Path('/home/mostafa/mmdetection3d/data/Custom/combined_bags')
OUT=Path(__file__).resolve().parents[1]/'work_dirs/bus_van_visual_audit_20261113'
data=json.loads((OUT/'index.json').read_text())
rows={str(r['token']):r for s in ('train','val') for r in load_annotations(ROOT/f'annotations/fisheye3dod_infos_{s}.pkl')['data_list']}
cams=load_cameras(DEFAULT_CALIBRATION)
edges=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
for number in (5,159,160,164,236,260,261,272,275,341,344,354):
    track=data['tracks'][number-1]
    for k,rep in enumerate(track['representatives']):
        rec=next(r for r in data['annotations'] if r['token']==rep['token'] and r['index']==rep['index'])
        tiles=[];box=np.array(rec['box']);cs=corners(box)
        for cam in cams:
            path=ROOT/camera_entry(rows[rep['token']]['cam_info']['cam_fisheye'],cam.name)['cam_path']
            im=cv2.imread(str(path))
            if im is None:raise FileNotFoundError(path)
            if im.shape[:2]!=(cam.height,cam.width):raise ValueError(path)
            im=cv2.resize(im,(800,640));scale=np.array([800/cam.width,640/cam.height])
            for i,j in edges:
                points=cs[i]+np.linspace(0,1,65)[:,None]*(cs[j]-cs[i])
                uv,valid,_=project_fisheye(points,cam);uv=np.rint((uv+.5)*scale-.5).astype(int)
                for t in range(64):
                    if valid[t] and valid[t+1]:cv2.line(im,tuple(uv[t]),tuple(uv[t+1]),(0,220,255),1,cv2.LINE_AA)
            uv,valid,_=project_fisheye(box[None,:3],cam)
            if valid[0]:cv2.drawMarker(im,tuple(np.rint((uv[0]+.5)*scale-.5).astype(int)),(0,0,255),cv2.MARKER_CROSS,18,2)
            cv2.putText(im,cam.name,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
            tiles.append(im)
        panel=np.vstack((np.hstack(tiles[:2]),np.hstack(tiles[2:])))
        cv2.putText(panel,f"#{number} {track['key']} {rec['split']} token={rep['token']}",(12,1265),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
        cv2.imwrite(str(OUT/f'detail_{number:03d}_{k}.jpg'),panel)
print('Detailed four-camera views rendered')
