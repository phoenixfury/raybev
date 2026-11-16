#!/usr/bin/env python3
"""Inventory and render every Bus/Truck/Van annotation; leave source labels intact."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from raybev.config import DEFAULT_CALIBRATION
from raybev.geometry import load_cameras,project_fisheye
from raybev.rosa_dataset import load_annotations,boxes_to_baselink,read_T_base_clidar,camera_entry
from compare_validation_rosa import corners,cloud,sha

LABELS=('bus','truck','van')


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pcd_path(root,sources,row):
    relative=Path(row['pc_info']['pts_path']);bag=str(row['metainfo']['sequence_id'])
    candidates=[root/relative,sources[bag]/relative,sources[bag]/'lidar_pcd'/relative.name]
    return next((p for p in candidates if p.is_file()),None)


def lidar_support(points,box):
    if points is None:return None
    xyz=points[:,:3];delta=xyz-box[:3]
    keep=(np.abs(delta[:,0])<=np.hypot(box[3],box[4])/2+.5)&(np.abs(delta[:,1])<=np.hypot(box[3],box[4])/2+.5)
    delta=delta[keep];c,s=np.cos(box[6]),np.sin(box[6])
    x=delta[:,0]*c+delta[:,1]*s;y=-delta[:,0]*s+delta[:,1]*c
    xy=(np.abs(x)<=box[3]/2)&(np.abs(y)<=box[4]/2)
    return dict(points_in_box=int((xy&(np.abs(delta[:,2])<=box[5]/2)).sum()),
                points_in_xy_column=int(xy.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--calibration',type=Path,default=Path(DEFAULT_CALIBRATION))
    a=p.parse_args();a.root=a.root.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);(a.output/'crops').mkdir();(a.output/'pages').mkdir();(a.output/'tracks').mkdir()
    cv2.setNumThreads(1);cameras=load_cameras(a.calibration);T=read_T_base_clidar(a.calibration)
    summary=json.loads((a.root/'summary.json').read_text());sources={r['name']:Path(r['dataset']) for r in summary['sequences']}
    records=[];groups=defaultdict(list);annotation_hashes={};rows={}
    for split in ('train','val'):
      ann=a.root/f'annotations/fisheye3dod_infos_{split}.pkl';annotation_hashes[split]=sha(ann)
      for row in load_annotations(ann)['data_list']:
        token=str(row['token']);rows[token]=row
        raw=np.asarray(row['annos']['gt_boxes']);boxes=boxes_to_baselink(raw,'CLIDAR',T)
        for index,(label,uid,box) in enumerate(zip(row['annos']['gt_names'],row['annos']['gt_ids'],boxes)):
          label=str(label).lower()
          if label not in LABELS:continue
          key='/'.join((str(row['metainfo']['sequence_id']),label,str(uid)))
          views=[]
          for cam in cameras:
            uv,valid,_=project_fisheye(np.r_[box[None,:3],corners(box)],cam)
            if not valid[0] or valid[1:].sum()<2:continue
            xy=uv[1:][valid[1:]];lo=np.maximum(xy.min(0),[0,0]);hi=np.minimum(xy.max(0),[cam.width-1,cam.height-1])
            area=float(np.maximum(hi-lo,0).prod())
            if area>=16:views.append(dict(camera=cam.name,area=area,bbox=[*lo.tolist(),*hi.tolist()],center=uv[0].tolist()))
          views.sort(key=lambda v:-v['area'])
          record=dict(id=len(records)+1,key=key,split=split,token=token,index=index,label=label,
            sequence=str(row['metainfo']['sequence_id']),track_id=str(uid),box_BASELINK=box.tolist(),
            box_CLIDAR=raw[index,:7].tolist(),in_range=bool(-48<=box[0]<48 and -48<=box[1]<48),
            bottom_z=float(box[2]-box[5]/2),views=views)
          records.append(record);groups[key].append(record)
    # Decode each source image/point cloud once for all annotations in a frame.
    by_frame=defaultdict(list)
    for rec in records:by_frame[rec['token']].append(rec)
    for number,(token,frame_records) in enumerate(by_frame.items(),1):
      row=rows[token];point_path=pcd_path(a.root,sources,row);points=cloud(point_path) if point_path else None
      images={}
      for rec in frame_records:
        rec['lidar']=lidar_support(points,np.asarray(rec['box_CLIDAR']))
        rec['lidar_path']=str(point_path) if point_path else None
        tile=np.full((210,280,3),23,np.uint8)
        title=f"A{rec['id']:04d} {rec['split']} {rec['label']} id={rec['track_id']}"
        cv2.putText(tile,title,(5,15),cv2.FONT_HERSHEY_SIMPLEX,.40,(245,245,245),1,cv2.LINE_AA)
        b=rec['box_BASELINK'];support=rec['lidar']
        caption='LWH '+('/'.join(f'{v:.1f}' for v in b[3:6]))+f" z0={rec['bottom_z']:.1f}"
        cv2.putText(tile,caption,(5,31),cv2.FONT_HERSHEY_SIMPLEX,.35,(240,240,240),1,cv2.LINE_AA)
        cv2.putText(tile,rec['sequence']+' n='+str(support['points_in_box'] if support else '?'),(5,46),cv2.FONT_HERSHEY_SIMPLEX,.34,(220,220,220),1,cv2.LINE_AA)
        rec['crop_path']=f"crops/A{rec['id']:04d}.jpg"
        if rec['views']:
          view=rec['views'][0];name=view['camera'];cam=next(c for c in cameras if c.name==name)
          path=a.root/camera_entry(row['cam_info']['cam_fisheye'],name)['cam_path']
          if name not in images:images[name]=cv2.imread(str(path))
          image=images[name]
          if image is None:raise FileNotFoundError(path)
          rec['image_path']=str(path)
          lo=np.asarray(view['bbox'][:2]);hi=np.asarray(view['bbox'][2:]);pad=np.maximum((hi-lo)*.35,30)
          lo=np.maximum(np.floor(lo-pad),0).astype(int);hi=np.minimum(np.ceil(hi+pad),[cam.width,cam.height]).astype(int)
          # Source images may already be calibrated network-sized; convert raw pixels exactly.
          scale=np.array([image.shape[1]/cam.width,image.shape[0]/cam.height])
          pixel_lo=np.floor(lo*scale).astype(int);pixel_hi=np.ceil(hi*scale).astype(int)
          crop=image[pixel_lo[1]:pixel_hi[1],pixel_lo[0]:pixel_hi[0]].copy()
          if not crop.size:raise ValueError((rec['id'],lo,hi))
          rec['crop_raw_xyxy']=[*lo.tolist(),*hi.tolist()]
          ratio=min(276/crop.shape[1],154/crop.shape[0]);small=cv2.resize(crop,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
          x=(280-small.shape[1])//2;y=52+(156-small.shape[0])//2
          tile[y:y+small.shape[0],x:x+small.shape[1]]=small
          pt=np.rint(((np.asarray(view['center'])+.5)*scale-.5-pixel_lo)*ratio+[x,y]).astype(int)
          cv2.drawMarker(tile,tuple(pt),(0,255,255),cv2.MARKER_CROSS,9,1)
        else:
          cv2.putText(tile,'NO ELIGIBLE VIEW',(24,135),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,180,255),1)
        cv2.imwrite(str(a.output/rec['crop_path']),tile)
      if number%100==0:print(f'Frames {number}/{len(by_frame)}, annotations {frame_records[-1]["id"]}/{len(records)}',flush=True)
    # Pages include EVERY annotation exactly once, grouped by track.
    ordered=[];tracks=[]
    for track_number,key in enumerate(sorted(groups),1):
      group=sorted(groups[key],key=lambda r:int(r['token']));ordered.extend(group)
      reps=[]
      for chunk in np.array_split(np.arange(len(group)),min(3,len(group))):
        if len(chunk):reps.append(max((group[int(i)] for i in chunk),key=lambda r:r['views'][0]['area'] if r['views'] else 0))
      tracks.append(dict(number=track_number,key=key,label=group[0]['label'],annotations=len(group),
        split_counts=dict(Counter(r['split'] for r in group)),in_range=sum(r['in_range'] for r in group),
        annotation_ids=[r['id'] for r in group],representatives=[r['id'] for r in reps]))
    for start in range(0,len(ordered),40):
      canvas=np.full((1680,1400,3),12,np.uint8);page=start//40+1
      for j,rec in enumerate(ordered[start:start+40]):
        rec['all_frames_page']=page
        canvas[(j//5)*210:(j//5+1)*210,(j%5)*280:(j%5+1)*280]=cv2.imread(str(a.output/rec['crop_path']))
      cv2.imwrite(str(a.output/f'pages/all_{page:03d}.jpg'),canvas)
    for start in range(0,len(tracks),12):
      canvas=np.full((1680,1260,3),12,np.uint8);page=start//12+1
      for j,track in enumerate(tracks[start:start+12]):
        rec=max((r for r in groups[track['key']] if r['id'] in track['representatives']),key=lambda r:r['views'][0]['area'] if r['views'] else 0)
        tile=cv2.imread(str(a.output/rec['crop_path']));tile=cv2.resize(tile,(420,315))
        tile[:50]=20
        cv2.putText(tile,f"T{track['number']:03d} "+track['key'],(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        cv2.putText(tile,f"N={track['annotations']} best A{rec['id']:04d}",(5,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
        top=(j//3)*420;left=(j%3)*420;canvas[top:top+315,left:left+420]=tile
        for k,rep in enumerate(track['representatives']):
          r=next(r for r in groups[track['key']] if r['id']==rep)
          small=cv2.imread(str(a.output/r['crop_path']));small=cv2.resize(small,(140,105))
          canvas[top+315:top+420,left+k*140:left+(k+1)*140]=small
      cv2.imwrite(str(a.output/f'tracks/representatives_{page:03d}.jpg'),canvas)
    write_json(a.output/'index.json',dict(root=str(a.root),annotation_sha256=annotation_hashes,
      calibration_sha256=sha(a.calibration),annotation_count=len(records),track_count=len(tracks),
      counts={s:dict(Counter(r['label'] for r in records if r['split']==s)) for s in ('train','val')},
      all_frames_pages=(len(ordered)+39)//40,representative_pages=(len(tracks)+11)//12,
      note='Inventory and evidence, not correctness labels. Every annotation has one all-frames tile.',tracks=tracks))
    with (a.output/'annotations.jsonl').open('w') as stream:
      for rec in records:stream.write(json.dumps(rec,allow_nan=False)+'\n')
    print('COMPLETE',len(records),'annotations',len(tracks),'tracks',flush=True)


if __name__=='__main__':main()
