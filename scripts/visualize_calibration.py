#!/usr/bin/env python3
"""Make raw camera overlays from BASELINK grid points to validate geometry.

Uses the rig's OpenCV fisheye model and BASELINK camera extrinsics.
Inspect these overlays before trusting labels/projected features.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageDraw
from raybev.rosa_dataset import load_annotations
from raybev.geometry import load_cameras,project_fisheye
from raybev.config import CAMERAS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--ann',default='native/rosa/fisheye3dod_infos_val.pkl')
    p.add_argument('--calibration',default='configs/calibration_baselink.json')
    p.add_argument('--output',default='calibration_overlays')
    p.add_argument('--samples',type=int,default=3)
    args=p.parse_args()
    root=Path(args.root);ds=load_annotations(root/args.ann)
    cams=load_cameras(args.calibration);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    xx=np.arange(2.,25.,2.); yy=np.arange(-10.,11.,2.)
    grid=np.stack(np.meshgrid(xx,yy,indexing='ij'),axis=-1).reshape(-1,2)
    points=np.column_stack((grid,np.zeros(len(grid))))
    for s in ds['data_list'][:args.samples]:
        for c,cam in zip(CAMERAS,cams):
            rel=Path(s['cam_info']['cam_fisheye'][c]['cam_path'])
            with Image.open(root/rel) as im: im=im.convert('RGB')
            uv,visible,_=project_fisheye(points,cam)
            # If using resized images, scale the raw projection accordingly.
            sx,sy=im.width/cam.width,im.height/cam.height
            canvas=ImageDraw.Draw(im)
            for (u,v),valid in zip(uv,visible):
                if valid:
                    u*=sx;v*=sy
                    canvas.ellipse([u-4,v-4,u+4,v+4],outline='yellow',width=2)
            im.save(out/f'{s["token"]}_{c}.jpg',quality=95)
    print('Check overlays in',out)

if __name__=='__main__':main()
