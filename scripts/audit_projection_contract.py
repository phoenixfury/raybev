#!/usr/bin/env python3
"""Read-only frame/calibration audit and native OpenCV projection examples."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from raybev.geometry import load_cameras,project_fisheye
from raybev.rosa_dataset import load_annotations
from compare_validation_rosa import CAMERA_KEYS,CAMERA_ORDER,corners,dashed,text,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--rosa-analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    current=json.loads(args.calibration.read_text())
    cameras={c.name:c for c in load_cameras(args.calibration)}
    rows=load_annotations(args.root/'rosa/fisheye3dod_infos_val.pkl')['data_list']
    plan=json.loads((args.rosa_analysis/'plan.json').read_text())
    old={str(r['token']):r for r in load_annotations(plan['union'])['data_list']}
    summary=json.loads((args.root/'summary.json').read_text())
    report={'annotation_frame':'CLIDAR: source CenterPoint/DetZero labels copied unchanged by dataset builder',
            'trained_box_frame':'BASELINK: checkpoint recipe; therefore no CLIDAR-to-BASELINK conversion occurred',
            'training_code_provenance':'tools/detzero_temporal_pipeline/integration_scripts/run_detzero_sequence.py::world_lidar_poses; tools/custom_data/append_improved_detzero_rosa_dataset.py::load_sequence',
            'bags':{}}
    for bag in sorted(summary['bag_stats']):
        source=Path(summary['bag_stats'][bag]['source']).parent/'calibration.json'
        document=json.loads(source.read_text());calibrated=load_cameras(source)
        T=np.asarray(document['extrinsics']['CLIDAR']['matrix'])
        members=[r for r in rows if r['metainfo']['sequence_id']==bag]
        boxes=np.concatenate([np.asarray(r['annos']['gt_boxes'])[:,:7] for r in members])
        transformed=boxes[:,:3]@T[:3,:3].T+T[:3,3]
        camera_results={}
        for cam in calibrated:
            raw=np.asarray(current['extrinsics'][cam.name]['matrix'])
            rotation=raw[:3,:3].T@cam.T_base_cam[:3,:3]
            deltas=[];extrinsic_only=[];template_residual=[]
            for r in members[::max(1,len(members)//30)]:
                xyz=np.asarray(r['annos']['gt_boxes'])[:,:3]
                if not len(xyz):continue
                recorded=np.asarray(old[str(r['token'])]['cam_info']['cam_fisheye'][CAMERA_KEYS[cam.name]]['lidar2cam'])
                expected=np.linalg.inv(cam.T_base_cam)@T
                template_residual.append(float(abs(expected-recorded).max()))
                good=replace(cameras[cam.name],T_base_cam=np.linalg.inv(recorded))
                uv,ok,_=project_fisheye(xyz,good)
                wrong,bad_ok,_=project_fisheye(xyz,cameras[cam.name])
                fixed_frame=xyz@T[:3,:3].T+T[:3,3]
                frame_uv,frame_ok,_=project_fisheye(fixed_frame,cameras[cam.name])
                both=ok&bad_ok
                deltas.extend((np.linalg.norm(uv[both]-wrong[both],axis=1)*800/cam.width).tolist())
                both=ok&frame_ok
                extrinsic_only.extend((np.linalg.norm(uv[both]-frame_uv[both],axis=1)*800/cam.width).tolist())
            camera_results[cam.name]={
                'source_vs_training_camera_translation_m':float(np.linalg.norm(cam.T_base_cam[:3,3]-raw[:3,3])),
                'source_vs_training_camera_rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),
                'K_max_difference':float(abs(cam.intrinsic-cameras[cam.name].intrinsic).max()),
                'D_max_difference':float(abs(cam.distortion-cameras[cam.name].distortion).max()),
                'source_lidar2cam_vs_ROSA_template_max_matrix_difference':max(template_residual,default=0.),
                'wrong_frame_and_extrinsics_pixel_error_median_p90_800wide':np.percentile(deltas,[50,90]).tolist() if deltas else None,
                'correct_frame_but_old_extrinsics_pixel_error_median_p90_800wide':np.percentile(extrinsic_only,[50,90]).tolist() if extrinsic_only else None}
        report['bags'][bag]={'source_calibration':str(source),'T_BASELINK_CLIDAR':T.tolist(),
            'raw_GT_Z_median':float(np.median(boxes[:,2])),
            'converted_BASELINK_GT_Z_median':float(np.median(transformed[:,2])),
            'cameras':camera_results}
        # Two explicit alternatives on the same native pixels: incorrect as-trained
        # frame/extrinsics versus source annotation-frame optical projection.
        row=max(members,key=lambda r:len(r['annos']['gt_boxes']))
        canvas=np.full((710,1600,3),15,np.uint8)
        text(canvas,f'{bag} | token {row["token"]} | OpenCV fisheye in BOTH rows',(10,24),.6)
        for ci,name in enumerate(CAMERA_ORDER):
            original=cv2.imread(str(args.root/row['cam_info']['cam_fisheye'][name]['cam_path']))
            recorded=np.asarray(old[str(row['token'])]['cam_info']['cam_fisheye'][CAMERA_KEYS[name]]['lidar2cam'])
            choices=[cameras[name],replace(cameras[name],T_base_cam=np.linalg.inv(recorded))]
            for ri,camera in enumerate(choices):
                image=original.copy()
                for box in np.asarray(row['annos']['gt_boxes']):
                    uv,visible,_=project_fisheye(corners(box),camera)
                    uv=(uv+.5)*np.array([image.shape[1]/camera.width,image.shape[0]/camera.height])-.5
                    for a,b in ((0,1),(0,2),(1,3),(2,3),(4,5),(4,6),(5,7),(6,7),(0,4),(1,5),(2,6),(3,7)):
                        if visible[a] and visible[b] and np.linalg.norm(uv[a]-uv[b])<image.shape[1]*.65:
                            cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),
                                     (40,70,255) if ri==0 else (30,255,70),2,cv2.LINE_AA)
                image=cv2.resize(image,(400,320),interpolation=cv2.INTER_AREA)
                top=55+ri*330;canvas[top:top+320,ci*400:(ci+1)*400]=image
                text(canvas,name,(ci*400+8,top+22),.6,thickness=2)
        text(canvas,'TOP RED: as-trained BASELINK assumption | BOTTOM GREEN: recorded CLIDAR-to-camera',(10,703),.56)
        cv2.imwrite(str(args.output/f'projection_audit_{bag}.jpg'),canvas)
    dump(args.output/'projection_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
