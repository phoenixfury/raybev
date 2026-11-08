import json
import pickle
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from raybev.config import CLASSES
from raybev.metrics import evaluate_detections
from raybev.rosa_dataset import parse_box_annotations,RosaCombinedDataset


def test_metrics_exclude_unseen_classes():
    g=np.asarray([[10,0,0,4,2,1.5,0,0]],dtype=float)
    p=np.asarray([[10,0,0,4,2,1.5,0,0,.9],[12,0,0,5,2,2,0,4,.7]],dtype=float)
    m=evaluate_detections([p],[g])
    assert m['active_classes']==['car']
    assert m['class_results']['van']['AP'] is None
    assert m['class_results']['car']['AP']==pytest.approx(1,abs=1e-5)
    assert m['NDS'] is None
    assert m['mATE']==pytest.approx(0)
    assert m['mASE']==pytest.approx(0)
    assert m['mAOE']==pytest.approx(0)
    assert m['geometry_score_3term_nonstandard']==pytest.approx(1)


def test_missing_class_ap_zero_if_gt_present():
    gt=np.asarray([[0,0,0,1,1,1,0,4]],dtype=float)
    m=evaluate_detections([np.empty((0,9))],[gt])
    assert m['active_classes']==['van']
    assert m['mAP']==0
    assert m['class_results']['van']['AP']==0


def test_box_parsing_and_lidar_yaw():
    s={'token':'x','annos':{'gt_boxes':np.asarray([[1,2,3,4,2,1,0,100,200]]), 'gt_names':['Truck']}}
    b=parse_box_annotations(s)
    assert b.shape==(1,8) and b[0,-1]==3
    T=np.eye(4);T[:2,:2]=[[-1,0],[0,-1]];T[0,3]=2
    b=parse_box_annotations(s,'CLIDAR',T,'bottom')
    assert b[0,0]==pytest.approx(1.)
    assert b[0,1]==pytest.approx(-2.)
    assert b[0,2]==pytest.approx(3.5)
    assert abs(float(b[0,6]))==pytest.approx(np.pi)


@pytest.mark.parametrize('camera_path_field', ['cam_path', 'img_path', None])
def test_organizer_native_and_rosa_pkl(tmp_path,camera_path_field):
    base=tmp_path/'combined';(base/'annotations').mkdir(parents=True)
    source=tmp_path/'native_source';(source/'images'/'arc_f').mkdir(parents=True)
    for cam in ('arc_f','arc_r','arc_sl','arc_sr'):
        d=source/'images'/cam;d.mkdir(parents=True,exist_ok=True)
        Image.new('RGB',(20,16),(40,60,80)).save(d/'111.jpg')
    row={'token':'t1','metainfo':{'sequence_id':'bag_1','temporal_split':'val','source_frame_id':0,'acquisition_timestamp_ns':100},
         'cam_info':{'cam_fisheye':{cam:dict({camera_path_field:f'images/{cam.lower()}/111.jpg'} if camera_path_field else {},unique_camera_field=42) for cam in ('ARC_F','ARC_R','ARC_SL','ARC_SR')}},
         'annos':{'gt_boxes':np.asarray([[1.,0,0,4,2,1.5,0]]),'gt_names':['Car'],'gt_ids':[1]}}
    info={'metainfo':{'classes':list(CLASSES)},'data_list':[row]}
    for split,rows in [('all',[row]),('val',[row]),('train',[])]:
        with (base/'annotations'/f'fisheye3dod_infos_{split}.pkl').open('wb') as f:pickle.dump(dict(info,data_list=rows),f)
    import csv
    with (source/'manifest.csv').open('w',newline='') as f:
        cols=['timestamp_ns','lidar_sensor_timestamp_ns','arc_f_path','arc_r_path','arc_sl_path','arc_sr_path']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerow(dict(timestamp_ns=111,lidar_sensor_timestamp_ns=100,**{f'{c}_path':f'images/{c}/111.jpg' for c in ('arc_f','arc_r','arc_sl','arc_sr')}))
    out=tmp_path/'prepared'
    script=Path(__file__).parents[1]/'scripts'/'prepare_combined_rosa.py'
    subprocess.run([sys.executable,str(script),'--base',str(base),'--output',str(out),
                    '--native-dataset','bag_1',str(source),'--native-width','20','--native-height','16'],check=True)
    with (out/'rosa/fisheye3dod_infos_val.pkl').open('rb') as f:info=pickle.load(f)
    s=info['data_list'][0]
    assert s['token']=='t1'
    assert s['annos']['gt_names']==['Car']
    assert s['cam_info']['cam_fisheye']['ARC_F']['unique_camera_field']==42
    if camera_path_field:
        assert s['cam_info']['cam_fisheye']['ARC_F'][camera_path_field] == s['cam_info']['cam_fisheye']['ARC_F']['cam_path']
    assert Image.open(out/s['cam_info']['cam_fisheye']['ARC_F']['cam_path']).size==(800,640)
    assert Image.open(out/'native_images/bag_1/ARC_F/t1.jpg').size==(20,16)
    assert (out/'annotations/by_bag/bag_1.pkl').is_file()
    ds=RosaCombinedDataset(out,'rosa/fisheye3dod_infos_val.pkl')
    image,boxes,token=ds[0]
    assert image.shape==(4,3,640,800) and boxes.shape==(1,8) and token=='t1'
    assert subprocess.run([sys.executable,str(script),'--base',str(base),'--output',str(out)],capture_output=True).returncode != 0
