"""Integration test with synthetic native synchronized bags and converter stubs."""
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from PIL import Image

P = Path(__file__).with_name('build_combined_from_synchronized.py')
spec = importlib.util.spec_from_file_location('rebuild', P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_stub_modules(modules):
    (modules / 'ocamcamera.py').write_text('class OcamCamera:\n    def __init__(self, filename, fov): self.filename=filename\n')
    (modules / 'build_improved_detzero_rosa_dataset.py').write_text('''
import numpy as np
CAMERAS=('ARC_F','ARC_R','ARC_SL','ARC_SR')
def build_rosa_anchor(train, count, seed):
    return np.zeros((count, 11), np.float32), sum(len(s['annos']['gt_names']) for s in train)
''')
    (modules / 'append_improved_detzero_rosa_dataset.py').write_text('''
import csv
import numpy as np
from pathlib import Path

def load_sequence(name,sequence,destination,ocam,storage,dry_run,next_id,some_flag):
    rows=list(csv.DictReader((Path(sequence)/'manifest.csv').open()))
    samples=[]
    for i,r in enumerate(rows):
        samples.append({'token':r['timestamp_ns'],
           'metainfo':{'sequence_id':name,'source_frame_id':i,'acquisition_timestamp_ns':int(r['lidar_sensor_timestamp_ns'])},
           'annos':{'gt_names':['Car'], 'gt_boxes':np.asarray([[float(i),0,1,4,2,2,0]],dtype=float),'gt_ids':[next_id+i]}})
    return samples, {'name':name},next_id+len(rows)

def assign_chronological_split(rows,val_fraction,purge,max_gap,membership=None):
    if membership is None:
        boundary=max(1,int(len(rows)*(1-val_fraction)))
        for i,s in enumerate(rows):s['metainfo']['temporal_split']='train' if i<boundary else 'val'
    else:
        for s in rows:s['metainfo']['temporal_split']=membership[str(s['token'])]
    tr=[s for s in rows if s['metainfo']['temporal_split']=='train']
    va=[s for s in rows if s['metainfo']['temporal_split']=='val']
    return tr,va,sum(s['metainfo']['temporal_split']=='gap' for s in rows)
''')


def make_bag(tmp, bag, stamp_start):
    root=tmp / bag
    root.mkdir()
    (root/'detzero_output').mkdir()
    (root/'detzero_output'/'improved_results.json').write_text('{}')
    (root/'fisheye3dod_inference_2hz').mkdir()
    with (root/'fisheye3dod_inference_2hz'/'fisheye3dod_infos_test.pkl').open('wb') as f:
        pickle.dump({'metainfo':{'classes':['Car','Pedestrian','Cyclist']}},f)
    fields=['timestamp_ns','lidar_sensor_timestamp_ns'] + list(m.MANIFEST_COLS.values())
    with (root/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for i in range(3):
            stamp=stamp_start+i*10_000_000
            row={'timestamp_ns':str(stamp),'lidar_sensor_timestamp_ns':str(stamp-5_000_000)}
            for cam,key in m.MANIFEST_COLS.items():
                rel=f'images/{cam.lower()}/{stamp}.jpg'
                path=root/rel;path.parent.mkdir(parents=True,exist_ok=True)
                Image.new('RGB',(40,32),(i*50+20,20,30)).save(path)
                row[key]=rel
            writer.writerow(row)
    return root


def test_full_build(tmp_path, monkeypatch):
    modules=tmp_path/'scripts';modules.mkdir();make_stub_modules(modules)
    # Stub imports are isolated by manipulating modules for this invocation.
    for n in ('ocamcamera','build_improved_detzero_rosa_dataset','append_improved_detzero_rosa_dataset'):
        sys.modules.pop(n,None)
    one=make_bag(tmp_path,'17_09',100000000000000000)
    two=make_bag(tmp_path,'16_09',200000000000000000)
    cal=tmp_path/'calib_results.txt';cal.write_text('example calibration')
    output=tmp_path/'combined_new'
    args=m.make_parser().parse_args(['--native-dataset','17_09',str(one), '--native-dataset','16_09',str(two),
        '--output',str(output),'--ocam-calibration',str(cal),'--builders-dir',str(modules),
        '--native-width','40','--native-height','32','--sequence-boundaries','17_09','1','2',
        '--sequence-boundaries','16_09','1','2'])
    result=m.build(args)
    assert result['frames']==6 and result['train_frames']==2 and result['val_frames']==2
    assert result['purged_frames']==2
    assert result['resized_images']==24
    rows=pickle.load((output/'rosa'/'fisheye3dod_infos_all.pkl').open('rb'))['data_list']
    assert len(rows)==6
    for row in rows:
        for cam in m.CAMERAS:
            rec=row['cam_info']['cam_fisheye'][cam]
            with Image.open(output/rec['cam_path']) as im:
                assert im.size==(800,640)
    bybag=pickle.load((output/'annotations'/'by_bag'/'17_09.pkl').open('rb'))
    assert len(bybag['data_list'])==3
    native=pickle.load((output/'rosa_native'/'fisheye3dod_infos_all.pkl').open('rb'))['data_list'][0]
    assert (output/native['cam_info']['cam_fisheye']['ARC_F']['cam_path']).exists()
    assert (output/'rosa'/'anchors'/'kmeans300_seed3407.npy').exists()
    assert (output/'annotations'/'fisheye3dod_infos_train.pkl').read_bytes()==(output/'rosa'/'fisheye3dod_infos_train.pkl').read_bytes()


def test_no_overwrite(tmp_path):
    p=tmp_path/'exists';p.mkdir()
    assert p.exists()
    # Access protection is validated before importing the builder modules.
    args=m.make_parser().parse_args(['--native-dataset','17_09',str(tmp_path),'--output',str(p),
            '--ocam-calibration',str(tmp_path/'calibration')])
    import pytest
    with pytest.raises(FileExistsError):m.build(args)
