"""Execute one combined-training epoch with teacher, tracks and heading reports.

Uses small synthetic camera images/teacher outputs, not an accuracy benchmark.
Checkpoint payloads are inspected in memory to avoid large test artifacts.
"""
import csv
import importlib.util
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from raybev.config import ModelConfig,DEFAULT_CALIBRATION,CAMERAS
from raybev.rosa_dataset import RosaCombinedDataset
from raybev.model import RayBEVLite
from raybev.training import fit_training_config,calibration_sha256
from raybev.config import CLASSES


def script(name):
    path=Path(__file__).parents[1]/'scripts'/f'{name}.py'
    spec=importlib.util.spec_from_file_location('test_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('recovery',[False,True])
@pytest.mark.parametrize('freeze',['none','early'])
def test_combined_training_connects_teacher_track_diagnostics_and_checkpoint_contract(tmp_path,monkeypatch,recovery,freeze):
    cfg=ModelConfig().downscale(5);rows=[]
    for i in range(3):
        cameras={}
        for j,cam in enumerate(CAMERAS):
            path=tmp_path/f'{i}_{cam}.jpg';Image.new('RGB',(160,128),(60+i*5,80+j*5,100)).save(path)
            cameras[cam]={'cam_path':path.name}
        rows.append(dict(token=str(i),cam_info={'cam_fisheye':cameras},annos=dict(
            gt_boxes=np.array([[4.,2.,0.,4.5,2.,1.5,.3]]),gt_names=['Car'],gt_ids=np.array([7])),
            metainfo=dict(temporal_split='train' if i<2 else 'val',sequence_id='bag',
                acquisition_timestamp_ns=1_000_000_000+i*500_000_000,T_world_CLIDAR=np.eye(4))))
    for name,data in [('train',rows[:2]),('val',rows[2:])]:
        with (tmp_path/f'{name}.pkl').open('wb') as f:pickle.dump({'data_list':data},f)
    ds=RosaCombinedDataset(tmp_path,'train.pkl',cfg=cfg)
    teacher=tmp_path/'rosa.pth';teacher.write_bytes(b'synthetic teacher fixture')
    doc=dict(box_origin='gravity_center',checkpoint=str(teacher),frames=[dict(token=r['token'],
        boxes_3d=r['annos']['gt_boxes'].tolist(),names=['Car'],scores=[.9]) for r in rows[:2]])
    predictions=tmp_path/'predictions.json';predictions.write_text(json.dumps(doc))
    cache=script('prepare_heading_teacher').build_cache(ds,doc,teacher,DEFAULT_CALIBRATION,DEFAULT_CALIBRATION,predictions)
    cachefile=tmp_path/'teacher.json';cachefile.write_text(json.dumps(cache))
    extra=[]
    if recovery:
        old_cfg=replace(fit_training_config(cfg,ds.boxes),bev_normalization='batchnorm',
                        size_encoding='shared_relative',heavy_classification_weight=.2,loss_normalization='batch_objects',
                        classification_mode='heatmap',foreground_label_smoothing=0.,center_classification_weight=.1)
        old=RayBEVLite(old_cfg,DEFAULT_CALIBRATION)
        checkpoint=tmp_path/'v4.pt'
        torch.save(dict(cfg=vars(old_cfg),state_dict=old.state_dict(),epoch=20,classes=CLASSES,
                        box_frame='CLIDAR',calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
                        lidar_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION)),checkpoint)
        extra=['--init-from',str(checkpoint)]
    train=script('train_combined_rosa')
    monkeypatch.setattr(train,'ModelConfig',lambda **kwargs:replace(cfg,**kwargs))
    def dataset(**kwargs):return RosaCombinedDataset(**kwargs,expected_size=(160,128))
    monkeypatch.setattr(train,'RosaCombinedDataset',dataset)
    saved={}
    def save(state,path):
        saved[Path(path).name]={k:state[k] for k in ('cfg','heading_teacher_sha256','epoch','best_heading_aoe','heading_diagnostics','warm_start')}
    monkeypatch.setattr(train.torch,'save',save)
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train','--root',str(tmp_path),'--train-ann','train.pkl','--val-ann','val.pkl',
        '--epochs','1','--batch-size','1','--workers','0','--no-balance','--no-augmentation',
        '--freeze-image-stages',freeze,
        '--warmup-steps','0','--heading-teacher',str(cachefile),'--out',str(out),'--progress','off']+
        (extra if recovery else ['--no-pretrained']))
    train.main()
    row=json.loads((out/'history.jsonl').read_text().splitlines()[0])
    assert row['train_heading_teacher_loss']>0 and row['train_heading_track_loss']>0
    assert row['train_heading_direction_loss']==row['train_heading_bin_loss']
    assert row['train_heading_axis_loss']>=0
    assert row['center_diagnostics']['train']['car']['count']==2
    assert row['center_diagnostics']['val']['car']['count']==1
    assert row['center_diagnostics']['val']['truck']['class_accuracy'] is None
    assert 'bag' in row['heading_diagnostics']['by_sequence']
    assert saved['last.pt']['cfg']['backbone']=='resnet18'
    assert saved['last.pt']['cfg']['heading_encoding']=='axis_direction'
    assert saved['last.pt']['cfg']['bev_normalization']=='frozen_batchnorm'
    assert saved['last.pt']['cfg']['size_encoding']=='class_relative'
    assert saved['last.pt']['cfg']['classification_mode']=='objectness_softmax'
    assert saved['last.pt']['epoch']==1
    assert bool(saved['last.pt']['warm_start'])==recovery
    assert 'train_heavy_classification_loss' in row and 'val_heavy_classification_loss' in row
    assert saved['last.pt']['heading_teacher_sha256']
    assert 'by_score' in saved['last.pt']['heading_diagnostics']
    assert 'best_map.pt' in saved and 'best_heading.pt' not in saved  # insufficient matched recall
    with (out/'epochs.csv').open() as f:flat=next(csv.DictReader(f))
    assert 'heading_score0.3_AOE_deg_car' in flat
