"""Audited V6 -> V6.1 weight migration and complete, resume-safe metric logs."""
import copy
import csv
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import pickle
import sys

import numpy as np
import pytest
import torch

from raybev import SparseConfig,YOLOROSALite
from raybev.config import CLASSES
from raybev.metrics import evaluate_detections,heading_diagnostics
from raybev.training import calibration_sha256


PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'scripts'))
spec=importlib.util.spec_from_file_location('sparse_refinement_trainer',PROJECT/'scripts/train_yolo_rosa_sparse.py')
trainer=importlib.util.module_from_spec(spec);spec.loader.exec_module(trainer)


def config(version=3):
    torch.set_num_threads(2)
    return SparseConfig(architecture_version=version,image_width=160,image_height=128,
                        channels=16,ffn_channels=32,queries=8,stages=4,max_detections=8).validate()


def contract():
    return dict(classes=list(CLASSES),box_frame='CLIDAR',z_reference='center',
                calibration_sha256='camera',lidar_calibration_sha256='lidar',
                annotation_sha256={'train':'raw-train','val':'raw-val'},
                semantic_annotation_sha256={'train':'semantic-train','val':'semantic-val'})


def checkpoint(model):
    return dict(architecture=model.cfg.architecture,cfg=model.cfg.to_dict(),
                state_dict=copy.deepcopy(model.state_dict()),heading_teacher_sha256='teacher',**contract())


def test_sparse_initialization_options_are_explicit_and_mutually_exclusive():
    parser=trainer.build_parser()
    assert parser.parse_args(['--root','data']).architecture_version==3
    args=parser.parse_args(['--root','data','--architecture-version','4','--init-from','best_map.pt'])
    assert args.architecture_version==4 and args.init_from=='best_map.pt' and args.resume is None
    with pytest.raises(SystemExit):parser.parse_args(['--root','data','--resume','last.pt','--init-from','best_map.pt'])


def test_sparse_warm_start_preserves_contract_but_allows_new_schedule_and_portable_pickle():
    source=checkpoint(YOLOROSALite(config()))
    current=contract();current['annotation_sha256']={'train':'repacked-train','val':'repacked-val'}
    assert trainer.check_warm_start(source,config(4),current,'teacher').architecture_version==3
    source4=checkpoint(YOLOROSALite(config(4)))
    assert trainer.check_warm_start(source4,config(4),current,'teacher').architecture_version==4
    with pytest.raises(ValueError,match='3 -> 4'):trainer.check_warm_start(source,config(),current,'teacher')
    with pytest.raises(ValueError,match='geometry/config.*queries'):
        trainer.check_warm_start(source,replace(config(4),queries=9,max_detections=9),current,'teacher')


@pytest.mark.parametrize('field',['classes','box_frame','z_reference','calibration_sha256',
                                 'lidar_calibration_sha256','semantic_annotation_sha256'])
def test_sparse_warm_start_rejects_data_or_calibration_changes(field):
    source=checkpoint(YOLOROSALite(config()));source[field]='different'
    with pytest.raises(ValueError,match='data/calibration.*'+field):
        trainer.check_warm_start(source,config(4),contract(),'teacher')


def test_sparse_warm_start_rejects_teacher_change():
    source=checkpoint(YOLOROSALite(config()))
    with pytest.raises(ValueError,match='teacher changed'):
        trainer.check_warm_start(source,config(4),contract(),'another-teacher')


def test_v3_to_v4_weights_are_enumerated_and_semantically_changed_outputs_reset():
    source_model=YOLOROSALite(config());target=YOLOROSALite(config(4))
    with torch.no_grad():
        for parameter in source_model.parameters():parameter.fill_(.125)
    source=checkpoint(source_model);initial=copy.deepcopy(target.state_dict())
    report=trainer.transfer_sparse_weights(target,source);actual=target.state_dict()
    assert len(report['expanded'])==8 and len(report['reset'])==20 and report['new']
    assert all(name in report['reset'] for name in ('layers.0.axis.weight','layers.3.size.bias','layers.2.center.weight'))
    assert 'layers.0.center.weight' in report['copied'] and 'layers.1.center.weight' in report['copied']
    for name in report['copied']:torch.testing.assert_close(actual[name],source['state_dict'][name])
    for name in report['reset']+report['new']:torch.testing.assert_close(actual[name],initial[name])
    for name in report['expanded']:
        count=7*target.cfg.heads
        torch.testing.assert_close(actual[name][:count],source['state_dict'][name])
        torch.testing.assert_close(actual[name][count:],initial[name][count:])


@pytest.mark.parametrize('mutation',['unexpected','missing','wrong_shape'])
def test_sparse_warm_start_rejects_unapproved_keys_before_mutating_model(mutation):
    source=checkpoint(YOLOROSALite(config()));target=YOLOROSALite(config(4))
    initial=copy.deepcopy(target.state_dict())
    if mutation=='unexpected':source['state_dict']['arbitrary.weight']=torch.ones(1)
    elif mutation=='missing':source['state_dict'].pop('query_features')
    else:source['state_dict']['query_features']=torch.ones(1,1)
    with pytest.raises(ValueError,match='Unexpected'):trainer.transfer_sparse_weights(target,source)
    for name,value in target.state_dict().items():torch.testing.assert_close(value,initial[name])


def test_v4_warm_start_copies_every_tensor_without_resetting_outputs():
    source=checkpoint(YOLOROSALite(config(4)));target=YOLOROSALite(config(4))
    report=trainer.transfer_sparse_weights(target,source)
    assert not report['reset'] and not report['expanded'] and not report['new']
    for name,value in target.state_dict().items():torch.testing.assert_close(value,source['state_dict'][name])


def metric_row():
    ground_truth=[np.array([[10.,0.,1.,4.,2.,2.,0.,0.],[20.,0.,1.,10.,3.,4.,0.,3.]])]
    predictions=[np.array([[10.2,0.,1.,4.,2.,2.,math.pi/4,0.,.8]])]
    metrics=evaluate_detections(predictions,ground_truth,score_threshold=.1)
    return dict(epoch=1,train_loss=10.,val_loss=8.,lr=.0002,seconds=1.,world_size=2,
                global_batch_size=32,train_samples_seen=4,val_samples_seen=1,metrics=metrics,
                measured_geometry=trainer.measured_geometry(metrics),
                train_components={key:1. for key in trainer.COMPONENTS},
                val_components={key:2. for key in trainer.COMPONENTS},
                heading_diagnostics=heading_diagnostics(predictions,ground_truth))


def test_sparse_flat_metrics_include_geometry_losses_and_match_coverage():
    row=metric_row();flat=trainer.flat_metrics(row)
    assert flat['mATE']==pytest.approx(.6) and flat['measured_mATE']==pytest.approx(.2)
    assert flat['measured_mAOE_deg']==pytest.approx(45.)
    assert flat['truck_geometry_status']=='no_matches' and flat['truck_measured_AOE'] is None
    assert flat['car_geometry_status']=='measured' and flat['car_heading_matches']==1
    assert flat['van_geometry_status']=='no_ground_truth'
    assert flat['heading_0.1_car_matched_count']==1
    for phase in ('train','val'):
        assert all(f'{phase}_{key}_loss' in flat for key in trainer.COMPONENTS)
    assert all(key in flat for key in ('mATE','mASE','mAOE','mAOE_deg','truck_AP','truck_ATE','truck_ASE','truck_AOE'))


def test_sparse_csv_upgrade_preserves_old_header_and_history(tmp_path):
    old=tmp_path/'epochs.csv';old.write_text('epoch,train_loss,val_loss,mAP,lr\n1,10,8,0.1,0.0002\n')
    history=tmp_path/'history.jsonl';history.write_text('{"epoch":1}\n')
    flat=trainer.flat_metrics(metric_row());path=trainer.append_metrics_csv(tmp_path,flat)
    assert path.name=='epochs_metrics_v2.csv' and old.read_text().startswith('epoch,train_loss,val_loss,mAP,lr\n')
    assert len(old.read_text().splitlines())==2 and history.read_text()=='{"epoch":1}\n'
    second=trainer.append_metrics_csv(tmp_path,{**flat,'epoch':2})
    assert second==path
    with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==2 and rows[1]['epoch']=='2' and rows[0]['measured_mAOE_deg']=='45.0'


def test_sparse_console_reports_train_val_components_and_geometric_coverage(capsys):
    trainer.print_epoch_metrics(metric_row());output=capsys.readouterr().out
    assert 'mATE=' in output and 'mASE=' in output and 'mAOE=' in output and 'geom_n' in output
    assert 'TRAIN components:' in output and 'VAL components:' in output and 'placeholders' in output
    assert 'truck' in output and 'Teacher disabled; VAL excludes denoising/extent losses' in output


def test_sparse_trainer_rejects_removed_teacher_options():
    parser=trainer.build_parser()
    for flags in (['--heading-teacher','teacher.json'],['--heading-teacher-weight','0.2']):
        with pytest.raises(SystemExit):parser.parse_args(['--root','data',*flags])


def test_sparse_resume_normalizes_only_inactive_teacher_weight():
    args=trainer.build_parser().parse_args(['--root','data'])
    cfg=replace(config(),teacher_weight=0.)
    state=dict(architecture=cfg.architecture,cfg=replace(cfg,teacher_weight=.2).to_dict(),heading_teacher_sha256=None,
               world_size=1,recipe=vars(args),**contract())
    trainer.check_resume(state,cfg,args,contract(),None,1)
    state['cfg']=replace(cfg,teacher_weight=0.).to_dict()
    trainer.check_resume(state,cfg,args,contract(),None,1)
    state['heading_teacher_sha256']='active-teacher'
    with pytest.raises(ValueError,match='teacher changed'):
        trainer.check_resume(state,cfg,args,contract(),None,1)
    state['heading_teacher_sha256']=None
    with pytest.raises(ValueError,match='architecture/config'):
        trainer.check_resume(state,replace(cfg,quality_score_power=0.),args,contract(),None,1)


def test_sparse_weight_warm_start_is_fresh_audited_and_needs_no_original_yolo_file(tmp_path,monkeypatch):
    from PIL import Image
    from raybev.config import CAMERAS
    rows=[]
    for i in range(4):
        cameras={}
        for camera in CAMERAS:
            name=f'{i}_{camera}.jpg';Image.new('RGB',(160,128),(70,80+i,90)).save(tmp_path/name)
            cameras[camera]=dict(cam_path=name)
        rows.append(dict(token=str(i),cam_info=dict(cam_fisheye=cameras),
            metainfo=dict(sequence_id='bag',temporal_split='train' if i<3 else 'val'),
            annos=dict(gt_boxes=np.array([[8.+i,2.,1.,8.,3.,3.,.4]]),gt_names=['Truck'],gt_ids=[10])))
    for name,subset in [('train',rows[:3]),('val',rows[3:])]:
        with (tmp_path/f'{name}.pkl').open('wb') as stream:pickle.dump(dict(data_list=subset),stream)
    arguments=['train','--root',str(tmp_path),'--train-ann','train.pkl','--val-ann','val.pkl',
               '--image-width','160','--image-height','128','--channels','16','--queries','8','--stages','4',
               '--epochs','1','--batch-size','2','--accum-steps','1','--workers','0','--allow-cpu',
               '--no-augmentation','--warmup-steps','0','--progress','off']
    previous=tmp_path/'previous';current=tmp_path/'current'
    monkeypatch.setattr(sys,'argv',arguments+['--out',str(previous),'--no-pretrained']);trainer.main()
    source=previous/'last.pt';source_digest=calibration_sha256(source)
    calls=[];constructor=trainer.YOLOROSALite
    def instrumented(*args,**kwargs):
        calls.append((kwargs.get('pretrained'),kwargs.get('backbone_weights')))
        return constructor(*args,**kwargs)
    monkeypatch.setattr(trainer,'YOLOROSALite',instrumented)
    new_arguments=arguments+['--out',str(current),'--architecture-version','4','--init-from',str(source),
                             '--backbone-weights',str(tmp_path/'not_present_yolo.pt'),'--lr','.0003']
    monkeypatch.setattr(sys,'argv',new_arguments);trainer.main()
    result=torch.load(current/'last.pt',map_location='cpu',weights_only=False)
    old=torch.load(source,map_location='cpu',weights_only=False)
    assert result['cfg']['teacher_weight']==0 and result['heading_teacher_sha256'] is None
    assert result['recipe'].get('heading_teacher') is None
    assert calls==[(False,None)]
    assert result['cfg']['architecture_version']==4 and result['epoch']==1
    assert result['warm_start']['checkpoint_sha256']==source_digest and result['warm_start']['epoch']==1
    assert result['warm_start']['optimizer']=='fresh' and result['warm_start']['transfer_report']['reset']
    assert result['recipe']['lr']==.0003 and old['recipe']['lr']==.0002
    assert 'val_components' in json.loads((current/'history.jsonl').read_text())
    monkeypatch.setattr(sys,'argv',new_arguments[:new_arguments.index('--init-from')]+
                        ['--resume',str(current/'last.pt'),'--lr','.0003'])
    trainer.main()
    assert len((current/'history.jsonl').read_text().splitlines())==1
