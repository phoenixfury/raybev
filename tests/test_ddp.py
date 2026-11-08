"""Tests for exact distributed validation and 2-GPU training settings."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'train_combined_rosa_ddp.py'
    spec = importlib.util.spec_from_file_location('raybev_ddp_test', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_val_sampler_strides_cover_each_sample_once():
    mod = module()
    for n in (1,2,3,7,674,675):
        a = list(mod.ExactEvalSampler(range(n),0,2))
        b = list(mod.ExactEvalSampler(range(n),1,2))
        assert sorted(a+b) == list(range(n))
        assert not set(a).intersection(b)


def test_merge_predictions_keeps_order_and_rejects_dupes():
    mod = module()
    shard0 = [(0,np.ones((1,9)),np.empty((0,8))),(2,np.empty((0,9)),np.ones((1,8)))]
    shard1 = [(1,np.zeros((2,9)),np.empty((0,8)))]
    predictions,labels = mod.combine_exact_predictions([shard0,shard1],3)
    assert [len(x) for x in predictions] == [1,2,0]
    assert [len(x) for x in labels] == [0,0,1]
    with pytest.raises(ValueError, match='Duplicate'):
        mod.combine_exact_predictions([shard0,shard0],3)
    with pytest.raises(ValueError, match='Missing'):
        mod.combine_exact_predictions([shard0],3)


def test_progress_text_fallback(capsys):
    mod = module()
    p=mod.Progress('train',1,2,2,'off',1)
    p.update(1,{'loss':'0.5'})
    p.update(1,{'loss':'0.3'})
    p.close()
    out=capsys.readouterr().out
    assert '2/2' in out and 'loss=0.3' in out and 'GPU0' in out


def test_flat_metrics_includes_ap_and_nullable_nds():
    mod=module()
    classes={c:dict(AP=None,gt_count=0,ATE=None,ASE=None,AOE=None,
                    precision=None,recall=None,AP_by_distance=None) for c in mod.CLASSES}
    classes['car'].update(AP=0.4,gt_count=8,prediction_count=10,AP_by_distance={'2.0':0.4})
    metrics=dict(class_results=classes,active_classes=['car'],excluded_classes=['van'],
                 AP_distance_thresholds=[.5,1.,2.,4.],mAP=0.4,NDS=None,mATE=0.5)
    row=dict(epoch=1,lr=0.0002,train_loss=1.,train_heatmap_loss=.2,
             train_regression_loss=.8,val_loss=1.1,val_heatmap_loss=.4,
             val_regression_loss=.7,seconds=12,world_size=2,
             global_batch_size=4,train_samples_seen=12,
             metrics=metrics,per_bag_metrics={})
    flattened=mod.make_flat_metrics(row)
    assert flattened['AP_car']==.4
    assert flattened['prediction_count_car']==10
    assert flattened['AP_van'] is None
    assert flattened['NDS'] is None
