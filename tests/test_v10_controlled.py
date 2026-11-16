from dataclasses import replace
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from raybev.config import DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_loss import SparseCriterion,matched_class_loss,match_queries
from raybev.sparse_model import YOLOROSALite,box_points,sparse_candidates
from raybev.sparse_training import fit_sparse_anchors
from raybev.sparse_v10 import (shape_hypotheses,objectness_loss,GeometryApprovals,fixed_class_recall,
                              geometry_stage_confusion,evaluation_slices)


def config(**options):
    torch.set_num_threads(2)
    return replace(SparseConfig(architecture_version=5,queries=6,max_detections=6,channels=32,ffn_channels=64,
        stages=2,image_width=160,image_height=128,proposal_queries=2,proposal_grid=4,
        stride4=True,observation_conditioning=True,proposal_cell_targets=True,
        teacher_weight=0.,class_positive_weights=(1.,)*6,training_class_counts=(100,2,1,5,4,3)),**options).validate()


def test_shape_mixture_preserves_compact_coverage_without_semantic_labels():
    boxes=np.array([[10,0,1,l,2,2,0,ci] for ci,l in enumerate((1.,4.,7.,11.,17.))])
    sizes=shape_hypotheses(boxes,100)
    assert np.bincount(np.digitize(sizes[:,0],[2.5,5.5,8.5,14.]),minlength=5).tolist()==[8,60,12,12,8]
    boxes[:,7]=0
    np.testing.assert_array_equal(sizes,shape_hypotheses(boxes,100))
    ds=SimpleNamespace(rows=[dict(metainfo=dict(temporal_split='train'))],boxes=[torch.tensor(boxes,dtype=torch.float32)])
    class Dataset:
        rows=ds.rows
        def __len__(self):return 1
    data=Dataset();data.boxes=ds.boxes
    c=config(shape_balanced_anchors=True,shape_balanced_proposals=True)
    _,first=fit_sparse_anchors(data,c,iterations=2)
    data.boxes[0][:,7]=torch.arange(5)
    _,second=fit_sparse_anchors(data,c,iterations=2)
    torch.testing.assert_close(first[:,3:6],second[:,3:6],rtol=0,atol=0)
    # Semantic labels may affect the shared legacy spatial fit, never sizes.
    data.boxes[0][:,7]=0
    _,control=fit_sparse_anchors(data,replace(c,shape_balanced_anchors=False,shape_balanced_proposals=False),iterations=2)
    torch.testing.assert_close(first[:,:2],control[:,:2],rtol=0,atol=0)


def test_foreground_balance_does_not_apply_to_background_queries():
    c=config(matched_class_weight=1.,foreground_balance=True)
    logits=torch.zeros(5,6,requires_grad=True)
    labels=torch.tensor([0,3])
    loss=matched_class_loss(logits[:2],labels,c,2.)
    loss.backward()
    assert logits.grad[2:].abs().sum()==0
    assert logits.grad[1,3].abs()>logits.grad[0,0].abs()
    objectness=torch.zeros(5,requires_grad=True)
    objectness_loss(objectness,torch.tensor([0,1]),torch.ones(5,dtype=torch.bool)).backward()
    torch.testing.assert_close(objectness.grad[0],objectness.grad[1])


def test_semantic_matching_uses_objectness_and_is_invariant_to_common_logit_shift():
    c=config(semantic_objectness=True)
    box=torch.tensor([8.,0,1.,8.,2.,2.,0.])
    logits=torch.zeros(6,6);logits[0,3]=2.
    stage=dict(boxes=box.repeat(6,1),class_logits=logits,objectness_logits=torch.zeros(6))
    gt=torch.cat((box,torch.tensor([3.])))[None]
    assert match_queries(stage,gt,c)[0].tolist()==[0]
    stage['class_logits'][1]+=100
    assert match_queries(stage,gt,c)[0].tolist()==[0]


@pytest.mark.parametrize('options',[
    dict(),dict(matched_class_weight=1.,foreground_balance=True),dict(semantic_objectness=True),
    dict(semantic_objectness=True,foreground_balance=True),dict(object_aligned_context=True),dict(early_stage_weight=2.),
])
def test_controlled_arm_forward_backward_and_strict_checkpoint(options):
    c=config(stage_diagnostics=True,**options);model=YOLOROSALite(c)
    images=torch.randn(2,4,3,128,160)
    gt=[torch.tensor([[8.,0,1.,8.,2.,2.,0.,3.],[20.,0,1.,4.,2.,2.,0.,0.]]),torch.empty(0,8)]
    output=model(images,gt);loss,parts=SparseCriterion(c)(output,gt,['a','b'])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert parts['supervision_stats'].shape==(2,6,8)
    assert parts['supervision_stats'][:,3,0].tolist()==[1,1]
    restored=YOLOROSALite(SparseConfig.from_checkpoint(dict(architecture=c.architecture,cfg=c.to_dict())))
    restored.load_state_dict(model.state_dict(),strict=True)
    if c.semantic_objectness:
        candidates=sparse_candidates(output['stages'][-1],c)
        assert torch.isfinite(candidates[2]).all()
        assert (candidates[2]<=output['stages'][-1]['objectness_logits'][:,:c.queries].sigmoid()+1e-6).all()
    if c.object_aligned_context:
        assert c.num_support_points==31
        assert box_points(torch.tensor([[[8.,0,1.,8.,2.,2.,0.]]]),torch.zeros(1,1,6,3),True).shape==(1,1,31,3)


def test_endpoint_arm_requires_explicit_credible_geometry(tmp_path):
    c=config(heavy_endpoint_weight=.25,geometry_approval_required=True)
    with pytest.raises(ValueError,match='approved credible cuboids'):SparseCriterion(c)
    from raybev.rosa_dataset import training_boxes_sha256
    from raybev.training import calibration_sha256
    gt=torch.tensor([[8.,0,1.,8.,2.,2.,0.,3.]])
    ds=SimpleNamespace(rows=[dict(token='a')],boxes=[gt],box_frame='CLIDAR',z_reference='center')
    path=tmp_path/'approvals.json'
    document=dict(type='raybev_geometry_approvals_v1',training_boxes_sha256=training_boxes_sha256(ds),
        calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),approvals=[])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError,match='No explicitly approved'):GeometryApprovals(path,ds,DEFAULT_CALIBRATION)
    document['approvals']=[dict(token='a',gt=gt[0].tolist(),geometry_reviewed=True,evidence='Independent LiDAR/image geometry review')]
    path.write_text(json.dumps(document));approved=GeometryApprovals(path,ds,DEFAULT_CALIBRATION)
    assert approved.mask('a',gt).tolist()==[True]
    assert approved.mask('different',gt).tolist()==[False]


def test_endpoint_gradients_exclude_unapproved_cuboids():
    c=config(heavy_endpoint_weight=.25,geometry_approval_required=True)
    approval=SimpleNamespace(mask=lambda token,gt:torch.tensor([True,False],device=gt.device))
    criterion=SparseCriterion(c,geometry_approvals=approval)
    gt=torch.tensor([[8.,0,1.,8.,2.,2.,0.,3.],[16.,0,1.,10.,2.,2.,0.,5.]])
    boxes=gt[:,:7].clone();boxes[:,0]+=1;boxes.requires_grad_()
    stage=dict(boxes=boxes,axis=torch.ones(2,2),facing=torch.zeros(2,8),class_logits=torch.zeros(2,6),quality_logits=torch.zeros(2))
    loss=criterion._regression(stage,torch.tensor([0,1]),gt,2.,'reviewed_train')['heavy_endpoint']
    assert loss>0
    loss.backward()
    assert boxes.grad[0].abs().sum()>0 and boxes.grad[1].abs().sum()==0


def test_fixed_recall_keeps_score_ties_and_marks_unreachable():
    gt=np.array([[8.,0,1.,4.,2.,2.,0.,0.]])
    prediction=np.array([[30.,0,1.,4.,2.,2.,0.,0.,.8],[8.,0,1.,4.,2.,2.,0.,0.,.8]])
    result=fixed_class_recall([prediction],[gt],0,1.)
    assert result['status']=='reached' and result['fp']==1
    assert fixed_class_recall([prediction[:1]],[gt],0,1.)['status']=='unreachable'


def test_geometry_stage_classification_does_not_match_using_semantic_labels():
    c=config();gt=torch.tensor([[8.,0,1.,8.,2.,2.,0.,3.]])
    boxes=torch.tensor([[[8.,0,1.,8.,2.,2.,0.],[30.,0,1.,8.,2.,2.,0.],
                         [40.,0,1.,8.,2.,2.,0.],[45.,0,1.,8.,2.,2.,0.],
                         [42.,10,1.,8.,2.,2.,0.],[45.,10,1.,8.,2.,2.,0.]]])
    logits=torch.zeros(1,6,6);logits[0,0,0]=5;logits[0,1,3]=10
    stage=dict(boxes=boxes,class_logits=logits,supported=torch.ones(1,6,dtype=torch.bool))
    counts=geometry_stage_confusion(dict(stages=[stage,stage]),[gt],c)
    assert counts[0,3,0]==1 and counts[0,3,3]==0


def test_seen_unseen_and_tram_heading_match_before_slicing():
    truth=np.array([[8.,0,1.,8.,2.,2.,0.,5.],[16.,0,1.,8.,2.,2.,0.,3.]])
    predictions=np.concatenate((truth,np.full((2,1),.9)),axis=1);predictions[0,6]=np.pi
    train=SimpleNamespace(rows=[dict(metainfo=dict(sequence_id='bag'),annos=dict(gt_ids=[1]))],box_source_indices=[np.array([0])])
    val=SimpleNamespace(rows=[dict(metainfo=dict(sequence_id='bag'),annos=dict(gt_ids=[1,2]))],box_source_indices=[np.array([0,1])])
    result=evaluation_slices([predictions],[truth],train,val,[dict(sequence='bag',track_id='1')])
    assert result['seen_unseen']['bus/seen']['recall']==1
    assert result['seen_unseen']['truck/unseen']['recall']==1
    assert result['seen_unseen']['truck/unseen']['track_count']==1
    assert len(result['heavy_per_track'])==2
    assert result['tram_heading']['tp']==1 and result['tram_heading']['flip_rate']==1
    assert result['tram_heading']['unsigned_axis_AOE_deg']==pytest.approx(0)


def test_real_trainer_writes_separate_losses_and_stage_reports(tmp_path):
    from PIL import Image
    from raybev.config import CAMERAS
    Image.new('RGB',(160,128),(80,110,140)).save(tmp_path/'image.png')
    for split in ('train','val'):
        rows=[]
        for i in range(2):
            boxes=np.array([[8,0,1,8,2,2,0],[16,2,1,4,1.8,1.5,0]],np.float32) if i==0 else np.empty((0,7),np.float32)
            rows.append(dict(token=f'{split}{i}',metainfo=dict(temporal_split=split,sequence_id='smoke'),
                cam_info=dict(cam_fisheye={c:dict(cam_path='image.png') for c in CAMERAS}),
                annos=dict(gt_boxes=boxes,gt_names=np.array(['Truck','Car'] if i==0 else [],dtype='U10'),
                           gt_ids=np.array([1,2] if i==0 else [],dtype=np.int64))))
        (tmp_path/f'{split}.pkl').write_bytes(pickle.dumps(dict(data_list=rows),protocol=4))
    script=Path(__file__).resolve().parents[1]/'scripts/train_yolo_rosa_sparse.py'
    command=[sys.executable,str(script),'--root',str(tmp_path),'--train-ann',str(tmp_path/'train.pkl'),
        '--val-ann',str(tmp_path/'val.pkl'),'--out',str(tmp_path/'run'),'--allow-cpu','--no-pretrained',
        '--epochs','1','--workers','0','--batch-size','2','--accum-steps','1','--architecture-version','5',
        '--queries','6','--stages','2','--channels','32','--image-width','160','--image-height','128',
        '--stride4','--observation-conditioning','--proposal-queries','2','--proposal-grid','4','--proposal-cell-targets',
        '--stage-diagnostics','--semantic-objectness','--foreground-balance','--unit-class-weights',
        '--object-aligned-context','--early-stage-weight','2','--dn-queries','2','--sampling','uniform',
        '--no-augmentation','--progress','off']
    environment=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=environment)
    assert result.returncode==0,result.stdout[-6000:]
    row=json.loads((tmp_path/'run/history.jsonl').read_text())
    assert row['stage_supervision']['train']['stage_1']['truck']['gt']==1
    assert row['stage_supervision']['val']['stage_1']['truck']['matched_semantic_nll'] is not None
    assert row['train_components']['objectness']>0 and row['train_components']['foreground_semantic']>0
    assert row['training_heavy_assignment']['objects']==1
    assert row['geometry_matched_stage_classification']['val']['stage_1']['truck']['gt']==1
    assert 'fixed_recall' in row['controlled_evaluation']
    assert (tmp_path/'run/last.pt').is_file()
