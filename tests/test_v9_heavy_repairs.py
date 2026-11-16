from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from raybev.sparse_config import SparseConfig
from raybev.sparse_loss import (class_match_weight, distance_quality, endpoint_error,
    hard_quality_negatives, match_queries, SparseCriterion)
from raybev.sparse_observation import select_proposal_targets,proposal_loss
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.sparse_audit import audit_queries,false_positive_audit,fixed_car_recall,raw_snapshot


def cfg(**options):
    torch.set_num_threads(2)
    return replace(SparseConfig(architecture_version=5,queries=6,max_detections=6,
        stages=2,channels=32,ffn_channels=64,proposal_queries=2,
        image_width=160,image_height=128,stride4=True,observation_conditioning=True),**options).validate()


def box(x=5., y=0., length=8., yaw=0.):
    return [x,y,1.,length,2.,2.,yaw]


def test_proposal_uses_in_cell_gt_even_when_external_gt_is_closer():
    delta=torch.tensor([[[2.1,0],[1.9,1.9]],[[2.,0],[-2.,0]],[[0.,0],[0.,0]]])
    nearest,positive=select_proposal_targets(delta,torch.tensor([4.,4.]),torch.tensor([True,True,False]))
    assert positive.tolist()==[True,True,False] and nearest[:2].tolist()==[1,1]
    centers=torch.tensor([[0.,0.,.9]])
    truth=[torch.tensor([box(2.1)+[0],box(1.9,1.9)+[3]])]
    data=dict(logits=torch.zeros(1,1,requires_grad=True),centers=centers[None].clone().requires_grad_(),
              valid=torch.ones(1,1,dtype=torch.bool),lattice=centers,cell_size=torch.tensor([4.,4.]))
    loss=proposal_loss(data,truth,cell_targets=True)
    loss.backward()
    assert data['logits'].grad.item()<0 and (data['centers'].grad[0,0,:2]<0).all()


def test_matching_schedule_and_better_geometric_supervision():
    c=cfg(match_class_start=0.,match_warmup_epochs=2,match_ramp_epochs=4)
    assert [class_match_weight(c,i) for i in (0,2,4,6,9)]==[0.,0.,.5,1.,1.]
    logits=torch.full((6,6),-4.);logits[0,3]=torch.logit(torch.tensor(.2));logits[1,3]=torch.logit(torch.tensor(.8))
    stage=dict(boxes=torch.tensor([box(),box(8),*[box(30+i) for i in range(4)]]),class_logits=logits)
    gt=torch.tensor([box()+[3]])
    assert match_queries(stage,gt,c,1.)[0].item()==1
    assert match_queries(stage,gt,c,0.)[0].item()==0


def test_endpoint_heading_invariance_and_distance_quality():
    gt=torch.tensor([box()])
    pred=torch.tensor([box(yaw=math.pi)],requires_grad=True)
    assert endpoint_error(pred,gt).item()<1e-8
    pred=torch.tensor([box(6.,length=6.)],requires_grad=True)
    endpoint_error(pred,gt).sum().backward()
    assert pred.grad[0,0]>0 and pred.grad[0,3]<0
    quality=distance_quality(torch.tensor([0.,1.,2.,4.,10.]))
    assert quality[0]>.98 and (quality.diff()<0).all()


def test_hard_negative_loss_protects_near_only_ignored_and_assigned_queries():
    q=torch.zeros(6,requires_grad=True)
    stage=dict(boxes=torch.tensor([box(x) for x in (5.,5.2,6.2,15.,25.,35.)]),
        quality_logits=q,class_logits=torch.ones(6,6))
    truth=torch.tensor([box()+[3]])
    loss=hard_quality_negatives(stage,truth,torch.tensor([0]),torch.tensor([0]),
                                torch.tensor([True,True,True,True,False,True]),.5)
    loss.backward()
    assert (q.grad[[0,2,4]]==0).all() and (q.grad[[1,3,5]]>0).all()


def fixture_output():
    c=cfg()
    logits=torch.full((1,6,6),-6.)
    for i,k in enumerate((0,3,0,5,3,0)):logits[0,i,k]=4.
    final=dict(boxes=torch.tensor([[box(5),box(5.15),box(11),box(15),box(25),box(10.8)]]),
               class_logits=logits,quality_logits=torch.tensor([[4.,1.,4.,-10.,4.,0.]]),
               axis=torch.ones(1,6,2),facing=torch.ones(1,6,8),
               supported=torch.tensor([[True,True,True,True,False,True]]))
    return c,dict(stages=[final])


def test_postprocess_duplicates_audit_and_checkpoint_only_replay():
    c,output=fixture_output()
    t=torch.tensor([box(5)+[3],box(11)+[0],box(15)+[5],box(25)+[3]])
    standard=decode_sparse(output,c,.1,return_indices=True)
    assert standard[0][1].tolist()==[0,2,1]
    merged=decode_sparse(output,c,.1,vehicle_duplicates=True,return_indices=True)
    assert merged[0][1].tolist()==[0,2] # Adjacent car stays; identical Truck/Car is grouped.
    audit=audit_queries(output,c,[t],standard)
    assert [x['reason'] for x in audit[0]]==[
        'correct_candidate_retained','below_score','only_unsupported_invalid_or_outside']
    report,_=false_positive_audit([standard[0][0].numpy()],[t.numpy()])
    assert report['by_class']['car']['wrong_class']==1
    assert fixed_car_recall([standard[0][0].numpy()],[t.numpy()],1)['fp_per_frame']==1
    _,ids=decode_sparse(output,c,.1,quality_power=0,return_indices=True)[0]
    assert 3 in ids.tolist()
    snapshot=raw_snapshot(output,c)
    assert torch.equal(snapshot['class_logits'],output['stages'][-1]['class_logits'])


def test_v9_forward_backward_empty_and_strict_legacy_weight_transfer():
    from scripts.train_yolo_rosa_sparse import transfer_sparse_weights
    base=cfg()
    options=dict(fresh_box_classification=True,independent_facing_views=True,
        proposal_cell_targets=True,match_class_start=.05,match_warmup_epochs=3,
        quality_distance_target=True,quality_hard_negative_weight=.2,heavy_endpoint_weight=.1)
    c=replace(base,**options).validate()
    legacy=YOLOROSALite(base).eval()
    model=YOLOROSALite(c)
    transfer=transfer_sparse_weights(model,dict(architecture=base.architecture,cfg=base.to_dict(),state_dict=legacy.state_dict()))
    assert transfer['new'] and all('refined_box_classifier' in n for n in transfer['new'])
    cls_only=YOLOROSALite(replace(c,independent_facing_views=False)).eval()
    transfer_sparse_weights(cls_only,dict(architecture=base.architecture,cfg=base.to_dict(),state_dict=legacy.state_dict()))
    x=torch.randn(2,4,3,128,160)
    with torch.no_grad():
        torch.testing.assert_close(cls_only(x)['stages'][-1]['class_logits'],
                                   legacy(x)['stages'][-1]['class_logits'])
    gt=[torch.tensor([box()+[3],box(9,2)+[5],box(20)+[-1]]),torch.empty(0,8)]
    criterion=SparseCriterion(c)
    output=model(x,gt);loss,parts=criterion(output,gt,['a','b'])
    assert torch.isfinite(loss) and parts['quality_hard_negative']>0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    restored=YOLOROSALite(SparseConfig.from_checkpoint(dict(architecture=c.architecture,cfg=c.to_dict())))
    restored.load_state_dict(model.state_dict(),strict=True)


def test_corrected_proposal_batch_average_is_independent_of_empty_frame_order():
    centers=torch.tensor([[0.,0.,.9]])
    data=dict(logits=torch.zeros(2,1),centers=centers.expand(2,1,3).clone(),
              valid=torch.ones(2,1,dtype=torch.bool),lattice=centers,cell_size=torch.tensor([4.,4.]))
    gt=torch.tensor([box(1.)+[3]]);empty=torch.empty(0,8)
    torch.testing.assert_close(proposal_loss(data,[gt,empty],True),proposal_loss(data,[empty,gt],True))


def test_class_operating_bias_and_thresholds_do_not_change_standard_decode():
    c,output=fixture_output()
    baseline=decode_sparse(output,c,.01)
    bias=[0.,0.,0.,9.,0.,0.]
    modified=decode_sparse(output,c,.1,class_bias=bias,class_thresholds=[.1,.1,.1,.98,.1,.1])
    assert len(modified[0])<len(baseline[0])
    assert torch.equal(baseline[0],decode_sparse(output,c,.01)[0])


def test_v9_checkpoint_audit_cli_and_reverse_label_review(tmp_path):
    import json
    import pickle
    import subprocess
    import sys
    from pathlib import Path
    from PIL import Image
    from raybev.config import CAMERAS,CLASSES,DEFAULT_CALIBRATION
    from raybev.rosa_dataset import RosaCombinedDataset
    from raybev.sparse_training import sparse_annotation_sha256
    from raybev.training import calibration_sha256
    c=cfg(proposal_cell_targets=True,fresh_box_classification=True)
    Image.new('RGB',(160,128),(188,60,40)).save(tmp_path/'img.png')
    row=dict(token='v',cam_info={'cam_fisheye':{k:{'cam_path':'img.png'} for k in CAMERAS}},
             metainfo=dict(temporal_split='val',sequence_id='s'),
             annos=dict(gt_boxes=[box(),box(12.,length=6.)],gt_names=['Truck','Car'],gt_ids=[9,10]))
    val=tmp_path/'val.pkl'
    val.write_bytes(pickle.dumps(dict(data_list=[row])))
    ds=RosaCombinedDataset(tmp_path,val,cfg=c,box_frame='BASELINK',expected_size=(160,128))
    model=YOLOROSALite(c)
    checkpoint=tmp_path/'last.pt'
    sha=calibration_sha256(DEFAULT_CALIBRATION)
    torch.save(dict(cfg=c.to_dict(),architecture=c.architecture,state_dict=model.state_dict(),
                    classes=CLASSES,calibration_sha256=sha,lidar_calibration_sha256=sha,
                    box_frame='BASELINK',z_reference='center',epoch=1,
                    semantic_annotation_sha256={'val':sparse_annotation_sha256(ds)}),checkpoint)
    project=Path(__file__).resolve().parents[1]
    out=tmp_path/'audit.json';raw=tmp_path/'queries.npz'
    result=subprocess.run([sys.executable,str(project/'scripts/infer_yolo_rosa_sparse.py'),
        '--checkpoint',str(checkpoint),'--root',str(tmp_path),'--ann',str(val),'--device','cpu',
        '--output',str(out),'--raw-queries',str(raw),'--audit','--sweep'],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(out.read_text())
    assert report['query_audit']['by_track'][0]['track_id']=='9'
    assert len(report['scoring_sweep'])==6
    with np.load(raw) as blob:
        assert blob['class_logits'].shape==(1,c.queries,len(CLASSES))
    row['metainfo']['temporal_split']='train'
    train=tmp_path/'train.pkl';train.write_bytes(pickle.dumps(dict(data_list=[row])))
    review=tmp_path/'review.json'
    result=subprocess.run([sys.executable,str(project/'scripts/audit_v9_training_labels.py'),
        '--train-ann',str(train),'--box-frame','BASELINK','--output',str(review)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(review.read_text())
    assert report['review_queue'][0]['track_id']=='10'
