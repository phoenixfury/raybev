"""Sparse geometry, matching, DN isolation, independent size, and training contracts."""
from dataclasses import replace
import importlib.util
import json
import math
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from raybev import SparseConfig,YOLOROSALite
from raybev.config import CLASSES,CAMERAS,DEFAULT_CALIBRATION
from raybev.geometry import load_cameras,project_fisheye
from raybev.sparse_model import OpenCVProjector,box_points,masked_softmax,decode_sparse
from raybev.sparse_loss import SparseCriterion,SparseYawTeacher,match_queries,ignored_queries
from raybev.sparse_training import fit_sparse_anchors,TrackBalancedSampler,sparse_annotation_sha256
from raybev.rosa_dataset import RosaCombinedDataset,training_boxes_sha256
from raybev.training import calibration_sha256

PROJECT=Path(__file__).parents[1]


def cfg():
    torch.set_num_threads(2)
    return SparseConfig(image_width=160,image_height=128,channels=32,ffn_channels=64,queries=12,stages=2,max_detections=12)


def script(name):
    import sys
    sys.path.insert(0,str(PROJECT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,PROJECT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('change',[dict(architecture_version=2),dict(channels=31),dict(max_range=48),
    dict(queries=0),dict(heading_bins=2),dict(dn_queries=-1),dict(class_positive_weights=(1,)*5)])
def test_sparse_config_rejects_invalid_contract(change):
    with pytest.raises(ValueError):replace(cfg(),**change).validate()


def test_sparse_config_rejects_dense_checkpoint_and_roundtrips_json():
    with pytest.raises(ValueError,match='Not a'):SparseConfig.from_checkpoint(dict(cfg={}))
    original=cfg();doc=json.loads(json.dumps(dict(architecture=original.architecture,cfg=original.to_dict())))
    assert SparseConfig.from_checkpoint(doc)==original


def test_dynamic_projector_matches_numpy_opencv_and_has_finite_gradients():
    config=cfg();projector=OpenCVProjector(config,DEFAULT_CALIBRATION)
    points=torch.tensor([[[[5.,2.,1.],[12.,-4.,2.],[-8.,-3.,1.]]]],requires_grad=True)
    actual,valid,cues=projector(points)
    for ci,camera in enumerate(load_cameras(DEFAULT_CALIBRATION)):
        uv,visible,_=project_fisheye(points.detach().numpy().reshape(-1,3),camera)
        expected=(uv+.5)*[config.image_width/camera.width,config.image_height/camera.height]-.5
        np.testing.assert_allclose(actual[0,0,:,ci].detach(),expected,atol=3e-4)
        np.testing.assert_array_equal(valid[0,0,:,ci],visible)
    actual[valid].sum().backward();assert torch.isfinite(points.grad).all() and points.grad.abs().sum()>0


def test_box_points_rotate_and_use_full_height_and_length():
    b=torch.tensor([[[10.,1.,2.,10.,3.,4.,math.pi/2]]])
    pts=box_points(b)[0,0]
    torch.testing.assert_close(pts[1],torch.tensor([10.,5.5,2.]),atol=1e-5,rtol=0)
    assert pts[5,2]==pytest.approx(3.8) and pts[6,2]==pytest.approx(.2)


def test_all_invalid_softmax_is_zero_with_finite_gradients():
    logits=torch.randn(2,12,requires_grad=True)
    value=masked_softmax(logits,torch.zeros_like(logits,dtype=torch.bool),-1)
    assert not value.any();value.sum().backward();assert torch.isfinite(logits.grad).all()


def test_forward_backward_every_parameter_and_separate_losses():
    model=YOLOROSALite(cfg()).train();targets=[torch.tensor([[8.,2.,1.,8.,2.5,3.,.3,3.]]),torch.empty(0,8)]
    output=model(torch.randn(2,4,3,128,160),targets)
    loss,parts=SparseCriterion(model.cfg)(output,targets)
    assert torch.isfinite(loss) and parts['classification']!=parts['center'] and parts['teacher']==0
    assert len(output['stages'])==2 and output['stages'][-1]['boxes'].shape==(2,13,7)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval();output=model(torch.randn(1,4,3,128,160))
    assert output['stages'][-1]['boxes'].shape==(1,12,7) and not output['dn_indices']
    assert all(torch.isfinite(v).all() and v.shape[1]==9 for v in decode_sparse(output,model.cfg,.001))


def test_empty_and_ignore_only_minibatches_have_finite_backward():
    model=YOLOROSALite(cfg()).train()
    for target in (torch.empty(0,8),torch.tensor([[10.,0.,1.,10.,3.,4.,0.,-1.]])):
        model.zero_grad(set_to_none=True)
        out=model(torch.randn(1,4,3,128,160),[target]);loss,parts=SparseCriterion(model.cfg)(out,[target])
        assert torch.isfinite(loss) and parts['center']==0 and parts['facing']==0 and parts['teacher']==0
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_denoising_queries_cannot_leak_into_normal_queries():
    config=replace(cfg(),dn_queries=4);model=YOLOROSALite(config).train()
    images=torch.randn(1,4,3,128,160);targets=[torch.tensor([[10.,2.,1.,8.,3.,3.,.4,3.]])]
    # Reuse identical feature tensors; train-mode BN is unrelated to attention isolation.
    features=model.image_encoder(images)
    normal=model.forward_features(features);with_dn=model.forward_features(features,targets)
    for a,b in zip(normal['stages'],with_dn['stages']):
        for key in ('boxes','class_logits','facing'):
            torch.testing.assert_close(a[key],b[key][:,:config.queries],atol=2e-5,rtol=1e-5)


def test_dimensions_do_not_change_when_class_winner_changes():
    config=cfg();model=YOLOROSALite(config).eval()
    with torch.no_grad():out=model(torch.randn(1,4,3,128,160))
    boxes=out['stages'][-1]['boxes'].clone()
    for ci in (0,3,5):
        out['stages'][-1]['class_logits'].fill_(-8);out['stages'][-1]['class_logits'][...,ci]=8
        out['stages'][-1]['supported'].fill_(True)
        predictions=decode_sparse(out,config,0,apply_nms=False)[0]
        assert torch.all(predictions[:,7]==ci)
        assert all(torch.isclose(boxes[0,:,:7],p[:7],atol=1e-5).all(1).any() for p in predictions)


def test_matching_is_one_to_one_and_ignored_footprint_respects_yaw():
    config=cfg();gt=torch.tensor([[10.,2.,1.,8.,3.,3.,0.,3.],[20.,-4.,1.,12.,3.,4.,0.,5.]])
    stage=dict(boxes=torch.cat((gt[:,:7],gt[:,:7])),class_logits=torch.zeros(4,6))
    pi,ti=match_queries(stage,gt,replace(config,queries=4,max_detections=4))
    assert len(pi)==2 and len(pi.unique())==2 and set(ti.tolist())=={0,1}
    boxes=torch.tensor([[10.,3.,1.,1.,1.,1.,0.],[13.,0.,1.,1.,1.,1.,0.]])
    ignored=torch.tensor([[10.,0.,1.,8.,2.,3.,math.pi/2,-1.]])
    assert ignored_queries(boxes,ignored).tolist()==[True,False]


def dataset_fixture(tmp_path):
    rows=[]
    for i in range(4):
        cams={}
        for camera in CAMERAS:
            name=f'{i}_{camera}.jpg';Image.new('RGB',(160,128),(70+i,80,90)).save(tmp_path/name)
            cams[camera]=dict(cam_path=name)
        rows.append(dict(token=str(i),cam_info=dict(cam_fisheye=cams),metainfo=dict(sequence_id='bag',temporal_split='train' if i<3 else 'val'),
            annos=dict(gt_boxes=np.array([[7.+i,2.,0.,8.,3.,3.,.4],[8.+i,-4.,0.,4.,2.,2.,.1]]),
                       gt_names=['Truck','Car'],gt_ids=[10,20],gt_ignore=[False,i==1])))
    for name,data in [('train',rows[:3]),('val',rows[3:])]:
        with (tmp_path/f'{name}.pkl').open('wb') as stream:pickle.dump(dict(data_list=data),stream)
    return RosaCombinedDataset(tmp_path,'train.pkl',cfg=cfg(),calibration=DEFAULT_CALIBRATION,expected_size=(160,128))


def test_train_only_anchors_and_balanced_sampler_are_reproducible(tmp_path):
    data=dataset_fixture(tmp_path);a,anchors=fit_sparse_anchors(data,cfg(),iterations=3)
    b,again=fit_sparse_anchors(data,cfg(),iterations=3)
    assert a==b and torch.equal(anchors,again) and a.training_class_counts[3]==3 and sum(a.training_class_counts)==5
    sampler=TrackBalancedSampler(data);assert list(sampler)==list(sampler)
    sampler.set_epoch(3);assert len(list(sampler))==len(data)
    data.rows[0]['metainfo']['temporal_split']='val'
    with pytest.raises(ValueError,match='TRAIN'):fit_sparse_anchors(data,cfg())
    with pytest.raises(ValueError,match='TRAIN'):TrackBalancedSampler(data)


def test_single_epoch_sparse_trainer_and_strict_resume(tmp_path,monkeypatch):
    dataset_fixture(tmp_path)
    train=script('train_yolo_rosa_sparse');import sys
    out=tmp_path/'run'
    arguments=['train','--root',str(tmp_path),'--train-ann','train.pkl','--val-ann','val.pkl','--out',str(out),
        '--image-width','160','--image-height','128','--channels','32','--queries','12','--stages','2',
        '--epochs','1','--batch-size','2','--accum-steps','2','--workers','0','--no-pretrained','--allow-cpu',
        '--no-augmentation','--warmup-steps','0','--progress','off']
    monkeypatch.setattr(sys,'argv',arguments);train.main()
    saved=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
    assert saved['architecture']=='yolo_rosa_lite_v1' and saved['epoch']==1
    assert saved['cfg']['queries']==12 and saved['cfg']['stages']==2 and saved['recipe']['sampling']=='track'
    assert (out/'best_map.pt').is_file() and len((out/'history.jsonl').read_text().splitlines())==1
    monkeypatch.setattr(sys,'argv',arguments+['--resume',str(out/'last.pt')]);train.main()
    assert len((out/'history.jsonl').read_text().splitlines())==1
    monkeypatch.setattr(sys,'argv',arguments+['--resume',str(out/'last.pt'),'--lr','.01'])
    with pytest.raises(ValueError,match='recipe changed'):train.main()
    inference=script('infer_yolo_rosa_sparse');prediction_file=tmp_path/'predictions.json'
    monkeypatch.setattr(sys,'argv',['infer','--root',str(tmp_path),'--ann','val.pkl','--checkpoint',str(out/'last.pt'),
        '--output',str(prediction_file),'--device','cpu'])
    inference.main();assert len(json.loads(prediction_file.read_text())['frames'])==1


def test_sparse_onnx_encoder_and_head_match_pytorch(tmp_path):
    pytest.importorskip('onnx');ort=pytest.importorskip('onnxruntime')
    model=YOLOROSALite(cfg()).eval();export=script('export_yolo_rosa_sparse')
    output=tmp_path/'onnx';metadata=export.export_model(model,output)
    assert not model.training and not model.image_encoder.training
    options=ort.SessionOptions();options.intra_op_num_threads=2
    images=torch.randn(1,4,3,128,160)
    with torch.inference_mode():features=model.image_encoder(images);expected=model.forward_features(features)['stages'][-1]
    session=ort.InferenceSession(str(output/'encoder_four_camera.onnx'),options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'images':images.numpy()})
    for a,b in zip(actual,features):np.testing.assert_allclose(a,b.numpy(),atol=1e-5,rtol=1e-3)
    session=ort.InferenceSession(str(output/'encoder_single_camera.onnx'),options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{'image':images[:,0].numpy()})
    for a,b in zip(actual,features):np.testing.assert_allclose(a,b[:,0].numpy(),atol=1e-5,rtol=1e-3)
    session=ort.InferenceSession(str(output/'sparse_head.onnx'),options,providers=['CPUExecutionProvider'])
    actual=session.run(None,{name:x.numpy() for name,x in zip(('p3','p4','p5'),features)})
    for a,key in zip(actual,('boxes','class_logits','quality_logits','axis','facing','supported')):
        if key=='supported':np.testing.assert_array_equal(a,expected[key].numpy())
        else:np.testing.assert_allclose(a,expected[key].numpy(),atol=3e-4,rtol=1e-3)
    assert metadata['queries']==12 and metadata['stages']==2 and metadata['support_points']==7
    assert 'GridSample' in metadata['operators']['sparse_head']
    assert not set(metadata['operators']['encoder_single_camera']).intersection({'GridSample','MatMul','Softmax','LayerNormalization'})


def test_sparse_ignore_mask_removes_background_but_keeps_known_matched_positive():
    config=replace(cfg(),queries=2,max_detections=2,mean_training_objects=1,dn_queries=0)
    boxes=torch.tensor([[[10.,0.,1.,8.,3.,3.,0.],[20.,0.,1.,8.,3.,3.,0.]]],requires_grad=True)
    def output():
        return dict(stages=[dict(boxes=boxes,axis=torch.tensor([[[0.,1.],[0.,1.]]],requires_grad=True),
            facing=torch.zeros(1,2,8,requires_grad=True),class_logits=torch.zeros(1,2,6,requires_grad=True),
            quality_logits=torch.zeros(1,2,requires_grad=True),supported=torch.ones(1,2,dtype=torch.bool))],dn_indices=[],num_queries=2)
    target=torch.tensor([[10.,0.,1.,8.,3.,3.,0.,-1.]])
    value=output();loss,_=SparseCriterion(config)(value,[target]);loss.backward()
    assert not value['stages'][0]['class_logits'].grad[0,0].any()
    assert (value['stages'][0]['class_logits'].grad[0,1]>0).all()
    value=output();target=torch.cat((target,torch.tensor([[10.,0.,1.,8.,3.,3.,0.,3.]])))
    loss,_=SparseCriterion(config)(value,[target]);loss.backward()
    assert value['stages'][0]['class_logits'].grad[0,0,3]<0


def test_sparse_head_can_fit_two_fixed_truck_bus_targets():
    torch.manual_seed(3407)
    config=replace(cfg(),dn_queries=0,mean_training_objects=2)
    gt=torch.tensor([[10.,2.,1.,8.,3.,3.,.3,3.],[20.,-4.,1.,12.,3.,4.,-2.,5.]])
    model=YOLOROSALite(config).eval()
    with torch.no_grad():
        model.anchors[:2]=gt[:,:7];model.anchors[:2,6]+=math.pi
        features=tuple(x.detach() for x in model.image_encoder(torch.randn(1,4,3,128,160)))
    criterion=SparseCriterion(config);optimizer=torch.optim.Adam(model.layers.parameters(),lr=.003)
    first=None
    for _ in range(110):
        optimizer.zero_grad();out=model.forward_features(features);loss,_=criterion(out,[gt]);loss.backward();optimizer.step()
        if first is None:first=float(loss)
    assert float(loss)<first*.5
    with torch.no_grad():
        out=model.forward_features(features);pred=decode_sparse(out,config,.1,apply_nms=False)[0]
    assert {3,5}.issubset(set(pred[:,7].tolist()))
    for target in gt:
        selected=pred[pred[:,7]==target[7]]
        best=selected[(selected[:,:2]-target[:2]).norm(dim=-1).argmin()]
        error=torch.atan2((best[6]-target[6]).sin(),(best[6]-target[6]).cos()).abs()
        assert error<.35  # learns to reverse the deliberately wrong anchor facing


def test_sparse_resume_signature_includes_ignore_geometry_and_track_ids(tmp_path):
    data=dataset_fixture(tmp_path);initial=sparse_annotation_sha256(data)
    data.ignore_boxes[1][0,0]+=1
    assert sparse_annotation_sha256(data)!=initial
    data.ignore_boxes[1][0,0]-=1;initial=sparse_annotation_sha256(data)
    data.rows[0]['annos']['gt_ids'][0]+=100
    assert sparse_annotation_sha256(data)!=initial


def test_sparse_teacher_validates_train_geometry_calibration_and_provenance(tmp_path):
    data=dataset_fixture(tmp_path);path=tmp_path/'teacher.json'
    doc=dict(type='raybev_heading_teacher',split='train',training_boxes_sha256=training_boxes_sha256(data),
        camera_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),
        lidar_calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),box_frame=data.box_frame,
        z_reference=data.z_reference,teacher_checkpoint_sha256='teacher',predictions_sha256='predictions',
        frames={str(row['token']):[dict(gt=gt[0].tolist(),yaw=float(gt[0,6])+.1,weight=.8)]
                for row,gt in zip(data.rows,data.boxes)})
    path.write_text(json.dumps(doc));teacher=SparseYawTeacher(path,data,DEFAULT_CALIBRATION)
    yaw,weight=teacher.targets('0',data.boxes[0])
    assert yaw[0]==pytest.approx(float(data.boxes[0][0,6])+.1) and weight.tolist()==pytest.approx([.8,0.])
    original=json.dumps(doc)
    for key,value in [('split','val'),('training_boxes_sha256','wrong'),('camera_calibration_sha256','wrong'),
                      ('lidar_calibration_sha256','wrong'),('teacher_checkpoint_sha256','')]:
        invalid=json.loads(original);invalid[key]=value;path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError,match='provenance mismatch'):SparseYawTeacher(path,data,DEFAULT_CALIBRATION)
    invalid=json.loads(original);invalid['frames']['0'][0]['gt'][0]+=1;path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError,match='Invalid sparse teacher'):SparseYawTeacher(path,data,DEFAULT_CALIBRATION)


@pytest.mark.parametrize('field',['class_logits','quality_logits','axis','facing'])
def test_sparse_decoder_rejects_nonfinite_used_logits(field):
    config=replace(cfg(),queries=2,max_detections=2)
    stage=dict(boxes=torch.tensor([[[10.,0.,1.,8.,3.,3.,0.],[20.,0.,1.,8.,3.,3.,0.]]]),
        class_logits=torch.zeros(1,2,6),quality_logits=torch.zeros(1,2),axis=torch.ones(1,2,2),
        facing=torch.zeros(1,2,8),supported=torch.ones(1,2,dtype=torch.bool))
    if stage[field].ndim==3:stage[field][0,0,0]=float('inf')
    else:stage[field][0,0]=float('inf')
    pred=decode_sparse(dict(stages=[stage]),config,0,False)[0]
    assert len(pred)==1 and pred[0,0]==20
