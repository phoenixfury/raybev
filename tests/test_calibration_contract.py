"""Default source-frame conversion, rig consistency, and safe resume."""
import importlib.util
import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from raybev.config import DEFAULT_CALIBRATION,ModelConfig
from raybev.geometry import load_cameras,project_fisheye
from raybev.rosa_dataset import (RosaCombinedDataset,boxes_to_baselink,
                                parse_box_annotations,read_T_base_clidar)
from raybev.training import calibration_sha256,check_resume

PROJECT=Path(__file__).parents[1]


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,PROJECT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_and_lidar_files_describe_the_same_rig():
    data=json.loads(Path(DEFAULT_CALIBRATION).read_text())
    full=json.loads((PROJECT/'configs/full_calibration_baselink_with_lidar.json').read_text())
    T=read_T_base_clidar()
    np.testing.assert_allclose(T[:3,3],[.84405,-.00185,1.41075])
    for camera in load_cameras(DEFAULT_CALIBRATION):
        name=camera.name
        assert data['intrinsics'][name]==full['intrinsics'][name]
        assert data['extrinsics'][name]==full['extrinsics'][name]
        np.testing.assert_allclose(np.linalg.inv(camera.T_base_cam)@T,
                                   data['lidar_to_camera'][name],atol=1e-7)
        assert data['intrinsics'][name]['projection_for_validation']=='opencv_fisheye'
    assert data['extrinsics']['CLIDAR']==full['extrinsics']['CLIDAR']
    # Preserve the exact camera calibration required by the old server weights.
    assert calibration_sha256(PROJECT/'configs/legacy_v2/calibration_baselink.json') == (
        'f7431585c121de9211ffe2d4d023fa581e76dd9aa77ac5be01f23cd04b578f70')


def test_clidar_is_default_and_baselink_is_an_explicit_noop():
    raw=np.array([[10.,2.,-1.,4.,2.,1.5,.4]])
    sample={'annos':{'gt_boxes':raw,'gt_names':['Car']}}
    actual=parse_box_annotations(sample).numpy()
    T=read_T_base_clidar()
    np.testing.assert_allclose(actual[0,:3],(T@np.r_[raw[0,:3],1])[:3],atol=1e-6)
    direction=T[:3,:3]@np.array([np.cos(.4),np.sin(.4),0])
    assert actual[0,6]==pytest.approx(np.arctan2(direction[1],direction[0]),abs=1e-6)
    np.testing.assert_allclose(actual[0,3:6],raw[0,3:6])
    np.testing.assert_array_equal(boxes_to_baselink(raw,'BASELINK'),raw)
    assert actual[0,7]==0


def test_bottom_centering_happens_before_pitched_lidar_rotation():
    angle=.1;c=np.cos(angle);s=np.sin(angle)
    T=np.eye(4);T[:3,:3]=[[c,0,s],[0,1,0],[-s,0,c]];T[:3,3]=[.8,0,1.4]
    raw=np.array([[10.,2.,-1.,4.,2.,2.,0.]])
    centered=np.r_[raw[0,:3]+[0,0,1],1]
    result=boxes_to_baselink(raw,'CLIDAR',T,'bottom')
    np.testing.assert_allclose(result[0,:3],(T@centered)[:3])
    np.testing.assert_array_equal(raw[0,:3],[10,2,-1])


def test_dataset_converts_once_before_range_filtering(tmp_path):
    raw=np.array([[10.,2.,-1.,4.,2.,1.5,.4],[47.8,0,0,4,2,2,0]])
    row={'token':'test','annos':{'gt_boxes':raw,'gt_names':['Car','Truck']}}
    with (tmp_path/'labels.pkl').open('wb') as handle:pickle.dump({'data_list':[row]},handle)
    dataset=RosaCombinedDataset(tmp_path,'labels.pkl',cfg=ModelConfig())
    assert dataset.box_frame=='CLIDAR' and dataset.range_filtered_boxes==1
    np.testing.assert_allclose(dataset.boxes[0],parse_box_annotations(row)[:1])
    np.testing.assert_array_equal(dataset.rows[0]['annos']['gt_boxes'],raw)


@pytest.mark.parametrize('script,required',[
    ('train',['--data','unused']),
    ('train_combined_rosa',['--root','unused']),
    ('train_combined_rosa_ddp',['--root','unused'])])
def test_all_trainers_default_to_clidar_and_one_calibration(script,required):
    args=load_script(script).build_parser().parse_args(required)
    assert args.box_frame=='CLIDAR'
    assert args.calibration==DEFAULT_CALIBRATION
    assert args.lidar_calibration is None  # Resolved to --calibration by main().


def test_npz_defaults_to_clidar_and_preserves_legacy_class_id(tmp_path):
    train=load_script('train')
    raw=np.array([[10.,2.,-1.,4.,2.,1.5,.4,12.,-3.,5.]],np.float32)
    np.savez(tmp_path/'frame.npz',images=np.zeros((4,3,32,40),np.uint8),boxes=raw)
    _,boxes=train.NPZDataset(tmp_path,ModelConfig().downscale(5))[0]
    np.testing.assert_allclose(boxes[:,:7],boxes_to_baselink(raw),atol=1e-6)
    assert boxes.shape==(1,8) and boxes[0,7]==5


def test_resume_requires_matching_label_frame_and_both_calibrations(tmp_path):
    cfg=ModelConfig()
    args=SimpleNamespace(calibration=DEFAULT_CALIBRATION,lidar_calibration=None,box_frame='CLIDAR')
    state={'cfg':vars(cfg),'box_frame':'CLIDAR','recipe':{},
           'calibration_sha256':calibration_sha256(DEFAULT_CALIBRATION),
           'lidar_calibration_sha256':calibration_sha256(DEFAULT_CALIBRATION)}
    check_resume(state,cfg,args)
    with pytest.raises(ValueError,match='label box frame'):
        check_resume(dict(state,box_frame='BASELINK'),cfg,args)
    with pytest.raises(ValueError,match='camera calibration'):
        check_resume(dict(state,calibration_sha256='old'),cfg,args)
    with pytest.raises(ValueError,match='CLIDAR calibration'):
        check_resume(dict(state,lidar_calibration_sha256=None),cfg,args)
    changed=tmp_path/'different.json';changed.write_text('{"different":"rig"}')
    args.lidar_calibration=str(changed)
    with pytest.raises(ValueError,match='CLIDAR calibration'):
        check_resume(state,cfg,args)


def test_converted_labels_reproject_like_raw_lidar_using_opencv():
    cv2=pytest.importorskip('cv2')
    data=json.loads(Path(DEFAULT_CALIBRATION).read_text())
    for camera in load_cameras(DEFAULT_CALIBRATION):
        lidar2cam=np.asarray(data['lidar_to_camera'][camera.name])
        camera_xyz=np.array([[0.,0.,10.],[1.,.4,2.],[-2.,.3,1.]])
        raw_xyz=camera_xyz@np.linalg.inv(lidar2cam)[:3,:3].T+np.linalg.inv(lidar2cam)[:3,3]
        raw=np.column_stack((raw_xyz,np.tile([4,2,1.5,0],(3,1))))
        base=boxes_to_baselink(raw)
        actual,_,_=project_fisheye(base[:,:3],camera)
        reference,_=cv2.fisheye.projectPoints(camera_xyz[None],np.zeros(3),np.zeros(3),
                                             camera.intrinsic,camera.distortion)
        np.testing.assert_allclose(actual,reference[0],atol=1e-6)


def test_rosa_comparison_uses_model_output_frame_before_range_filtering():
    comparison=load_script('compare_validation_rosa')
    raw=np.array([[10.,2.,-1.,4.,2.,1.5,.4,0,0],[47.8,0,0,4,2,2,0,0,0]])
    frame={'boxes_3d':raw,'names':['Car','Truck'],'scores':[.9,.8]}
    legacy=comparison.as_rosa_array(frame,ModelConfig())
    assert len(legacy)==2
    np.testing.assert_allclose(legacy[:,:7],raw[:,:7],atol=1e-6)
    corrected=comparison.as_rosa_array(frame,ModelConfig(),read_T_base_clidar())
    assert len(corrected)==1
    np.testing.assert_allclose(corrected[:,:7],boxes_to_baselink(raw)[:1],atol=1e-6)
    assert corrected[0,7]==0 and corrected[0,8]==pytest.approx(.9)
