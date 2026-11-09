import importlib.util
from pathlib import Path
from copy import deepcopy
import json
import pickle
import numpy as np
import pytest
from raybev.rosa_dataset import names_to_ids,parse_box_annotations


def module():
    path=Path(__file__).parents[1]/'scripts/prepare_label_revision.py'
    spec=importlib.util.spec_from_file_location('label_revision_test',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture():
    return dict(metainfo={},data_list=[dict(token='t1',metainfo=dict(sequence_id='bag',temporal_split='val'),
          annos=dict(gt_ids=np.array([7,8]),gt_names=np.array(['Bus','Bus'],dtype='<U3'),
          gt_mesh=np.array(['detzero.bus','detzero.bus']),gt_boxes=np.array([[2,0,1,6,2,3,.2],[4,2,1,12,3,3.5,.3]])))])


def rule():return dict(sequence_id='bag',track_id='7',**{'from':'Bus','to':'Truck'},expected_raw_changes={'train':0,'val':1})


def test_tram_policy_maps_to_bus_but_unknown_taxonomies_fail():
    np.testing.assert_array_equal(names_to_ids(['Tram',' tram ',b'TRAM','Bus','Truck']),[5,5,5,5,3])
    with pytest.raises(ValueError,match='Unknown class'):names_to_ids(['train'])


def test_review_changes_only_approved_track_class_without_unicode_truncation():
    original=fixture();revised,changes=module().revise_document(original,[rule()],'val')
    assert revised['data_list'][0]['annos']['gt_names'].tolist()==['Truck','Bus']
    assert original['data_list'][0]['annos']['gt_names'].tolist()==['Bus','Bus']
    np.testing.assert_array_equal(original['data_list'][0]['annos']['gt_boxes'],revised['data_list'][0]['annos']['gt_boxes'])
    assert len(changes)==1 and revised['data_list'][0]['annos']['gt_mesh'][0]=='detzero.truck'


@pytest.mark.parametrize('problem',['count','old_class','duplicate','split'])
def test_review_rejects_wrong_input_contracts(problem):
    r=rule();doc=fixture();rules=[r]
    if problem=='count':r['expected_raw_changes']['val']=2
    if problem=='old_class':r['from']='Car'
    if problem=='duplicate':rules.append(deepcopy(r))
    if problem=='split':doc['data_list'][0]['metainfo']['temporal_split']='train'
    with pytest.raises(ValueError):module().revise_document(doc,rules,'val')


def test_revision_is_dry_by_default_keeps_train_byte_identical_and_never_overwrites(tmp_path):
    m=module();root=tmp_path/'data';(root/'rosa').mkdir(parents=True)
    train=dict(metainfo={},data_list=[]);val=fixture()
    for split,doc in [('train',train),('val',val)]:
        with (root/'rosa'/('fisheye3dod_infos_'+split+'.pkl')).open('wb') as f:pickle.dump(doc,f)
    review=tmp_path/'rules.json';review.write_text(json.dumps(dict(type='raybev_track_class_review_v1',approved=True,rules=[rule()])))
    output=tmp_path/'revision';report=m.create_revision(root,review,output)
    assert not output.exists() and report['splits']['val']['raw_changes']==1
    m.create_revision(root,review,output,write=True)
    assert (output/'train.pkl').read_bytes()==(root/'rosa/fisheye3dod_infos_train.pkl').read_bytes()
    m.verify_revision(root,output)
    # Same labels can be serialized differently after transfer to the server.
    with (root/'rosa/fisheye3dod_infos_val.pkl').open('wb') as f:pickle.dump(val,f,protocol=2)
    m.verify_revision(root,output)
    val['data_list'][0]['annos']['gt_boxes'][0,0]+=1
    with (root/'rosa/fisheye3dod_infos_val.pkl').open('wb') as f:pickle.dump(val,f)
    with pytest.raises(ValueError,match='source labels differ'):m.verify_revision(root,output)
    with pytest.raises(FileExistsError):m.create_revision(root,review,output,write=True)
