"""V7 track segmentation is explicit, collision-free and independently auditable."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from test_desktop_heavy_review import session, review
from audit_manual_release import audit

spec=importlib.util.spec_from_file_location('build_v7',Path(__file__).resolve().parents[1]/'scripts/prepare_manual_review_v7.py')
build=importlib.util.module_from_spec(spec);spec.loader.exec_module(build)


def test_v7_track_split_preserves_boxes_and_has_independent_source_audit(session):
    source=session.root/'source';session.export(source)
    s=review.ReviewSession(session.root,source/'fisheye3dod_infos_train.pkl',source/'fisheye3dod_infos_val.pkl',
        source/'train_yolo_extents_v1.json',session.calibration,session.root/'v7_state.json',classes=review.CLASSES)
    s.apply('train:t1:0','bus')
    out=session.root/'v7';s.export(out)
    before=review.load_annotations(out/'fisheye3dod_infos_val.pkl')
    changes=build.split_mixed_track_ids(out,s,{('bag','7')})
    assert len(changes)==1 and changes[0]['id']=='val:v1:0'
    assert changes[0]['new_track_id']>108
    after=review.load_annotations(out/'fisheye3dod_infos_val.pkl')
    np.testing.assert_array_equal(before['data_list'][0]['annos']['gt_boxes'],after['data_list'][0]['annos']['gt_boxes'])
    checked=audit(s.root,out,source,s.calibration)
    assert checked['track_id_edits']==1 and not checked['track_class_conflicts']
    assert checked['yolo']['pixel_evidence_preserved']
    ledger=out/'track_id_revisions.json';ledger.write_text(ledger.read_text()+' ')
    with pytest.raises(ValueError,match='track revision hash'):review.verify_export(s.root,out,s.calibration)


def test_v7_track_split_handles_removal_shift_without_touching_neighbor(session):
    s=review.ReviewSession(session.root,session.paths['train'],session.paths['val'],session.extents,
        session.calibration,session.root/'all_state.json',classes=review.CLASSES)
    s.remove('train:t1:0');out=s.root/'removed';s.export(out)
    assert not build.split_mixed_track_ids(out,s,{('bag','7')})
    train=review.load_annotations(out/'fisheye3dod_infos_train.pkl')
    assert train['data_list'][0]['annos']['gt_ids'].tolist()==[107]


def test_v7_audit_rejects_unapproved_track_id_change(session):
    import pickle
    source=session.root/'source';session.export(source)
    s=review.ReviewSession(session.root,source/'fisheye3dod_infos_train.pkl',source/'fisheye3dod_infos_val.pkl',
        source/'train_yolo_extents_v1.json',session.calibration,session.root/'v7_state.json',classes=review.CLASSES)
    s.apply('train:t1:0','bus');out=session.root/'v7';s.export(out)
    build.split_mixed_track_ids(out,s,{('bag','7')})
    path=out/'fisheye3dod_infos_val.pkl';doc=review.load_annotations(path)
    doc['data_list'][0]['annos']['gt_ids'][1]=999
    path.write_bytes(pickle.dumps(doc,protocol=4))
    manifest=json.loads((out/'review_manifest.json').read_text())
    manifest['splits']['val']['output_sha256']=review.sha(path)
    review.atomic_json(out/'review_manifest.json',manifest)
    with pytest.raises(AssertionError):audit(s.root,out,source,s.calibration)
