"""Class-only corrections and weak YOLO-target synchronization must fail closed."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

spec = importlib.util.spec_from_file_location('heavy_review', Path(__file__).resolve().parents[1] / 'scripts/prepare_heavy_annotation_review.py')
review = importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


def extent_release(tmp_path):
    old = dict(frames={'1': [dict(gt=[1., 2., 3., 4., 1.8, 1.5, 0., 5.], bbox=[1, 2, 3, 4])]})
    new = copy.deepcopy(old); new['frames']['1'][0]['gt'][7] = 0.
    source = tmp_path / 'original.json'; source.write_text(json.dumps(old))
    release = tmp_path / 'train_yolo_extents_v1.json'; release.write_text(json.dumps(new))
    expected = dict(source_sha256=review.sha(source), output_sha256=review.sha(release))
    return source, release, expected


def test_extent_comparison_when_original_exists(tmp_path):
    source, _, expected = extent_release(tmp_path)
    assert review.verify_extent_files(tmp_path, source, expected) == (1, 1)


def test_missing_original_requires_opt_in_and_reports_unavailable(tmp_path):
    source, _, expected = extent_release(tmp_path)
    missing = tmp_path / 'deleted.json'
    with pytest.raises(FileNotFoundError):
        review.verify_extent_files(tmp_path, missing, expected)
    assert review.verify_extent_files(tmp_path, missing, expected, True) == (1, None)


def test_missing_original_never_skips_release_hash(tmp_path):
    _, release, expected = extent_release(tmp_path)
    release.write_text(release.read_text() + ' ')
    with pytest.raises(ValueError, match='Revised extent file hash differs'):
        review.verify_extent_files(tmp_path, tmp_path / 'deleted.json', expected, True)


def test_opt_in_does_not_accept_wrong_existing_source(tmp_path):
    source, release, expected = extent_release(tmp_path)
    with pytest.raises(ValueError, match='Source extent file hash differs'):
        review.verify_extent_files(tmp_path, release, expected, True)
    source.write_text('{}')
    with pytest.raises(ValueError, match='Source extent file hash differs'):
        review.verify_extent_files(tmp_path, source, expected, True)


def sample():
    return dict(metainfo={'kept': True}, data_list=[dict(token='1', metainfo=dict(sequence_id='bag', temporal_split='train'),
        cam_info={'unchanged': 'image.jpg'}, annos=dict(gt_names=np.array(['Bus'], dtype='U3'),
        gt_mesh=np.array(['detzero.bus']), gt_ids=np.array([12]),
        gt_boxes=np.array([[1., 2., 3., 6., 2., 2., .1]])))])


def decision():
    return {('1', 0): dict(old_label='bus', new_label='Truck', track_id='12',
                          box_CLIDAR=[1., 2., 3., 6., 2., 2., .1])}


def test_string_dtype_expands_and_original_is_unchanged():
    src = sample(); dst = review.revise_document(src, decision())
    assert src['data_list'][0]['annos']['gt_names'].tolist() == ['Bus']
    assert dst['data_list'][0]['annos']['gt_names'].tolist() == ['Truck']
    assert dst['data_list'][0]['annos']['gt_mesh'].tolist() == ['detzero.truck']
    assert review.verify_document(src, dst, decision()) == 1


def test_user_dimensions_change_only_lwh_and_do_not_mutate_source():
    src = sample(); decisions = decision()
    decisions[('1', 0)].update(new_label='Car', new_dimensions_lwh=[4., 1.8, 1.5],
                             geometry_authority='Explicit user request')
    dst = review.revise_document(src, decisions)
    old = src['data_list'][0]['annos']['gt_boxes'][0]
    new = dst['data_list'][0]['annos']['gt_boxes'][0]
    np.testing.assert_array_equal(new[3:6], [4., 1.8, 1.5])
    np.testing.assert_array_equal(old[3:6], [6., 2., 2.])
    np.testing.assert_array_equal(new[[0, 1, 2, 6]], old[[0, 1, 2, 6]])
    assert review.verify_document(src, dst, decisions) == 1
    dst['data_list'][0]['annos']['gt_boxes'][0, 3] = 4.1
    with pytest.raises(ValueError):
        review.verify_document(src, dst, decisions)


@pytest.mark.parametrize('dimensions', [[4., 1.8], [4., -1.8, 1.5], [4., float('nan'), 1.5]])
def test_invalid_dimensions_rejected(dimensions):
    d = decision(); d[('1', 0)].update(new_dimensions_lwh=dimensions, geometry_authority='User')
    with pytest.raises(ValueError):
        review.revise_document(sample(), d)


def test_dimensions_require_explicit_authority():
    d = decision(); d[('1', 0)]['new_dimensions_lwh'] = [4., 1.8, 1.5]
    with pytest.raises(ValueError):
        review.revise_document(sample(), d)


@pytest.mark.parametrize('field', ['gt_boxes', 'gt_ids'])
def test_second_sweep_rejects_geometry_and_identity_mutation(field):
    src = sample(); dst = review.revise_document(src, decision())
    dst['data_list'][0]['annos'][field].flat[0] += 1
    with pytest.raises(ValueError):
        review.verify_document(src, dst, decision())


def dataset(label):
    return SimpleNamespace(rows=[dict(token='1')], boxes=[torch.tensor([[1., 2., 3., 6., 2., 2., .1, label]])],
                           box_frame='CLIDAR', z_reference='center')


def test_extent_changes_only_gt_label_and_fingerprint():
    old = dataset(5.); new = dataset(3.)
    record = dict(gt=old.boxes[0][0].tolist(), bbox=[10., 20., 50., 60.],
                  edge_valid=[True, False, True, True], camera=2, weight=.2,
                  verification=dict(cache_sha256='source evidence unchanged'))
    src = dict(training_boxes_sha256=review.training_boxes_sha256(old), frames={'1': [record]}, counts={'bus': 1, 'rejected': 7})
    snapshot = copy.deepcopy(src)
    out, stats = review.synchronize_extents(src, old, new)
    assert src == snapshot and stats['changed_gt_records'] == 1
    assert out['frames']['1'][0]['gt'][7] == 3.
    for key in record.keys() - {'gt'}:
        assert out['frames']['1'][0][key] == record[key]
    assert out['counts'] == {'truck': 1, 'rejected': 7}
    assert out['training_boxes_sha256'] == review.training_boxes_sha256(new)
    with pytest.raises(ValueError):
        review.synchronize_extents(src, new, old)


def test_extent_rejects_geometry_change_or_missing_association():
    old = dataset(5.); new = dataset(3.)
    src = dict(training_boxes_sha256=review.training_boxes_sha256(old), frames={'1': [dict(gt=old.boxes[0][0].tolist())]}, counts={})
    new.boxes[0][0, 0] += .1
    with pytest.raises(ValueError):
        review.synchronize_extents(src, old, new)
    new = dataset(3.); src['frames']['1'][0]['gt'][0] += .1
    with pytest.raises(ValueError):
        review.synchronize_extents(src, old, new)


def test_size_edits_allowed_only_without_associated_extent_evidence():
    old = dataset(5.); new = dataset(0.)
    src = dict(training_boxes_sha256=review.training_boxes_sha256(old), frames={'1': []}, counts={})
    new.boxes[0][0, 3:6] = torch.tensor([4., 1.8, 1.5])
    with pytest.raises(ValueError):
        review.synchronize_extents(src, old, new)
    out, stats = review.synchronize_extents(src, old, new, allow_unreferenced_size_changes=True)
    assert stats['unreferenced_size_changes'] == 1 and stats['records'] == 0
    assert out['training_boxes_sha256'] == review.training_boxes_sha256(new)
    src['frames']['1'] = [dict(gt=old.boxes[0][0].tolist())]
    with pytest.raises(ValueError, match='fresh association evidence'):
        review.synchronize_extents(src, old, new, allow_unreferenced_size_changes=True)
    src['frames']['1'] = []
    new.boxes[0][0, 0] += 1
    with pytest.raises(ValueError):
        review.synchronize_extents(src, old, new, allow_unreferenced_size_changes=True)
