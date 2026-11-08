import math
from pathlib import Path
import numpy as np
import torch
from raybev import ModelConfig, RayBEVLite, CLASSES, CLASS_TO_ID
from raybev.geometry import build_projection_lut, load_cameras, letterbox_params, project_fisheye
from raybev.preprocessing import letterbox_images
from raybev.detection import make_targets, detection_loss, decode_detections, metric_to_cell

CAL = Path(__file__).parents[1] / 'configs/calibration_baselink.json'


def test_96_meter_grid_800x640_input_and_six_classes():
    cfg = ModelConfig().validate()
    assert (cfg.image_height, cfg.image_width) == (640, 800)
    assert cfg.bev_height == cfg.bev_width == 384
    assert math.isclose(cfg.bev_x_max - cfg.bev_x_min, 96.)
    assert math.isclose(cfg.bev_y_max - cfg.bev_y_min, 96.)
    assert math.isclose(cfg.bev_resolution, .25)
    assert cfg.num_classes == 6
    assert (cfg.bev_x_min, cfg.bev_x_max) == (-48., 48.)
    assert (cfg.bev_y_min, cfg.bev_y_max) == (-48., 48.)
    assert CLASSES == ('car', 'pedestrian', 'cyclist', 'truck', 'van', 'bus')
    assert CLASS_TO_ID['truck'] == 3
    assert CLASS_TO_ID['van'] == 4
    assert CLASS_TO_ID['bus'] == 5


def test_native_source_aspect_1920x1536_resizes_without_padding():
    assert letterbox_params(1920, 1536, 800, 640) == (800, 640, 0, 0)
    img = torch.ones((1, 4, 3, 1536, 1920))
    out = letterbox_images(img, 800, 640)
    assert out.shape == (1, 4, 3, 640, 800)
    assert out.min().item() == 1.
    assert out.max().item() == 1.


def test_lut_coordinates_no_padding():
    cfg = ModelConfig()
    cams = load_cameras(CAL)
    lut, prior, valid = build_projection_lut(cams, cfg)
    assert lut.shape == (4, 4, 384, 384, 2)
    assert np.any(valid)
    assert np.all(prior[~valid] == 0)
    z, c, row, col = np.argwhere(valid)[len(np.argwhere(valid)) // 2]
    x = cfg.bev_x_max - (row + .5) * cfg.bev_resolution
    y = cfg.bev_y_max - (col + .5) * cfg.bev_resolution
    uv, ok, theta = project_fisheye([[x, y, cfg.heights[z]]], cams[c], cfg.theta_max_degrees)
    assert ok[0]
    gx = 2 * ((uv[0, 0] + .5) * (800 / cams[c].width)) / 800 - 1
    gy = 2 * ((uv[0, 1] + .5) * (640 / cams[c].height)) / 640 - 1
    assert abs(lut[z, c, row, col, 0] - gx) < 1e-5
    assert abs(lut[z, c, row, col, 1] - gy) < 1e-5


def test_encode_decode_exact_grid_point():
    cfg = ModelConfig.from_checkpoint({'cfg':{'architecture_version':2}})
    box = torch.tensor([[2.5, -4., 0.8, 4.5, 1.8, 1.5, 1.1, 0.]])
    heat, reg, mask = make_targets([box], cfg)
    assert mask.sum().item() == 1
    row, col = metric_to_cell(2.5, -4., cfg)
    i, j = math.floor(row + .5), math.floor(col + .5)
    logits = torch.full_like(heat, -20.)
    logits[0, 0, i, j] = 15.
    det = decode_detections((logits, reg), cfg, score_threshold=.5)[0]
    assert det.shape == (1, 9)
    assert torch.allclose(det[0, :7], box[0, :7], atol=1e-4)
    loss, _ = detection_loss((logits, reg), (heat, reg, mask),cfg=cfg)
    assert torch.isfinite(loss)


def test_all_six_class_target_and_decode():
    cfg = ModelConfig.from_checkpoint({'cfg':{'architecture_version':2}})
    boxes = []
    for class_id in range(6):
        boxes.append([15. - class_id * 5., -10. + class_id * 4., 0.9,
                      4.0, 1.9, 1.8, 0.2 * class_id, class_id])
    boxes = torch.tensor(boxes)
    heat, reg, mask = make_targets([boxes], cfg)
    assert heat.shape == (1, 6, 384, 384)
    assert mask.sum().item() == 6
    logits = torch.full_like(heat, -20.0)
    for box in boxes:
        class_id = int(box[-1].item())
        row, col = metric_to_cell(float(box[0]), float(box[1]), cfg)
        i, j = math.floor(row + .5), math.floor(col + .5)
        assert heat[0, class_id, i, j] == 1.0
        logits[0, class_id, i, j] = 15.
    decoded = decode_detections((logits, reg), cfg, score_threshold=0.5)[0]
    assert decoded.shape == (6, 9)
    assert {int(v) for v in decoded[:, 7].tolist()} == set(range(6))
    for pred in decoded:
        expected = boxes[int(pred[7])]
        assert torch.allclose(pred[:7], expected[:7], atol=1e-4)
    loss, metrics = detection_loss((logits, reg), (heat, reg, mask),cfg=cfg)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics['heatmap'])


def test_small_forward_6_class_head():
    torch.set_num_threads(2)
    cfg = ModelConfig().downscale(5)
    model = RayBEVLite(cfg, str(CAL)).eval()
    with torch.no_grad():
        outputs = model(torch.zeros((1, 4, 3, 128, 160)))
    assert outputs[0].shape == (1, 6, cfg.bev_height, cfg.bev_width)
    assert outputs[1].shape == (1, 8, cfg.bev_height, cfg.bev_width)
    assert outputs[2].shape == (1, 1, cfg.bev_height, cfg.bev_width)
    assert torch.isfinite(outputs[0]).all()


def test_edge_of_range_and_outside_targets():
    cfg = ModelConfig()
    # Cell centers closest to the four configured boundaries.
    boxes = torch.tensor([
        [47.875, 47.875, 1., 4., 2., 2., 0., 0.],
        [-47.875, -47.875, 1., 4., 2., 2., 0., 3.],
        [49.0, 0., 1., 4., 2., 2., 0., 1.],
        [0., -49., 1., 4., 2., 2., 0., 5.],
    ])
    heat, reg, mask = make_targets([boxes], cfg)
    assert mask.sum().item() == 2
    assert mask[0, 0, 0, 0]
    assert mask[0, 0, 383, 383]
    assert reg.shape == (1, 8, 384, 384)
    for x, y in [(47.875,47.875),(-47.875,-47.875)]:
        r, c = metric_to_cell(x,y,cfg)
        assert 0 <= round(r) < 384 and 0 <= round(c) < 384


def test_targets_reject_legacy_velocity_columns():
    cfg = ModelConfig()
    import pytest
    with pytest.raises(ValueError, match='velocities are not used'):
        make_targets([torch.tensor([[1.,2.,.5,4.,2.,2.,0.,0.,0.,0.]])], cfg)


def test_training_legacy_npz_drops_velocity_columns(tmp_path):
    import importlib.util
    module_path = Path(__file__).parents[1] / 'scripts/train.py'
    spec = importlib.util.spec_from_file_location('raybev_train', module_path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    np.savez(tmp_path / 'frame.npz',
             images=np.zeros((4,3,32,40), dtype=np.uint8),
             boxes=np.array([[2.,3.,1.,5.,2.,2.,0.2,15.,-3.,4.]], dtype=np.float32))
    ims, boxes = train.NPZDataset(tmp_path, ModelConfig(),box_frame='BASELINK')[0]
    assert ims.shape == (4,3,640,800)
    assert boxes.shape == (1,8)
    assert torch.allclose(boxes[0], torch.tensor([2.,3.,1.,5.,2.,2.,0.2,4.]))
