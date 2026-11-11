"""Static, calibration-dependent fisheye sampling LUT.

ARC_* uses the supplied OpenCV fisheye/equidistant calibration (four k terms).
No OCam conversion is applied. All trigonometry and lens cues are precomputed.

T_BASELINK_CAMERA maps camera XYZ into BASELINK XYZ; never invert twice.
BEV: row zero = forward (+x), column zero = left (+y).
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .config import CAMERAS, ModelConfig


@dataclass
class Camera:
    name: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    T_base_cam: np.ndarray
    width: int
    height: int
    model: str


def load_cameras(path):
    data = json.loads(Path(path).read_text())
    result = []
    for name in CAMERAS:
        i = data['intrinsics'][name]
        e = data['extrinsics'][name]
        if e['from'] != name or e['to'] != 'BASELINK':
            raise ValueError(f'Unexpected extrinsic convention for {name}')
        if i['width'] <= 0 or i['height'] <= 0:
            raise ValueError(f'Invalid camera image size: {name}')
        projection = i.get('projection_for_validation', 'opencv_fisheye')
        if projection != 'opencv_fisheye':
            raise ValueError(f'{name}: expected OpenCV fisheye calibration, got {projection}')
        K = np.asarray(i['camera_matrix'], np.float64)
        D = np.asarray(i['distortion_coefficients'], np.float64).reshape(-1)
        if (K.shape != (3, 3) or D.shape != (4,) or not np.isfinite(K).all()
                or not np.isfinite(D).all() or min(K[0,0], K[1,1]) <= 0):
            raise ValueError(f'Invalid OpenCV fisheye intrinsics: {name}')
        t = np.asarray(e['matrix'], dtype=np.float64)
        if t.shape != (4, 4) or not np.allclose(t[3], [0, 0, 0, 1]):
            raise ValueError(f'Malformed extrinsics for {name}')
        R = t[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-2) or np.linalg.det(R) < 0.95:
            raise ValueError(f'Non-rigid camera transform for {name}')
        result.append(Camera(name, K, D,
                             t, int(i['width']), int(i['height']),
                             i.get('camera_model', 'unspecified')))
    return result


def project_fisheye(points_base, camera, theta_max_degrees=105.0):
    """Project Nx3 BASELINK points using OpenCV fisheye theta polynomial.

    Returns Nx2 raw-image pixel positions, Nx visibility mask, Nx incidence
    angle [radians]. Supports rays >90deg from optical axis (true fisheye).
    """
    pts = np.asarray(points_base, dtype=np.float64).reshape(-1, 3)
    T_cam_base = np.linalg.inv(camera.T_base_cam)
    xyz = pts @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    X, Y, Z = xyz.T
    rho = np.hypot(X, Y)
    theta = np.arctan2(rho, Z)
    coeffs = list(camera.distortion)
    if len(coeffs) != 4:
        raise ValueError('Expected four OpenCV fisheye distortion coefficients')
    k1, k2, k3, k4 = coeffs
    t2 = theta * theta
    theta_d = theta * (1 + k1*t2 + k2*t2*t2 + k3*t2**3 + k4*t2**4)
    scale = np.divide(theta_d, rho, out=np.zeros_like(theta_d), where=rho > 1.e-10)
    xy = np.stack((X*scale, Y*scale, np.ones_like(X)), axis=-1)
    pixels = xy @ camera.intrinsic.T
    u, v = pixels[:, 0], pixels[:, 1]
    valid = (theta <= math.radians(theta_max_degrees)) & \
        (np.linalg.norm(xyz, axis=1) > 0.05) & \
        (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)
    return np.column_stack([u, v]), valid, theta


def resized_camera_intrinsics(intrinsic,src_width,src_height,dst_width,dst_height):
    """K_network = A_resize @ K_native; OpenCV distortion is unchanged.

    Pixel centres follow u'=(u+.5)*sx-.5+left, likewise for v. Use the
    actual integer resize dimensions, not an approximate scalar scale.
    """
    width,height,left,top=letterbox_params(src_width,src_height,dst_width,dst_height)
    sx,sy=width/src_width,height/src_height
    transform=np.array([[sx,0,(sx-1)/2+left],[0,sy,(sy-1)/2+top],[0,0,1]],dtype=np.float64)
    return transform@np.asarray(intrinsic,dtype=np.float64)


def letterbox_params(src_width: int, src_height: int, dst_width: int, dst_height: int):
    """Exact integer resize-and-pad transform shared by preprocess and projection.

    OpenCV-compatible half-pixel center convention. Scale is computed using
    INTEGER resized dimensions to avoid a geometry/image mismatch.
    """
    scale = min(dst_width / src_width, dst_height / src_height)
    new_w = max(1, round(src_width * scale))
    new_h = max(1, round(src_height * scale))
    pad_left = (dst_width - new_w) // 2
    pad_top = (dst_height - new_h) // 2
    return new_w, new_h, pad_left, pad_top


def build_projection_lut(cameras, cfg: ModelConfig):
    """grid [K,N,H,W,2] for grid_sample(..., align_corners=False).

    The +0.5 terms are raw pixel centers. The same resize + letterbox
    transform must be applied to the raw images before feeding the model.
    Then convert resized input pixels to the actual stride-8 CNN lattice.
    """
    h, w = cfg.bev_height, cfg.bev_width
    x = cfg.bev_x_max - (np.arange(h, dtype=np.float64)+.5)*cfg.bev_resolution
    y = cfg.bev_y_max - (np.arange(w, dtype=np.float64)+.5)*cfg.bev_resolution
    xx, yy = np.meshgrid(x, y, indexing='ij')
    grids, priors, visibilities = [], [], []
    theta_max = math.radians(cfg.theta_max_degrees)
    for z in cfg.heights:
        points = np.stack([xx, yy, np.full_like(xx, z)], axis=-1).reshape(-1, 3)
        row_grid, row_prior, row_valid = [], [], []
        for cam in cameras:
            uv, visible, theta = project_fisheye(points, cam, cfg.theta_max_degrees)
            # Weight central rays more, but avoid dropping valid edge observations.
            margin = np.minimum.reduce((uv[:,0]/cam.width, 1-uv[:,0]/cam.width,
                                        uv[:,1]/cam.height, 1-uv[:,1]/cam.height))
            reliability = np.maximum(.05, 1. - (theta/theta_max)**2) * np.clip(4*margin, .1, 1.)
            prior = np.where(visible, reliability, 0.)
            new_w, new_h, pad_left, pad_top = letterbox_params(
                cam.width, cam.height, cfg.image_width, cfg.image_height)
            grid = np.stack([
                2*((uv[:, 0]+.5)*(new_w/cam.width)+pad_left+3.5-cfg.feature_pixel_origin)/cfg.image_width-1,
                2*((uv[:, 1]+.5)*(new_h/cam.height)+pad_top+3.5-cfg.feature_pixel_origin)/cfg.image_height-1,
            ], axis=-1)
            grid[~visible] = 2.0  # out of bounds for grid_sample
            row_grid.append(grid.reshape(h, w, 2).astype(np.float32))
            row_prior.append(prior.reshape(h, w).astype(np.float32))
            row_valid.append(visible.reshape(h, w))
        grids.append(row_grid)
        priors.append(row_prior)
        visibilities.append(row_valid)
    return (np.asarray(grids, np.float32),
            np.asarray(priors, np.float32),
            np.asarray(visibilities, np.bool_))


def build_image_geometry(cameras, cfg):
    """Eight static cues at stride 8: camera ray XYZ, UV, angle, margin, scale.

    Invert only the supplied OpenCV theta polynomial. This small table is an
    explicit per-camera encoder input at deployment; it needs no runtime trig.
    """
    h, w = cfg.image_height // 8, cfg.image_width // 8
    yy, xx = np.meshgrid(np.arange(h)*8+cfg.feature_pixel_origin,
                         np.arange(w)*8+cfg.feature_pixel_origin, indexing='ij')
    maps = []
    ts = np.linspace(0, math.radians(cfg.theta_max_degrees), 8193)
    for cam in cameras:
        nw, nh, left, top = letterbox_params(cam.width,cam.height,cfg.image_width,cfg.image_height)
        u = (xx+.5-left)*cam.width/nw-.5
        v = (yy+.5-top)*cam.height/nh-.5
        inv = np.linalg.inv(cam.intrinsic)
        lens = np.stack((u,v,np.ones_like(u)),axis=-1) @ inv.T
        rho = np.hypot(lens[...,0],lens[...,1])
        k1,k2,k3,k4 = cam.distortion
        td = ts*(1+k1*ts**2+k2*ts**4+k3*ts**6+k4*ts**8)
        if np.any(np.diff(td) <= 0):
            raise ValueError(f'{cam.name}: OpenCV lens polynomial is not invertible within configured FOV')
        theta = np.interp(rho,td,ts)
        radial = np.divide(np.sin(theta),rho,out=np.ones_like(rho),where=rho>1e-10)
        margin = np.minimum.reduce((u/cam.width,1-u/cam.width,v/cam.height,1-v/cam.height))
        derivative = 1+3*k1*theta**2+5*k2*theta**4+7*k3*theta**6+9*k4*theta**8
        cues = np.stack((lens[...,0]*radial,lens[...,1]*radial,np.cos(theta),
                         2*(u+.5)/cam.width-1,2*(v+.5)/cam.height-1,
                         theta/ts[-1],np.clip(4*margin,0,1),np.clip(derivative,0,2)/2))
        valid = (rho <= td[-1]) & (margin >= 0)
        maps.append(np.where(valid[None],cues,0).astype(np.float32))
    return np.stack(maps)
