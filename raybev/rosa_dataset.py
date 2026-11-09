"""ROS/Fisheye3DOD combined annotations -> RayBEV training tensors.

Preserves sample tokens, source sequence, and temporal split. Reads the *existing*
ROSA `data_list` layout without rewriting boxes or creating pseudo labels.
"""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
import pickle
import numpy as np
import torch
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from .config import CAMERAS,CLASSES,CLASS_TO_ID,DEFAULT_BOX_FRAME,DEFAULT_CALIBRATION
from .preprocessing import normalize_imagenet
from .detection import in_bev_range


class SerializedCameraMetadata:
    """Passive legacy PKL camera state; never used to project OpenCV images."""


class AnnotationUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module in ('ocamcamera','ocamcamera.ocamcamera') and name=='OcamCamera':
            # ROSA PKLs may retain an unused OCam object even for native paths.
            # Reading paths/labels must not require the OCam projection package.
            return SerializedCameraMetadata
        return super().find_class(module,name)


def load_annotations(path):
    # Load trusted locally generated pkl only (Python pickle is executable).
    with Path(path).open('rb') as f:obj=AnnotationUnpickler(f).load()
    if not isinstance(obj,dict) or not isinstance(obj.get('data_list'),list):
        raise ValueError(f'Expected trusted ROSA dict with data_list: {path}')
    return obj


def training_boxes_sha256(dataset):
    """Portable TRAIN-label identity: token + ordered BASELINK boxes, not paths.

    Preserve box order because ties in shared regression cells use that order.
    Ignore pickle protocol, row ordering and camera/source paths across machines.
    """
    digest=hashlib.sha256()
    digest.update(json.dumps([list(CLASSES),dataset.box_frame,dataset.z_reference],separators=(',',':')).encode())
    for i in sorted(range(len(dataset.rows)),key=lambda i:str(dataset.rows[i]['token'])):
        row=[str(dataset.rows[i]['token']),dataset.boxes[i].tolist()]
        digest.update(json.dumps(row,separators=(',',':'),allow_nan=False).encode())
        digest.update(b'\n')
    return digest.hexdigest()


def names_to_ids(names):
    aliases={'vehicle':'car','bicycle':'cyclist','person':'pedestrian',
             'motorbike':'cyclist','construction_vehicle':'van',
             'tram':'bus'}  # Explicit custom taxonomy: rail passenger vehicles are Bus.
    ids=[]
    for x in names:
        name=str(x).strip().lower()
        if name.startswith("b'"):name=name[2:-1]
        name=aliases.get(name,name)
        if name not in CLASS_TO_ID:raise ValueError(f'Unknown class {x!r}. Define explicit label taxonomy, do not guess.')
        ids.append(CLASS_TO_ID[name])
    return np.asarray(ids,dtype=np.int64)


def boxes_to_baselink(boxes,box_frame=DEFAULT_BOX_FRAME,T_base_clidar=None,z_reference='center'):
    """Convert raw centers/headings once; targets and predictions use BASELINK.

    Boxes are upright yaw boxes, so the rotated heading is projected onto XY.
    A bottom-Z input is centered in its source frame before rotating/translating.
    """
    boxes=np.asarray(boxes,dtype=np.float64)
    if boxes.ndim!=2 or boxes.shape[1]<7:raise ValueError(f'Invalid 3D boxes {boxes.shape}')
    boxes=boxes[:,:7].copy()
    if z_reference=='bottom':boxes[:,2]+=boxes[:,5]/2
    elif z_reference!='center':raise ValueError(z_reference)
    box_frame=box_frame.upper()
    if box_frame=='CLIDAR':
        T=read_T_base_clidar() if T_base_clidar is None else np.asarray(T_base_clidar,dtype=np.float64)
        xyz_h=np.concatenate((boxes[:,:3],np.ones((len(boxes),1))),axis=1)
        boxes[:,:3]=(xyz_h@T.T)[:,:3]
        dirs=np.stack((np.cos(boxes[:,6]),np.sin(boxes[:,6]),np.zeros(len(boxes))),axis=1)
        rot=dirs@T[:3,:3].T
        boxes[:,6]=np.arctan2(rot[:,1],rot[:,0])
    elif box_frame!='BASELINK':raise ValueError(f'Unsupported box_frame {box_frame}')
    return boxes


def parse_box_annotations(sample,box_frame=DEFAULT_BOX_FRAME,T_base_clidar=None,z_reference='center'):
    annos=sample.get('annos',{})
    for key in ('gt_boxes','gt_boxes_3d','boxes_3d'):
        if key in annos:
            v=annos[key]
            if hasattr(v,'tensor'):v=v.tensor
            if isinstance(v,torch.Tensor):v=v.detach().cpu().numpy()
            arr=np.asarray(v,dtype=np.float64)
            break
    else:raise KeyError(f'Sample {sample.get("token")} annos missing 3D boxes')
    if arr.size==0:arr=np.empty((0,7),dtype=np.float64)
    if arr.ndim!=2 or arr.shape[1]<7:raise ValueError(f'Invalid 3D boxes {arr.shape} in token {sample.get("token")}')
    if 'gt_names' in annos:
        labels=names_to_ids(annos['gt_names'])
    else:
        raise KeyError(f'Sample {sample.get("token")} missing gt_names: numeric IDs may be in a different ROSA class order')
    if len(labels)!=len(arr):raise ValueError(f'Class labels != boxes for {sample.get("token")}')
    boxes=boxes_to_baselink(arr,box_frame,T_base_clidar,z_reference)
    # Local visibility is clipped later by target encoder, but remove malformed boxes.
    keep=np.isfinite(boxes).all(axis=1)&(boxes[:,3:6]>0).all(axis=1)
    out=np.concatenate((boxes[keep],labels[keep,None].astype(np.float64)),axis=1)
    return torch.as_tensor(out,dtype=torch.float32)


def read_T_base_clidar(path=None):
    data=json.loads(Path(path or DEFAULT_CALIBRATION).read_text())
    extr=data['extrinsics']['CLIDAR']
    if extr['from']!='CLIDAR' or extr['to']!='BASELINK':raise ValueError('CLIDAR extrinsics must map CLIDAR to BASELINK')
    T=np.asarray(extr['matrix'],np.float64)
    if (T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1])
            or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-2)
            or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-2)):
        raise ValueError('Invalid CLIDAR transform')
    return T


def photometric_augment(images):
    """TRAIN-only appearance regularization; never move pixels or change the rig."""
    cameras=len(images)
    gain=images.new_empty(cameras,1,1,1).uniform_(.7,1.3)
    contrast=images.new_empty(cameras,1,1,1).uniform_(.75,1.25)
    color=images.new_empty(cameras,3,1,1).uniform_(.85,1.15)
    gamma=images.new_empty(cameras,1,1,1).uniform_(.85,1.15)
    mean=images.mean(dim=(1,2,3),keepdim=True)
    return (((images-mean)*contrast+mean)*gain*color).clamp(0,1).pow(gamma)


class RosaCombinedDataset(Dataset):
    def __init__(self,root,ann_file,box_frame=DEFAULT_BOX_FRAME,calibration=None,z_reference='center',
                 strict_size=True,expected_size=(800,640),cfg=None,augment=False):
        self.root=Path(root).resolve()
        self.annotation_path=self.root/ann_file if not Path(ann_file).is_absolute() else Path(ann_file)
        self.annotation=load_annotations(self.annotation_path)
        self.rows=self.annotation['data_list']
        self.box_frame=box_frame.upper()
        self.z_reference=z_reference
        self.T_base_clidar=read_T_base_clidar(calibration) if self.box_frame=='CLIDAR' else None
        self.strict_size=strict_size;self.expected_size=expected_size
        self.augment=augment
        self.cfg=cfg
        self.range_filtered_boxes=0
        self.gt_counts={name:0 for name in CLASSES}
        # Precache compact GT tensors; this also rejects mismatched taxonomies before training.
        self.boxes=[]
        for s in self.rows:
            b=parse_box_annotations(s,self.box_frame,self.T_base_clidar,self.z_reference)
            if cfg is not None:
                keep=in_bev_range(b,cfg)
                self.range_filtered_boxes+=int((~keep).sum())
                b=b[keep]
            self.boxes.append(b)
            for i in range(len(CLASSES)):
                self.gt_counts[CLASSES[i]]+=int((b[:,-1]==i).sum())

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        sample=self.rows[index]
        cam_dict=sample['cam_info']['cam_fisheye']
        image_list=[]
        for cam in CAMERAS:
            rel=Path(cam_dict[cam]['cam_path'])
            if rel.is_absolute() or '..' in rel.parts:raise ValueError(f'Unsafe camera path: {rel}')
            path=self.root/rel
            with Image.open(path) as im:
                im=im.convert('RGB')
                if im.size!=self.expected_size:
                    if self.strict_size:
                        raise ValueError(f'Expected pre-resized {self.expected_size} image, got {im.size}: {path}')
                    if (im.width*self.expected_size[1]) != (im.height*self.expected_size[0]):
                        raise ValueError(f'Nonmatching aspect ratio of {im.size} for {path}; prepare images first')
                    im=im.resize(self.expected_size,Image.Resampling.BILINEAR)
                array=np.asarray(im,dtype=np.uint8)
            image_list.append(torch.from_numpy(array.copy()).permute(2,0,1).float()/255.)
        images=torch.stack(image_list)
        if self.augment:
            # Photometry only: pixel coordinates and the OpenCV LUT stay fixed.
            images=photometric_augment(images)
        images=normalize_imagenet(images[None])[0]
        return images,self.boxes[index],str(sample['token'])


def collate_samples(examples):
    images,boxes,tokens=zip(*examples)
    return torch.stack(images),list(boxes),list(tokens)
