"""Verified image-region supervision. Never manufacture 2D truth from 3D GT."""
import json
import itertools
import torch
from torch.nn import functional as F
from .training import calibration_sha256
from .rosa_dataset import training_boxes_sha256
from .config import CAMERAS


def extent_points(boxes):
    # Corners plus edge samples: fisheye-projected straight edges are curved.
    corners=list(itertools.product((-.5,.5),repeat=3));points=list(corners)
    for i,a in enumerate(corners):
        for b in corners[i+1:]:
            if sum(x!=y for x,y in zip(a,b))==1:
                points.extend(tuple(x+(y-x)*t for x,y in zip(a,b)) for t in (.25,.5,.75))
    offsets=boxes.new_tensor(points)*boxes[...,None,3:6]
    x,y,z=offsets.unbind(-1);yaw=boxes[...,6,None]
    rotated=torch.stack((x*yaw.cos()-y*yaw.sin(),x*yaw.sin()+y*yaw.cos(),z),-1)
    return boxes[...,None,:3]+rotated


class VerifiedExtents:
    def __init__(self,path,dataset,calibration):
        doc=json.loads(open(path).read());self.sha256=calibration_sha256(path)
        if (doc.get('type')!='raybev_verified_extents_v1' or doc.get('split')!='train' or
                doc.get('training_boxes_sha256')!=training_boxes_sha256(dataset) or
                doc.get('camera_calibration_sha256')!=calibration_sha256(calibration) or
                doc.get('coordinates')!='network_pixels' or
                doc.get('image_size')!=[dataset.cfg.image_width,dataset.cfg.image_height] or
                doc.get('camera_order')!=list(CAMERAS)):
            raise ValueError('Verified extent data/calibration/coordinates mismatch')
        self.frames=doc['frames'];by_token={str(r['token']):g for r,g in zip(dataset.rows,dataset.boxes)}
        if not set(self.frames)<=set(by_token):raise ValueError('Extent supervision must be TRAIN only')
        count=0
        for token,records in self.frames.items():
            for record in records:
                gt=torch.tensor(record['gt'],dtype=torch.float32);bbox=torch.tensor(record['bbox'],dtype=torch.float32)
                if (gt.shape!=(8,) or not torch.isfinite(gt).all() or
                        not torch.isclose(by_token[token],gt,atol=1e-4,rtol=0).all(-1).any() or
                        bbox.shape!=(4,) or not torch.isfinite(bbox).all() or
                        not (bbox[2:]>bbox[:2]).all() or (bbox<0).any() or
                        bbox[2]>dataset.cfg.image_width or bbox[3]>dataset.cfg.image_height or
                        record.get('verified') is not True or record.get('source') not in ('manual_2d','verified_mask','postprocess_yolo') or
                        type(record.get('camera')) is not int or not 0<=record['camera']<4 or
                        len(record.get('edge_valid',[]))!=4 or any(type(x) is not bool for x in record['edge_valid']) or
                        not any(record['edge_valid']) or type(record.get('occluded')) is not bool or
                        type(record.get('truncated')) is not bool or
                        ((record['occluded'] or record['truncated']) and all(record['edge_valid']))):
                    raise ValueError('Unverified/malformed extent or missing visibility edge mask')
                if record['source']=='postprocess_yolo':
                    proof=record.get('verification',{})
                    if (proof.get('policy')!='unique_geometry_confidence_v1' or
                            not proof.get('cache_sha256') or not proof.get('manifest_sha256') or
                            proof.get('iou',0)<.4 or proof.get('confidence',0)<.6 or
                            not 0<record.get('weight',0)<=.25):
                        raise ValueError('YOLO extent lacks audited weak-supervision provenance')
                count+=1
        if not count:raise ValueError('Extent file has no verified visible edges')

    def loss(self,pred,gt,token,projector):
        total=pred.sum()*0;count=0
        if token is None:return total
        records=self.frames.get(str(token),[])
        if not records:return total
        uv,valid,cues=projector(extent_points(pred[None]))
        # No gradient through visibility decisions; lens-invalid boxes are
        # excluded. Cropping cannot become spurious length supervision.
        angle_valid=(cues[...,3]<=1).all(2)[0]
        lo=uv.amin(2)[0];hi=uv.amax(2)[0]
        extent=torch.cat((lo,hi),-1)
        for record in records:
            matching=torch.isclose(gt,gt.new_tensor(record['gt']),atol=1e-4,rtol=0).all(-1)
            for i in torch.nonzero(matching,as_tuple=False).flatten():
                cam=record['camera']
                if not bool(angle_valid[i,cam]):continue
                edges=torch.tensor(record['edge_valid'],device=pred.device,dtype=torch.bool)
                target=pred.new_tensor(record['bbox'])
                # Relative image extent keeps far/small objects from receiving
                # vanishing gradients merely because the canvas is large.
                scale=(target[2:]-target[:2]).clamp_min(4.).repeat(2)
                error=F.smooth_l1_loss((extent[i,cam]-target)/scale,torch.zeros_like(target),beta=.1,reduction='none')
                total=total+error[edges].sum()*record.get('weight',1.);count+=int(edges.sum())
        return total/max(count,1)
