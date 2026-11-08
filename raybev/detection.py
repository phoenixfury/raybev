"""Center heatmap supervision and deterministic 3D box decode for RayBEV-Lite.

Annotation box convention: (x, y, z, length, width, height, yaw, class_id)
All values except class_id are in BASELINK; x forward, y left, z up.
Default class IDs: 0 car, 1 pedestrian, 2 cyclist, 3 truck, 4 van, 5 bus.
Yaw rotates an object along +x about +z. z is the box center (not bottom).
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from .config import ModelConfig, CLASSES


def metric_to_cell(x, y, cfg: ModelConfig):
    """Top of BEV is +X (front); left of grid is +Y (left)."""
    return ((cfg.bev_x_max-x)/cfg.bev_resolution-.5,
            (cfg.bev_y_max-y)/cfg.bev_resolution-.5)


def in_bev_range(boxes, cfg):
    """Same center bounds for training, validation, and decoded predictions."""
    return ((boxes[:,0] >= cfg.bev_x_min) & (boxes[:,0] < cfg.bev_x_max) &
            (boxes[:,1] >= cfg.bev_y_min) & (boxes[:,1] < cfg.bev_y_max))


def gaussian_radius(length_cells, width_cells, overlap=0.1):
    """CenterNet minimum-overlap radius using the full object footprint."""
    h,w = length_cells,width_cells
    b1 = h+w
    r1 = (b1+math.sqrt(max(0,b1*b1-4*w*h*(1-overlap)/(1+overlap))))/2
    b2 = 2*(h+w)
    r2 = (b2+math.sqrt(max(0,b2*b2-16*(1-overlap)*w*h)))/2
    b3 = -2*overlap*(h+w)
    r3 = (b3+math.sqrt(max(0,b3*b3-16*overlap*(overlap-1)*w*h)))/2
    return max(0,int(min(r1,r2,r3)))


def make_targets(batch_boxes, cfg: ModelConfig, device='cpu',return_class_ids=False):
    """Batch of tensors or sequences of boxes, variable N each; no autocasting.

    Returns heatmap[B,K,H,W], regression[B,8,H,W], mask[B,1,H,W].
    With return_class_ids=True, appends owner classes[B,1,H,W] (-1 outside
    centers). Trainers use this to weight the exact regression owner, even
    when different classes share one cell. Three-target callers remain valid
    when the class at each regression center is unambiguous.
    Avoid redundant targets when multiple objects share a cell by retaining
    the closer center to the cell midpoint.
    """
    b = len(batch_boxes)
    h,w = cfg.bev_height,cfg.bev_width
    # Construct sparse labels on CPU, then transfer once. Per-cell CUDA writes
    # and float(tensor) synchronizations made large-grid target creation costly.
    heat = torch.zeros((b,cfg.num_classes,h,w),dtype=torch.float32)
    regression = torch.zeros((b,8,h,w),dtype=torch.float32)
    mask = torch.zeros((b,1,h,w),dtype=torch.bool)
    owners = torch.full((b,1,h,w),-1,dtype=torch.long) if return_class_ids else None
    best = torch.full((b,1,h,w),float('inf'))
    for batch, boxes in enumerate(batch_boxes):
        if isinstance(boxes,torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        for box in boxes:
            if len(box) != 8:
                raise ValueError("Each box must be [x,y,z,length,width,height,yaw,class_id] (8 values); velocities are not used")
            x,y,z,l,width,height,yaw,cls = (float(v) for v in box)
            if not all(math.isfinite(v) for v in box):
                continue
            k=int(cls)
            if (not 0 <= k < cfg.num_classes or min(l,width,height) <= 0 or
                    not cfg.bev_x_min <= x < cfg.bev_x_max or
                    not cfg.bev_y_min <= y < cfg.bev_y_max):
                continue
            r,c = metric_to_cell(x,y,cfg)
            i,j = min(h-1,math.floor(r+.5)),min(w-1,math.floor(c+.5))
            radius = min(max(h,w),gaussian_radius(l/cfg.bev_resolution,width/cfg.bev_resolution,cfg.gaussian_overlap))
            sigma = (2*radius+1)/6
            y0,y1 = max(0,i-radius),min(h,i+radius+1)
            x0,x1 = max(0,j-radius),min(w,j+radius+1)
            yy = torch.arange(y0,y1,dtype=torch.float32)-i
            xx = torch.arange(x0,x1,dtype=torch.float32)-j
            patch = torch.exp(-.5*(yy[:,None]**2+xx[None,:]**2)/(sigma*sigma))
            torch.maximum(heat[batch,k,y0:y1,x0:x1],patch,out=heat[batch,k,y0:y1,x0:x1])
            distance=(r-i)**2+(c-j)**2
            if distance >= float(best[batch,0,i,j]):
                continue
            best[batch,0,i,j] = distance
            mask[batch,0,i,j] = True
            if owners is not None:owners[batch,0,i,j]=k
            regression[batch,:,i,j] = torch.tensor([
                r-i,c-j,z,math.log(l),math.log(width),math.log(height),
                math.sin(yaw),math.cos(yaw)],dtype=torch.float32)
    targets=(heat.to(device),regression.to(device),mask.to(device))
    return targets+(owners.to(device),) if return_class_ids else targets


def detection_loss(output, targets, heat_weight=1., regression_weight=2., cfg=None):
    cfg = ModelConfig() if cfg is None else cfg
    heat_logits, reg_pred = (v.float() for v in output[:2])
    heat_target, reg_target, reg_mask = targets[:3]
    if heat_logits.shape != heat_target.shape or reg_pred.shape != reg_target.shape:
        raise ValueError('Mismatch between predictions and target map shapes')
    # CornerNet/CenterNet Gaussian focal loss in logit space for stability.
    p=torch.sigmoid(heat_logits).clamp(1e-5,1-1e-5)
    is_pos=(heat_target >= 1-1e-5)
    is_neg=~is_pos
    neg_w=(1-heat_target).pow(4)
    class_weights=p.new_tensor(cfg.class_positive_weights)[None,:,None,None]
    pos_loss=-(torch.log(p) * (1-p).pow(2)*is_pos*class_weights).sum()
    neg_loss=-(torch.log(1-p)*p.pow(2)*neg_w*is_neg).sum()
    heat_loss=(pos_loss+neg_loss)/is_pos.sum().clamp(min=1)
    # Select centers before normalizing yaw; no boolean branch for empty DDP ranks.
    pred=reg_pred.permute(0,2,3,1)[reg_mask[:,0]]
    truth=reg_target.permute(0,2,3,1)[reg_mask[:,0]]
    count=reg_mask.sum().clamp(min=1)
    if len(targets)==4:
        class_ids=targets[3][:,0][reg_mask[:,0]]
    else:
        center_classes=is_pos.permute(0,2,3,1)[reg_mask[:,0]]
        if cfg.class_regression_weights!=(1.0,)*cfg.num_classes and bool((center_classes.sum(dim=1)!=1).any()):
            raise ValueError('Ambiguous regression class; use make_targets(..., return_class_ids=True)')
        class_ids=center_classes.long().argmax(dim=1)
    object_weights=pred.new_tensor(cfg.class_regression_weights)[class_ids]
    norm=torch.linalg.vector_norm(pred[:,6:8],dim=1,keepdim=True)
    unit_yaw=pred[:,6:8]/norm.clamp(min=.1)
    normalized=torch.cat((pred[:,:6],unit_yaw),dim=1)
    weighted=F.smooth_l1_loss(normalized,truth,reduction='none')*pred.new_tensor(cfg.regression_weights)
    # Divide by the unweighted object count: a single Bus retains its 4x
    # gradient instead of its weight cancelling in a weighted denominator.
    yaw_penalty=((norm[:,0]-1).square()*object_weights).sum()/count
    reg_loss=(weighted.sum(dim=1)*object_weights).sum()/count+cfg.yaw_norm_weight*yaw_penalty
    quality_loss=reg_pred.sum()*0.
    if len(output)==3:
        # Detached localization quality: meter-space center and relative size error.
        xy_error=torch.linalg.vector_norm((pred[:,:2]-truth[:,:2])*cfg.bev_resolution,dim=1)
        size_error=(pred[:,3:6]-truth[:,3:6]).abs().mean(dim=1)
        quality_target=torch.exp(-xy_error-size_error).detach()
        quality=output[2].float()[:,0][reg_mask[:,0]]
        quality_loss=(F.binary_cross_entropy_with_logits(quality,quality_target,reduction='none')*object_weights).sum()/count
    return heat_weight*heat_loss + regression_weight*reg_loss+cfg.quality_loss_weight*quality_loss, {
        'heatmap': heat_loss.detach(), 'regression': reg_loss.detach(),
        'quality':quality_loss.detach(),'yaw_norm':yaw_penalty.detach(),
    }


def detection_scores(output,cfg: ModelConfig):
    """Shared FP32 confidence/peak rule for decode, analysis and host deployment.

    New runs: p_class * sqrt(p_quality), with peaks selected AFTER fusion.
    Scores never exceed p_class. Difficulty weights affect losses, not scores.
    """
    heat_logits,reg=output[:2]
    heat=heat_logits.float().sigmoid()
    score=heat
    valid=torch.isfinite(heat_logits)&torch.isfinite(reg).all(dim=1,keepdim=True)
    if len(output)==3 and cfg.quality_score_power>0:
        quality_logits=output[2].float()
        score=score*quality_logits.sigmoid().pow(cfg.quality_score_power)
        valid=valid&torch.isfinite(quality_logits)
    score=torch.where(valid,score,torch.zeros_like(score))
    peak_map=score if cfg.score_peak_mode=='fused' else torch.where(valid,heat,torch.zeros_like(heat))
    return score*(peak_map==F.max_pool2d(peak_map,3,1,1))


@torch.no_grad()
def decode_detections(output, cfg: ModelConfig, score_threshold=0.30, topk=300,
                      apply_nms=True):
    """Return [N,9] x,y,z,l,w,h,yaw,class_id,score per frame.

    Peak scores use the checkpoint's fusion rule, then class-aware rotated NMS.
    NMS stays on the host (Arm/C7x), outside the TIDL learned graphs.
    """
    if not 0 <= score_threshold <= 1 or topk<1:
        raise ValueError('score_threshold must be in [0,1] and topk must be positive')
    heat,reg=output[0],output[1].float()
    b,k,h,w=heat.shape
    score=detection_scores(output,cfg)
    results=[]
    for bi in range(b):
        scores,indices=score[bi].flatten().topk(min(topk,k*h*w))
        selected=(scores >= score_threshold)&(scores>0)
        scores,indices=scores[selected],indices[selected]
        class_ids=indices//(h*w)
        cells=indices%(h*w)
        rows=cells//w
        cols=cells%w
        pred=reg[bi,:,rows,cols].T
        x=cfg.bev_x_max - (rows.to(pred.dtype)+.5+pred[:,0])*cfg.bev_resolution
        y=cfg.bev_y_max - (cols.to(pred.dtype)+.5+pred[:,1])*cfg.bev_resolution
        z=pred[:,2]
        sizes=pred[:,3:6].clamp(-3.,5.).exp()
        yaw=torch.atan2(pred[:,6],pred[:,7])
        detection=torch.cat([x[:,None],y[:,None],z[:,None],sizes,
                             yaw[:,None],class_ids.to(pred.dtype)[:,None],
                             scores[:,None]],dim=1)
        finite=torch.isfinite(detection).all(dim=1)
        detection=detection[finite & in_bev_range(detection,cfg)]
        if apply_nms and len(detection):
            keep=rotated_bev_nms(detection,cfg.nms_iou_threshold,cfg.max_detections)
            detection=detection[keep]
        results.append(detection[:cfg.max_detections])
    return results


def _corners(box):
    l,w=box[3:5]/2
    xy=np.array([[l,w],[-l,w],[-l,-w],[l,-w]])
    c,s=math.cos(box[6]),math.sin(box[6])
    return xy @ np.array([[c,s],[-s,c]])+box[:2]


def _cross(a,b):
    return a[0]*b[1]-a[1]*b[0]


def _intersection_area(subject,clip):
    polygon=list(subject)
    for a,b in zip(clip,np.roll(clip,-1,axis=0)):
        source,polygon=polygon,[]
        if not source: break
        previous=source[-1]
        for current in source:
            inside=_cross(b-a,current-a)>=-1e-9
            before=_cross(b-a,previous-a)>=-1e-9
            if inside != before:
                delta=current-previous
                denominator=_cross(delta,b-a)
                if abs(denominator)>1e-12:
                    t=_cross(a-previous,b-a)/denominator
                    polygon.append(previous+t*delta)
            if inside: polygon.append(current)
            previous=current
    if len(polygon)<3:return 0.
    p=np.asarray(polygon)
    return .5*abs(np.dot(p[:,0],np.roll(p[:,1],-1))-np.dot(p[:,1],np.roll(p[:,0],-1)))


def rotated_bev_nms(detections,threshold=.25,max_detections=150):
    """Portable polygon-IoU reference with vectorized axis-aligned rejection."""
    if not len(detections):
        return torch.empty(0,device=detections.device,dtype=torch.long)
    boxes=detections.detach().float().cpu().numpy()
    order=np.argsort(-boxes[:,8],kind='stable')
    corners=np.asarray([_corners(b) for b in boxes])
    lo,hi=corners.min(axis=1),corners.max(axis=1)
    areas=boxes[:,3]*boxes[:,4]
    keep=[]
    while len(order) and len(keep)<max_detections:
        i=int(order[0]);keep.append(i)
        remaining=order[1:]
        if not len(remaining):break
        overlap=(np.minimum(hi[i],hi[remaining])>np.maximum(lo[i],lo[remaining])).all(axis=1)
        candidates=np.where(overlap & (boxes[remaining,7]==boxes[i,7]))[0]
        suppress=np.zeros(len(remaining),dtype=bool)
        for pos in candidates:
            j=remaining[pos]
            inter=_intersection_area(corners[i],corners[j])
            suppress[pos]=inter/max(float(areas[i]+areas[j]-inter),1e-9)>threshold
        order=remaining[~suppress]
    return torch.tensor(keep,device=detections.device,dtype=torch.long)
