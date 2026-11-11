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


def heading_targets(yaw,cfg):
    """Nearest coarse direction and a local unit residual (within half a bin)."""
    if cfg.heading_encoding!='binned_residual' or not cfg.heading_bins:
        raise ValueError('heading_targets is only for historical binned-residual headings')
    step=2*math.pi/cfg.heading_bins
    bins=torch.remainder(torch.floor(torch.remainder(yaw+step/2,2*math.pi)/step).long(),cfg.heading_bins)
    delta=yaw-bins.to(yaw.dtype)*step
    return bins,torch.stack((delta.sin(),delta.cos()),dim=-1)


def facing_direction(logits,cfg):
    """Selected facing mode, not the mean of potentially opposite directions.

    Circular-bin probabilities can retain front/back ambiguity without averaging
    their two unit vectors to zero. All bin decoding stays in host postprocess.
    """
    if cfg.facing_loss!='circular_bins':return logits.float()
    angle=logits.argmax(dim=-1).to(torch.float32)*(2*math.pi/cfg.heading_bins)
    return torch.stack((angle.sin(),angle.cos()),dim=-1)


def decode_heading(vectors,bin_logits,cfg):
    """Decode radians and a separate bin confidence; never boost object scores."""
    angle=torch.atan2(vectors[...,0],vectors[...,1])
    confidence=torch.ones_like(angle)
    if cfg.heading_bins and cfg.heading_encoding=='axis_direction':
        if bin_logits is None:raise ValueError('Axis heading requires facing vectors')
        angle=angle*.5
        unit=torch.stack((angle.sin(),angle.cos()),dim=-1)
        # Two continuous facing features form a binary front/back classifier
        # along the predicted axis. No discontinuous global quadrant targets.
        facing=(facing_direction(bin_logits,cfg)*unit).sum(dim=-1)
        angle=angle+(facing<0).to(angle.dtype)*math.pi
        if cfg.facing_loss=='circular_bins':
            centers=torch.arange(cfg.heading_bins,device=unit.device,dtype=torch.float32)*(2*math.pi/cfg.heading_bins)
            directions=torch.stack((centers.sin(),centers.cos()),dim=-1)
            same=(unit[...,None,:]*directions).sum(-1)>=0
            hemisphere=(bin_logits.float().softmax(-1)*same).sum(-1)
            if getattr(cfg,'architecture_version',0)>=5:
                # Bayes decision for the supervised binary hemisphere event.
                facing=2*hemisphere-1
                angle=.5*torch.atan2(vectors[...,0],vectors[...,1])+(facing<0).to(angle.dtype)*math.pi
            # Confidence must describe the hemisphere actually selected by the
            # strongest mode, not whichever hemisphere has more total mass.
            confidence=torch.where(facing>=0,hemisphere,1-hemisphere)
        else:confidence=(2*facing.abs()).sigmoid()
        return torch.atan2(angle.sin(),angle.cos()),confidence
    if cfg.heading_bins:
        if bin_logits is None:raise ValueError('Binned-heading config requires heading logits')
        bins=bin_logits.argmax(dim=-1)
        angle=angle+bins.to(angle.dtype)*(2*math.pi/cfg.heading_bins)
        confidence=bin_logits.float().softmax(dim=-1).max(dim=-1).values
    return torch.atan2(angle.sin(),angle.cos()),confidence


def facing_vector_loss(prediction,target,cfg):
    """Per-object loss; unit regression also constrains the sideways component.

    Contrast alone has an unbounded orthogonal nullspace: a correct projection
    onto GT can still flip the decoder when its predicted axis has a small error.
    Raw (not normalized) regression retains gradients at a 180-degree reversal.
    """
    if cfg.facing_loss=='circular_bins':
        # Adjacent-bin interpolation is continuous through every bin boundary,
        # including -pi/pi and 0/2pi. CE models a distribution instead of forcing
        # a mean vector that cancels when front/rear evidence is ambiguous.
        yaw=torch.atan2(target[...,0],target[...,1])
        location=torch.remainder(yaw,2*math.pi)*(cfg.heading_bins/(2*math.pi))
        lower=location.floor().long()%cfg.heading_bins
        fraction=location-location.floor()
        logp=prediction.log_softmax(-1)
        a=logp.gather(-1,lower[...,None])[...,0]
        b=logp.gather(-1,((lower+1)%cfg.heading_bins)[...,None])[...,0]
        return -(1-fraction)*a-fraction*b
    if cfg.facing_loss == 'unit_vector':
        return F.smooth_l1_loss(prediction,target,reduction='none').sum(dim=-1)
    projection=2*(prediction*target).sum(dim=-1)
    return F.binary_cross_entropy_with_logits(projection,torch.ones_like(projection),reduction='none')


def decode_sizes(log_sizes,class_ids,cfg):
    if cfg.size_encoding=='class_relative':
        log_sizes=log_sizes+log_sizes.new_tensor(cfg.class_size_priors)[class_ids].log()
    elif cfg.size_encoding=='shared_relative':
        log_sizes=log_sizes+log_sizes.new_tensor(cfg.size_reference).log()
    return log_sizes.clamp(-3.,5.).exp()


def loss_denominator(mask,cfg):
    if cfg.loss_normalization=='fixed_train_mean':
        return mask.new_tensor(mask.shape[0]*cfg.mean_training_objects,dtype=torch.float32)
    return mask.sum().clamp(min=1)


def objectness_logits(output,cfg):
    index=4 if cfg.heading_bins else 3
    expected=(output[0].shape[0],1,*output[0].shape[2:])
    if len(output)!=index+1 or tuple(output[index].shape)!=expected:
        raise ValueError('objectness_softmax requires a separate one-channel objectness output')
    return output[index].float()


def heavy_vehicle_classification(logits,class_ids,weights,count):
    """Hard-negative vehicle ranking at GT centers, including light-car negatives.

    Heavy GT competes with car/van AND the other heavy class; car/van GT
    counters false truck/bus scores. No class-macro average or inference boost.
    """
    vehicle_ids=class_ids.new_tensor((0,3,4,5))
    heavy=(class_ids==3)|(class_ids==5)
    light=(class_ids==0)|(class_ids==4)
    rivals=logits[:,vehicle_ids]
    allowed=(vehicle_ids[None]!=class_ids[:,None]) & (heavy[:,None]|((vehicle_ids==3)|(vehicle_ids==5))[None])
    hardest=rivals.masked_fill(~allowed,float('-inf')).max(dim=1).values
    correct=logits.gather(1,class_ids[:,None])[:,0]
    return (F.softplus(.5+hardest-correct)*weights*(heavy|light)).sum()/count


def _object_loss_weights(class_ids,cfg,reference):
    weights=reference.new_tensor(cfg.class_regression_weights)[class_ids]
    if cfg.class_balanced_losses:
        counts=torch.bincount(class_ids,minlength=cfg.num_classes).to(reference.dtype)
        present=(counts>0).sum().clamp(min=1)
        weights=weights*class_ids.numel()/(present*counts[class_ids].clamp(min=1))
    return weights


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
    Explicit TRAIN-only ignored boxes use class_id=-1 and require
    return_class_ids=True. They append a fifth valid-background mask, but never
    create a foreground/regression target or remove an overlapping known center.
    Avoid redundant targets when multiple objects share a cell by retaining
    the closer center to the cell midpoint.
    """
    b = len(batch_boxes)
    h,w = cfg.bev_height,cfg.bev_width
    # Construct sparse labels on CPU, then transfer once. Per-cell CUDA writes
    # and float(tensor) synchronizations made large-grid target creation costly.
    heat = torch.zeros((b,cfg.num_classes,h,w),dtype=torch.float32)
    valid_background=torch.ones((b,1,h,w),dtype=torch.bool)
    has_ignored=False
    regression = torch.zeros((b,8,h,w),dtype=torch.float32)
    mask = torch.zeros((b,1,h,w),dtype=torch.bool)
    owners = torch.full((b,1,h,w),-1,dtype=torch.long) if return_class_ids else None
    best = torch.full((b,1,h,w),float('inf'))
    log_priors=torch.tensor(cfg.class_size_priors).log() if cfg.size_encoding=='class_relative' else torch.zeros(cfg.num_classes,3)
    if cfg.size_encoding=='shared_relative':
        log_priors=torch.tensor(cfg.size_reference).log().expand(cfg.num_classes,-1)
    for batch, boxes in enumerate(batch_boxes):
        if isinstance(boxes,torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        for box in boxes:
            if len(box) != 8:
                raise ValueError("Each box must be [x,y,z,length,width,height,yaw,class_id] (8 values); velocities are not used")
            x,y,z,l,width,height,yaw,cls = (float(v) for v in box)
            if cls==-1:
                if not return_class_ids:raise ValueError('Ignored TRAIN boxes require return_class_ids=True')
                if not all(math.isfinite(v) for v in box) or min(l,width,height)<=0:
                    raise ValueError('Invalid ignored TRAIN box')
                has_ignored=True
                # Conservative rotated-footprint envelope plus one cell; no
                # positive/regression target and no false background inside it.
                dx=(abs(math.cos(yaw))*l+abs(math.sin(yaw))*width)/2+cfg.bev_resolution
                dy=(abs(math.sin(yaw))*l+abs(math.cos(yaw))*width)/2+cfg.bev_resolution
                r0,c0=metric_to_cell(x+dx,y+dy,cfg);r1,c1=metric_to_cell(x-dx,y-dy,cfg)
                r0=max(0,math.floor(r0));c0=max(0,math.floor(c0))
                r1=min(h,math.ceil(r1)+1);c1=min(w,math.ceil(c1)+1)
                if r1>r0 and c1>c0:valid_background[batch,:,r0:r1,c0:c1]=False
                continue
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
                r-i,c-j,z,math.log(l)-float(log_priors[k,0]),math.log(width)-float(log_priors[k,1]),math.log(height)-float(log_priors[k,2]),
                math.sin(yaw),math.cos(yaw)],dtype=torch.float32)
    targets=(heat.to(device),regression.to(device),mask.to(device))
    if return_class_ids:targets=targets+(owners.to(device),)
    return targets+(valid_background.to(device),) if has_ignored else targets


def detection_loss(output, targets, heat_weight=1., regression_weight=2., cfg=None):
    cfg = ModelConfig() if cfg is None else cfg
    heat_logits, reg_pred = (v.float() for v in output[:2])
    heat_target, reg_target, reg_mask = targets[:3]
    if heat_logits.shape != heat_target.shape or reg_pred.shape != reg_target.shape:
        raise ValueError('Mismatch between predictions and target map shapes')
    # CornerNet/CenterNet Gaussian focal loss in logit space for stability.
    foreground_pos=(heat_target>=1-1e-5)
    decoupled=cfg.classification_mode=='objectness_softmax'
    focal_logits=objectness_logits(output,cfg) if decoupled else heat_logits
    focal_target=heat_target.amax(dim=1,keepdim=True) if decoupled else heat_target
    p=torch.sigmoid(focal_logits).clamp(1e-5,1-1e-5)
    is_pos=(focal_target >= 1-1e-5)
    is_neg=~is_pos
    neg_w=(1-focal_target).pow(4)
    class_weights=p.new_tensor(cfg.class_positive_weights)[None,:,None,None]
    if decoupled:
        classes=targets[3].clamp(min=0) if len(targets)>=4 else heat_target.argmax(dim=1,keepdim=True)
        class_weights=p.new_tensor(cfg.class_positive_weights)[classes]
    positive=-(torch.log(p) * (1-p).pow(2)*is_pos*class_weights)
    background=targets[4] if len(targets)==5 else 1.
    neg_loss=-(torch.log(1-p)*p.pow(2)*neg_w*is_neg*background).sum()
    if cfg.class_balanced_losses:
        # Historical v3 only; new runs use global normalization below. Mixing
        # macro positives with globally normalized negatives distorts priors.
        per_class_count=is_pos.sum(dim=(0,2,3))
        pos_loss=(positive.sum(dim=(0,2,3))/per_class_count.clamp(min=1)).sum()/(per_class_count>0).sum().clamp(min=1)
    heat_count=loss_denominator(reg_mask,cfg) if cfg.loss_normalization=='fixed_train_mean' else is_pos.sum().clamp(min=1)
    if not cfg.class_balanced_losses:pos_loss=positive.sum()/heat_count
    heat_loss=pos_loss+neg_loss/heat_count
    # Select centers before normalizing yaw; no boolean branch for empty DDP ranks.
    pred=reg_pred.permute(0,2,3,1)[reg_mask[:,0]]
    truth=reg_target.permute(0,2,3,1)[reg_mask[:,0]]
    count=loss_denominator(reg_mask,cfg)
    if len(targets)>=4:
        class_ids=targets[3][:,0][reg_mask[:,0]]
    else:
        center_classes=foreground_pos.permute(0,2,3,1)[reg_mask[:,0]]
        class_aware=(cfg.class_balanced_losses or cfg.center_classification_weight>0 or cfg.heavy_classification_weight>0 or
                     tuple(cfg.class_regression_weights)!=(1.0,)*cfg.num_classes)
        if class_aware and bool((center_classes.sum(dim=1)!=1).any()):
            raise ValueError('Ambiguous regression class; use make_targets(..., return_class_ids=True)')
        class_ids=center_classes.long().argmax(dim=1)
    object_weights=_object_loss_weights(class_ids,cfg,pred)
    norm=torch.linalg.vector_norm(pred[:,6:8],dim=1,keepdim=True)
    yaw_prediction=pred[:,6:8]/norm.clamp(min=.1) if cfg.normalize_yaw_loss else pred[:,6:8]
    regression_prediction=torch.cat((pred[:,:6],yaw_prediction),dim=1)
    heading_loss=reg_pred.sum()*0.
    if cfg.heading_bins:
        if len(output)<4:raise ValueError('Binned-heading loss requires heading logits')
        heading_logits=output[3].float().permute(0,2,3,1)[reg_mask[:,0]]
        yaw=torch.atan2(truth[:,6],truth[:,7])
        if cfg.heading_encoding=='axis_direction':
            heading_loss=(facing_vector_loss(heading_logits,truth[:,6:8],cfg)*object_weights).sum()/count
            truth=torch.cat((truth[:,:6],torch.stack(((2*yaw).sin(),(2*yaw).cos()),dim=1)),dim=1)
        else:
            bin_target,residual_target=heading_targets(yaw,cfg)
            truth=torch.cat((truth[:,:6],residual_target),dim=1)
            heading_loss=(F.cross_entropy(heading_logits,bin_target,reduction='none')*object_weights).sum()/count
    weighted=F.smooth_l1_loss(regression_prediction,truth,reduction='none')*pred.new_tensor(cfg.regression_weights)
    # Use one unweighted object normalizer (fixed TRAIN mean in recipe 4.1):
    # a Bus retains 4x gradient, rather than cancelling its weight below.
    yaw_penalty=((norm[:,0]-1).square()*object_weights).sum()/count
    reg_loss=(weighted.sum(dim=1)*object_weights).sum()/count+cfg.yaw_norm_weight*yaw_penalty
    quality_loss=reg_pred.sum()*0.
    if len(output)>=3:
        # Detached localization quality: meter-space center and relative size error.
        xy_error=torch.linalg.vector_norm((pred[:,:2]-truth[:,:2])*cfg.bev_resolution,dim=1)
        size_error=(pred[:,3:6]-truth[:,3:6]).abs().mean(dim=1)
        quality_target=torch.exp(-xy_error-size_error).detach()
        quality=output[2].float()[:,0][reg_mask[:,0]]
        quality_loss=(F.binary_cross_entropy_with_logits(quality,quality_target,reduction='none')*object_weights).sum()/count
    # Explicit foreground class competition without adding inference softmax.
    center_logits=heat_logits.permute(0,2,3,1)[reg_mask[:,0]]
    unique_center=(foreground_pos.permute(0,2,3,1)[reg_mask[:,0]].sum(dim=1)==1)
    foreground_weights=center_logits.new_tensor(cfg.class_positive_weights)[class_ids] if decoupled else object_weights
    center_loss=(F.cross_entropy(center_logits,class_ids,reduction='none',
                               label_smoothing=cfg.foreground_label_smoothing)*foreground_weights*unique_center).sum()/count
    heavy_loss=heavy_vehicle_classification(center_logits,class_ids,object_weights*unique_center,count)
    classification=heat_loss+cfg.center_classification_weight*center_loss+cfg.heavy_classification_weight*heavy_loss
    return heat_weight*classification + regression_weight*reg_loss+cfg.quality_loss_weight*quality_loss+cfg.heading_loss_weight*heading_loss, {
        'heatmap': classification.detach(), 'regression': reg_loss.detach(),
        'quality':quality_loss.detach(),'yaw_norm':yaw_penalty.detach(),
        'heading_bin':heading_loss.detach(),'center_classification':center_loss.detach(),
        'heavy_classification':heavy_loss.detach(),
        'objectness':heat_loss.detach(),
        'heading_axis':(weighted[:,6:8].sum(dim=1)*object_weights).sum().detach()/count,
        'size_regression':(weighted[:,3:6].sum(dim=1)*object_weights).sum().detach()/count,
    }


def detection_scores(output,cfg: ModelConfig):
    """Shared FP32 confidence/peak rule for decode, analysis and host deployment.

    Recipe 4.2: p_object * softmax(class) * sqrt(p_quality).
    Old checkpoints retain sigmoid(class). Peaks follow score fusion; training
    difficulty weights never multiply inference confidence.
    """
    heat_logits,reg=output[:2]
    decoupled=cfg.classification_mode=='objectness_softmax'
    if decoupled:
        finite=torch.isfinite(heat_logits)
        logits=torch.where(finite,heat_logits.float(),float('-inf'))
        heat=torch.nan_to_num(logits.softmax(dim=1))
    else:heat=heat_logits.float().sigmoid()
    score=heat
    valid=torch.isfinite(heat_logits)&torch.isfinite(reg).all(dim=1,keepdim=True)
    if decoupled:
        objects=objectness_logits(output,cfg)
        valid=valid&torch.isfinite(objects)
        score=score*objects.sigmoid()
    if len(output)>=3 and cfg.quality_score_power>0:
        quality_logits=output[2].float()
        score=score*quality_logits.sigmoid().pow(cfg.quality_score_power)
        valid=valid&torch.isfinite(quality_logits)
    if cfg.heading_bins:
        if len(output)<4:raise ValueError('Binned-heading config requires heading logits')
        valid=valid&torch.isfinite(output[3]).all(dim=1,keepdim=True)
    score=torch.where(valid,score,torch.zeros_like(score))
    if cfg.single_label:
        winner=score.argmax(dim=1,keepdim=True)
        selected=torch.zeros_like(score,dtype=torch.bool).scatter_(1,winner,True)
        score=score*selected
    peak_map=score if cfg.score_peak_mode=='fused' else torch.where(valid,heat,torch.zeros_like(heat))
    return score*(peak_map==F.max_pool2d(peak_map,3,1,1))


@torch.no_grad()
def decode_detections(output, cfg: ModelConfig, score_threshold=0.30, topk=300,
                      apply_nms=True,per_class_topk=None):
    """Return [N,9] x,y,z,l,w,h,yaw,class_id,score per frame.

    Peak scores use the checkpoint's fusion rule, then class-aware rotated NMS.
    NMS stays on the host (Arm/C7x), outside the TIDL learned graphs.
    """
    if not 0 <= score_threshold <= 1 or topk<1:
        raise ValueError('score_threshold must be in [0,1] and topk must be positive')
    heat,reg=output[0],output[1].float()
    b,k,h,w=heat.shape
    score=detection_scores(output,cfg)
    if per_class_topk is not None:
        if not isinstance(per_class_topk,int) or per_class_topk<1:raise ValueError('per_class_topk must be positive')
        # Select on FINAL scores. Masking conditional class logits beforehand
        # renormalizes softmax and spuriously boosts the surviving classes.
        flat=score.flatten(2)
        ids=flat.topk(min(per_class_topk,h*w),dim=2).indices
        allowed=torch.zeros_like(flat,dtype=torch.bool).scatter_(2,ids,True)
        score=score.masked_fill(~allowed.reshape_as(score),0.)
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
        sizes=decode_sizes(pred[:,3:6],class_ids,cfg)
        bins=output[3][bi,:,rows,cols].T.float() if cfg.heading_bins else None
        yaw,_=decode_heading(pred[:,6:8],bins,cfg)
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
