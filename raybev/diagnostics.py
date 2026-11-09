"""Oracle-center diagnostics: isolate representation errors from proposal/NMS errors.

These are NOT detection AP/recall. Sufficient statistics can be SUM-reduced
across DDP ranks, including an empty validation shard, without averaging ratios.
"""
import math
import torch
from .config import CLASSES
from .detection import decode_heading,decode_sizes


@torch.no_grad()
def center_statistics(output,targets,cfg):
    mask=targets[2][:,0]
    ids=targets[3][:,0][mask].long()
    pred=output[1].float().permute(0,2,3,1)[mask]
    truth=targets[1].float().permute(0,2,3,1)[mask]
    logits=output[0].float().permute(0,2,3,1)[mask]
    p=logits.softmax(-1) if cfg.classification_mode=='objectness_softmax' else logits.sigmoid()
    obj=output[4].float()[:,0][mask].sigmoid() if cfg.classification_mode=='objectness_softmax' else torch.ones_like(ids,dtype=torch.float32)
    quality=output[2].float()[:,0][mask].sigmoid() if len(output)>2 else torch.ones_like(obj)
    winner=p.argmax(-1)
    true_p=p.gather(1,ids[:,None]).squeeze(1)
    facing=output[3].float().permute(0,2,3,1)[mask] if cfg.heading_bins else None
    yaw,_=decode_heading(pred[:,6:8],facing,cfg)
    gt_yaw=torch.atan2(truth[:,6],truth[:,7])
    error=torch.atan2((yaw-gt_yaw).sin(),(yaw-gt_yaw).cos()).abs()
    flips=error>math.pi/2
    gt_size=decode_sizes(truth[:,3:6],ids,cfg)
    ratios=decode_sizes(pred[:,3:6],ids,cfg)/gt_size
    actual_l=decode_sizes(pred[:,3:6],winner,cfg)[:,0]/gt_size[:,0]
    perp=norm=wrong=unstable=torch.zeros_like(obj)
    if cfg.heading_bins and cfg.heading_encoding=='axis_direction':
        projection=(facing*truth[:,6:8]).sum(-1)
        perp=(facing*torch.stack((truth[:,7],-truth[:,6]),-1)).sum(-1).abs()
        norm=facing.norm(dim=-1);wrong=(projection<0).float()
        unstable=((projection>0)&flips).float()
    values=torch.stack((torch.ones_like(obj),(winner==ids).float(),obj,(obj>=.1).float(),true_p,
                        obj*true_p*quality.pow(cfg.quality_score_power),error,torch.minimum(error,math.pi-error),
                        flips.float(),perp,unstable,ratios[:,0],ratios[:,1],ratios[:,2],actual_l,
                        (pred[:,:2]-truth[:,:2]).norm(dim=-1)*cfg.bev_resolution,norm,wrong),dim=-1)
    result=pred.new_zeros((len(CLASSES),18))
    result.index_add_(0,ids,values)
    return result


def summarize_centers(statistics):
    keys=('class_accuracy','mean_objectness','objectness_ge_0_1_rate','mean_true_class_probability',
          'mean_true_class_fused_score','AOE_deg','axis_AOE_deg','flip_rate','mean_facing_perpendicular',
          'wrong_decode_despite_positive_GT_projection_rate','mean_L_ratio_GT_class','mean_W_ratio_GT_class',
          'mean_H_ratio_GT_class','mean_L_ratio_predicted_class','mean_center_error_m','mean_facing_norm',
          'negative_GT_facing_projection_rate')
    result={}
    for name,values in zip(CLASSES,statistics.detach().cpu().double().tolist()):
        count=int(values[0]);row={'count':count}
        means=[v/count for v in values[1:]] if count else [None]*len(keys)
        if count:
            means[5]=math.degrees(means[5]);means[6]=math.degrees(means[6])
        row.update(zip(keys,means));result[name]=row
    return result
