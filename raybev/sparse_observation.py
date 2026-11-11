"""Compact OpenCV observation conditioning and image-scored ground hypotheses."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def distortion_descriptor(cues,visible,boxes,projector,strides):
    ray=cues[...,:3]
    theta=cues[...,3]*math.radians(projector.cfg.theta_max_degrees)
    # Derivative of OpenCV's theta polynomial, in the local spherical tangent
    # basis. Use the full resized 2x2 intrinsic map (including skew).
    # Squared pixel Jacobians exceed FP16 range even for ordinary focal
    # lengths; no_grad alone does not disable autocast's einsum lowering.
    with torch.no_grad(),torch.autocast(device_type=cues.device.type,enabled=False):
        t=theta.detach();t2=t.square();k=projector.distortion
        radial=1+3*k[:,0]*t2+5*k[:,1]*t2**2+7*k[:,2]*t2**3+9*k[:,3]*t2**4
        distorted=t*(1+k[:,0]*t2+k[:,1]*t2**2+k[:,2]*t2**3+k[:,3]*t2**4)
        tangential=torch.where(t.abs()<1e-4,torch.ones_like(t),distorted/t.sin().clamp_min(1e-6))
        xy=ray[...,:2].detach();rho=xy.norm(dim=-1,keepdim=True)
        direction=torch.where(rho>1e-6,xy/rho.clamp_min(1e-6),xy.new_tensor([1.,0.]))
        x,y=direction.unbind(-1)
        basis=torch.stack((torch.stack((x*radial,-y*tangential),-1),
                           torch.stack((y*radial,x*tangential),-1)),-2)
        intrinsic=projector.intrinsics[:,:2,:2]*projector.resize_scale[:,:,None]
        jac=torch.einsum('nij,bqpnjk->bqpnik',intrinsic,basis)
        a,b=jac[...,0,:].unbind(-1);c,d=jac[...,1,:].unbind(-1)
        trace=a*a+b*b+c*c+d*d
        root=(trace.square()-4*(a*d-b*c).square()).clamp_min(0).sqrt()
        singular=torch.stack((((trace+root)/2).clamp_min(1e-8).sqrt(),
                              ((trace-root)/2).clamp_min(1e-8).sqrt()),-1).clamp_min(1e-4)
        logs=singular.log()
    # Camera-relative range, rather than distance to the rig origin.
    distance=(cues[...,5]*math.log(projector.cfg.max_range)).exp().clamp_min(.5)
    extent=torch.stack((boxes[...,3:5].norm(dim=-1),boxes[...,5]),-1)
    angular=(extent[:,:,None,None,:]/distance[...,None]).clamp_min(1e-5).log()
    level=logs.new_tensor(strides).log()
    compatibility=-((level[None,None,None,None,:,None]-logs[...,None,:]-angular[...,None,:])**2).mean(-1)
    overlap=visible.float().mean(-1,keepdim=True).expand_as(theta)
    tangent=ray[...,:2]*theta[...,None]/ray[...,:2].norm(dim=-1,keepdim=True).clamp_min(1e-6)
    base=torch.cat((logs,(logs[...,0:1]-logs[...,1:2]).abs(),overlap[...,None],tangent),-1)
    descriptor=torch.cat((base[...,None,:].expand(*base.shape[:-1],len(strides),6),compatibility[...,None]),-1)
    return torch.where(visible[...,None,None],descriptor,torch.zeros_like(descriptor))


class GroundProposals(nn.Module):
    """A fixed calibrated XY lattice, scored from all visible cameras at /8.

    Ground height is BASELINK metres. Only centres assume ground support;
    decoder dimensions and heading remain class independent. No GT is used
    in inference. Top-k indices are discrete; dense score/offset supervision
    trains every valid lattice cell independently of that selection.
    """
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        x=torch.linspace(cfg.bev_x_min,cfg.bev_x_max,cfg.proposal_grid+1)
        y=torch.linspace(cfg.bev_y_min,cfg.bev_y_max,cfg.proposal_grid+1)
        xx,yy=torch.meshgrid((x[:-1]+x[1:])/2,(y[:-1]+y[1:])/2,indexing='ij')
        centers=torch.stack((xx.flatten(),yy.flatten(),torch.full_like(xx.flatten(),cfg.ground_z+.9)),-1)
        self.register_buffer('centers',centers)
        self.register_buffer('cell_size',torch.tensor([(cfg.bev_x_max-cfg.bev_x_min)/cfg.proposal_grid,
                                                     (cfg.bev_y_max-cfg.bev_y_min)/cfg.proposal_grid]))
        self.context=nn.Sequential(nn.Linear(cfg.channels,cfg.channels),nn.ReLU())
        self.score=nn.Linear(cfg.channels,1);self.offset=nn.Linear(cfg.channels,2)
        nn.init.zeros_(self.offset.weight);nn.init.zeros_(self.offset.bias)
        nn.init.constant_(self.score.bias,-4.)

    def forward(self,features,projector,anchors):
        feature=features[self.cfg.feature_strides.index(8)];b,n,c,h,w=feature.shape
        points=self.centers[None,:,None].expand(b,-1,-1,-1)
        uv,valid,_=projector(points);uv=uv[:,:,0];valid=valid[:,:,0]
        grid=2*(uv/8+.5)/uv.new_tensor([w,h])-1
        grid=torch.where(valid[...,None],grid,torch.full_like(grid,2.))
        grid=grid.permute(0,2,1,3).reshape(b*n,-1,1,2)
        with torch.autocast(device_type=feature.device.type,enabled=False):
            sampled=F.grid_sample(feature.float().reshape(b*n,c,h,w),grid.float(),align_corners=False)
        values=sampled.reshape(b,n,c,-1).permute(0,3,1,2)
        values=(values*valid[...,None]).sum(2)/valid.sum(2,keepdim=True).clamp_min(1)
        values=self.context(values);logits=self.score(values)[...,0].float()
        offsets=self.offset(values).float().tanh()*self.cell_size/2
        centers=torch.cat((self.centers[None,:,:2]+offsets,self.centers[None,:,2:].expand(b,-1,-1)),-1)
        visible=valid.any(-1)
        indices=logits.masked_fill(~visible,-1e4).topk(self.cfg.proposal_queries,dim=1).indices
        chosen=centers.gather(1,indices[...,None].expand(-1,-1,3))
        features_out=values.gather(1,indices[...,None].expand(-1,-1,c))
        # Preserve the TRAIN-derived mixture of object sizes/yaws, not a Car
        # template or a predicted-class lookup. Invisible slots fall back.
        prior=anchors[:,-self.cfg.proposal_queries:]
        center=torch.cat((chosen[...,:2],prior[...,5:6]/2+self.cfg.ground_z),-1)
        boxes=torch.cat((center,prior[...,3:]),-1)
        supported=visible.gather(1,indices)
        boxes=torch.where(supported[...,None],boxes,prior)
        features_out=features_out*supported[...,None]
        return boxes,features_out,dict(logits=logits,centers=centers,valid=visible,
            lattice=self.centers,cell_size=self.cell_size)


def proposal_loss(proposal,targets):
    from .sparse_loss import ignored_queries
    zero=proposal['logits'].sum()*0+proposal['centers'].sum()*0
    total=zero
    for bi,target in enumerate(targets):
        target=target.to(proposal['logits'].device)
        gt=target[target[:,7]>=0];truth=torch.zeros_like(proposal['logits'][bi])
        keep=proposal['valid'][bi].clone()
        placeholder=torch.cat((proposal['lattice'],proposal['lattice'].new_ones(len(keep),4)),1)
        keep &= ~ignored_queries(placeholder,target[target[:,7]<0])
        regression=zero
        if len(gt):
            distance=torch.cdist(proposal['lattice'][:,:2],gt[:,:2])
            nearest=distance.argmin(1)
            delta=gt[nearest,:2]-proposal['lattice'][:,:2]
            positive=(delta.abs()<=proposal['cell_size']/2).all(-1)&proposal['valid'][bi]
            truth[positive]=1;keep|=positive
            regression=F.smooth_l1_loss(proposal['centers'][bi,positive,:2],gt[nearest[positive],:2],beta=.1,reduction='sum')/max(int(positive.sum()),1)
        logits=proposal['logits'][bi];p=logits.sigmoid()
        pt=torch.where(truth.bool(),p,1-p)
        focal=F.binary_cross_entropy_with_logits(logits,truth,reduction='none')*(1-pt).square()
        total+=((focal*keep).sum()/truth.sum().clamp_min(1)+regression)/len(targets)
    return total
