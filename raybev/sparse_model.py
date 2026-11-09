"""Calibration-aware sparse 3D detection; no dense BEV or ROSA dependency.

Camera order follows config.CAMERAS. Public images are ImageNet-normalized RGB;
the YOLO stages consume RGB [0,1]. Projection and state arithmetic stay FP32.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .config import DEFAULT_CALIBRATION
from .geometry import load_cameras,letterbox_params
from .detection import decode_heading,rotated_bev_nms,in_bev_range
from .model import conv_bn_relu
from .sparse_config import SparseConfig
from .yolo26 import build_backbone,PROFILE


class YOLOPyramid(nn.Module):
    def __init__(self,cfg,pretrained=False,backbone_weights=None):
        super().__init__()
        network,self.initialization=build_backbone(pretrained,backbone_weights)
        self.backbone_profile=dict(PROFILE,ultralytics_version=network.ultralytics_version)
        self.stage8=network[:5]
        self.stage16=nn.Sequential(*list(network.children())[5:7])
        self.stage32=nn.Sequential(*list(network.children())[7:10])
        self.adapters=nn.ModuleList(nn.Sequential(conv_bn_relu(n,cfg.channels,1),
            nn.Conv2d(cfg.channels,cfg.channels,3,padding=1,groups=cfg.channels,bias=False),
            nn.BatchNorm2d(cfg.channels),nn.ReLU6()) for n in (128,128,256))
        self.register_buffer('mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer('std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)

    def forward_single(self,image):
        p3=self.stage8(image*self.std+self.mean)
        p4=self.stage16(p3);p5=self.stage32(p4)
        return tuple(adapter(x) for adapter,x in zip(self.adapters,(p3,p4,p5)))

    def forward(self,images):
        b,n,c,h,w=images.shape
        if n!=4 or c!=3:raise ValueError('Expected four RGB camera images')
        return tuple(x.reshape(b,4,*x.shape[1:]) for x in self.forward_single(images.reshape(b*4,c,h,w)))


class OpenCVProjector(nn.Module):
    def __init__(self,cfg,calibration):
        super().__init__();self.cfg=cfg
        cameras=load_cameras(calibration)
        def buffer(name,value):self.register_buffer(name,torch.as_tensor(np.asarray(value),dtype=torch.float32),persistent=False)
        buffer('transforms',[np.linalg.inv(c.T_base_cam) for c in cameras])
        buffer('intrinsics',[c.intrinsic for c in cameras]);buffer('distortion',[c.distortion for c in cameras])
        buffer('raw_size',[[c.width,c.height] for c in cameras])
        resize=[letterbox_params(c.width,c.height,cfg.image_width,cfg.image_height) for c in cameras]
        buffer('resize_scale',[[r[0]/c.width,r[1]/c.height] for r,c in zip(resize,cameras)])
        buffer('padding',[[r[2],r[3]] for r in resize])

    def forward(self,points):
        with torch.autocast(device_type=points.device.type,enabled=False):
            return self._project(points.float())

    def _project(self,points):
        # [B,Q,P,3] -> [B,Q,P,camera,2]. No pinhole Z>0 visibility test:
        # a fisheye legitimately observes rays beyond 90 degrees.
        points=points.float()
        xyz=torch.einsum('ncj,bqpj->bqpnc',self.transforms[:,:3,:3],points)
        xyz=xyz+self.transforms[None,None,None,:,:3,3]
        x,y,z=xyz.unbind(-1)
        rho=torch.sqrt(x.square()+y.square()+1e-12)
        radius=xyz.norm(dim=-1).clamp_min(1e-6)
        theta=torch.atan2(rho,z);t2=theta.square()
        k=self.distortion
        distorted=theta*(1+k[:,0]*t2+k[:,1]*t2.square()+k[:,2]*t2**3+k[:,3]*t2**4)
        xy=torch.stack((x*distorted/rho,y*distorted/rho,torch.ones_like(x)),-1)
        uv=torch.einsum('nij,bqpnj->bqpni',self.intrinsics,xy)[...,:2]
        margin=torch.minimum(uv/self.raw_size,1-uv/self.raw_size).amin(-1)
        valid=(theta<=math.radians(self.cfg.theta_max_degrees))&(radius>.05)&(margin>=0)
        valid=valid&(uv<self.raw_size).all(-1)&torch.isfinite(xyz).all(-1)
        net=(uv+.5)*self.resize_scale-.5+self.padding
        cues=torch.cat((xyz/radius[...,None],(theta/math.radians(self.cfg.theta_max_degrees))[...,None],
                        (4*margin).clamp(0,1)[...,None],(radius.log()/math.log(self.cfg.max_range))[...,None]),-1)
        return net,valid,cues


def box_points(boxes):
    offsets=boxes.new_tensor([[0,0,0],[.45,0,0],[-.45,0,0],[0,.45,0],[0,-.45,0],[0,0,.45],[0,0,-.45]])
    local=boxes[...,None,3:6]*offsets
    x,y,z=local.unbind(-1);angle=boxes[...,6,None]
    rotated=torch.stack((x*angle.cos()-y*angle.sin(),x*angle.sin()+y*angle.cos(),z),-1)
    return boxes[...,None,:3]+rotated


def masked_softmax(logits,valid,dim):
    # All-invalid observations must yield exactly zero, not NaN or a uniform
    # distribution over invalid cameras. Multiplication also kills gradients.
    weights=logits.float().masked_fill(~valid,-1e4).softmax(dim)*valid
    return weights/weights.sum(dim,keepdim=True).clamp_min(1e-8)


class QueryAttention(nn.Module):
    """Explicit static-shape attention, avoiding version-dependent MHA exporters."""
    def __init__(self,cfg):
        super().__init__();self.heads=cfg.heads;self.width=cfg.channels//cfg.heads
        self.qkv=nn.Linear(cfg.channels,3*cfg.channels);self.output=nn.Linear(cfg.channels,cfg.channels)

    def forward(self,x,mask=None,key_valid=None):
        b,q,c=x.shape
        a=self.qkv(x).reshape(b,q,3,self.heads,self.width).permute(2,0,3,1,4)
        query,key,value=a.unbind(0)
        logits=(query.float()@key.float().transpose(-1,-2))/math.sqrt(self.width)
        valid=torch.ones_like(logits,dtype=torch.bool)
        if mask is not None:valid=valid&~mask[None,None]
        if key_valid is not None:valid=valid&key_valid[:,None,None,:]
        weights=masked_softmax(logits,valid,-1)
        attended=(weights@value.float()).transpose(1,2).reshape(b,q,c)
        return self.output(attended)


class SparseEvidence(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.query=nn.Linear(cfg.channels,cfg.attention_channels)
        self.key=nn.Linear(cfg.channels,cfg.attention_channels)
        # ray XYZ, angle, margin, log-range, expected angular size, log stride.
        self.geometry=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,cfg.heads))
        self.ray=nn.Linear(9,cfg.channels,bias=False);nn.init.zeros_(self.ray.weight)
        self.point_query=nn.Linear(cfg.channels,7*cfg.heads)
        self.point_key=nn.Linear(cfg.channels,cfg.heads)
        self.output=nn.Linear(cfg.channels,cfg.channels)

    def forward(self,query,boxes,features,projector):
        uv,visible,cues=projector(box_points(boxes))
        b,q,p,n,_=uv.shape;h=self.cfg.heads;c=self.cfg.channels
        sampled=[]
        for feature,stride in zip(features,(8,16,32)):
            fh,fw=feature.shape[-2:]
            grid=2*(uv/stride+.5)/uv.new_tensor([fw,fh])-1
            grid=torch.where(visible[...,None],grid,torch.full_like(grid,2.))
            grid=grid.permute(0,3,1,2,4).reshape(b*n,q*p,1,2)
            # FP32 geometry/sampling is intentional. Casts preserve gradients
            # into the AMP encoder; no detach in the observation feedback loop.
            with torch.autocast(device_type=feature.device.type,enabled=False):
                value=F.grid_sample(feature.float().reshape(b*n,c,fh,fw),grid.float(),align_corners=False)
            sampled.append(value.reshape(b,n,c,q,p).permute(0,3,4,1,2))
        values=torch.stack(sampled,dim=-2)  # B,Q,P,N,L,C
        x,y,z=cues[...,:3].unbind(-1)
        basis=torch.stack((torch.ones_like(x),x,y,z,x*y,y*z,z*x,x*x-y*y,3*z*z-1),-1)
        values=values+self.ray(basis)[...,None,:]
        key=self.key(values).reshape(b,q,p,n,3,h,-1)
        attention_query=self.query(query).reshape(b,q,h,-1)
        logits=(key.float()*attention_query[:,:,None,None,None].float()).sum(-1)/math.sqrt(self.cfg.attention_channels//h)
        angular=(boxes[...,3:6].norm(dim=-1)/boxes[...,:3].norm(dim=-1).clamp_min(.5)).clamp(.001,10).log()
        descriptor=torch.cat((cues,angular[:,:,None,None,None].expand(b,q,p,n,1)),-1)
        descriptor=descriptor[...,None,:].expand(b,q,p,n,3,7)
        stride=values.new_tensor([math.log(s/8) for s in (8,16,32)]).view(1,1,1,1,3,1).expand(b,q,p,n,3,1)
        logits=logits+self.geometry(torch.cat((descriptor,stride),-1))
        valid=visible[...,None,None].expand(b,q,p,n,3,h).reshape(b,q,p,n*3,h)
        weights=masked_softmax(logits.reshape(b,q,p,n*3,h),valid,3)
        split=values.reshape(b,q,p,n*3,h,c//h)
        per_point=(weights[...,None]*split).sum(3).reshape(b,q,p,c)
        point_logits=self.point_query(query).reshape(b,q,p,h)+self.point_key(per_point)
        point_valid=visible.any(-1)[...,None].expand(b,q,p,h)
        point_weights=masked_softmax(point_logits,point_valid,2)
        fused=(point_weights[...,None]*per_point.reshape(b,q,p,h,c//h)).sum(2).reshape(b,q,c)
        return self.output(fused),per_point,visible.any(-1).any(-1)


class SparseStage(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.position=nn.Sequential(nn.Linear(8,cfg.channels),nn.ReLU(),nn.Linear(cfg.channels,cfg.channels))
        self.attention=QueryAttention(cfg);self.norm1=nn.LayerNorm(cfg.channels)
        self.evidence=SparseEvidence(cfg);self.norm2=nn.LayerNorm(cfg.channels)
        self.ffn=nn.Sequential(nn.Linear(cfg.channels,cfg.ffn_channels),nn.ReLU(),nn.Linear(cfg.ffn_channels,cfg.channels))
        self.norm3=nn.LayerNorm(cfg.channels)
        self.classifier=nn.Linear(cfg.channels,cfg.num_classes)
        self.center=nn.Linear(cfg.channels,3);self.size=nn.Linear(cfg.channels,3)
        self.axis=nn.Linear(cfg.channels,2);self.quality=nn.Linear(cfg.channels,1)
        self.facing=nn.Sequential(nn.Linear(3*cfg.channels,cfg.channels//2),nn.ReLU(),nn.Linear(cfg.channels//2,8))
        nn.init.constant_(self.classifier.bias,math.log(.01/.99))
        for module in (self.center,self.size,self.axis,self.facing[-1]):
            nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        self.axis.bias.data[1]=1.

    def forward(self,query,boxes,features,projector,attention_mask=None,key_valid=None):
        boxes=boxes.float()
        position=torch.cat((boxes[...,:3]/48,boxes[...,3:6].clamp_min(.05).log()/3,
                            boxes[...,6,None].sin(),boxes[...,6,None].cos()),-1)
        query=self.norm1(query+self.attention(query+self.position(position),attention_mask,key_valid))
        evidence,points,supported=self.evidence(query,boxes,features,projector)
        query=self.norm2(query+evidence);query=self.norm3(query+self.ffn(query))
        raw=self.center(query).float()
        radius=boxes[...,:3].norm(dim=-1,keepdim=True).clamp_min(self.cfg.min_range)
        direction=F.normalize(boxes[...,:3],dim=-1,eps=1e-6)
        tangent=torch.stack((-direction[...,1],direction[...,0],torch.zeros_like(direction[...,0])),-1)
        fallback=direction.new_tensor([1,0,0]).expand_as(tangent)
        tangent=F.normalize(torch.where(tangent.norm(dim=-1,keepdim=True)>.001,tangent,fallback),dim=-1)
        vertical=torch.cross(direction,tangent,dim=-1)
        direction=F.normalize(direction+.5*raw[...,:1].tanh()*tangent+.5*raw[...,1:2].tanh()*vertical,dim=-1)
        distance=(radius.log()+.7*raw[...,2:3].tanh()).clamp(math.log(self.cfg.min_range),math.log(self.cfg.max_range)).exp()
        center=direction*distance
        size=(boxes[...,3:6].clamp_min(.05).log()+.5*self.size(query).float().tanh()).clamp(-3,5).exp()
        rotation=F.normalize(self.axis(query).float(),dim=-1,eps=1e-4)
        old=2*boxes[...,6]
        axis=torch.stack((old.sin()*rotation[...,1]+old.cos()*rotation[...,0],
                          old.cos()*rotation[...,1]-old.sin()*rotation[...,0]),-1)
        facing_input=torch.cat((query,points[:,:,1]-points[:,:,2],points[:,:,1]+points[:,:,2]),-1)
        centers=query.new_tensor([i*2*math.pi/8 for i in range(8)])
        facing=self.facing(facing_input).float()+torch.cos(centers-boxes[...,6,None])
        yaw,_=decode_heading(axis,facing,self.cfg)
        updated=torch.cat((center,size,yaw[...,None]),-1)
        return query,dict(boxes=updated,axis=axis,facing=facing,class_logits=self.classifier(query),
                          quality_logits=self.quality(query)[...,0],supported=supported)


class YOLOROSALite(nn.Module):
    def __init__(self,cfg=None,calibration=DEFAULT_CALIBRATION,anchors=None,pretrained=False,backbone_weights=None):
        super().__init__();self.cfg=(cfg or SparseConfig()).validate()
        self.image_encoder=YOLOPyramid(self.cfg,pretrained,backbone_weights)
        self.projector=OpenCVProjector(self.cfg,calibration)
        if anchors is None:
            angles=torch.arange(self.cfg.queries)*2*math.pi/self.cfg.queries
            anchors=torch.stack((20*angles.cos(),20*angles.sin(),torch.ones_like(angles),
                angles.new_full(angles.shape,4.5),angles.new_full(angles.shape,2.),angles.new_full(angles.shape,1.8),angles),-1)
        anchors=torch.as_tensor(anchors,dtype=torch.float32)
        if anchors.shape!=(self.cfg.queries,7) or not torch.isfinite(anchors).all() or (anchors[:,3:6]<=0).any():
            raise ValueError('Expected finite [queries,7] anchors with positive dimensions')
        if (anchors[:,:3].norm(dim=-1)<self.cfg.min_range).any():raise ValueError('Anchor centers cannot be at the rig origin')
        self.register_buffer('anchors',anchors.clone())
        self.query_features=nn.Parameter(torch.randn(self.cfg.queries,self.cfg.channels)*.02)
        self.layers=nn.ModuleList(SparseStage(self.cfg) for _ in range(self.cfg.stages))

    def forward_features(self,features,targets=None):
        b=features[0].shape[0];q=self.cfg.queries
        boxes=self.anchors[None].expand(b,-1,-1)
        query=self.query_features[None].expand(b,-1,-1)
        indices=[];attention_mask=None;valid=None
        if self.training and targets is not None and self.cfg.dn_queries:
            for gt in targets:
                known=torch.nonzero(gt[:,7]>=0,as_tuple=False)[:,0]
                if len(known):
                    weights=gt.new_tensor(self.cfg.class_positive_weights)[gt[known,7].long()]
                    known=known[torch.multinomial(weights,min(len(known),self.cfg.dn_queries),replacement=False)]
                indices.append(known)
            count=max(map(len,indices),default=0)
            if count:
                extra=boxes.new_zeros(b,count,7);extra[...,0]=10;extra[...,3:6]=1
                valid=torch.ones(b,q+count,dtype=torch.bool,device=boxes.device)
                valid[:,q:]=False
                for bi,ids in enumerate(indices):
                    value=targets[bi][ids,:7].to(boxes.device).clone();n=len(value)
                    value[:,:3]+=(torch.rand_like(value[:,:3])-.5)*value[:,3:6]*.4
                    value[:,3:6]*=torch.exp(torch.randn_like(value[:,3:6])*.1)
                    value[:,6]+=(torch.rand_like(value[:,6])-.5)*.6
                    extra[bi,:n]=value;valid[bi,q:q+n]=True
                boxes=torch.cat((boxes,extra),1)
                query=torch.cat((query,query.new_zeros(b,count,self.cfg.channels)),1)
                attention_mask=torch.zeros(q+count,q+count,dtype=torch.bool,device=boxes.device)
                attention_mask[:q,q:]=True
                attention_mask[q:,q:]=~torch.eye(count,dtype=torch.bool,device=boxes.device)
        outputs=[]
        for layer in self.layers:
            query,output=layer(query,boxes,features,self.projector,attention_mask,valid)
            boxes=output['boxes'];outputs.append(output)
        return dict(stages=outputs,dn_indices=indices,num_queries=q)

    def forward(self,images,targets=None):
        if images.shape[-2:]!=(self.cfg.image_height,self.cfg.image_width):raise ValueError('Sparse input image size mismatch')
        return self.forward_features(self.image_encoder(images),targets)


@torch.no_grad()
def decode_sparse(output,cfg,score_threshold=.1,apply_nms=True):
    if not 0<=score_threshold<=1:raise ValueError('Invalid sparse score threshold')
    final=output['stages'][-1];q=cfg.queries
    probabilities=final['class_logits'][:,:q].float().sigmoid()
    score,classes=probabilities.max(-1)
    score=score*final['quality_logits'][:,:q].float().sigmoid().pow(cfg.quality_score_power)
    finite=torch.isfinite(final['class_logits'][:,:q]).all(-1)
    finite=finite&torch.isfinite(final['axis'][:,:q]).all(-1)&torch.isfinite(final['facing'][:,:q]).all(-1)
    if cfg.quality_score_power>0:finite=finite&torch.isfinite(final['quality_logits'][:,:q])
    results=[]
    for boxes,labels,scores,support,valid in zip(final['boxes'][:,:q].float(),classes,score,final['supported'][:,:q],finite):
        detections=torch.cat((boxes,labels[:,None].float(),scores[:,None]),-1)
        keep=valid&torch.isfinite(detections).all(-1)&in_bev_range(detections,cfg)&(scores>=score_threshold)&(scores>0)&support
        detections=detections[keep]
        detections=detections[detections[:,8].argsort(descending=True)]
        if apply_nms and len(detections):detections=detections[rotated_bev_nms(detections,cfg.nms_iou_threshold,cfg.max_detections)]
        results.append(detections[:cfg.max_detections])
    return results
