"""RayBEV: YOLO26n / ResNet-18 / MobileNetV2, fisheye planes and 3D head.

The geometric view transform is a reference PyTorch implementation using
fixed projection grids. It is NOT claimed to be TIDL-compatible as a graph;
use a C7x/Arm implementation of the fused LUT sampler for on-chip deployment.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2, ResNet18_Weights, resnet18

from .config import ModelConfig
from .geometry import build_image_geometry, build_projection_lut, load_cameras


def conv_bn_relu(cin, cout, kernel=3, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel//2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU6(inplace=True))


class FixedStatsBatchNorm2d(nn.BatchNorm2d):
    """Learnable affine, fixed statistics in BOTH modes; folds into Conv at export.

    Ordinary BatchNorm state keys are retained for strict v4 weight migration.
    model.train(), including DDP, cannot re-enable per-frame normalization.
    """
    def forward(self,x):
        return F.batch_norm(x,self.running_mean,self.running_var,self.weight,self.bias,
                            training=False,momentum=0.,eps=self.eps)


def fix_batchnorm_statistics(module):
    for name,child in module.named_children():
        if isinstance(child,nn.BatchNorm2d):
            fixed=FixedStatsBatchNorm2d(child.num_features,eps=child.eps,
                                       momentum=child.momentum,affine=child.affine)
            fixed.load_state_dict(child.state_dict(),strict=True)
            setattr(module,name,fixed)
        else:fix_batchnorm_statistics(child)


class SeparableBlock(nn.Module):
    def __init__(self, cin, cout=None, stride=1):
        super().__init__()
        cout = cin if cout is None else cout
        self.block = nn.Sequential(
            nn.Conv2d(cin,cin,3,stride,1,groups=cin,bias=False),
            nn.BatchNorm2d(cin),nn.ReLU6(inplace=True),
            nn.Conv2d(cin,cout,1,bias=False),nn.BatchNorm2d(cout))
        self.residual = cin == cout and stride == 1
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        y = self.block(x)
        return self.act(x+y if self.residual else y)


def separable_block(ch):
    return SeparableBlock(ch)


class MobileNetEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig, pretrained=False, backbone_weights=None):
        super().__init__()
        if pretrained and cfg.mobile_width != 1.0:
            raise ValueError('TorchVision ImageNet MobileNetV2 weights require mobile_width=1.0')
        weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained and not backbone_weights else None
        network = mobilenet_v2(weights=weights, width_mult=cfg.mobile_width)
        if backbone_weights:
            state = torch.load(backbone_weights,map_location='cpu',weights_only=True)
            network.load_state_dict(state.get('state_dict',state),strict=True)
        self.initialization = str(backbone_weights) if backbone_weights else ('IMAGENET1K_V2' if pretrained else 'random_explicit')
        mobile = network.features
        self.stage8 = mobile[:7]
        self.stage16 = mobile[7:14]
        self.stage32 = mobile[14:18]
        def round8(ch):
            # TorchVision MobileNet's _make_divisible behavior.
            z = max(8, int(ch * cfg.mobile_width + 4) // 8 * 8)
            return z + 8 if z < .9 * ch * cfg.mobile_width else z
        c8, c16, c32 = round8(32), round8(96), round8(320)
        self.init_pyramid(cfg,c8,c16,c32)

    def init_pyramid(self,cfg,c8,c16,c32):
        self.lat8 = conv_bn_relu(c8, cfg.feature_channels, 1)
        self.lat16 = conv_bn_relu(c16, cfg.feature_channels, 1)
        self.lat32 = conv_bn_relu(c32, cfg.feature_channels, 1)
        self.fuse = separable_block(cfg.feature_channels)
        # Geometry-conditioned level selection and ray feature lifting, all 1x1.
        self.ray_embedding = nn.Conv2d(8,cfg.feature_channels,1,bias=False)
        self.level_gate = nn.Conv2d(cfg.feature_channels+8,3,1)
        self.confidence = nn.Conv2d(cfg.feature_channels,len(cfg.heights),1)
        nn.init.zeros_(self.level_gate.weight)
        nn.init.zeros_(self.level_gate.bias)

    def forward_single(self, x, geometry):
        """One image [B,3,H,W] -> feature tensor + confidence logits.

        This 4D subgraph should be exported for TIDL and invoked four times
        with shared weights. No camera-specific layers live here.
        """
        low = self.stage8(x)
        high = self.stage16(low)
        deep = self.stage32(high)
        p8 = self.lat8(low)
        # Input dimensions are multiples of 32: use fixed scales in ONNX,
        # avoiding Shape/Slice subgraphs for runtime-computed resize sizes.
        p16 = F.interpolate(self.lat16(high),scale_factor=2.,mode='nearest')
        p32 = F.interpolate(self.lat32(deep),scale_factor=4.,mode='nearest')
        gates = torch.sigmoid(self.level_gate(torch.cat((p8,geometry),dim=1)))
        features = p8*gates[:,0:1] + p16*gates[:,1:2] + p32*gates[:,2:3]
        features = self.fuse(features + self.ray_embedding(geometry))
        logits = self.confidence(features)
        return features, logits

    def forward(self, images, geometry):
        """images: [B,4,3,H,W], output: [B,4,C,H/8,W/8], confidence logits."""
        b, n, c, h, w = images.shape
        if n != 4 or c != 3 or h % 32 or w % 32:
            raise ValueError('Expected [B,4,3,H,W], with H and W divisible by 32')
        cues = geometry.unsqueeze(0).expand(b,-1,-1,-1,-1).reshape(b*n,8,h//8,w//8)
        features, logits = self.forward_single(images.reshape(b*n,c,h,w),cues)
        return (features.reshape(b,n,*features.shape[1:]),
                logits.reshape(b,n,*logits.shape[1:]))


class ResNet18Encoder(MobileNetEncoder):
    """Shared /8,/16,/32 geometry pyramid; conventional ImageNet ResNet-18."""
    def __init__(self,cfg,pretrained=False,backbone_weights=None):
        nn.Module.__init__(self)
        weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained and not backbone_weights else None
        network=resnet18(weights=weights)
        if backbone_weights:
            state=torch.load(backbone_weights,map_location='cpu',weights_only=True)
            network.load_state_dict(state.get('state_dict',state),strict=True)
        self.initialization=str(backbone_weights) if backbone_weights else ('RESNET18_IMAGENET1K_V1' if pretrained else 'random_explicit')
        self.stage8=nn.Sequential(network.conv1,network.bn1,network.relu,network.maxpool,network.layer1,network.layer2)
        self.stage16=network.layer3
        self.stage32=network.layer4
        self.init_pyramid(cfg,128,256,512)


class YOLO26nEncoder(MobileNetEncoder):
    """Pretrained nano CNN stages + the shared geometry-conditioned RayBEV FPN.

    The external RGB/ImageNet input contract remains identical to legacy models.
    Restore RGB [0,1] BEFORE the YOLO stages, with no BGR swap or letterboxing.
    Only layers 0..9 are retained: no self-attention or YOLO 2D detector.
    """
    def __init__(self,cfg,pretrained=False,backbone_weights=None):
        nn.Module.__init__(self)
        from .yolo26 import build_backbone,PROFILE
        backbone,self.initialization=build_backbone(pretrained,backbone_weights)
        self.backbone_profile=dict(PROFILE)
        self.stage8=backbone[:5]
        self.stage16=nn.Sequential(*list(backbone.children())[5:7])
        self.stage32=nn.Sequential(*list(backbone.children())[7:10])
        self.register_buffer('input_mean',torch.tensor([.485,.456,.406]).view(1,3,1,1),persistent=False)
        self.register_buffer('input_std',torch.tensor([.229,.224,.225]).view(1,3,1,1),persistent=False)
        self.init_pyramid(cfg,128,128,256)

    def forward_single(self,x,geometry):
        return super().forward_single(x*self.input_std+self.input_mean,geometry)


class FisheyeRayTransport(nn.Module):
    def __init__(self, cfg, calibration_path):
        super().__init__()
        self.cfg = cfg.validate()
        self.cameras = load_cameras(calibration_path)
        grid, prior, _ = build_projection_lut(self.cameras, cfg)
        self.register_buffer('sample_grid', torch.from_numpy(grid), persistent=False)
        self.register_buffer('geometric_prior', torch.from_numpy(prior), persistent=False)
        self.register_buffer('image_geometry',torch.from_numpy(build_image_geometry(self.cameras,cfg)),persistent=False)
        x = cfg.bev_x_max-(torch.arange(cfg.bev_height)+.5)*cfg.bev_resolution
        y = cfg.bev_y_max-(torch.arange(cfg.bev_width)+.5)*cfg.bev_resolution
        xx,yy = torch.meshgrid(x,y,indexing='ij')
        sx = max(abs(cfg.bev_x_min),abs(cfg.bev_x_max))
        sy = max(abs(cfg.bev_y_min),abs(cfg.bev_y_max))
        positions = torch.stack((xx/sx,yy/sy,torch.sqrt((xx/sx)**2+(yy/sy)**2)/2**.5))
        self.register_buffer('position_cues',positions[None],persistent=False)

    def forward(self, features, confidence_logits):
        b, n, c, _, _ = features.shape
        if n != 4 or confidence_logits.shape[:2] != (b, 4):
            raise ValueError('Expected four camera features and gates')
        numerator = denominator = height_sum = None
        coverage = []
        planes = []
        for k in range(len(self.cfg.heights)):
            plane_denominator = plane_numerator = None
            for camera_idx in range(4):
                # Small reference implementation; replace with one fused LUT gather
                # + weighted accumulation kernel in a production TI pipeline.
                image_map = torch.cat((features[:,camera_idx],
                                       confidence_logits[:,camera_idx,k:k+1]), dim=1)
                grid = self.sample_grid[k,camera_idx].unsqueeze(0).expand(b,-1,-1,-1)
                sampled = F.grid_sample(image_map, grid, mode='bilinear',
                                        align_corners=False, padding_mode='zeros')
                prior = self.geometric_prior[k,camera_idx][None,None].to(sampled.dtype)
                weight = torch.sigmoid(sampled[:, -1:]) * prior
                contribution = sampled[:, :-1] * weight
                if self.cfg.height_fusion == 'concat':
                    plane_numerator = contribution if plane_numerator is None else plane_numerator + contribution
                else:
                    numerator = contribution if numerator is None else numerator + contribution
                denominator = weight if denominator is None else denominator + weight
                hz = weight * (self.cfg.heights[k]/max(1.,max(abs(z) for z in self.cfg.heights)))
                height_sum = hz if height_sum is None else height_sum+hz
                plane_denominator = weight if plane_denominator is None else plane_denominator+weight
            coverage.append(plane_denominator.clamp(0,1))
            if self.cfg.height_fusion == 'concat':planes.append(plane_numerator)
        # A COMMON denominator preserves camera/height reliability and gives
        # sum(plane features) == the legacy pooled map. Do not normalize each
        # height separately: that would amplify nearly invisible height planes.
        norm = denominator.clamp(min=1e-4)
        visual = torch.cat(planes,dim=1)/norm if planes else numerator/norm
        height = height_sum / denominator.clamp(min=1e-4)
        positions = self.position_cues.to(visual.dtype).expand(b,-1,-1,-1)
        return torch.cat([visual,*coverage,height,positions],dim=1)


class BEVDetector(nn.Module):
    """Purely convolutional subgraph, suitable for separate ONNX/TIDL evaluation."""
    def __init__(self, cfg):
        super().__init__()
        c,h = cfg.bev_channels,cfg.head_channels
        self.exact_pyramid = cfg.bev_height%8==0 and cfg.bev_width%8==0
        self.fine = conv_bn_relu(cfg.bev_input_channels,h,1)
        self.down2 = nn.Sequential(SeparableBlock(h,c,2),separable_block(c))
        self.down4 = nn.Sequential(SeparableBlock(c,c+32,2),separable_block(c+32))
        self.down8 = nn.Sequential(SeparableBlock(c+32,c+64,2),
                                   *[separable_block(c+64) for _ in range(4)])
        self.lat8 = conv_bn_relu(c+64,c+32,1)
        self.lat4 = conv_bn_relu(c+32,c,1)
        self.lat2 = conv_bn_relu(c,h,1)
        self.fuse4 = separable_block(c+32)
        self.fuse2 = separable_block(c)
        self.fuse1 = separable_block(h)
        def head(channels):
            return nn.Sequential(separable_block(h),nn.Conv2d(h,channels,1))
        self.heatmap = head(cfg.num_classes)
        self.objectness = head(1) if cfg.classification_mode=='objectness_softmax' else None
        self.offset = head(2)
        self.height = head(1)
        self.size = head(3)
        self.yaw = head(2)
        self.heading = head(cfg.heading_bins) if cfg.heading_bins else None
        self.quality = head(1)
        # Regression channels: offset_r,c (2), z (1), log_lwh (3),
        # sin(2*yaw),cos(2*yaw) axis in recipe 4, local bin residual in v3,
        # global sin/cos for legacy v2. Facing features are a separate output.
        # At 384x384 most cells are background. A 1% initial probability avoids
        # the dense-negative spike produced by the old 10% prior on this grid.
        nn.init.normal_(self.heatmap[-1].weight,std=.001)
        nn.init.constant_(self.heatmap[-1].bias,math.log(.01/.99))
        if self.objectness is not None:
            nn.init.normal_(self.objectness[-1].weight,std=.001)
            nn.init.constant_(self.objectness[-1].bias,math.log(.01/.99))
        if cfg.size_encoding in ('class_relative','shared_relative'):
            nn.init.zeros_(self.size[-1].weight)
            nn.init.zeros_(self.size[-1].bias)
        if self.heading is not None:
            nn.init.normal_(self.heading[-1].weight,std=.001)
            nn.init.zeros_(self.heading[-1].bias)
        nn.init.zeros_(self.yaw[-1].weight)
        with torch.no_grad():
            self.yaw[-1].bias.copy_(torch.tensor([0.,1.]))
        if cfg.bev_normalization=='frozen_batchnorm':
            fix_batchnorm_statistics(self)

    def forward(self, bev):
        fine = self.fine(bev)
        s2 = self.down2(fine)
        s4 = self.down4(s2)
        s8 = self.down8(s4)
        s4 = self.fuse4(s4+self.upsample(self.lat8(s8),s4))
        s2 = self.fuse2(s2+self.upsample(self.lat4(s4),s2))
        features = self.fuse1(fine+self.upsample(self.lat2(s2),fine))
        regression = torch.cat((self.offset(features),self.height(features),
                                self.size(features),self.yaw(features)),dim=1)
        outputs=(self.heatmap(features),regression,self.quality(features))
        if self.heading is not None:outputs=outputs+(self.heading(features),)
        if self.objectness is not None:outputs=outputs+(self.objectness(features),)
        return outputs

    def upsample(self, x, reference):
        if self.exact_pyramid:
            return F.interpolate(x,scale_factor=2.,mode='nearest')
        # Nonmultiple-of-eight grids are used only by compact smoke fixtures.
        return F.interpolate(x,size=reference.shape[-2:],mode='nearest')


class RayBEVLite(nn.Module):
    def __init__(self, cfg: ModelConfig, calibration_path, pretrained=False, backbone_weights=None):
        super().__init__()
        self.cfg = cfg.validate()
        encoder={'resnet18':ResNet18Encoder,'mobilenet_v2':MobileNetEncoder,
                 'yolo26n':YOLO26nEncoder}[cfg.backbone]
        self.image_encoder = encoder(cfg,pretrained=pretrained,backbone_weights=backbone_weights)
        self.transport = FisheyeRayTransport(cfg, calibration_path)
        self.bev_detector = BEVDetector(cfg)

    def forward(self, images):
        f, confidence = self.image_encoder(images,self.transport.image_geometry)
        bev = self.transport(f, confidence)
        return self.bev_detector(bev)
