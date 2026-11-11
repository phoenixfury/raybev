"""Versioned sparse architectures; legacy Q100/C64 and Q225 observation experiments."""
from dataclasses import dataclass,asdict
import math
from .config import CLASSES


@dataclass(frozen=True)
class SparseConfig:
    architecture: str = 'yolo_rosa_lite_v1'
    architecture_version: int = 3
    backbone: str = 'yolo26n'
    image_width: int = 800
    image_height: int = 640
    channels: int = 64
    queries: int = 100
    stages: int = 4
    heads: int = 4
    attention_channels: int = 32
    ffn_channels: int = 128
    num_classes: int = len(CLASSES)
    theta_max_degrees: float = 105.
    bev_x_min: float = -48.
    bev_x_max: float = 48.
    bev_y_min: float = -48.
    bev_y_max: float = 48.
    min_range: float = .5
    max_range: float = 68.
    class_positive_weights: tuple = (1.,2.,4.,4.,4.,4.)
    mean_training_objects: float = 1.
    training_class_counts: tuple = (0,)*len(CLASSES)
    facing_loss: str = 'circular_bins'
    heading_encoding: str = 'axis_direction'
    heading_bins: int = 8
    quality_score_power: float = .5
    nms_iou_threshold: float = .5
    max_detections: int = 100
    dn_queries: int = 24
    dn_weight: float = .5
    teacher_weight: float = .2
    stride4: bool = False
    observation_conditioning: bool = False
    proposal_queries: int = 0
    proposal_grid: int = 24
    ground_z: float = 0.
    proposal_weight: float = 1.
    facing_cosine_prior: float = 1.
    facing_temperature: float = 1.
    multi_camera_facing: bool = True
    extent_weight: float = 0.

    @property
    def is_refined(self):return self.architecture_version>=4

    @property
    def feature_strides(self):return (4,8,16,32) if self.stride4 else (8,16,32)

    @property
    def num_learned_points(self):return 6 if self.is_refined else 0

    @property
    def num_support_points(self):return 7+self.num_learned_points

    @property
    def center_loss_beta(self):return .1 if self.is_refined else 1.

    def validate(self):
        if self.architecture!='yolo_rosa_lite_v1' or self.architecture_version not in (3,4,5) or self.backbone!='yolo26n':
            raise ValueError('Sparse architecture contract mismatch; dense/ROSA checkpoints cannot be resumed')
        if min(self.image_width,self.image_height)<=0 or self.image_width%32 or self.image_height%32:
            raise ValueError('Sparse image size must be divisible by 32')
        if (self.channels<16 or self.queries<1 or self.stages<1 or self.heads<1 or
                self.channels%self.heads or self.attention_channels%self.heads or
                self.attention_channels<self.heads or self.ffn_channels<self.channels):
            raise ValueError('Invalid sparse dimensions')
        if self.num_classes!=len(CLASSES) or len(self.class_positive_weights)!=len(CLASSES):
            raise ValueError('Sparse class order must follow CLASSES')
        if (self.facing_loss!='circular_bins' or self.heading_encoding!='axis_direction' or self.heading_bins!=8):
            raise ValueError('Sparse heading requires eight circular facing bins and a continuous doubled axis')
        values=(*self.class_positive_weights,self.mean_training_objects,self.min_range,self.max_range)
        if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Invalid sparse weights/ranges')
        if not (0<self.theta_max_degrees<180 and self.min_range<self.max_range and
                self.bev_x_min<self.bev_x_max and self.bev_y_min<self.bev_y_max):
            raise ValueError('Invalid sparse geometry')
        corner=max(math.hypot(x,y) for x in (self.bev_x_min,self.bev_x_max) for y in (self.bev_y_min,self.bev_y_max))
        if self.max_range<corner:raise ValueError('Radial limit must include corners of the square evaluation region')
        if (not 0<=self.quality_score_power<=1 or not 0<=self.nms_iou_threshold<=1 or
                not 1<=self.max_detections<=self.queries or self.dn_queries<0 or
                not all(math.isfinite(x) and x>=0 for x in (self.dn_weight,self.teacher_weight))):
            raise ValueError('Invalid sparse loss/decoder settings')
        if len(self.training_class_counts)!=len(CLASSES) or any(x<0 for x in self.training_class_counts):
            raise ValueError('Invalid sparse TRAIN counts')
        if (self.architecture_version<5 and (self.stride4 or self.observation_conditioning or self.proposal_queries or self.extent_weight)):
            raise ValueError('Observation/proposal/extent changes require architecture 5')
        if (not 0<=self.proposal_queries<self.queries or self.proposal_grid<2 or
                self.proposal_grid**2<self.proposal_queries or not math.isfinite(self.ground_z) or
                not all(math.isfinite(x) and x>=0 for x in (self.proposal_weight,self.extent_weight,self.facing_cosine_prior)) or
                not math.isfinite(self.facing_temperature) or self.facing_temperature<=0):
            raise ValueError('Invalid proposal/extent/facing settings')
        return self

    def to_dict(self):return asdict(self)

    @classmethod
    def from_checkpoint(cls,state):
        if state.get('architecture')!='yolo_rosa_lite_v1':raise ValueError('Not a YOLO-ROSA-Lite checkpoint')
        values=dict(state['cfg'])
        for key in ('class_positive_weights','training_class_counts'):
            if key in values:values[key]=tuple(values[key])
        return cls(**values).validate()
