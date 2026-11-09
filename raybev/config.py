"""RayBEV-Lite configuration. BASELINK uses x forward, y left, z up."""
from dataclasses import dataclass, replace
import math
from pathlib import Path

CAMERAS = ("ARC_F", "ARC_R", "ARC_SL", "ARC_SR")
CLASSES = ("car", "pedestrian", "cyclist", "truck", "van", "bus")
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}
DEFAULT_BOX_FRAME = 'CLIDAR'  # Input labels only; the network operates in BASELINK.
DEFAULT_CALIBRATION = str(Path(__file__).resolve().parents[1]/'configs/calibration_baselink.json')


@dataclass(frozen=True)
class ModelConfig:
    # Each 1920x1536 source image is aspect-preservingly resized to 800x640.
    # No padding is required for this source camera aspect ratio.
    image_width: int = 800
    image_height: int = 640
    architecture_version: int = 2
    feature_channels: int = 32
    bev_channels: int = 64
    head_channels: int = 32
    backbone: str = 'resnet18'
    bev_normalization: str = 'frozen_batchnorm'
    classification_mode: str = 'objectness_softmax'
    foreground_label_smoothing: float = 0.05
    mobile_width: float = 1.0
    # 96 m square, 384x384 output. The expensive BEV context runs at /2,/4,/8.
    bev_x_min: float = -48.0
    bev_x_max: float = 48.0
    bev_y_min: float = -48.0
    bev_y_max: float = 48.0
    bev_resolution: float = 0.25
    heights: tuple = (0.4, 1.4, 2.4, 3.4)
    theta_max_degrees: float = 105.0
    num_classes: int = len(CLASSES)
    gaussian_overlap: float = 0.1
    # Foreground-only difficulty weights; dense background penalties stay unchanged.
    class_positive_weights: tuple = (1.0, 2.0, 4.0, 4.0, 4.0, 4.0)
    class_regression_weights: tuple = (1.0, 2.0, 4.0, 4.0, 4.0, 4.0)
    regression_weights: tuple = (2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    quality_loss_weight: float = 0.5
    yaw_norm_weight: float = 0.1
    quality_score_power: float = 0.5
    score_peak_mode: str = 'fused'
    # Recipe 4: continuous sin(2*yaw),cos(2*yaw) axis + facing classifier.
    heading_encoding: str = 'axis_direction'
    heading_bins: int = 2  # facing-vector channels; legacy bins remain 0/4
    heading_loss_weight: float = 1.0
    normalize_yaw_loss: bool = False
    # Task recipe 3.1: keep the same object denominator for foreground and
    # background. Macro averaging + difficulty weights + repeats collapsed v3
    # class scores. Retain the flag only to reproduce historical checkpoints.
    class_balanced_losses: bool = False
    center_classification_weight: float = 1.0
    heavy_classification_weight: float = 0.0
    # Fixed TRAIN mean makes accumulation independent of per-frame crowding.
    loss_normalization: str = 'fixed_train_mean'
    mean_training_objects: float = 1.0  # fitted before training
    size_encoding: str = 'class_relative'
    size_reference: tuple = (4.5,1.9,1.6)  # fitted from TRAIN, never predicted class
    # Replaced by TRAIN-ONLY medians before constructing the training model.
    class_size_priors: tuple = ((1.0,1.0,1.0),)*len(CLASSES)
    training_class_counts: tuple = (0,)*len(CLASSES)
    sampling_mode: str = 'frame_presence'
    single_label: bool = True
    nms_iou_threshold: float = 0.25
    max_detections: int = 150

    @property
    def bev_input_channels(self):
        # Height-pooled features, per-height coverage, mean height, XY/range cues.
        return self.feature_channels + len(self.heights) + 4

    @property
    def bev_height(self) -> int:
        n = (self.bev_x_max - self.bev_x_min) / self.bev_resolution
        if abs(n - round(n)) > 1.e-5:
            raise ValueError('BEV x extent must be divisible by resolution')
        return round(n)

    @property
    def bev_width(self) -> int:
        n = (self.bev_y_max - self.bev_y_min) / self.bev_resolution
        if abs(n - round(n)) > 1.e-5:
            raise ValueError('BEV y extent must be divisible by resolution')
        return round(n)

    def validate(self):
        if self.architecture_version != 2:
            raise ValueError('Legacy architecture: v2 requires a fresh training run')
        if min(self.image_width, self.image_height) <= 0 or self.image_width % 32 or self.image_height % 32:
            raise ValueError('Input dimensions must be divisible by 32')
        if (self.bev_resolution <= 0 or not self.heights or
                not all(math.isfinite(z) for z in self.heights) or
                not 0 < self.theta_max_degrees < 180 or
                self.bev_x_min >= self.bev_x_max or self.bev_y_min >= self.bev_y_max):
            raise ValueError('Invalid geometry configuration')
        if min(self.feature_channels, self.bev_channels, self.head_channels) < 8:
            raise ValueError('Network channel counts must be >= 8')
        if self.bev_height < 8 or self.bev_width < 8:
            raise ValueError('BEV must be at least 8x8')
        if (self.num_classes != len(CLASSES) or len(self.class_positive_weights) != self.num_classes
                or len(self.class_regression_weights) != self.num_classes):
            raise ValueError('Class weights must follow CLASSES')
        if len(self.regression_weights) != 8 or min(self.regression_weights) <= 0:
            raise ValueError('Eight positive regression weights required')
        if (not all(math.isfinite(w) and w>0 for w in (*self.class_positive_weights,*self.class_regression_weights))
                or not 0 < self.gaussian_overlap < 1):
            raise ValueError('Invalid positive weights or Gaussian overlap')
        if (not math.isfinite(self.quality_score_power) or not 0 <= self.quality_score_power <= 1
                or self.score_peak_mode not in ('heatmap','fused')):
            raise ValueError('Invalid quality score power or peak mode')
        if not 0 <= self.nms_iou_threshold <= 1 or self.max_detections < 1:
            raise ValueError('Invalid NMS configuration')
        if self.backbone not in ('mobilenet_v2','resnet18'):
            raise ValueError('Invalid image backbone')
        if self.bev_normalization not in ('batchnorm','frozen_batchnorm'):
            raise ValueError('Invalid BEV normalization')
        if self.classification_mode not in ('heatmap','objectness_softmax'):
            raise ValueError('Invalid classification mode')
        if not math.isfinite(self.foreground_label_smoothing) or not 0<=self.foreground_label_smoothing<1:
            raise ValueError('Invalid foreground label smoothing')
        if (self.loss_normalization not in ('batch_objects','fixed_train_mean') or
                not math.isfinite(self.mean_training_objects) or self.mean_training_objects<=0):
            raise ValueError('Invalid loss normalization')
        if len(self.size_reference)!=3 or not all(math.isfinite(v) and v>0 for v in self.size_reference):
            raise ValueError('Invalid shared size reference')
        if self.heading_encoding not in ('axis_direction','binned_residual'):
            raise ValueError('Invalid heading encoding')
        if (self.heading_bins not in (0,2,4) or
                (self.heading_bins and self.heading_bins!=(2 if self.heading_encoding=='axis_direction' else 4)) or
                self.size_encoding not in ('absolute','class_relative','shared_relative')):
            raise ValueError('Invalid heading bins or size encoding')
        if (len(self.class_size_priors)!=len(CLASSES) or
                any(len(p)!=3 or not all(math.isfinite(v) and v>0 for v in p) for p in self.class_size_priors)):
            raise ValueError('Six finite positive L/W/H class priors required')
        if len(self.training_class_counts)!=len(CLASSES) or any(n<0 for n in self.training_class_counts):
            raise ValueError('Invalid training class counts')
        if self.sampling_mode not in ('frame_presence','object_frequency'):
            raise ValueError('Invalid sampling mode')
        if not all(math.isfinite(v) and v>=0 for v in (self.heading_loss_weight,self.center_classification_weight,self.heavy_classification_weight)):
            raise ValueError('Invalid auxiliary loss weights')
        return self

    @classmethod
    def from_checkpoint(cls, state):
        if 'cfg' not in state or state['cfg'].get('architecture_version') != 2:
            raise ValueError('Checkpoint predates RayBEV v2; start fresh with pretrained image weights')
        values=dict(state['cfg'])
        values.setdefault('backbone','mobilenet_v2')
        values.setdefault('heading_encoding','binned_residual')
        values.setdefault('bev_normalization','batchnorm')
        values.setdefault('classification_mode','heatmap')
        values.setdefault('foreground_label_smoothing',0.)
        values.setdefault('heavy_classification_weight',0.)
        values.setdefault('loss_normalization','batch_objects')
        values.setdefault('mean_training_objects',1.)
        values.setdefault('size_reference',(4.5,1.9,1.6))
        # Historical v2 weights must keep their original losses and score rule.
        values.setdefault('class_regression_weights',(1.0,)*len(CLASSES))
        values.setdefault('quality_score_power',1.0)
        values.setdefault('score_peak_mode','heatmap')
        for key,value in dict(heading_bins=0,heading_loss_weight=0.,normalize_yaw_loss=True,
                class_balanced_losses=False,center_classification_weight=0.,size_encoding='absolute',
                class_size_priors=((1.,1.,1.),)*len(CLASSES),training_class_counts=(0,)*len(CLASSES),
                sampling_mode='frame_presence',single_label=False).items():
            values.setdefault(key,value)
        # JSON-based tooling may serialize tuple fields as lists.
        for key in ('heights','class_positive_weights','class_regression_weights','regression_weights','training_class_counts','size_reference'):
            if key in values:values[key]=tuple(values[key])
        if 'class_size_priors' in values:values['class_size_priors']=tuple(tuple(p) for p in values['class_size_priors'])
        return cls(**values).validate()

    def downscale(self, factor: int):
        """Small smoke-test model with same FOV, height planes, and BEV coverage.

        For the default 800x640 configuration factor=5 produces 160x128.
        """
        if factor <= 0 or self.image_width % factor or self.image_height % factor:
            raise ValueError('Downscale must divide the image dimensions exactly')
        # Smoke tests may use non-integral 384/factor; preserve exact physical bounds.
        cells = max(8, round(self.bev_height / factor))
        if self.bev_height != self.bev_width:
            raise ValueError('Smoke downscale expects square BEV')
        return replace(self, image_width=self.image_width // factor,
                       image_height=self.image_height // factor,
                       bev_resolution=(self.bev_x_max-self.bev_x_min)/cells).validate()
