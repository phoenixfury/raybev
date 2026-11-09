"""Official YOLO26n convolutional backbone, without C2PSA or the 2D head.

Ultralytics is an optional dependency. Only trusted official/local Ultralytics
pickles may be supplied as pretrained weights; RayBEV checkpoints remain plain
state dictionaries. No Ultralytics predictor/trainer runs inside RayBEV.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile

import torch
from torch import nn


ULTRALYTICS_VERSION = '8.4.182'
WEIGHTS_URL = 'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'
WEIGHTS_SHA256 = '9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1]/'weights/yolo26n.pt'

# Exact official YOLO26 definition through SPPF. Retain convolutional /8,/16,/32
# features; do not construct/register unused attention/PAN/Detect parameters.
BACKBONE = [
    [-1,1,'Conv',[64,3,2]],
    [-1,1,'Conv',[128,3,2]],
    [-1,2,'C3k2',[256,False,0.25]],
    [-1,1,'Conv',[256,3,2]],
    [-1,2,'C3k2',[512,False,0.25]],
    [-1,1,'Conv',[512,3,2]],
    [-1,2,'C3k2',[512,True]],
    [-1,1,'Conv',[1024,3,2]],
    [-1,2,'C3k2',[1024,True]],
    [-1,1,'SPPF',[1024,5,3,True]],
]
PROFILE = dict(name='yolo26n_conv_sppf',ultralytics_version=ULTRALYTICS_VERSION,
               retained_layers=list(range(10)),feature_layers=[4,6,9],
               feature_strides=[8,16,32],feature_channels=[128,128,256],
               excluded_layers='C2PSA (10), YOLO PAN neck and all 2D Detect heads (11+)',
               self_attention=False,pretrained_source=WEIGHTS_URL,
               encoder_input='ImageNet-normalized RGB; internally x*std+mean -> RGB [0,1]',
               external_mean=[.485,.456,.406],external_std=[.229,.224,.225])


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def ensure_official_weights(path=DEFAULT_WEIGHTS):
    """Download once atomically, verify pinned bytes; never silently random-init."""
    path=Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True,exist_ok=True)
        # Separate temporary names also make concurrent cache population safe.
        with tempfile.TemporaryDirectory(prefix='yolo26n-',dir=str(path.parent)) as temporary:
            download=Path(temporary)/'yolo26n.pt'
            try:
                torch.hub.download_url_to_file(WEIGHTS_URL,str(download),progress=True)
            except Exception as error:
                raise RuntimeError(f'YOLO26n download failed. Copy official yolo26n.pt to {path} '
                                   'or pass --backbone-weights; no random fallback.') from error
            if sha256(download)!=WEIGHTS_SHA256:
                raise ValueError('Official YOLO26n download SHA256 mismatch')
            download.replace(path)
    if sha256(path)!=WEIGHTS_SHA256:
        raise ValueError(f'Official YOLO26n cache SHA256 mismatch: {path}')
    return path


def build_backbone(pretrained=False,backbone_weights=None):
    """Build the same offline architecture for fresh training and strict reloads."""
    try:
        import ultralytics
        from ultralytics.nn.tasks import parse_model
        from ultralytics.nn.modules.block import C2f
        from ultralytics.utils.torch_utils import initialize_weights
    except ImportError as error:
        raise ImportError('YOLO26n requires requirements-yolo26.txt; legacy backbones do not.') from error
    if ultralytics.__version__!=ULTRALYTICS_VERSION:
        raise RuntimeError(f'YOLO26n checkpoint architecture requires ultralytics=={ULTRALYTICS_VERSION}; '
                           f'found {ultralytics.__version__}. Install requirements-yolo26.txt.')
    definition=dict(nc=80,scale='n',scales={'n':[.5,.25,1024]},
                    backbone=deepcopy(BACKBONE),head=[])
    backbone,_=parse_model(definition,ch=3,verbose=False)
    # BN eps/momentum are NOT in state_dict. Match official DetectionModel init.
    initialize_weights(backbone)
    # Ultralytics' own export uses explicit constant splits instead of chunk.
    # Numerically identical, but avoids dynamic Shape/Gather/Div slicing in ONNX.
    for module in backbone.modules():
        if isinstance(module,C2f):module.forward=module.forward_split
    initialization='YOLO26N_random_explicit'
    if pretrained or backbone_weights:
        path=Path(backbone_weights) if backbone_weights else ensure_official_weights()
        if not path.is_file():raise FileNotFoundError(f'YOLO26n weights not found: {path}')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        network=(checkpoint.get('ema') or checkpoint.get('model')) if isinstance(checkpoint,dict) else None
        yaml=getattr(network,'yaml',{})
        head=yaml.get('head',[])
        if (not isinstance(network,nn.Module) or yaml.get('scale')!='n' or
                yaml.get('backbone',[])[:10]!=BACKBONE or yaml.get('reg_max')!=1 or
                yaml.get('end2end') is not True or yaml.get('nc')!=80 or
                not head or head[-1][2]!='Detect'):
            raise ValueError('Expected unfused official COCO YOLO26n detection weights (not YOLO11n, s, cls, seg or RayBEV)')
        # Strictly load EVERY retained parameter and running statistic. Shape/key
        # mismatches or fused checkpoints are errors, never partial success.
        retained=nn.Sequential(*list(network.float().model.children())[:10])
        backbone.load_state_dict(retained.state_dict(),strict=True)
        digest=sha256(path)
        source='official_COCO' if digest==WEIGHTS_SHA256 else 'trusted_local_architecture_verified'
        initialization=(f'YOLO26N_{source}_conv_sppf; path={path}; sha256={digest}; '
                        f'ultralytics={ULTRALYTICS_VERSION}; layers=0..9; taps=4,6,9; no_C2PSA_no_2D_head')
    return backbone,initialization
