#!/usr/bin/env python3
"""Shape/MAC/memory inventory for a four-camera frame. No target FPS inference."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev import ModelConfig,RayBEVLite


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration',default='configs/calibration_baselink.json')
    parser.add_argument('--checkpoint')
    parser.add_argument('--output',default='model_profile.json')
    args=parser.parse_args()
    torch.set_num_threads(2)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False) if args.checkpoint else None
    cfg=ModelConfig.from_checkpoint(state) if state else ModelConfig()
    model=RayBEVLite(cfg,args.calibration).eval()
    if state:model.load_state_dict(state['state_dict'])
    macs={'image_encoder':0,'bev_detector':0}
    largest_conv_output=0
    handles=[]
    def hook(group):
        def record(module,inputs,output):
            nonlocal largest_conv_output
            macs[group]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            largest_conv_output=max(largest_conv_output,output.numel())
        return record
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook(name.split('.')[0])))
    started=time.monotonic()
    with torch.inference_mode():
        outputs=model(torch.zeros(1,4,3,cfg.image_height,cfg.image_width))
    for handle in handles:handle.remove()
    bev_elements=cfg.bev_input_channels*cfg.bev_height*cfg.bev_width
    report={'cfg':vars(cfg),'parameters':sum(p.numel() for p in model.parameters()),
            'convolution_MACs_per_four_camera_frame':macs,
            'total_convolution_GMACs':sum(macs.values())/1e9,
            'output_shapes':[list(x.shape) for x in outputs],
            'transport_output_elements':bev_elements,
            'transport_output_MiB_by_dtype':{name:bev_elements*n/2**20 for name,n in [('int8',1),('fp16',2),('fp32',4)]},
            'fixed_geometry_MiB_fp32':sum(b.numel()*b.element_size() for b in model.transport.buffers())/2**20,
            'largest_conv_output_elements_batch_four_cameras':largest_conv_output,
            'host_CPU_reference_forward_seconds':time.monotonic()-started,
            'limits':'MAC count excludes transport, resize, elementwise operations and NMS. Tensor sizes are not peak memory. CPU timing is not TDA4VH latency.',
            'TDA4VH_status':'requires matching TIDL SDK compile/offload report, INT8 accuracy evaluation and device latency/memory measurements'}
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
