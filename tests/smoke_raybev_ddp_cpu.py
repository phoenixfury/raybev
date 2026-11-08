"""Check real RayBEVLite model backward with 2-rank CPU DDP (no GPU claims)."""
import os
from pathlib import Path
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raybev import ModelConfig,RayBEVLite
from raybev.detection import make_targets,detection_loss

def main():
    rank=int(os.environ['RANK'])
    dist.init_process_group('gloo')
    try:
        torch.manual_seed(21)
        cfg=ModelConfig().downscale(5)
        model=RayBEVLite(cfg,str(Path(__file__).resolve().parents[1]/'configs/calibration_baselink.json'))
        ddp=DDP(model,broadcast_buffers=False,find_unused_parameters=False)
        x=torch.rand(1,4,3,128,160)
        for step in range(2):
            ddp.zero_grad(set_to_none=True)
            box=[4.,1.,1.6,10.,2.8,3.5,.1,5.] if step==0 else [4.,1.,1.3,7.,2.5,2.8,.1,3.]
            gt=[torch.tensor([box]) if (rank+step)%2==0 else torch.empty(0,8)]
            output=ddp(x)
            loss,_=detection_loss(output,make_targets(gt,cfg,return_class_ids=True),cfg=cfg)
            loss.backward()
            assert torch.isfinite(loss)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        if rank==0: print('Real RayBEVLite 2-rank DDP CPU weighted Truck/Bus + empty-rank backward PASSED',flush=True)
    finally:
        dist.destroy_process_group()
if __name__=='__main__':main()
