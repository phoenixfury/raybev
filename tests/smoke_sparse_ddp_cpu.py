"""Two-rank sparse DDP backward: empty/ignored rank, DN and repeated optimizer steps."""
import os
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite
from raybev.sparse_loss import SparseCriterion


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--architecture-version',type=int,choices=[3,4],default=3)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.manual_seed(34)
    rank=int(os.environ['RANK']);dist.init_process_group('gloo')
    try:
        cfg=SparseConfig(architecture_version=args.architecture_version,image_width=160,image_height=128,queries=12,stages=4,channels=32,ffn_channels=64,max_detections=12)
        model=YOLOROSALite(cfg).train();wrapped=DDP(model,broadcast_buffers=False,find_unused_parameters=False)
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);criterion=SparseCriterion(cfg)
        images=torch.randn(1,4,3,128,160)
        for i in range(3):
            gt=torch.tensor([[10.,2.,1.,8.,3.,3.,.5,3.],[12.,-3.,1.,12.,3.,4.,-.4,5.]])
            if rank:gt=torch.empty(0,8) if i!=1 else torch.tensor([[10.,2.,1.,8.,3.,3.,.5,-1.]])
            output=wrapped(images,[gt]);loss,_=criterion(output,[gt]);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step();optimizer.zero_grad(set_to_none=True)
        value=next(model.parameters()).detach().flatten()[:8].clone();values=[torch.zeros_like(value) for _ in range(dist.get_world_size())]
        dist.all_gather(values,value)
        assert all(torch.equal(value,other) for other in values)
        if rank==0:print('PASS sparse DDP: three iterations, empty/ignored rank, DN isolation and synchronized weights',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
