"""Two-process CPU/gloo runtime smoke for distributed aggregation semantics.

Run: torchrun --standalone --nproc-per-node=2 tests/smoke_distributed_runtime.py
"""
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('train_ddp',HERE/'scripts/train_combined_rosa_ddp.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    rank=int(os.environ['RANK'])
    world=int(os.environ['WORLD_SIZE'])
    dist.init_process_group('gloo')
    try:
        torch.manual_seed(3407)
        model=DDP(torch.nn.Linear(5,2))
        opt=torch.optim.SGD(model.parameters(),lr=.01)
        opt.zero_grad(set_to_none=True)
        for i in range(3):
            x=torch.randn(3,5)
            ctx=model.no_sync() if i<2 else __import__('contextlib').nullcontext()
            with ctx:
                (model(x).square().mean()/3).backward()
        opt.step()
        v=torch.tensor([float(sum(p.sum() for p in model.parameters()))])
        vs=[torch.zeros_like(v) for _ in range(world)]
        dist.all_gather(vs,v)
        assert all(torch.allclose(a,vs[0]) for a in vs)

        valid_indices=list(module.ExactEvalSampler(range(7),rank,world))
        local=[(j,np.full((1,9),float(j)),np.zeros((0,8))) for j in valid_indices]
        gathered=[None]*world
        dist.all_gather_object(gathered,local)
        if rank==0:
            preds,gts=module.combine_exact_predictions(gathered,7)
            assert [int(p[0,0]) for p in preds] == list(range(7))
            print('Distributed smoke PASSED: 2 ranks, accumulated gradients, exact validation gather')
    finally:
        dist.destroy_process_group()

if __name__=='__main__':main()
