#!/usr/bin/env python3
"""Paired synthetic-input latency ablations; NOT an accuracy measurement."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite
from raybev.sparse_loss import SparseCriterion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda');p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--backward',action='store_true')
    p.add_argument('--batch-size',type=int,default=1);p.add_argument('--amp',action='store_true')
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(3407)
    if a.output.exists():raise FileExistsError('Choose a new benchmark report')
    if a.iterations<1 or a.batch_size<1:raise ValueError('Positive iterations/batch required')
    base=SparseConfig(architecture_version=5,queries=225,max_detections=225,facing_cosine_prior=0.)
    variants=[('q225_c64',base),('q225_c64_proposals',replace(base,proposal_queries=64)),
        ('q225_c96_proposals',replace(base,proposal_queries=64,channels=96,ffn_channels=192)),
        ('q225_c96_p2_proposals',replace(base,proposal_queries=64,channels=96,ffn_channels=192,stride4=True)),
        ('q225_c96_p2_observation',replace(base,proposal_queries=64,channels=96,ffn_channels=192,stride4=True,observation_conditioning=True))]
    gpu=str(a.device).startswith('cuda');results={}
    def sync():
        if gpu:torch.cuda.synchronize()
    images=torch.randn(a.batch_size,4,3,base.image_height,base.image_width,device=a.device)
    for name,cfg in variants:
        model=YOLOROSALite(cfg).to(a.device).eval();samples=[]
        with torch.inference_mode():
            for i in range(a.iterations+3):
                sync();start=time.perf_counter()
                with torch.autocast(device_type='cuda' if gpu else 'cpu',enabled=a.amp):model(images)
                sync()
                if i>=3:samples.append(1000*(time.perf_counter()-start))
        results[name]=dict(batch_latency_ms=sum(samples)/len(samples),parameters=sum(p.numel() for p in model.parameters()),cfg=cfg.to_dict())
        if a.backward:
            model.train()
            gt=[images.new_tensor([[12.,2.,1.5,8.,2.5,3.,.3,5.]]) for _ in range(a.batch_size)]
            if gpu:torch.cuda.reset_peak_memory_stats()
            with torch.autocast(device_type='cuda' if gpu else 'cpu',enabled=a.amp):
                out=model(images,gt);loss,parts=SparseCriterion(cfg)(out,gt)
            loss.backward();sync()
            finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            if not finite:raise FloatingPointError('Missing or nonfinite training gradient in '+name)
            results[name].update(training_loss=float(loss.detach()),training_peak_allocated_mb=torch.cuda.max_memory_allocated()/2**20 if gpu else None)
            del out,loss,parts
        print(name,results[name]['batch_latency_ms'],'ms/batch',flush=True)
        del model
        if gpu:torch.cuda.empty_cache()
    report=dict(device=torch.cuda.get_device_name() if gpu else 'CPU',torch_version=torch.__version__,batch_size=a.batch_size,amp=a.amp,
        scope='Four native 800x640 cameras; encoder+proposals+decoder only. Random weights/input. No IO/NMS. Not TDA4VH or accuracy.',results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
