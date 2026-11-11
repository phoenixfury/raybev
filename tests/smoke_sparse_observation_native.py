"""Two real-data optimizer checks across eight bags; not production training."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev.config import DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.sparse_loss import SparseCriterion
from raybev.sparse_extent import VerifiedExtents
from raybev.sparse_training import fit_sparse_anchors
from raybev.rosa_dataset import RosaCombinedDataset


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=4)
    a=p.parse_args();torch.set_num_threads(2);torch.manual_seed(3407)
    if a.output.exists():raise FileExistsError(a.output)
    cfg=SparseConfig(architecture_version=5,queries=225,max_detections=225,channels=96,ffn_channels=192,
        stride4=True,observation_conditioning=True,proposal_queries=64,facing_cosine_prior=0.,extent_weight=1.)
    ds=RosaCombinedDataset(a.root,a.root/'annotations/fisheye3dod_infos_train.pkl',cfg=cfg,
        calibration=DEFAULT_CALIBRATION,strict_size=False)
    cfg,anchors=fit_sparse_anchors(ds,cfg,iterations=5)
    extents=VerifiedExtents('datasets/combined_8bags/train_yolo_extents_v1.json',ds,DEFAULT_CALIBRATION)
    sequences=sorted({str(r['metainfo']['sequence_id']) for r in ds.rows})
    assert len(sequences)==8
    ids=[]
    for sequence in sequences:
        choices=[i for i,r in enumerate(ds.rows) if str(r['metainfo']['sequence_id'])==sequence and str(r['token']) in extents.frames]
        # Exercise long-vehicle extent gradients when available.
        heavy=[i for i in choices if any(int(v['gt'][7]) in (3,5) for v in extents.frames[str(ds.rows[i]['token'])])]
        ids.append((heavy or choices)[0])
    model=YOLOROSALite(cfg,anchors=anchors,pretrained=True).cuda().train()
    criterion=SparseCriterion(cfg,projector=model.projector,extents=extents)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    report=dict(type='eight_bag_native_AMP_structural_smoke_NOT_accuracy',cfg=cfg.to_dict(),
        sequences=sequences,train_counts=ds.gt_counts,initialization=model.image_encoder.initialization,
        extent_sha256=extents.sha256,steps=[])
    torch.cuda.reset_peak_memory_stats()
    for start in range(0,len(ids),a.batch_size):
        samples=[ds[i] for i in ids[start:start+a.batch_size]]
        images=torch.stack([s[0] for s in samples]).cuda();targets=[s[1].cuda() for s in samples];tokens=[s[2] for s in samples]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda'):
            output=model(images,targets);loss,parts=criterion(output,targets,tokens)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        assert torch.isfinite(loss)
        for name,param in model.named_parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all(),name
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
        scaler.step(optimizer);scaler.update()
        report['steps'].append(dict(tokens=tokens,loss=float(loss),gradient_norm=float(norm),
            components={k:float(v) for k,v in parts.items()}))
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda'):
        output=model(images[:1]);pred=decode_sparse(output,cfg,.01)
    assert output['stages'][-1]['boxes'].shape==(1,225,7)
    assert not output['dn_indices'] and all(torch.isfinite(p).all() for p in pred)
    report['peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS native eight-bag AMP, extents and inference:',a.output,flush=True)


if __name__=='__main__':main()
