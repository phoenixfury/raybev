"""Native Q100/L4/C64 pretrained real-data AMP smoke; not an accuracy benchmark."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from raybev.config import DEFAULT_CALIBRATION
from raybev.sparse_config import SparseConfig
from raybev.sparse_model import YOLOROSALite,decode_sparse
from raybev.sparse_loss import SparseCriterion,SparseYawTeacher
from raybev.sparse_training import fit_sparse_anchors
from raybev.rosa_dataset import RosaCombinedDataset
from raybev.training import build_optimizer,calibration_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-output',type=Path)
    a=p.parse_args()
    if a.output.exists() or (a.export_output and a.export_output.exists()):raise FileExistsError('Choose new smoke/export outputs')
    if not torch.cuda.is_available():raise RuntimeError('Native AMP smoke requires CUDA')
    torch.set_num_threads(2);torch.manual_seed(3407)
    cfg=SparseConfig();ds=RosaCombinedDataset(a.root,str(Path('datasets/heavy_review_v4/train.pkl').resolve()),cfg=cfg,calibration=DEFAULT_CALIBRATION)
    cfg,anchors=fit_sparse_anchors(ds,cfg)
    teacher=SparseYawTeacher('teacher/raybev_heading_teacher_train_reviewv4.json',ds,DEFAULT_CALIBRATION)
    model=YOLOROSALite(cfg,anchors=anchors,pretrained=True).cuda().train()
    optimizer=build_optimizer(model,2e-4,.001,.1);criterion=SparseCriterion(cfg,teacher)
    scaler=torch.amp.GradScaler('cuda')
    report=dict(type='native_sparse_structural_smoke_NOT_accuracy',cfg=cfg.to_dict(),teacher_sha256=teacher.sha256,
        calibration_sha256=calibration_sha256(DEFAULT_CALIBRATION),backbone_initialization=model.image_encoder.initialization,
        parameters=sum(p.numel() for p in model.parameters()),steps=[],amp_backoffs=[])
    for sequence,uid in [('30_09','2303'),('17_09','370'),('17_09','420'),('16_09','1193')]:
        ids=[i for i,r in enumerate(ds.rows) if str(r['metainfo']['sequence_id'])==sequence and uid in set(map(str,r['annos']['gt_ids']))]
        index=ids[len(ids)//2];images,boxes,token=ds[index];images=images[None].cuda();target=[boxes.cuda()]
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda'):
                output=model(images,target);loss,parts=criterion(output,target,[token])
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<scale:
                report['amp_backoffs'].append(dict(index=index,old_scale=scale,new_scale=scaler.get_scale()));continue
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            assert torch.isfinite(norm)
            report['steps'].append(dict(index=index,sequence_id=sequence,track_id=uid,loss=float(loss),gradient_norm=float(norm),
                components={k:float(v) for k,v in parts.items()},matching_queries=cfg.queries,
                train_queries=int(output['stages'][-1]['boxes'].shape[1]),loss_scale=scaler.get_scale()))
            break
        else:raise RuntimeError('Native sparse AMP never reached finite update')
    model.eval()
    with torch.inference_mode(),torch.autocast('cuda'):
        inference=model(images);detections=decode_sparse(inference,cfg,.001)
    assert inference['stages'][-1]['boxes'].shape==(1,100,7) and not inference['dn_indices']
    assert all(torch.isfinite(p).all() and p.shape[1]==9 for p in detections)
    report['inference_shape']=list(inference['stages'][-1]['boxes'].shape)
    report['peak_cuda_allocated_MiB']=torch.cuda.max_memory_allocated()/1024**2
    report['deployment']='No TIDL/INT8 or TDA4VH latency claim'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS native sparse AMP and inference:',a.output,flush=True)
    if a.export_output:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from export_yolo_rosa_sparse import export_model,verify_export
        optimizer.zero_grad(set_to_none=True);model.cpu().eval()
        export_model(model,a.export_output);parity=verify_export(model,a.export_output)
        proof=dict(type='native_sparse_export_after_four_smoke_updates_NOT_accuracy',
                   cfg=cfg.to_dict(),parity=parity,TDA4VH_latency_measured=False)
        (a.output.parent/'yolo_rosa_sparse_native_export.json').write_text(json.dumps(proof,indent=2)+'\n')
        print('PASS native sparse ONNX parity:',a.export_output,flush=True)


if __name__=='__main__':main()
