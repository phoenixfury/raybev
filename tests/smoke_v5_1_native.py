"""Four real-image AMP recovery updates; optional native ONNX parity.

Structural verification ONLY: not retraining or an accuracy benchmark. Never
writes a production checkpoint. Run from the project directory with --root.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raybev import ModelConfig,RayBEVLite
from raybev.config import DEFAULT_CALIBRATION
from raybev.detection import make_targets,detection_loss,decode_detections
from raybev.heading_supervision import HeadingSupervision
from raybev.rosa_dataset import RosaCombinedDataset
from raybev.training import fit_training_config,initialize_recovery,configure_image_training,build_optimizer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--export-output',type=Path)
    p.add_argument('--init-from',default='work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti/best_heading.pt')
    p.add_argument('--revision',type=Path,default=Path('datasets/heavy_review_v4'))
    p.add_argument('--heading-teacher',default='teacher/raybev_heading_teacher_train_reviewv4.json')
    a=p.parse_args()
    for path in (a.output,a.export_output):
        if path and path.exists():raise FileExistsError('Choose new output; refusing smoke-report overwrite')
    if not torch.cuda.is_available():raise RuntimeError('Native AMP smoke requires CUDA')
    torch.set_num_threads(2);torch.manual_seed(3407)
    cfg=ModelConfig(backbone='yolo26n',facing_loss='circular_bins',heading_bins=8)
    dataset=RosaCombinedDataset(a.root,str((a.revision/'train.pkl').resolve()),calibration=DEFAULT_CALIBRATION,cfg=cfg)
    cfg=fit_training_config(cfg,dataset.boxes)
    options=SimpleNamespace(init_from=a.init_from,resume=None,box_frame='CLIDAR',z_reference='center',
        calibration=DEFAULT_CALIBRATION,lidar_calibration=DEFAULT_CALIBRATION,
        heading_teacher=a.heading_teacher,heading_teacher_weight=.2,track_heading_weight=.1)
    supervision=HeadingSupervision(dataset,cfg,options)
    model=RayBEVLite(cfg,DEFAULT_CALIBRATION).cuda()
    recovery=initialize_recovery(model,cfg,options)
    model.train();configure_image_training(model,'early')
    def frozen_state():
        return {k:v.detach().cpu().clone() for k,v in model.image_encoder.state_dict().items()
                if k.startswith(('stage8.','stage16.'))}
    frozen=frozen_state();optimizer=build_optimizer(model,1e-4,.01,.1)
    scaler=torch.cuda.amp.GradScaler()
    report=dict(type='four_step_native_structural_smoke_NOT_accuracy',cfg=vars(cfg),
                warm_start=recovery,steps=[],amp_backoffs=[])
    # Corrected cargo Truck, true tram Bus, ambulance Van, uncertain ignored GT.
    selected=[('30_09','2303'),('17_09','370'),('17_09','420'),('16_09','1193')]
    for sequence,uid in selected:
        indices=[i for i,row in enumerate(dataset.rows) if str(row['metainfo']['sequence_id'])==sequence
                 and uid in {str(v) for v in row['annos']['gt_ids']}]
        if not indices:raise ValueError(f'Missing real TRAIN track {sequence}/{uid}')
        index=indices[len(indices)//2]
        images,boxes,token=dataset[index];images=images[None].cuda()
        targets=make_targets([boxes],cfg,device='cuda',return_class_ids=True)
        for attempt in range(9):
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                output=model(images)
                loss,parts=detection_loss(output,targets,cfg=cfg)
                auxiliary,aux=supervision.loss(output,targets,[token]);loss=loss+auxiliary
            assert torch.isfinite(loss)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            if scaler.get_scale()<old_scale:
                report['amp_backoffs'].append(dict(index=index,old_scale=old_scale,new_scale=scaler.get_scale()))
                continue
            assert torch.isfinite(norm) and all(torch.isfinite(v).all() for v in output)
            with torch.no_grad():
                decoded=decode_detections(output,cfg,score_threshold=.01)
            assert all(torch.isfinite(v).all() and v.shape[1]==9 for v in decoded)
            report['steps'].append(dict(sequence_id=sequence,track_id=uid,index=index,token=token,
                loss=float(loss),gradient_norm_before_clip=float(norm),loss_scale=scaler.get_scale(),
                teacher_centers=aux['teacher_centers'],track_centers=aux['track_centers'],
                ignored_boxes=int((boxes[:,7]<0).sum()),heading_shape=list(output[3].shape)))
            break
        else:raise RuntimeError('AMP never reached a finite update')
    after=frozen_state()
    assert all(torch.equal(v,after[k]) for k,v in frozen.items())
    report['frozen_weights_and_BN_unchanged']=True
    report['peak_cuda_allocated_MiB']=torch.cuda.max_memory_allocated()/1024**2
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS four real-image AMP updates:',a.output,flush=True)
    if a.export_output:
        optimizer.zero_grad(set_to_none=True);model.cpu().eval()
        import onnx
        import onnxruntime as ort
        with tempfile.TemporaryDirectory(prefix='raybev-v51-export-') as directory:
            temporary=Path(directory);checkpoint=temporary/'smoke.pt';exported=temporary/'export'
            torch.save(dict(cfg=vars(cfg),state_dict=model.state_dict()),checkpoint)
            spec=importlib.util.spec_from_file_location('native_export',Path(__file__).parents[1]/'scripts/export_onnx.py')
            export=importlib.util.module_from_spec(spec);spec.loader.exec_module(export)
            sys.argv=['export','--checkpoint',str(checkpoint),'--output',str(exported)]
            export.main()
            metadata=json.loads((exported/'metadata.json').read_text())
            result=dict(type='native_shape_recovery_smoke_export_NOT_accuracy',cfg=vars(cfg),
                        metadata_heading={k:v for k,v in metadata.items() if k.startswith(('heading_','facing_'))})
            ort_options=ort.SessionOptions();ort_options.intra_op_num_threads=2
            examples=[('image_encoder',model.image_encoder.forward_single,
                       (torch.rand(1,3,cfg.image_height,cfg.image_width),model.transport.image_geometry[:1])),
                      ('bev_detector',model.bev_detector,(torch.rand(1,cfg.bev_input_channels,cfg.bev_height,cfg.bev_width)*.1,))]
            for name,module,inputs in examples:
                path=exported/(name+'.onnx');graph=onnx.load(str(path));onnx.checker.check_model(graph)
                operators={node.op_type for node in graph.graph.node}
                assert not operators.intersection({'Softmax','Attention','GridSample','MatMul','LayerNormalization'})
                session=ort.InferenceSession(str(path),ort_options,providers=['CPUExecutionProvider'])
                actual=session.run(None,{i.name:x.numpy() for i,x in zip(session.get_inputs(),inputs)})
                with torch.inference_mode():expected=module(*inputs)
                errors=[]
                for value,reference in zip(actual,expected):
                    np.testing.assert_allclose(value,reference.numpy(),atol=2e-5,rtol=1e-3)
                    errors.append(float(np.max(np.abs(value-reference.numpy()))))
                result[name]=dict(operators=sorted(operators),max_absolute_errors=errors,
                                  output_shapes=[list(x.shape) for x in actual])
            a.export_output.parent.mkdir(parents=True,exist_ok=True)
            a.export_output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('PASS native learned-graph ONNX parity:',a.export_output,flush=True)


if __name__=='__main__':main()
