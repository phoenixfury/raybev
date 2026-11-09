"""ROSA teacher inference only; same learned model/geometry as epoch-6 ROSA.

Use the validated comparison run's runtime settings. Do not run the official
evaluation metric on TRAIN or load unused evaluation GT into each batch.
"""
_base_ = ['../../ROSA/configs/fisheye3dod/rosa_vov99_baseline_adapt_8e.py']
model = dict(data_preprocessor=dict(non_blocking=True))
test_dataloader = dict(dataset=dict(load_eval_anns=False))
test_evaluator = []  # ROSA run.py adds DumpResults through --dump.
env_cfg = dict(cudnn_benchmark=True)
randomness = dict(seed=3407,deterministic=False)
custom_hooks = []
default_hooks = dict(checkpoint=None,logger=dict(interval=25))
