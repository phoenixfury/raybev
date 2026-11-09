"""Inspect launcher arguments without starting a real training process."""
import os
from pathlib import Path
import subprocess

import pytest


PROJECT=Path(__file__).parents[1]
LAUNCHER=PROJECT/'scripts/launch_v5_yolo26n.sh'


@pytest.fixture
def launch_environment(tmp_path):
    commands=tmp_path/'bin';commands.mkdir()
    # The real launcher still runs; intercept only validation/training processes.
    for name,body in [('python','exit 0'),('torchrun','printf "%s\\n" "$@"')]:
        path=commands/name;path.write_text('#!/bin/sh\n'+body+'\n');path.chmod(0o755)
    weights=tmp_path/'yolo26n.pt';weights.write_bytes(b'argument-test-only')
    return dict(os.environ,PATH=str(commands)+os.pathsep+os.environ['PATH'],
                YOLO26_WEIGHTS=str(weights),ROOT=str(tmp_path/'data'))


def test_new_launcher_selects_pretrained_yolo_and_isolates_output(launch_environment):
    env=launch_environment.copy()
    for key in ('OUT','BATCH_SIZE','ACCUM_STEPS','EPOCHS','LR'):env.pop(key,None)
    result=subprocess.run(['bash',str(LAUNCHER)],env=env,text=True,capture_output=True,check=True)
    args=result.stdout.splitlines()
    index=max(i for i,arg in enumerate(args) if arg=='--backbone')
    assert args[index+1]=='yolo26n'
    assert args[args.index('--backbone-weights')+1]==env['YOLO26_WEIGHTS']
    assert args[args.index('--out')+1]=='work_dirs/raybev_v5_yolo26n_48m_clidar_2x3080ti'
    assert args[args.index('--batch-size')+1]=='4'
    assert args[args.index('--accum-steps')+1]=='1'
    assert args[args.index('--freeze-image-stages')+1]=='none'
    assert args[args.index('--train-ann')+1]==str(PROJECT/'datasets/heavy_review_v1/train.pkl')
    assert '--init-from' not in args and '--resume' not in args


@pytest.mark.parametrize('argument',['--init-from=old.pt','--no-pretrained','--backbone=resnet18','--backbone-weights=wrong.pt'])
def test_launcher_rejects_incompatible_or_random_initialization(launch_environment,argument):
    result=subprocess.run(['bash',str(LAUNCHER),argument],env=launch_environment,capture_output=True,text=True)
    assert result.returncode==2 and 'fixes the pretrained backbone' in result.stderr


def test_missing_pretrained_weights_fail_fresh_but_do_not_block_resume(launch_environment):
    env=launch_environment.copy();env['YOLO26_WEIGHTS']+='.missing'
    fresh=subprocess.run(['bash',str(LAUNCHER)],env=env,capture_output=True,text=True)
    assert fresh.returncode==1 and 'Missing YOLO26n weights' in fresh.stderr
    resumed=subprocess.run(['bash',str(LAUNCHER),'--resume','yolo_run/last.pt'],env=env,capture_output=True,text=True,check=True)
    assert '--resume\nyolo_run/last.pt' in resumed.stdout
