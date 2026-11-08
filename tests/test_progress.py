from __future__ import annotations
import importlib.util
from pathlib import Path


def load_training_module():
    path = Path(__file__).resolve().parents[1] / 'scripts' / 'train_combined_rosa.py'
    spec = importlib.util.spec_from_file_location('train_combined_rosa_progress_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_progress_emits_start_batch_finish(capsys):
    module = load_training_module()
    progress = module.BatchProgress('train', 1, 3, 2, mode='off', log_every=2)
    progress.update(2, {'loss': '1.2000', 'lr': '2e-4'})
    progress.update(2, {'loss': '1.0000', 'lr': '2e-4'})
    progress.close()
    text = capsys.readouterr().out
    assert 'train starting: 2 batches' in text
    assert '1/2 (50.0%)' in text
    assert '2/2 (100.0%)' in text
    assert '4 samples' in text
    assert 'images/s=' in text


def test_batch_progress_rejects_zero_interval():
    module = load_training_module()
    import pytest
    with pytest.raises(ValueError):
        module.BatchProgress('val', 1, 2, 5, mode='off', log_every=0)
