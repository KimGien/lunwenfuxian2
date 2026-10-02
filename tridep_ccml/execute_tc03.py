"""Run only the three TC-03 inference variants, with runtime no-training guards."""
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
import sys
from unittest.mock import patch

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
from .records import ROOT, digest, read_json, require
from . import train


def snapshot():
    config = read_json(ROOT / 'configs/experiment.json')
    result = {}
    for seed in config['seeds']:
        for fold in config['folds']:
            directory = train.run_directory(ROOT, 'TC-02-37-fulltext', seed, fold)
            for name in ('checkpoint.pt', 'checkpoint.json', 'run.json', 'train_history.json', 'predictions.jsonl'):
                path = directory / name
                result[path.relative_to(ROOT).as_posix()] = digest(path)
    return result


def main():
    report_path = ROOT / 'reports/tc03_execution_guard.json'
    require(not report_path.exists(), 'Existing TC03 execution audit; refusing to overwrite')
    before = snapshot()
    report = {'started_utc': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'training_performed': False, 'optimizer_updates': 0,
              'runner_sha256': digest(ROOT / 'train.py'), 'guard_sha256': digest(__file__),
              'runtime_guards': ['train.fit_heads', 'torch.optim.Adamax', 'torch.Tensor.backward'],
              'TC02_files_before': before}
    try:
        args = ['tridep_ccml.train', '--experiment', 'TC-03-mean-37-fulltext',
                '--experiment', 'TC-03-sum-37-fulltext', '--experiment', 'TC-03-ccml-37-fulltext',
                '--execute', '--device', 'cpu', '--report', str(ROOT / 'reports/tc03_launch_preflight.json')]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sys, 'argv', args))
            for name in ('tridep_ccml.train.fit_heads', 'torch.optim.Adamax', 'torch.Tensor.backward'):
                stack.enter_context(patch(name, side_effect=RuntimeError('Training forbidden in TC03: ' + name)))
            train.main()
        after = snapshot()
        require(before == after, 'TC02 artifacts changed during inference')
        report.update(passed=True, TC02_files_after=after, inherited_files_unchanged=True,
                      completed_inference_jobs=45)
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        with report_path.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


if __name__ == '__main__':
    main()
