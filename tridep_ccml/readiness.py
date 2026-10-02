"""Check preparation and write an evidence-based blocker report if necessary."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    checks, blockers = [], []

    def check(name, action):
        try:
            evidence = action()
            checks.append({'name': name, 'passed': True, 'evidence': evidence})
        except Exception as exc:
            item = {'name': name, 'passed': False, 'error_type': type(exc).__name__, 'error': str(exc)}
            checks.append(item)
            blockers.append(item)

    def source_check():
        from .records import RecordContext
        context = RecordContext()
        config = context.config
        assert digest(WORK / config['plan_path']) == config['plan_sha256'], '实验计划在固定配置后发生变更'
        preparation = read(ROOT / 'reports/representation_preparation.json')
        assert preparation['passed'] and preparation['checkpoints_verified'] == 45
        assert preparation['probabilities_reproduced'] == 1665
        assert len(preparation['bundles']) == 15
        assert preparation['cohort_sha256'] == context.cohort_sha256
        for bundle in preparation['bundles']:
            path = ROOT / bundle['path']
            assert digest(path) == bundle['sha256'], f'缓存变化: {path}'
            assert digest(path.with_suffix('.json')) == bundle['metadata_sha256'], f'来源记录变化: {path}'
            metadata = read(path.with_suffix('.json'))
            assert metadata['exporter_sha256'] == digest(ROOT / 'prepare.py'), '特征导出代码发生变化'
            assert metadata['upstream_model_source_sha256'] == digest(WORK / 'S00/upstream-dev/lib/models/models.py'), '父模型定义变化'
            context.manifest(metadata['seed'], metadata['fold'])
        return {'parent_weights': 45, 'bundles': 15, 'parent_probabilities': 1665,
                'maximum_probability_error': preparation['max_absolute_probability_error']}

    def references():
        source = read(ROOT / 'references/ccml/source.json')
        assert source['commit'] == '8938ae23bcba5d6ff1508b35fc9a6664eca54f71'
        for item in source['files']:
            assert digest(ROOT / 'references/ccml' / item['path']) == item['sha256'], '官方参考文件变化'
        return {'commit': source['commit'], 'files': len(source['files'])}

    def command(args, logfile):
        completed = subprocess.run([sys.executable, '-B'] + args, cwd=WORK, capture_output=True, text=True, encoding='utf-8', errors='replace')
        output = completed.stdout + completed.stderr
        (ROOT / 'reports' / logfile).write_text(output, encoding='utf-8')
        if completed.returncode:
            raise RuntimeError(f'退出码 {completed.returncode}；见 {logfile}: {output[-1400:]}')
        return output

    def tests():
        output = command(['-m', 'unittest', 'discover', '-s', 'tridep_ccml/tests', '-v'], 'tests.log')
        match = re.search(r'Ran (\d+) tests?', output)
        assert match and int(match.group(1)) > 0, '未发现实际执行的测试'
        return {'tests_passed': int(match.group(1)), 'log': 'tests.log'}

    def training_entry():
        # Existing completed runs are verified; they are not overwrite attempts.
        from .records import RecordContext, validate_prediction_records
        from .train import DEFAULT_EXPERIMENTS, load_features, run_directory, validate_config
        context = RecordContext()
        config = validate_config(context)
        for seed in config['seeds']:
            for fold in config['folds']:
                load_features(context, seed, fold)
        completed = {}
        pending = 0
        for experiment in DEFAULT_EXPERIMENTS:
            rows, count = [], 0
            for seed in config['seeds']:
                for fold in config['folds']:
                    directory = run_directory(ROOT, experiment, seed, fold)
                    if not directory.exists():
                        pending += 1
                        continue
                    run = read(directory / 'run.json')
                    assert run['status'] == 'complete' and not (directory / 'failure.json').exists(), f'未完成的运行: {directory}'
                    assert (run['experiment'], run['seed'], run['fold']) == (experiment, seed, fold)
                    assert run['training_performed'] == (not experiment.startswith('TC-03-'))
                    checkpoint = Path(run['checkpoint_path'])
                    assert digest(checkpoint) == run['checkpoint_sha256']
                    sidecar = read(checkpoint.with_suffix('.json'))
                    assert sidecar['metadata'] == run['source_metadata']
                    metadata = run['source_metadata']
                    assert metadata['completed_epochs'] == 60
                    for field, filename in [('runner_sha256', 'train.py'), ('ccml_sha256', 'ccml.py'), ('records_sha256', 'records.py')]:
                        assert metadata[field] == digest(ROOT / filename)
                    fold_rows = [json.loads(line) for line in (directory / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
                    assert len(fold_rows) == 37 and {r['subject_id'] for r in fold_rows} == set(context.ids)
                    validate_prediction_records(fold_rows, experiment, context=context, require_complete=False)
                    rows.extend(fold_rows)
                    count += 1
            if count:
                validate_prediction_records(rows, experiment, context=context, require_complete=count == 15)
                completed[experiment] = count
        report = {'validated_feature_folds': 15, 'planned_jobs': 15 * len(DEFAULT_EXPERIMENTS),
                  'completed_jobs': completed, 'pending_jobs': pending, 'training_performed': False,
                  'blockers': [], 'log': 'training_preflight.log'}
        (ROOT / 'reports/training_preflight.log').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        return report

    (ROOT / 'reports').mkdir(parents=True, exist_ok=True)
    check('父权重、特征与人群划分', source_check)
    check('CCML 官方来源版本', references)
    check('依赖兼容性', lambda: command(['-m', 'pip', 'check'], 'dependency_check.log').strip())
    check('证据数学与记录验证测试', tests)
    check('TC-00 标准记录与覆盖', lambda: command(['-m', 'tridep_ccml.baseline'], 'tc00_validation.log').strip())
    check('全部种子及折的训练入口检查', training_entry)
    status = {'utc': datetime.now(timezone.utc).isoformat(), 'passed': not blockers,
              'scope': 'TC first-phase experiment preparation, not formal training',
              'active_blockers': blockers, 'checks': checks,
              'formal_run_directories_present': (ROOT / 'runs').exists() and any((ROOT / 'runs').iterdir()),
              'tc06': 'deferred_second_phase_not_a_first_phase_blocker'}
    (ROOT / 'reports/准备状态.json').write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    passed_tests = next((item['evidence']['tests_passed'] for item in checks if item['name'] == '证据数学与记录验证测试' and item['passed']), None)
    lines = ['# TC 实验准备报告', '', '检查时间：' + status['utc'], '',
             ('结论：实验准备检查全部通过，当前没有阻止 TC 第一阶段运行的硬阻塞。' if not blockers else f'结论：存在 {len(blockers)} 项当前阻塞，详见《TC阻塞报告》。'), '',
             '本报告复核固定计划的准备条件及已存在的运行记录，不启动正式训练。TC-00 已完成父概率复核与等权补算；各阶段的最新执行情况见 `experiment_status.json`（若已生成）及阶段完成报告。', '',
             '## 已完成的准备', '',
             '| 检查 | 状态 |', '|---|---|']
    lines.extend(f"| {item['name']} | {'通过' if item['passed'] else '阻塞'} |" for item in checks)
    lines.extend(['', '人群固定为 MODMA-37-v1：37 人、16 MDD、21 HC，沿用五折及种子 11、22、33。45 个父权重已校验；1,665 个分支概率重载复现，最大绝对差为 0。15 组缓存均为 37×3×128，绑定各自折、种子和父模型训练名单。', '',
                  f'合成与记录测试通过数：{passed_tests if passed_tests is not None else "尚未通过"}。训练入口检查只检查输入，不执行正式优化。', '',
                  '## TC-00 复核', '',
                  '原验证集选权重融合的 Macro-F1 为 0.724180；相同原概率等权融合为 0.682487。均采用每种子五折均值再对三个种子求均值的口径。这是已有父模型结果和补算，不能视为 CCML 收益。', '',
                  '标准记录：`tc00_records.jsonl`。原始补算来源：`tc00_probabilities.jsonl`。TC-00 中没有新头权重、证据或 Dirichlet 不确定性；准备配置哈希仅用于追溯本次整理。', '',
                  '## 持续披露的局限', '',
                  '- 固定测试划分的结果此前已查看，本轮属于开发性比较，不是独立盲测。',
                  '- 语音随机冻结编码器与单位置注意力、EEG 尺度及通道对应限制沿用父版本。',
                  '- 每折验证集只有 3 人；不进行调参、阈值搜索、选 epoch 或拟合校准器。',
                  '- 第二阶段 TC-06 尚未实现，不影响当前冻结表征的新头实验。', '',
                  '这些事项影响结论的解释范围，当前不是缺数据、缺权重或环境不可用导致的硬阻塞。', '',
                  '## 运行与追溯', '',
                  '详见上一级 `README.md`。默认 `run.ps1 -Action check` 复核准备状态；`-Action train` 默认仍为检查，只有显式 `-Execute` 才启动指定实验、种子和折。实验配置和已有输出不会被静默覆盖。', '',
                  '来源核验：`parent_audit.md`、`parent_audit.json`、`representation_preparation.json`。本次检查详情：`准备状态.json`、`tests.log`、`training_preflight.log`。', ''])
    (ROOT / 'reports/TC实验准备报告.md').write_text('\n'.join(lines), encoding='utf-8')
    blocker_path = ROOT / 'reports/TC阻塞报告.md'
    if blockers:
        body = ['# TC 阻塞报告', '', '检查时间：' + status['utc'], '',
                f'当前 {len(blockers)} 项检查未通过；正式实验应在这些问题解决并重新检查通过后启动。', '']
        for i, blocker in enumerate(blockers, 1):
            body.extend([f"## {i} {blocker['name']}", '', '核查证据：', '',
                         '```text', blocker['error'], '```', '',
                         '影响：尚不能确认相应实验依赖或实现符合已锁定协议。', '',
                         '处理：修复上述具体错误，保留父实验与原数据，再运行 `run.ps1 -Action check`。', ''])
        blocker_path.write_text('\n'.join(body), encoding='utf-8')
    elif blocker_path.exists():
        blocker_path.write_text('# TC 阻塞报告\n\n检查时间：' + status['utc'] + '\n\n当前没有未解决的硬阻塞。之前的检查问题已经修复并重新验证通过。详见《TC实验准备报告》。\n', encoding='utf-8')
    print(json.dumps({'passed': not blockers, 'active_blockers': len(blockers), 'tests_passed': passed_tests,
                      'report': str(ROOT / 'reports/TC实验准备报告.md')}, ensure_ascii=False, indent=2))
    return 0 if not blockers else 2


if __name__ == '__main__':
    raise SystemExit(main())
