"""Registered I/O-only recovery of the interrupted matched four-second study."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

from . import common as C

REC = C.OUT / 'recovery'
ANNEX = C.WORK / 'output/plans/TriDep_EEGPT文件写入故障恢复附表_v2_20261007.md'
PLAIN_READ = C.read
BUDGET = {
    'prior_completed_fits': 23, 'prior_completed_updates': 2760,
    'discarded_partial_fit_starts': 1, 'discarded_partial_updates': 82,
    'discarded_partial_gradient_passes': 82, 'discarded_partial_person_examples': 1066,
    'new_complete_fits': 7, 'new_updates': 840, 'new_gradient_passes': 840,
    'canonical_completed_models': 30, 'canonical_optimizer_updates': 3600,
    'canonical_gradient_passes': 3600, 'canonical_person_examples': 47880,
    'actual_fit_starts': 31, 'actual_optimizer_updates': 3682,
    'actual_training_gradient_passes': 3682, 'actual_person_examples': 48946,
    'producer_head_person_forward_calls': 2220,
    'independent_replay_head_person_forward_calls': 1110,
}


def execution_registration():
    return C.desc(REC / 'registration.json')


def check_registration():
    C.registration()  # The unchanged original method and all protected stages.
    path = C.checked(PLAIN_READ(REC / 'registration_anchor.json')['registration'])
    reg = PLAIN_READ(path)
    C.require(reg['budget'] == BUDGET and reg['model_execution_allowed'] and
              reg['original_registration'] == C.desc(C.OUT / 'registration.json') and
              reg['old_failed_action_must_remain_failed'] and
              not reg['model_math_or_hyperparameters_changed'], 'Recovery contract differs')
    for name, sha in reg['sources'].items():
        C.checked({'path': name, 'sha256': sha})
    return reg


def original_completed_metadata():
    reg = check_registration()
    prior = PLAIN_READ(C.checked(reg['prior_training_ledger']))
    C.require(prior['fits_completed'] == 23 and prior['updates_completed'] == 2760, 'Prior canonical ledger differs')
    indexed = {(m['condition'], m['seed'], m['fold']): m for m in prior['fits']}
    result = []
    for entry in reg['preserved_completed_fits']:
        complete = PLAIN_READ(C.checked(entry['complete']))
        C.require(complete['artifacts'] == entry['artifacts'] and complete['passed'], 'Preserved completion manifest differs')
        for artifact in entry['artifacts']:
            C.checked(artifact)
        identity = tuple(entry[k] for k in ('condition', 'seed', 'fold'))
        folder = C.OUT / f'runs/{identity[0]}/seed_{identity[1]}/fold_{identity[2]}'
        meta = PLAIN_READ(folder / 'metadata.json')
        C.require(meta == indexed[identity], 'Preserved metadata differs from prior ledger')
        result.append(meta)
    C.require(len(result) == 23, 'Reuse exactly23 completed models')
    return result


def guard_training():
    reg = check_registration()
    receipt = PLAIN_READ(REC / 'registration_independent_verification.json')
    C.require(receipt['passed'] and receipt['mode'] == 'registration_only' and receipt['read_only'] and
              receipt['execution_registration'] == execution_registration() and
              receipt['source'] == C.desc(C.CODE / 'recovery_verify.py'), 'Current independent recovery acceptance required')
    marker = PLAIN_READ(C.OUT / 'actions/fit_all_recovery.json')
    C.require(marker['status'] == 'started' and marker['execution_registration'] == execution_registration(),
              'Persisted recovery action start required before any model operation')
    C.require(not (C.REPORT / 'metrics.json').exists() and not (C.OUT / 'evaluation').exists(),
              'Recovery was not authorized from unreleased training only')
    return reg


def prepare():
    C.require(not (REC / 'registration.json').exists(), 'Recovery already registered; preserve it')
    original = C.registration()
    snapshot = PLAIN_READ(REC / 'failure_snapshot.json')
    archive = PLAIN_READ(REC / 'archive_record.json')
    C.require(PLAIN_READ(C.OUT / 'actions/fit_all.json')['status'] == 'failed' and
              C.desc(C.OUT / 'actions/fit_all.json') == snapshot['failed_action'], 'Original failure changed')
    C.require(snapshot['completed_models'] == 23 and snapshot['partial_updates'] == 82 and
              snapshot['next_step_not_forwarded'] == 83 and snapshot['no_evaluation_or_scores_released'] and
              archive['all_five_files_byte_identical'], 'Failure boundary/archive differs')
    from . import recovery_fit
    C.require([(e['condition'], e['seed'], e['fold']) for e in snapshot['preserved_completed_fits']] ==
              recovery_fit.completed_identities(), 'Preserved23 order differs')
    pending = [{'condition': c, 'seed': s, 'fold': f} for c, s, f in recovery_fit.pending_identities()]
    C.require(len(pending) == 7 and all(not (C.OUT / f'runs/{e["condition"]}/seed_{e["seed"]}/fold_{e["fold"]}').exists()
              for e in pending), 'Pending seven paths are not empty after failure preservation')
    C.require((C.CODE / 'recovery_verify.py').is_file() and ANNEX.is_file(), 'Finish recovery sources first')
    source_paths = [C.OUT / 'registration.json', C.OUT / 'registration_anchor.json',
                    C.OUT / 'features_manifest.json', C.OUT / 'feature_ledger.json',
                    C.REPORT / 'input_independent_verification.json',
                    C.OUT / 'actions/features.json', C.OUT / 'actions/fit_all.json',
                    C.OUT / 'pause_record_20261007.json', ANNEX]
    for entry in snapshot['preserved_completed_fits']:
        source_paths.append(C.WORK / entry['complete']['path'])
        source_paths.extend(C.WORK / d['path'] for d in entry['artifacts'])
    for entry in archive['failed_partial']['artifacts']:
        source_paths.append(C.checked(entry['archived']))
    manifest = PLAIN_READ(C.OUT / 'features_manifest.json')
    for entry in manifest['records']:
        source_paths.extend([C.checked(entry['features']), C.checked(entry['sidecar'])])
    for p in REC.glob('*'):
        if p.is_file():
            source_paths.append(p)
    # Bind all newly added recovery sources and their pure tests, including this entry.
    source_paths.extend(p for p in C.CODE.glob('*recovery*.py') if p.is_file())
    sources = dict(original['sources'])
    for path in source_paths:
        source = C.desc(path)
        C.require(source['path'] not in sources or sources[source['path']] == source['sha256'], 'Conflicting recovery source')
        sources[source['path']] = source['sha256']
    for name, sha in sources.items():
        C.checked({'path': name, 'sha256': sha})
    evidence = recovery_fit.ast_evidence()
    C.require(evidence['epoch_ast_sha256']['identical'] and
              evidence['model_rng_dataset_loader_optimizer_ast_sha256']['identical'], 'Recovery changed numerical training')
    reg = {
        'experiment': C.OUT.name, 'kind': 'registered_IO_failure_recovery', 'registered_at': C.now(),
        'client_date': '2026-10-07', 'user_resume_instruction_verbatim': '继续',
        'sources': sources, 'original_registration': C.desc(C.OUT / 'registration.json'),
        'original_registration_anchor': C.desc(C.OUT / 'registration_anchor.json'),
        'prior_training_ledger': C.desc(REC / 'prior_training_ledger.json'),
        'failed_action': C.desc(C.OUT / 'actions/fit_all.json'),
        'pause_record': C.desc(C.OUT / 'pause_record_20261007.json'),
        'failure_snapshot': C.desc(REC / 'failure_snapshot.json'),
        'preserved_completed_fits': snapshot['preserved_completed_fits'],
        'failed_partial': archive['failed_partial'], 'replacement_fits': pending,
        'budget': BUDGET, 'math_ast_evidence': evidence, 'appendix': C.desc(ANNEX),
        'model_execution_allowed': True, 'model_math_or_hyperparameters_changed': False,
        'restart_from_saved_optimizer_state': False,
        'old_failed_action_must_remain_failed': True,
        'IO_only_retry': {'max_seconds': 5., 'max_attempts': 40, 'max_sleep_seconds': .1,
                          'same_already_serialized_temporary_file': True, 'model_computation_retried': False},
        'evaluation_dependency_bridge': 'Only old failed fit_all completion dependency redirects to actual new fit_all_recovery completion; original failed file remains immutable',
        'new_test_selection_or_threshold_searches': 0,
    }
    C.save(REC / 'registration.json', reg, immutable=True)
    C.save(REC / 'registration_anchor.json', {'registration': execution_registration()}, immutable=True)
    C.save(C.REPORT / 'status.json', {'status': 'recovery_registered_pending_independent_audit',
        'registration': reg['original_registration'], 'execution_registration': execution_registration(),
        'fits_completed': 23, 'actual_fit_starts': 24, 'actual_optimizer_updates': 2842,
        'metrics_released': False})
    print(f'Recovery registered:{len(sources)} sources;7 new fullfits/840 updates;actual total31 starts/3682 updates', flush=True)


def train():
    reg = check_registration()
    approval = PLAIN_READ(REC / 'registration_independent_verification.json')
    C.require(approval['passed'] and approval['execution_registration'] == execution_registration(),
              'Independent recovery registration audit required')
    C.require(approval['source'] == C.desc(C.CODE / 'recovery_verify.py'), 'Recovery registration verifier source differs')
    C.require(approval['mode'] == 'registration_only' and approval['read_only'], 'Receipt is not the readonly pretraining registration gate')
    action_path = C.OUT / 'actions/fit_all_recovery.json'
    C.require(not action_path.exists(), 'Recovery already started; preserve partial work')
    action = {'action': 'fit_all_recovery', 'status': 'started', 'started_at': C.now(),
              'registration': reg['original_registration'], 'execution_registration': execution_registration()}
    C.save(action_path, action, immutable=True)
    from .recovery_io import installed_io
    try:
        with installed_io():
            from .recovery_fit import fit_remaining
            fit_remaining()
            from .train_eval import completed_fits
            lock, folds, labels = C.context()
            completed_fits(lock, folds)
            entries = []
            for identity in reg['replacement_fits']:
                folder = C.OUT / f'runs/{identity["condition"]}/seed_{identity["seed"]}/fold_{identity["fold"]}'
                entries.append({**identity, 'metadata': C.desc(folder / 'metadata.json'), 'complete': C.desc(folder / 'complete.json')})
            C.save(REC / 'training_completion.json', {
                'passed': True, 'method_registration': reg['original_registration'],
                'execution_registration': execution_registration(), 'preserved_completed_models': 23,
                'new_completed_models': 7, 'canonical_completed_models': 30,
                'canonical_optimizer_updates': 3600, 'actual_fit_starts': 31,
                'actual_optimizer_updates': 3682, 'actual_training_gradient_passes': 3682,
                'actual_person_examples': 48946, 'new_fit_bindings': entries,
                'training_ledger': C.desc(C.OUT / 'training_ledger.json')}, immutable=True)
            action.update(status='complete', finished_at=C.now())
            C.save(action_path, action)
    except BaseException as error:
        action.update(status='failed', stopped_at=C.now(), error_type=type(error).__name__, error=str(error))
        C.save(action_path, action)
        raise
    print('All30 canonical models complete;actual31 starts/3682 updates;no test evaluation yet', flush=True)


@contextmanager
def evaluation_bridge():
    """Redirect the sole completion dependency to a real registered new record."""
    reg = check_registration()
    new_action = PLAIN_READ(C.OUT / 'actions/fit_all_recovery.json')
    C.require(new_action['status'] == 'complete' and new_action['execution_registration'] == execution_registration(),
              'Recovery action incomplete')
    old_read = C.read
    from . import run
    old_entry_read = run.read
    failed_path = (C.OUT / 'actions/fit_all.json').resolve()
    C.require(C.desc(failed_path) == reg['failed_action'] and PLAIN_READ(failed_path)['status'] == 'failed',
              'Original failure was modified')
    def redirected(path):
        if Path(path).resolve() == failed_path:
            return PLAIN_READ(C.OUT / 'actions/fit_all_recovery.json')
        return old_read(path)
    C.read = redirected
    run.read = redirected
    try:
        yield
    finally:
        C.read = old_read
        run.read = old_entry_read
        C.require(C.desc(failed_path) == reg['failed_action'], 'Original failed action changed')


def evaluate():
    reg = check_registration()
    C.require(PLAIN_READ(REC / 'training_completion.json')['passed'], 'Completed recovery required')
    C.save(REC / 'evaluation_dependency_redirect.json', {
        'original_failed_action': reg['failed_action'],
        'new_completed_action': C.desc(C.OUT / 'actions/fit_all_recovery.json'),
        'execution_registration': execution_registration(), 'old_failed_file_unchanged': True,
        'only_this_dependency_redirected': True}, immutable=True)
    from .recovery_io import installed_io
    from . import run
    with installed_io(), evaluation_bridge():
        run.action('evaluate')


def complete():
    reg = check_registration()
    extra = PLAIN_READ(C.REPORT / 'recovery_independent_verification.json')
    C.require(extra['passed'] and extra['execution_registration'] == execution_registration(), 'Independent recovery cost verification required')
    inventory_path = C.REPORT / 'deliverable_hashes.json'
    C.require(not inventory_path.exists(), 'Already closed; use readonly recovery verification')
    # The original report mathematics is reused; sealing is deferred until the
    # actual interrupted-work accounting and its verification have been added.
    from . import run
    from .recovery_io import installed_io
    deferred_inventory = []
    with installed_io(event_log_name='finalization_io_events.jsonl'):
        io_save = run.save
        def report_save(path, data, immutable=False):
            if Path(path).resolve() == inventory_path.resolve():
                deferred_inventory.append(data)
                return
            return C.save(path, data, immutable=immutable)
        run.save = report_save
        try:
            run.complete()
        finally:
            run.save = io_save
        C.require(len(deferred_inventory) == 1 and not inventory_path.exists(), 'Canonical builder did not defer sealing')
        report_path = C.REPORT / 'TriDep_EEGPT匹配4秒备选完成报告.md'
        text = report_path.read_text(encoding='utf-8').replace('完成：2026-10-06。', '完成：2026-10-07。')
        old_work = '每条件15次正式拟合；合计30次、3600次优化更新和训练梯度计算。'
        C.require(old_work in text, 'Canonical report accounting wording changed')
        text = text.replace(old_work, '每条件15个完整模型；有效模型内计30次完整拟合及3600次更新和梯度。实际拟合启动31次、实际更新及梯度各3682次（含中断82次），详见恢复计账。')
        text += '''

## 文件写入故障、暂停与恢复的实际计账

原运行完成23个模型的2760次更新后，第24次拟合（EEGPT seed33/fold2）完成41轮、82次更新与梯度。第83次前置进度文件的原子替换被Windows拒绝，尚未进入该次前向、梯度或更新。未保存82步的模型/优化器状态，因此无法无损续接；全部5份证据（含.part）归档原字节，旧fit_all失败记录始终保留failed，用户暂停记录保留。

2026-10-07按用户“继续”先追加冻结恢复登记，再复用23个完成模型；从原固定初始化、种子、数据顺序和算法重建中断模型，补齐其他6个，共新增7次完整拟合、840次更新。最终有效模型仍为30个、有效模型内更新3600次；**实际拟合启动31次、实际优化更新及训练梯度计算各3682次**，含中断后废弃的82次。有效优化人例47880，实际48946，额外1066来自中断模型41轮×26人，不省略为0。重建模型的前41轮逐批损失、初始化与已归档历史精确一致；第42轮首16人也匹配旧第83次前置名单。

恢复只增加已完成模型复用、真实执行来源和文件写入持久性处理。训练epoch与模型/随机数/数据/优化器设置AST逐节点相同。I/O重试限同一已序列化临时文件的replace，最多40次/5秒，绝不重复神经前向、梯度或step。旧评价仅将完成依赖读取桥接到实际新恢复完成记录，没有把磁盘上的旧失败文件改成成功；独立核验单独验证此桥接和实际总成本。原S/T、输入、阈值和门槛均保持原登记，不根据成绩选择是否重建。

[恢复执行附表](../../plans/TriDep_EEGPT文件写入故障恢复附表_v2_20261007.md) · [恢复登记](../../../tridep_ccml/experiments/TriDep-Stage4-EEGPT-Matched4s-37-fulltext-20261006/recovery/registration.json) · [实际成本独立核验](recovery_independent_verification.json)

封存后的完整只读核验入口：`S00/.venv/Scripts/python.exe -B -m tridep_ccml.tridep_eegpt.recovery_verify --post-closeout`。原核验器仍核对30个有效模型的数学，恢复核验器另外核对31次启动、3682次实际更新及全部失败档案。
'''
        report_path.write_text(text, encoding='utf-8')
        status = PLAIN_READ(C.REPORT / 'status.json')
        status.update(formal_fits=31, formal_updates=3682, training_gradient_passes=3682,
                      canonical_completed_models=30, canonical_optimizer_updates=3600,
                      actual_person_examples=48946, execution_registration=execution_registration(),
                      recovery_budget=reg['budget'], report=C.desc(report_path),
                      recovery_independent_verification=C.desc(C.REPORT / 'recovery_independent_verification.json'))
        C.save(C.REPORT / 'status.json', status)
        files = sorted(p for base in (C.OUT, C.REPORT) for p in base.rglob('*') if p.is_file() and p != inventory_path)
        C.save(inventory_path, {p.relative_to(C.WORK).as_posix(): C.digest(p) for p in files}, immutable=True)
    # Include the IO bridge's final closing event in the actual final inventory.
    files = sorted(p for base in (C.OUT, C.REPORT) for p in base.rglob('*') if p.is_file() and p != inventory_path)
    C.save(inventory_path, {p.relative_to(C.WORK).as_posix(): C.digest(p) for p in files})
    print(f'Recovered study closed:{len(files)} files;30 successful models,31 starts,3682 actual updates', flush=True)


def accept_registration():
    from . import recovery_verify
    result = recovery_verify.verify(registration_only=True)
    C.save(REC / 'registration_independent_verification.json', result, immutable=True)
    print('Independent readonly registration audit receipt saved;0 neural forwards', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'accept-registration', 'train', 'evaluate', 'complete'))
    args = parser.parse_args()
    {'prepare': prepare, 'accept-registration': accept_registration, 'train': train,
     'evaluate': evaluate, 'complete': complete}[args.action]()
