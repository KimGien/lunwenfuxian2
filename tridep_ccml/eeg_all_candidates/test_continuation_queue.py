"""Pure-standard-library queue regressions. All process launches are mocked."""
from __future__ import annotations

from contextlib import ExitStack
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import continuation_queue as Q


class FakeProcess:
    def __init__(self, *, returncode=0, on_wait=None):
        self.pid = 424242
        self.returncode = None
        self.result = returncode
        self.on_wait = on_wait
        self.alive = True
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        if self.on_wait is not None:
            callback, self.on_wait = self.on_wait, None
            callback()
        self.alive = False
        self.returncode = self.result
        return self.result

    def poll(self):
        return None if self.alive else self.returncode

    def terminate(self):
        self.terminated = True
        self.alive = False
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.alive = False
        self.returncode = -9


class ContinuationQueue(unittest.TestCase):
    def test_exact195_matrix_fixed64_and_all9_finetunes_without_score_filter(self):
        jobs = Q.jobs()
        fitted = [condition for job in jobs if job.get('fits') for condition in job['conditions']]
        self.assertEqual(set(fitted), set(Q.REMAINING))
        self.assertEqual(len(fitted), 13)
        self.assertEqual(sum(job.get('fits', 0) for job in jobs), 195)
        self.assertEqual(len(Q.FIRST) * 15, 165)
        self.assertEqual(len(Q.FINETUNE), 9)
        self.assertEqual(set(Q.FIRST).intersection(fitted), set())
        ssl = next(job for job in jobs if job['id'] == 'ssl-prepare')
        self.assertEqual(ssl['args'][ssl['args'].index('--logical-batch') + 1], '64')
        for job in jobs:
            command = Q.command(job, resume=True)
            self.assertEqual('--resume' in command, bool(job.get('fits')))
        # Matrix construction has no access to any frozen scores or predictions.
        self.assertNotIn('test', {job['module'] for job in jobs})

    def fixture(self, jobs):
        context = ExitStack()
        directory = context.enter_context(tempfile.TemporaryDirectory(prefix='queue-stdlib-', dir=Q.S.REPORT))
        root = Path(directory)
        registration = {'jobs': jobs, 'sources': []}
        Q.S.save(root / 'registration.json', registration)
        context.enter_context(patch.object(Q, 'ROOT', root))
        context.enter_context(patch.object(Q, 'register', return_value=registration))
        context.enter_context(patch.object(Q, 'wait_for_first'))
        context.enter_context(patch.object(Q, 'command', return_value=['mocked-process-no-executable']))
        context.enter_context(patch.object(Q.R, 'fit_status', return_value={'status': 'engineering_pending'}))
        recorded = context.enter_context(patch.object(Q.R, 'record_fit_status'))
        context.enter_context(patch.object(Q.R, 'status_summary', return_value={'formal_fits': 360, 'test_evaluation_frozen': True}))
        # Any platform-specific tree termination also remains completely mocked.
        context.enter_context(patch.object(Q.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='')))
        return context, root, recorded

    def test_failed_prerequisite_blocks_dependent_but_runs_independent_route(self):
        jobs = [
            {'id': 'bad-prepare', 'module': 'ssl', 'args': [], 'conditions': ['EEG2Rep-fold-local-ssl'], 'dependencies': []},
            {'id': 'dependent-fit', 'module': 'supervised', 'args': [], 'conditions': ['EEG2Rep-supervised'], 'dependencies': ['bad-prepare'], 'fits': 15},
            {'id': 'independent-fit', 'module': 'finetune', 'args': [], 'conditions': ['EEGPT-lastblock-finetune'], 'dependencies': [], 'fits': 15},
        ]
        context, root, recorded = self.fixture(jobs)
        launches, live = [], []
        def launch(command, **kwargs):
            self.assertFalse(any(process.alive for process in live), 'Two children concurrently own GPU')
            process = FakeProcess(returncode=1 if not launches else 0)
            launches.append(command); live.append(process)
            return process
        with context, patch.object(Q.subprocess, 'Popen', side_effect=launch), \
             patch.object(Q, 'all_complete', side_effect=lambda conditions: conditions == ['EEGPT-lastblock-finetune']):
            Q.run()
            value = Q.S.read(root / 'status.json')
            self.assertEqual(value['outcomes'], {'bad-prepare': 'failed', 'dependent-fit': 'blocked', 'independent-fit': 'complete'})
            self.assertEqual(value['status'], 'required_work_remaining')
            self.assertTrue(value['test_evaluation_frozen'])
            self.assertEqual(len(launches), 2)
            blocked = {call.args[0] for call in recorded.call_args_list if call.args[3] == 'blocked'}
            self.assertTrue({'EEG2Rep-fold-local-ssl', 'EEG2Rep-supervised'}.issubset(blocked))
            self.assertNotIn('EEGPT-lastblock-finetune', blocked)

    def test_parent_interruption_terminates_and_reaps_its_child(self):
        job = {'id': 'interrupted', 'module': 'supervised', 'args': [], 'conditions': ['EEGConformer-scratch'], 'dependencies': [], 'fits': 15}
        context, root, _ = self.fixture([job])
        def interrupt():
            raise KeyboardInterrupt('injected parent interruption')
        process = FakeProcess(on_wait=interrupt)
        with context, patch.object(Q.subprocess, 'Popen', return_value=process), patch.object(Q, 'all_complete', return_value=True):
            with self.assertRaises(KeyboardInterrupt):
                Q.run()
            self.assertTrue(process.terminated or process.killed, 'Interrupted parent left its CUDA child alive')
            self.assertIsNotNone(process.poll(), 'Child was not reaped')
            self.assertEqual(Q.S.read(root / 'status.json')['status'], 'failed')

    def test_second_dispatcher_is_rejected_before_launching_any_child(self):
        job = {'id': 'single-owner', 'module': 'supervised', 'args': [], 'conditions': ['EEGConformer-scratch'], 'dependencies': [], 'fits': 15}
        context, _, _ = self.fixture([job])
        rejected, launches = [], []
        def second_invocation():
            try:
                Q.run()
            except (RuntimeError, ValueError, FileExistsError):
                rejected.append(True)
        def launch(command, **kwargs):
            launches.append(command)
            return FakeProcess(on_wait=second_invocation if len(launches) == 1 else None)
        with context, patch.object(Q.subprocess, 'Popen', side_effect=launch), patch.object(Q, 'all_complete', return_value=True):
            Q.run()
            self.assertTrue(rejected, 'A second active dispatcher was not rejected')
            self.assertEqual(len(launches), 1, 'Duplicate invocation launched another child')

    def test_completed_marker_with_changed_checkpoint_bytes_rejects_gpu_transfer(self):
        with tempfile.TemporaryDirectory(prefix='queue-artifact-stdlib-', dir=Q.S.REPORT) as directory:
            folder = Path(directory)
            evidence = {'passed': True, 'checkpoint_reload_verified': True, 'parameter_update_verified': True,
                        'frozen_parameters_verified': True, 'epochs': 3, 'optimizer_updates': 6, 'stop_reason': 'engineering_fixture'}
            for key, name in (('checkpoint', 'best.pt'), ('train_history', 'history.json'),
                              ('validation_history', 'val.json'), ('metadata', 'metadata.json')):
                path = folder / name
                Q.S.write(path, lambda stream: stream.write(b'bound artifact bytes'))
                evidence[key] = Q.S.desc(path)
            lock = {'path': 'synthetic-engineering-lock', 'sha256': 'fixture'}
            status = {'status': 'training_complete', 'evidence': evidence, 'condition_source_lock': lock}
            with patch.object(Q.R, 'fit_status', return_value=status), patch.object(Q.R, 'fit_directory', return_value=folder), \
                 patch.object(Q.R, '_condition_lock', return_value=lock):
                self.assertTrue(Q.all_complete(['EEGConformer-scratch']))
                (folder / 'best.pt').write_bytes(b'changed after completion')
                with self.assertRaisesRegex(ValueError, 'Source bytes changed'):
                    Q.all_complete(['EEGConformer-scratch'])

    def test_zero_exit_prepare_without_real_engineering_lock_fails_but_independent_continues(self):
        jobs = [
            {'id': 'conformer-prepare', 'module': 'supervised', 'args': ['--prepare'], 'conditions': ['EEGConformer-scratch'], 'dependencies': []},
            {'id': 'independent-fit', 'module': 'finetune', 'args': [], 'conditions': ['EEGPT-lastblock-finetune'], 'dependencies': [], 'fits': 15},
        ]
        context, root, _ = self.fixture(jobs)
        launches = []
        def launch(*_, **__):
            process = FakeProcess(returncode=0)
            launches.append(process)
            return process
        def condition_lock(condition):
            if condition == 'EEGConformer-scratch':
                raise FileNotFoundError('Missing actual engineering lock')
            return {'path': 'fixture', 'sha256': 'engineering'}
        with context, patch.object(Q.subprocess, 'Popen', side_effect=launch), patch.object(Q.R, '_condition_lock', side_effect=condition_lock), \
             patch.object(Q, 'all_complete', side_effect=lambda conditions: conditions == ['EEGPT-lastblock-finetune']):
            Q.run()
            status = Q.S.read(root / 'status.json')
            self.assertEqual(status['outcomes'], {'conformer-prepare': 'failed', 'independent-fit': 'complete'})
            self.assertEqual(len(launches), 2)

    def test_linux_child_cleanup_signals_its_registered_group_and_preserves_unverified_lease(self):
        with tempfile.TemporaryDirectory(prefix='queue-linux-stdlib-', dir=Q.S.REPORT) as directory:
            owner_file = Path(directory) / 'linux_owner.json'
            Q.S.save(owner_file, {'pid': 43210, 'pgid': 43210, 'action': 'mamba-check'})
            process = FakeProcess()
            calls = []
            def run(arguments, **_):
                calls.append(arguments)
                return SimpleNamespace(returncode=1 if '-0' in arguments else 0)
            with patch.object(Q.subprocess, 'run', side_effect=run):
                Q.terminate_child(process, {'linux': True}, owner_file)
            self.assertTrue(any('/bin/kill' in args and '-TERM' in args and '-43210' in args for args in calls))
            self.assertTrue(any('taskkill' in args and str(process.pid) in args for args in calls))
            self.assertFalse(process.alive)
            process = FakeProcess()
            with patch.object(Q.subprocess, 'run', return_value=SimpleNamespace(returncode=0)), patch.object(Q.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'still alive'):
                    Q.terminate_child(process, {'linux': True}, owner_file)

    def test_codebrain_repair_waits_for_both_initial_owners_then_serially_resumes_original_queue(self):
        with tempfile.TemporaryDirectory(prefix='queue-repair-stdlib-', dir=Q.S.REPORT) as directory:
            running, children = [], []
            failure = {'status': 'failed', 'condition': 'CodeBrain-frozen-smallhead'}
            def read(path):
                if Path(path).name == 'initial_owner.json':
                    return {'process_ids': [10001, 10002]}
                return failure if not children else {'status': 'registered_ready_batch_complete'}
            def pid(pid):
                running.append(pid)
                return False
            def run_child(job, owner, **_):
                children.append(job['id'])
                return 'complete'
            with patch.object(Q, 'ROOT', Path(directory)), patch.object(Q.S, 'read', side_effect=read), \
                 patch.object(Q, 'pid_running', side_effect=pid), patch.object(Q, 'run_child', side_effect=run_child), \
                 patch.object(Q, 'all_complete', return_value=True):
                Q.wait_for_first({'id': 'fixture-owner'})
            self.assertEqual(running, [10001, 10002])
            self.assertEqual(children, ['first-codebrain-repair', 'resume-first-frozen135'])

    def test_preceding_terminal_marker_alone_does_not_transfer_gpu(self):
        with tempfile.TemporaryDirectory(prefix='queue-wait-stdlib-', dir=Q.S.REPORT) as directory:
            class WaitObserved(Exception):
                pass
            with patch.object(Q, 'ROOT', Path(directory)), patch.object(Q.S, 'read', return_value={'status': 'registered_ready_batch_complete'}), \
                 patch.object(Q, 'all_complete', return_value=False), patch.object(Q.time, 'sleep', side_effect=WaitObserved):
                with self.assertRaises(WaitObserved):
                    Q.wait_for_first()

    def test_failed_full360_gate_launches_no_evaluation_child(self):
        context, root, _ = self.fixture([])
        with context, patch.object(Q.R, 'assert_test_unlocked', side_effect=ValueError('359/360 incomplete')), \
             patch.object(Q, 'run_child') as child:
            self.assertFalse(Q.evaluate_tail({'id': 'fixture-owner'}))
            child.assert_not_called()
            self.assertFalse(Q.S.read(root / 'evaluation_gate_status.json')['passed'])
            failures = list((root / 'evaluation_gate_failures').glob('*.json'))
            self.assertEqual(len(failures), 1)
            self.assertEqual(Q.S.read(failures[0])['test_forward_calls_this_attempt'], 0)

    def test_changed_registered_source_launches_no_evaluation_child(self):
        context, root, _ = self.fixture([])
        source = root / 'bound-source.py'
        source.write_text('original source', encoding='utf-8')
        Q.S.save(root / 'registration.json', {'sources': [Q.S.desc(source)]})
        source.write_text('changed source', encoding='utf-8')
        with context, patch.object(Q.R, 'assert_test_unlocked') as gate, patch.object(Q, 'run_child') as child:
            self.assertFalse(Q.evaluate_tail({'id': 'fixture-owner'}))
            gate.assert_not_called()
            child.assert_not_called()
            self.assertFalse(Q.S.read(root / 'evaluation_gate_status.json')['passed'])

    def test_successful_gate_dispatches_linux_mamba_before_windows_aggregate(self):
        context, root, _ = self.fixture([])
        events = []
        def gate():
            events.append('gate')
            return {'passed': True, 'registration': {'path': 'synthetic', 'sha256': 'fixture'},
                    'formal_fits_verified': 360, 'statuses': []}
        def child(job, owner, **_):
            events.append(job['id'])
            return 'complete'
        with context, patch.object(Q.R, 'assert_test_unlocked', side_effect=gate), patch.object(Q, 'run_child', side_effect=child):
            self.assertTrue(Q.evaluate_tail({'id': 'fixture-owner'}))
            self.assertEqual(events, ['gate', 'evaluate-mamba-finetune', 'evaluate-all'])
            self.assertTrue(Q.S.read(root / 'evaluation_gate_status.json')['passed'])

    def test_linux_evaluation_failure_does_not_dispatch_windows_aggregate(self):
        context, _, _ = self.fixture([])
        gate = {'passed': True, 'registration': {'path': 'synthetic', 'sha256': 'fixture'},
                'formal_fits_verified': 360, 'statuses': []}
        with context, patch.object(Q.R, 'assert_test_unlocked', return_value=gate), patch.object(Q, 'run_child', return_value='failed') as child:
            self.assertFalse(Q.evaluate_tail({'id': 'fixture-owner'}))
            self.assertEqual(child.call_count, 1)
            self.assertEqual(child.call_args.args[0]['id'], 'evaluate-mamba-finetune')

    def test_zero_exit_evaluation_without_real_probability_artifacts_is_not_complete(self):
        for job in (
            {'id': 'evaluate-mamba-finetune', 'module': 'linux_entry', 'args': [], 'conditions': ['EEGMamba-lastblock-finetune'], 'dependencies': [], 'linux': True},
            {'id': 'evaluate-all', 'module': 'evaluate', 'args': [], 'conditions': [], 'dependencies': []},
        ):
            with self.subTest(job=job['id']):
                context, _, _ = self.fixture([])
                with context, patch.object(Q.subprocess, 'Popen', return_value=FakeProcess(returncode=0)), \
                     patch.object(Q.R, '_condition_lock', return_value={'path': 'synthetic', 'sha256': 'fixture'}):
                    owner = Q.acquire_owner()
                    try:
                        self.assertEqual(Q.run_child(job, owner), 'failed', 'Evaluation exit0 without real artifacts cannot count complete')
                    finally:
                        Q.release_owner(owner)

    def test_evaluator_entry_rechecks_full360_gate_before_any_model_construction(self):
        from . import evaluate as Evaluation
        with patch.object(Evaluation.R, 'assert_test_unlocked', side_effect=ValueError('child gate incomplete')), \
             patch.object(Evaluation, '_materialize_predictor') as construct:
            with self.assertRaisesRegex(ValueError, 'child gate incomplete'):
                Evaluation.run(('EEGMamba-lastblock-finetune',), no_report=True)
            construct.assert_not_called()

    def test_existing_evaluation_artifact_with_stale_binding_or_gate_is_rejected(self):
        from . import evaluate as Evaluation
        mamba_job = {'id': 'evaluate-mamba-finetune', 'module': 'linux_entry', 'args': [],
                     'conditions': ['EEGMamba-lastblock-finetune'], 'dependencies': [], 'linux': True}
        context, root, _ = self.fixture([])
        corrupted = root / 'existing-evaluation-complete.json'
        expected = {'condition': 'EEGMamba-lastblock-finetune', 'test_gate': 'current verified gate'}
        Q.S.save(corrupted, {'passed': True, 'binding': {**expected, 'test_gate': 'stale gate'}})
        def verify_mamba(_):
            return Evaluation._verified_completed_fit(corrupted, expected, {}, {})
        with context, patch.object(Q.subprocess, 'Popen', return_value=FakeProcess(returncode=0)), \
             patch.object(Evaluation, 'completed_condition_verified', side_effect=verify_mamba):
            owner = Q.acquire_owner()
            try:
                self.assertEqual(Q.run_child(mamba_job, owner), 'failed')
            finally:
                Q.release_owner(owner)
        context, root, _ = self.fixture([])
        current_gate = {'path': 'current-gate.json', 'sha256': 'verified'}
        Q.S.save(root / 'complete.json', {'passed': True, 'test_gate': {'path': 'stale.json', 'sha256': 'stale'},
                 'formal_fits_verified': 360, 'supervised_conditions': 23, 'supervised_fits_evaluated': 345,
                 'evaluated_checkpoints': 435, 'candidate_papers_count': 9})
        aggregate = {'id': 'evaluate-all', 'module': 'evaluate', 'args': [], 'conditions': [], 'dependencies': []}
        with context, patch.object(Evaluation, 'ROOT', root), patch.object(Evaluation, '_gate', return_value=current_gate), \
             patch.object(Q.subprocess, 'Popen', return_value=FakeProcess(returncode=0)):
            owner = Q.acquire_owner()
            try:
                self.assertEqual(Q.run_child(aggregate, owner), 'failed')
            finally:
                Q.release_owner(owner)


if __name__ == '__main__':
    unittest.main(verbosity=2)
