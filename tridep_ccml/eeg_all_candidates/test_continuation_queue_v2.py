"""Pure-standard-library simulated handoff: never open/suspend/terminate real processes."""
from __future__ import annotations

import copy
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import continuation_queue as Q1, continuation_queue_v2 as Q2
from . import handoff_supervisor_v2 as H, storage as S


class FakeBackend:
    def __init__(self, *, gpu_exit=0, unexpected=False, wrong_identity=False):
        self.events = []; self.ticks = 0; self.parent_terminated = False
        self.gpu_exit = gpu_exit; self.unexpected = unexpected; self.wrong_identity = wrong_identity

    def capture(self, role, *, arm=False):
        self.events.append(('capture', role))
        value = {'role': role, 'pid': H.ROLE_PIDS[role], 'parent_pid': H.ROLE_PARENTS[role],
                 'creation_100ns': 'exact-' + role, 'created_local': H.CREATED_LOCAL[role],
                 'image': 'synthetic_python.exe', 'handle': 'mock-' + role}
        if self.wrong_identity and role == 'gpu_child':
            value['creation_100ns'] = 'reused-pid-different-object'
        return value

    def exited(self, ref):
        return self.parent_terminated if ref['role'] == 'dispatcher' else self.ticks > 0

    def descendants(self):
        return {2768, 33744, 98765} if self.unexpected else ({2768, 33744} if self.ticks == 0 else set())

    def exit_code(self, ref):
        return H.DISPATCHER_TERMINATION_CODE if ref['role'] == 'dispatcher' else (self.gpu_exit if ref['role'] == 'gpu_child' else 0)

    def suspend_dispatcher(self, ref):
        self.events.append(('suspend', ref['role']))

    def terminate_dispatcher(self, ref):
        self.events.append(('terminate', ref['role'])); self.parent_terminated = True

    def wait_tick(self):
        self.events.append(('children_continue_normally',)); self.ticks += 1

    def spawn_v2(self):
        self.events.append(('spawn_v2',)); return SimpleNamespace(pid=76543)

    def wait_v2(self, process):
        assert S.read(Q2.ROOT / 'handoff_complete.json')['passed'] is True
        self.events.append(('guardian_wait_v2', process.pid)); return 0

    def close(self, ref):
        self.events.append(('close_handle', ref['role']))


class V2Handoff(unittest.TestCase):
    def expected(self):
        fake = FakeBackend()
        return {'old_owner_id': H.OLD_OWNER_ID,
                'roles': {role: H.public_identity(fake.capture(role)) for role in H.ROLE_PIDS}}

    def fixture(self):
        context = ExitStack()
        directory = context.enter_context(tempfile.TemporaryDirectory(prefix='handoff-v2-synthetic-', dir=S.REPORT))
        root = Path(directory)
        old, new = root / 'v1', root / 'v2'
        owner = {'id': H.OLD_OWNER_ID, 'pid': 39128, 'child_pid': 2768, 'child_job': 'conformer-fit',
                 'linux_owner': (old / 'attempts/current-conformer/linux_owner.json').relative_to(S.WORK).as_posix()}
        S.save(old / 'owner.json', owner)
        S.save(old / 'registration.json', {'sources': []})
        S.save(old / 'attempts/current-conformer/status.json', {'status': 'running', 'child_pid': 2768,
               'job': {'id': 'conformer-fit', 'module': 'supervised'}})
        S.save(new / 'registration.json', {'version': 2})
        context.enter_context(patch.object(Q1, 'ROOT', old))
        context.enter_context(patch.object(Q2, 'ROOT', new))
        context.enter_context(patch.object(H, 'ROOT', new / 'handoff'))
        context.enter_context(patch.object(Q2, 'register', return_value={'version': 2}))
        context.enter_context(patch.object(H, 'check_old_boundary', return_value=owner))
        context.enter_context(patch.object(H, 'no_old_ssl_started'))
        context.enter_context(patch.object(H, 'conformer_artifacts', return_value=[]))
        return context, old, new

    def test_v2_matrix_only_replaces_ssl_module_and_leaves_v1_globals_untouched(self):
        old_root, old_jobs = Q1.ROOT, Q1.jobs()
        new_jobs = Q2.jobs()
        self.assertEqual(sum(j.get('fits', 0) for j in new_jobs), 195)
        self.assertEqual([j['id'] for j in new_jobs], [j['id'] for j in old_jobs])
        self.assertEqual(len([j for j in new_jobs if j['module'] == 'ssl_v2']), 2)
        for before, after in zip(old_jobs, new_jobs):
            restored = dict(after)
            if restored['module'] == 'ssl_v2': restored['module'] = 'ssl'
            self.assertEqual(before, restored)
        driver = Q2.isolated_driver()
        self.assertEqual(driver.ROOT, Q2.ROOT)
        self.assertEqual(Q1.ROOT, old_root)
        self.assertEqual(Q1.jobs(), old_jobs)

    def test_parent_only_pause_then_both_children_normal_exit_then_verified_takeover(self):
        context, old, new = self.fixture()
        original_owner = (old / 'owner.json').read_bytes()
        backend = FakeBackend()
        with context:
            proof = H.arm(self.expected(), backend)
            self.assertTrue(proof['passed'])
            self.assertEqual([e for e in backend.events if e[0] == 'suspend'], [('suspend', 'dispatcher')])
            self.assertEqual([e for e in backend.events if e[0] == 'terminate'], [('terminate', 'dispatcher')])
            self.assertLess(backend.events.index(('children_continue_normally',)), backend.events.index(('terminate', 'dispatcher')))
            self.assertLess(backend.events.index(('terminate', 'dispatcher')), backend.events.index(('spawn_v2',)))
            self.assertLess(backend.events.index(('spawn_v2',)), backend.events.index(('guardian_wait_v2', 76543)))
            self.assertEqual((old / 'owner.json').read_bytes(), original_owner)
            self.assertEqual(S.read(new / 'handoff_complete.json')['conformer_fits_verified'], 15)
            self.assertEqual(S.read(new / 'handoff/state.json')['v2_exit_code'], 0)

    def test_exact_creation_identity_mismatch_refuses_before_suspending(self):
        context, _, _ = self.fixture()
        backend = FakeBackend(wrong_identity=True)
        with context:
            with self.assertRaisesRegex(ValueError, 'process identity'):
                H.arm(self.expected(), backend)
        self.assertFalse(any(e[0] in ('suspend', 'terminate', 'spawn_v2') for e in backend.events))

    def test_new_ssl_child_race_is_quarantined_without_child_kill_or_old_resume(self):
        context, _, new = self.fixture()
        backend = FakeBackend(unexpected=True)
        with context:
            with self.assertRaisesRegex(ValueError, 'Unexpected child'):
                H.arm(self.expected(), backend)
            state = S.read(new / 'handoff/state.json')
            self.assertEqual(state['phase'], 'failed_parent_kept_suspended')
            self.assertFalse(state['auto_resume_v1'])
            self.assertFalse(any(e[0] in ('terminate', 'spawn_v2') for e in backend.events))

    def test_child_failure_or_incomplete_conformer_never_terminates_parent_or_starts_v2(self):
        for failure in ('child', 'artifacts'):
            context, _, new = self.fixture()
            backend = FakeBackend(gpu_exit=1 if failure == 'child' else 0)
            with context:
                with patch.object(H, 'conformer_artifacts', side_effect=ValueError('Conformer15 incomplete') if failure == 'artifacts' else None,
                                  return_value=[]):
                    with self.assertRaises(ValueError):
                        H.arm(self.expected(), backend)
                self.assertFalse((new / 'handoff_complete.json').exists())
            self.assertFalse(any(e[0] in ('terminate', 'spawn_v2') for e in backend.events))

    def test_v2_refuses_to_dispatch_without_completed_handoff_proof(self):
        context, _, _ = self.fixture()
        with context, patch.object(Q2, 'isolated_driver') as driver:
            with self.assertRaises(FileNotFoundError):
                Q2.run(resume=True)
            driver.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
