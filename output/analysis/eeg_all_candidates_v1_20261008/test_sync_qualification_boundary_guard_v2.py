"""Fake-only Guardian2 lifecycle tests; never controls a real process or GPU."""
import copy
import importlib.util
from pathlib import Path
import types
import unittest

REPORT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


B = load('_g2_fake_fixture_delegate', REPORT / 'test_sync_qualification_boundary_guard_v1.py')


class Guardian2Tests(unittest.TestCase):
    # Reuse only the existing fake fixture setup/helpers, never its native API.
    desc, checked, read, save = B.GuardianTests.desc, B.GuardianTests.checked, B.GuardianTests.read, B.GuardianTests.save
    tick, spawn, arm, resumed, state = B.GuardianTests.tick, B.GuardianTests.spawn, B.GuardianTests.arm, B.GuardianTests.resumed, B.GuardianTests.state
    gpu_idle = B.GuardianTests.gpu_idle
    pause, tearDown = B.GuardianTests.pause, B.GuardianTests.tearDown

    def setUp(self):
        B.GuardianTests.setUp(self)
        self.g1, self.oldroot = self.g, self.root
        self.g = load('_g2_fake_guard', REPORT / 'sync_qualification_boundary_guard_v2.py')
        self.root = self.out / 'execution_acceleration_v1/sync_boundary_v2'; self.root.mkdir()
        self.source = self.report / 'sync_qualification_boundary_guard_v2.py'
        self.source.write_bytes((REPORT / self.source.name).read_bytes())
        self.g.SOURCE, self.g.ROOT, self.g.G1 = self.source, self.root, self.g1
        self.g.S, self.g.Q4, self.g.Q1, self.g.R = self.g1.S, self.g1.Q4, self.g1.Q1, self.g1.R
        self.g.S.digest = lambda p: self.desc(p)['sha256']
        self.g.time = types.SimpleNamespace(time_ns=__import__('time').time_ns, sleep=self.tick)
        self.g.subprocess = types.SimpleNamespace(Popen=self.spawn, STDOUT=-2)
        self.q5root = self.out / 'continuation_queue_v5'; self.q5root.mkdir()
        self.save(self.q5root / 'registration.json', {'version': 5, 'sources': self.sources})
        self.register_calls = 0
        def register():
            self.register_calls += 1
            for binding in self.sources: self.checked(binding)
            return self.read(self.q5root / 'registration.json')
        self.g.Q5 = types.SimpleNamespace(ROOT=self.q5root, register=register)
        self.identity = {'pid': 999, 'creation_100ns': '777', 'parent_pid': 42, 'source': self.desc(self.source)}
        self.g.guardian_identity = lambda native: copy.deepcopy(self.identity)
        self.contexts = []; self.gpu_hook = None; self.gpu_reads = 0
        self.g.gpu_observation = self.observation
        for name in self.g.QUALIFIERS: (self.report / name).write_text('# fake qualifier\n')
        (self.work / 'tridep_ccml/eeg_all_candidates/sync_qualification_gate_v2.py').write_text('# fake gate\n')
        self.parent_done = False; self.terminate_ok = True; self.terminate_hook = None
        self.q5code = 0; self.q5_spawn_error = False
        native_exited = self.native.exited
        self.native.exited = lambda ref: self.parent_done if ref['role'] == 'dispatcher' else native_exited(ref)
        native_exit_code = self.native.exit_code
        self.native.exit_code = lambda ref: 0xEED80008 if ref['role'] == 'dispatcher' else native_exit_code(ref)
        kernel_wait = self.native.kernel.WaitForSingleObject
        self.native.kernel.WaitForSingleObject = lambda handle, ms: (0 if self.parent_done else 0x102) if handle == 'fake-dispatcher' else kernel_wait(handle, ms)
        self.native.kernel.TerminateProcess = self.terminate
        self.g.spawn_q5 = self.spawn_q5
        self.g.completed_artifacts = self.completed
        self.expected = {'source': self.desc(self.source), 'owner': self.desc(self.q4root / 'owner.json'),
                         'q4_registration': self.desc(self.q4root / 'registration.json'),
                         'q5_registration': self.desc(self.q5root / 'registration.json'),
                         'roles': {r: self.g1.H2.public_identity(v) for r, v in self.native.refs.items()}}
        self.expected_path = self.root / 'inspection_fixture.json'; self.save(self.expected_path, self.expected)

    def observation(self):
        self.gpu_reads += 1
        if self.gpu_hook: self.gpu_hook()
        return {'contexts': copy.deepcopy(self.contexts), 'idle': not self.contexts,
                'query': 'FAKE XML', 'checked_at': 'fixture-' + str(self.gpu_reads)}

    def completed(self):
        self.actions.append(('full15_evidence',))
        self.g.require(self.complete, 'fake fifteen incomplete')
        return [self.desc(self.fit / 'train_history.json')]

    def terminate(self, handle, code):
        self.assertEqual(handle, 'fake-dispatcher'); self.assertEqual(code, 0xEED80008)
        self.assertTrue(self.native.children_done)
        self.assertTrue(self.complete)
        self.native.calls.append(('terminate', handle))
        if self.terminate_hook: self.terminate_hook()
        if self.terminate_ok: self.parent_done = True
        return self.terminate_ok

    def spawn_q5(self):
        self.assertTrue(self.parent_done); self.assertFalse((self.q4root / 'owner.json').exists())
        self.assertTrue((self.q5root / 'handoff_complete.json').is_file())
        self.actions.append(('spawn_q5',))
        if self.q5_spawn_error: raise RuntimeError('fake Q5 launch failure')
        return types.SimpleNamespace(pid=2222, wait=lambda: self.q5code)

    def terminated(self): return [x for x in self.native.calls if x[0] == 'terminate']
    def q5_started(self): return bool([x for x in self.actions if x[0] == 'spawn_q5'])

    def live_proof(self):
        self.native.children_done = True
        lease = self.root / 'guardian_owner.json'; self.save(lease, {'identity': self.identity})
        proof = {'schema': 'sync_qualification_safe_boundary_v2', 'passed': True, 'source': self.desc(self.source),
                 'queue_owner': self.desc(self.q4root / 'owner.json'), 'registration': self.desc(self.q4root / 'registration.json'),
                 'guardian_owner': self.desc(lease), 'guardian_identity': self.identity,
                 'held_process_identities': self.expected['roles'], 'gpu_children_exit_verified': True,
                 'launcher_exit_code': 0, 'gpu_child_exit_code': 0}
        p = self.root / 'proof.json'; self.save(p, proof); binding = self.desc(p)
        self.save(self.root / 'state.json', {'phase': 'qualification_running', 'dispatcher_suspended': True,
                  'boundary_proof': binding, 'source': proof['source'], 'guardian_identity': self.identity,
                  'held_process_identities': proof['held_process_identities']})
        return binding

    def test_quiet_natural_exit_profiles_then_only_parent_termination_then_q5(self):
        value = self.arm()
        self.assertTrue(value['passed']); self.assertEqual(len(self.processes), 2)
        self.assertEqual(self.terminated(), [('terminate', 'fake-dispatcher')]); self.assertFalse(self.resumed())
        self.assertTrue(self.q5_started()); self.assertTrue(self.state()['all_qualifications_passed'])
        self.assertTrue(all(not x[2] for x in self.native.calls if x[0] == 'capture' and x[1] != 'dispatcher'))
        self.assertEqual([x for x in self.native.calls if x[0] == 'suspend'], [('suspend', 'dispatcher')])
        self.assertEqual(self.read(self.work / value['old_owner_snapshot']['path']), self.owner)

    def test_known_graphics_defers_all_profiles_but_handoffs(self):
        self.contexts = [{'pid': i + 1, 'type': 'C+G', 'process_name': p} for i, p in enumerate(self.g.OBSERVED_GRAPHICS_PATHS)]
        self.contexts += [{'pid': 10, 'type': 'G', 'process_name': 'desktop'}]
        value = self.arm()
        self.assertFalse(self.processes); self.assertTrue(self.q5_started()); self.assertFalse(self.state()['all_qualifications_passed'])
        deferred = self.read(self.work / value['deferred_qualification']['path'])
        self.assertEqual(deferred['model_constructions'], 0); self.assertFalse(deferred['GPU_speedup_claimed'])

    def test_unknown_compute_keeps_boundary_held(self):
        self.contexts = [{'pid': 77, 'type': 'C', 'process_name': 'python.exe'}]
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.processes or self.terminated() or self.resumed()); self.assertFalse(self.q5_started())

    def test_unknown_combined_or_truncated_name_is_refused(self):
        for kind, path in [('C+G', 'LeagueClientUxRender.exe'), ('C+G', 'C:/other/app.exe'), ('unknown', '')]:
            with self.assertRaises(ValueError): self.g.training_gpu_scope({'contexts': [{'pid': 1, 'type': kind, 'process_name': path}]})

    def test_live_orphan_or_stale_lease_never_handoffs(self):
        self.behaviors = [{'orphan': True}, {}]
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.terminated() or self.resumed()); self.assertFalse(self.q5_started())

    def test_stale_shared_lease_never_handoffs(self):
        self.save(self.probe_lease, {'id': 'foreign'})
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.processes or self.terminated() or self.resumed())

    def test_timeout_cpu_qualifier_keeps_parent_held_even_idle(self):
        self.behaviors = [{'timeout': True}, {}]
        with self.assertRaises(TimeoutError): self.arm()
        self.assertFalse(self.terminated() or self.resumed()); self.assertTrue(self.state()['dispatcher_suspended'])

    def test_failed_clean_probe_preserved_then_q5_without_second_probe(self):
        self.behaviors = [{'code': 1}, {}]; value = self.arm()
        self.assertEqual(len(self.processes), 1); self.assertTrue(self.q5_started())
        self.assertTrue(value['optional_qualification_outcomes'][0]['execution_failed'])
        self.assertFalse(self.state()['all_qualifications_passed'])

    def test_partial_success_then_graphics_race_is_not_all_qualified(self):
        def later_graphics():
            if self.processes and self.processes[0].returncode == 0:
                self.contexts = [{'pid': 4, 'type': 'G', 'process_name': 'desktop'}]
        self.gpu_hook = later_graphics
        value = self.arm()
        self.assertEqual(len(self.processes), 1); self.assertFalse(self.state()['all_qualifications_passed'])
        self.assertTrue(value['GPU_training_scope']['contexts'])

    def test_old_pause_marker_revokes_waiting_resume(self):
        self.wait_hook = lambda: self.save(self.oldroot / 'user_pause.json', {'paused': True})
        with self.assertRaises(self.g1.UserPauseRequested): self.arm()
        self.assertFalse(self.processes or self.resumed() or self.terminated())

    def test_new_pause_marker_revokes_waiting_resume(self):
        self.wait_hook = self.pause
        with self.assertRaises(self.g1.UserPauseRequested): self.arm()
        self.assertEqual(self.state()['phase'], 'user_pause_Q4_held'); self.assertFalse(self.resumed())

    def test_malformed_pause_marker_fails_closed(self):
        self.wait_hook = lambda: self.pause({'paused': False})
        with self.assertRaises(ValueError): self.arm()
        self.assertEqual(self.state()['phase'], 'invalid_user_pause_control'); self.assertFalse(self.resumed())

    def test_timeout_with_pause_is_recognized_as_pause(self):
        self.behaviors = [{'timeout': True, 'pause': True}, {}]
        with self.assertRaises(TimeoutError): self.arm()
        self.assertEqual(self.state()['phase'], 'user_pause_Q4_held'); self.assertFalse(self.resumed())

    def test_pause_after_old_parent_exit_does_not_claim_held_or_spawn(self):
        self.terminate_hook = self.pause
        with self.assertRaises(self.g1.UserPauseRequested): self.arm()
        self.assertEqual(self.state()['phase'], 'user_pause_after_Q4_exit')
        self.assertFalse(self.state()['dispatcher_suspended']); self.assertFalse(self.q5_started())
        self.assertTrue((self.q4root / 'owner.json').exists())

    def test_termination_failure_never_resumes_old_parent(self):
        self.terminate_ok = False
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.resumed()); self.assertFalse(self.q5_started())

    def test_q5_spawn_failure_retains_handoff_and_never_resumes_dead_parent(self):
        self.q5_spawn_error = True
        with self.assertRaises(RuntimeError): self.arm()
        self.assertFalse(self.resumed()); self.assertTrue((self.q5root / 'handoff_complete.json').is_file())

    def test_nonzero_q5_exit_not_passed(self):
        self.q5code = 1; self.arm()
        self.assertFalse(self.state()['passed']); self.assertEqual(self.state()['q5_exit_code'], 1)

    def test_incomplete_or_failed_children_do_not_terminate(self):
        self.complete = False
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse(self.terminated()); self.assertEqual(len(self.resumed()), 1)

    def test_identity_mismatch_prevents_any_suspend(self):
        self.expected['roles']['gpu_child']['creation_100ns'] = 'wrong'; self.save(self.expected_path, self.expected)
        with self.assertRaises(ValueError): self.arm()
        self.assertFalse([x for x in self.native.calls if x[0] in ('suspend', 'terminate', 'resume')])

    def test_source_mutation_before_handoff_is_refused(self):
        self.wait_hook = lambda: ((self.work / 'source_0.py').write_text('# changed'), setattr(self.native, 'children_done', True))
        with self.assertRaises(AssertionError): self.arm()
        self.assertFalse(self.terminated()); self.assertFalse(self.q5_started())

    def test_live_verifier_binds_source_identity_and_no_children(self):
        binding = self.live_proof(); self.assertTrue(self.g.verify_live_boundary(binding)['passed'])
        self.assertFalse(self.terminated() or self.resumed())
        state = self.state(); state['guardian_identity'] = {}; self.save(self.root / 'state.json', state)
        with self.assertRaises(ValueError): self.g.verify_live_boundary(binding)

    def test_live_verifier_rejects_birth_change_and_user_pause(self):
        binding = self.live_proof(); self.native.guardian_birth += 1
        with self.assertRaises(ValueError): self.g.verify_live_boundary(binding)
        self.native.guardian_birth -= 1; self.pause()
        with self.assertRaises(self.g1.UserPauseRequested): self.g.verify_live_boundary(binding)


if __name__ == '__main__': unittest.main(verbosity=2)
