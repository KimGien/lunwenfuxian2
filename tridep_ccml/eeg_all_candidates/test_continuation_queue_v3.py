"""Purestdlib scoped v3 routing/handoff simulations; no real processes or GPU."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from . import continuation_queue_v2 as Q2, continuation_queue_v3 as Q3
from . import handoff_supervisor_v3 as H3, linux_entry as L1, linux_entry_v2 as L2, storage as S


class FakeProcess:
    pid = 98760
    def __init__(self): self.returncode = None
    def wait(self, timeout=None): self.returncode = 0; return 0
    def poll(self): return self.returncode


class FakeBackend:
    def __init__(self, *, unexpected=False, fail=False, wrong=False):
        self.events=[]; self.ticks=0; self.dead=False; self.unexpected=unexpected; self.fail=fail; self.wrong=wrong
    def capture(self, role, *, arm=False):
        self.events.append(('capture',role))
        return {'role':role,'pid':H3.ROLE_PIDS[role],'parent_pid':H3.ROLE_PARENTS[role],
                'creation_100ns':('wrong' if self.wrong and role=='gpu_child' else 'exact-'+role),
                'created_local':H3.CREATED_LOCAL[role],'image':'synthetic-python','handle':'fake-'+role}
    def exited(self,ref): return self.dead if ref['role']=='dispatcher' else self.ticks>0
    def descendants(self): return {48392,40680,99991} if self.unexpected else ({48392,40680} if self.ticks==0 else set())
    def suspend_dispatcher(self,ref): self.events.append(('suspend',ref['role']))
    def terminate_dispatcher(self,ref): self.events.append(('terminate',ref['role'])); self.dead=True
    def exit_code(self,ref): return H3.TERMINATION_CODE if ref['role']=='dispatcher' else (1 if self.fail and ref['role']=='gpu_child' else 0)
    def wait_tick(self): self.events.append(('ssl_children_continue',)); self.ticks+=1
    def wait_v2(self,process): self.events.append(('guardian_wait_v3',process.pid)); return 0
    def close(self,ref): self.events.append(('close',ref['role']))


class V3Scoped(unittest.TestCase):
    def test_exact195_matrix_ssl_stays_v2_ft_and_linux_routes_change_only_modules(self):
        old=Q2.jobs(); new=Q3.jobs()
        self.assertEqual(sum(j.get('fits',0) for j in new),195)
        self.assertEqual([j['id'] for j in old],[j['id'] for j in new])
        for a,b in zip(old,new):
            restored={**b,'module':{'finetune_v2':'finetune','linux_entry_v2':'linux_entry'}.get(b['module'],b['module'])}
            self.assertEqual(a,restored)
        self.assertEqual([j['module'] for j in new if j['id'] in ('ssl-prepare','ssl-fit')],['ssl_v2','ssl_v2'])
        self.assertEqual(len([j for j in new if j['module']=='finetune_v2']),16)
        driver=Q3.isolated_driver()
        self.assertEqual(driver.ROOT,Q3.ROOT)
        for job in ({'module':'evaluate','args':[]},{'module':'linux_entry','args':['--action','mamba-evaluate'],'linux':True}):
            command=driver.command(job)
            self.assertIn('tridep_ccml.eeg_all_candidates.'+('evaluate_v2' if job['module']=='evaluate' else 'linux_entry_v2'),command)

    def test_corrected_linux_owner_is_accepted_by_sealed_entry_while_status_stays_root3(self):
        with tempfile.TemporaryDirectory(prefix='q3-',dir=S.WORK) as temporary:
            work=Path(temporary); out=work/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'; root=out/'continuation_queue_v3'
            driver=Q3.isolated_driver(); driver.ROOT=root
            seen=[]
            def launch(arguments,**_):
                owner=arguments[arguments.index('--owner-record')+1]
                seen.append(owner)
                with patch.object(L1.sys,'platform','linux'),patch.object(L1.os,'getpgrp',return_value=8000,create=True),patch.object(L1.os,'getpid',return_value=8000):
                    path=L1._register_owner('mamba-check',owner)
                self.assertTrue(path.is_relative_to(out/'continuation_queue/attempts'))
                self.assertTrue(path.parent.name.startswith('v3_'))
                return FakeProcess()
            with patch.object(S,'WORK',work),patch.object(S,'OUT',out),patch.object(driver,'command',return_value=['mock-only']), \
                 patch.object(driver.subprocess,'Popen',side_effect=launch),patch.object(driver,'postcondition',return_value=True):
                S.save(root/'registration.json',{'synthetic':True})
                owner=driver.acquire_owner()
                try:
                    self.assertEqual(driver.run_child({'id':'mamba-check','module':'linux_entry_v2','args':[],'linux':True,'conditions':[]},owner),'complete')
                    self.assertEqual(len(list((root/'attempts').glob('*/status.json'))),1)
                    self.assertEqual(len(list((out/'continuation_queue/attempts').glob('v3_*/linux_owner.json'))),1)
                finally: driver.release_owner(owner)
            self.assertEqual(len(seen),1)

    def test_linux_check_export_delegation_and_eval_v2_route_do_not_import_ft(self):
        with patch.object(L2.V1,'execute',return_value='delegated') as delegated:
            self.assertEqual(L2.execute('mamba-check',owner_record='reviewed'),'delegated')
            delegated.assert_called_once_with('mamba-check',resume=False,owner_record='reviewed')
        # Actual evaluation routing is checked with a synthetic module, never an encoder.
        import sys
        import tridep_ccml.eeg_all_candidates as package
        fake=SimpleNamespace(run=lambda conditions,no_report:('v2-eval',conditions,no_report))
        with patch.dict(sys.modules,{'tridep_ccml.eeg_all_candidates.evaluate_v2':fake}), \
             patch.object(package,'evaluate_v2',fake,create=True), \
             patch.object(L2.V1,'_register_owner') as owner,patch.object(L2.V1,'_require_runtime'):
            self.assertEqual(L2.execute('mamba-evaluate',owner_record='reviewed'),('v2-eval',('EEGMamba-lastblock-finetune',),True))
            owner.assert_called_once_with('mamba-evaluate','reviewed')

    def test_windows_owner_locator_retains_actual_root3_attempt_for_future_handoff(self):
        with tempfile.TemporaryDirectory(prefix='q3-',dir=S.WORK) as temporary:
            work=Path(temporary);out=work/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008';root=out/'continuation_queue_v3'
            driver=Q3.isolated_driver();driver.ROOT=root
            with patch.object(S,'WORK',work),patch.object(S,'OUT',out),patch.object(driver,'command',return_value=['mock-only']), \
                 patch.object(driver.subprocess,'Popen',return_value=FakeProcess()),patch.object(driver,'postcondition',return_value=True):
                S.save(root/'registration.json',{'synthetic':True})
                owner=driver.acquire_owner()
                try:
                    def wait(process,timeout=None):
                        current=S.read(root/'owner.json')
                        locator=(work/current['linux_owner']).parent
                        self.assertTrue(locator.is_relative_to(root/'attempts'))
                        self.assertEqual(S.read(locator/'status.json')['child_pid'],process.pid)
                        process.returncode=0;return 0
                    with patch.object(FakeProcess,'wait',wait):
                        self.assertEqual(driver.run_child({'id':'example-windows-fit','module':'finetune_v2','args':[],'conditions':[]},owner),'complete')
                finally:driver.release_owner(owner)

    def fixture(self):
        context=ExitStack(); temporary=context.enter_context(tempfile.TemporaryDirectory(prefix='h3-scoped-stdlib-',dir=S.REPORT))
        base=Path(temporary); old=base/'q2'; new=base/'q3'
        owner={'id':H3.OWNER_ID,'pid':41664,'child_pid':48392,'child_job':'ssl-fit'}
        attempt=old/'attempts/current-ssl/status.json'
        S.save(old/'registration.json',{'sources':[]});S.save(old/'owner.json',owner);S.save(attempt,{'status':'running'})
        S.save(new/'registration.json',{'version':3})
        context.enter_context(patch.object(Q2,'ROOT',old));context.enter_context(patch.object(Q3,'ROOT',new))
        context.enter_context(patch.object(H3,'ROOT',new/'handoff'));context.enter_context(patch.object(Q3,'register',return_value={'version':3}))
        context.enter_context(patch.object(H3,'boundary',return_value=(owner,attempt,{'status':'running'})))
        context.enter_context(patch.object(H3,'ssl_artifacts',return_value=[]))
        fake=FakeBackend()
        expected={'owner_id':H3.OWNER_ID,'q2_registration':S.desc(old/'registration.json'),
                  'roles':{role:H3.H2.public_identity(fake.capture(role)) for role in H3.ROLE_PIDS}}
        return context,new,expected

    def test_ssl_batch_parent_only_pause_both_exit0_then_handoff_and_guardian_wait(self):
        context,new,expected=self.fixture();backend=FakeBackend()
        with context:
            proof=H3.arm(expected,backend,spawn=lambda:SimpleNamespace(pid=98989))
            self.assertTrue(proof['passed'])
            self.assertEqual([e for e in backend.events if e[0]=='suspend'],[('suspend','dispatcher')])
            self.assertEqual([e for e in backend.events if e[0]=='terminate'],[('terminate','dispatcher')])
            self.assertLess(backend.events.index(('ssl_children_continue',)),backend.events.index(('terminate','dispatcher')))
            self.assertIn(('guardian_wait_v3',98989),backend.events)
            self.assertFalse(proof['h2_guardian_or_ancestors_operated'])
            self.assertEqual(S.read(new/'handoff_complete.json')['ssl_fits_verified'],15)

    def test_wrong_object_or_unexpected_child_or_child_failure_never_starts_v3(self):
        for kind in ('identity','race','failure'):
            context,new,expected=self.fixture();backend=FakeBackend(wrong=kind=='identity',unexpected=kind=='race',fail=kind=='failure')
            spawn=lambda: (_ for _ in ()).throw(AssertionError('Never start v3'))
            with context:
                with self.assertRaises(ValueError): H3.arm(expected,backend,spawn=spawn)
                self.assertFalse((new/'handoff_complete.json').exists())
            self.assertFalse(any(e[0]=='terminate' for e in backend.events))

    def test_missing_ssl_batch_proof_rejects_v3_before_dispatch(self):
        with patch.object(Q3,'register'),patch.object(Q3,'require_handoff',side_effect=ValueError('SSL15 incomplete')),patch.object(Q3,'isolated_driver') as driver:
            with self.assertRaisesRegex(ValueError,'SSL15 incomplete'):Q3.run(resume=True)
            driver.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
