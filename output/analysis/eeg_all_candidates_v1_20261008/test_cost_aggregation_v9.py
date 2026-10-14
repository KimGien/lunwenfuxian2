"""Pure-stdlib fixtures; no actual collector or recovery verifier is executed."""
import copy
import importlib.util
from pathlib import Path
import unittest
import hashlib
import sys
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_cost9_guards', HERE/'cost_aggregation_v9.py')
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)


def descriptor(path): return {'path': path, 'sha256': 'a'*64}


def proof(event, updates, backs):
    return {'passed': True, 'persisted_counters_are_lower_bounds': True, 'unflushed_tail_unknown': True,
            'extra_work_not_added_again': True, 'original_status': 'running', 'graceful_terminal_flush': False,
            'unflushed_tail_counters': None, 'final_elapsed_seconds': None,
            'known_extra_optimizer_returns_outside_checkpoint': updates,
            'known_extra_backward_returns_outside_checkpoint': backs,
            **{k: descriptor(event+'/'+k+'.json') for k in ('closure','recovery_registration','closure_source','checkpoint_cpu_resume_proof')}}


class CostV9Guards(unittest.TestCase):
    def test_package_relative_verifier_import(self):
        # Exercise the same fully qualified module convention without importing
        # any project verifier or its transitive weight-hashing dependencies.
        with tempfile.TemporaryDirectory(dir=HERE, prefix='_cost7_import_') as folder:
            root=Path(folder);package=root/'fixture_package';package.mkdir()
            (package/'__init__.py').write_text('',encoding='utf-8')
            (package/'storage.py').write_text('MARKER = 73\n',encoding='utf-8')
            source=package/'verifier.py';source.write_text('from . import storage\nVALUE = storage.MARKER\n',encoding='utf-8')
            sys.path.insert(0,str(root))
            try:
                module=C.load_module(source,hashlib.sha256(source.read_bytes()).hexdigest(),'fixture_package._bound_verifier')
                self.assertEqual(module.VALUE,73)
            finally:
                sys.path.remove(str(root))
                for name in tuple(sys.modules):
                    if name=='fixture_package' or name.startswith('fixture_package.'):sys.modules.pop(name)
    def setUp(self):
        self.records = {}
        self.proofs = {}
        rows=[]
        for condition,event,updates,backs in [('CSBrain','old',4,27),('LUNA','new',1,9)]:
            path='out/finetune_v2_attempts/'+condition+'/'+event+'/attempt.json'
            self.records[path]={'implementation':{'version':3},'condition':condition,'seed':33,'fold':3,'status':'running'}
            self.proofs[path]=proof(event,updates,backs)
            rows.append({'artifact':descriptor(path),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation',
                         'status':'running','seconds_observed':90000.,'observed_counters':{'optimizer_returned':100,'backward_returned':700}})
        self.base={'schema_version':5,'source':descriptor('v5.py'),'cost_records':rows,
            'category_observations':{'formal_training_attempt':{'counter_totals':{'optimizer_returned':{'known_observed_subtotal':200},'backward_returned':{'known_observed_subtotal':1400}}}},
            'inputs':[],'input_read_errors':[],'rules':[],'formal_durable_updates_known_subtotal':196,
            'source_bound_stopped_attempts':[{'extra':[0,2]}],'legacy_resume_events':[{'extra':[4,27]}]}
    def run_case(self,base=None,verifier=None):
        return C.augment(copy.deepcopy(base or self.base),
            verifier=verifier or (lambda d,*_:copy.deepcopy(self.proofs[d['path']])),
            reader=lambda d:copy.deepcopy(self.records[d['path']]),verifier_source=descriptor('L3.py'))
    def test_two_distinct_closures_deltas_not_added(self):
        value=self.run_case()
        self.assertEqual(len(value['source_bound_lost_f3_attempts']),2)
        self.assertEqual(value['formal_durable_updates_known_subtotal'],196)
        self.assertEqual([r['observed_counters'] for r in value['cost_records']],[r['observed_counters'] for r in self.base['cost_records']])
        self.assertEqual(value['category_observations']['formal_training_attempt']['counter_totals']['optimizer_returned']['known_observed_subtotal'],200)
        self.assertEqual(value['source_bound_stopped_attempts'],self.base['source_bound_stopped_attempts'])
        self.assertEqual(value['legacy_resume_events'],self.base['legacy_resume_events'])
        self.assertTrue(all(r['additional_optimizer_returns_added_for_recovery']==r['additional_backward_returns_added_for_recovery']==0 for r in value['cost_records']))
    def test_new_event_not_forced_into_old_four_twentyseven(self):
        value=self.run_case();self.assertFalse(value['input_read_errors'])
        self.assertEqual(value['source_bound_lost_f3_attempts'][1]['known_extra_optimizer_returns_outside_checkpoint'],1)
    def test_unknown_new_boundary_remains_unknown(self):
        p=list(self.proofs.values())[-1];p['known_extra_optimizer_returns_outside_checkpoint']=None;p['known_extra_backward_returns_outside_checkpoint']=None
        value=self.run_case();self.assertFalse(value['input_read_errors'])
        self.assertIsNone(value['source_bound_lost_f3_attempts'][-1]['known_extra_optimizer_returns_outside_checkpoint'])
    def test_running_elapsed_and_tail_never_zero_filled(self):
        value=self.run_case()
        self.assertTrue(all(r['seconds_observed'] is None and r['unflushed_tail_counters'] is None for r in value['cost_records']))
        self.assertIsNone(value['category_observations']['formal_training_attempt']['known_recorded_scope_seconds_subtotal'])
        self.assertIsNone(value['all_plan_wall_seconds_exact']);self.assertIsNone(value['all_plan_physical_optimizer_returns_exact'])
    def test_validation_failure_does_not_retain_inferred_elapsed(self):
        def invalid(*args): raise ValueError('broken closure bytes')
        value=self.run_case(verifier=invalid)
        self.assertEqual(len(value['input_read_errors']),2)
        self.assertFalse(value['lost_f3_closure_verification']['passed_for_observed_closures'])
        self.assertTrue(all(r['seconds_observed'] is None for r in value['cost_records']))
    def test_unmatched_running_has_no_invented_closure(self):
        value=self.run_case(verifier=lambda *_:None)
        self.assertFalse(value['source_bound_lost_f3_attempts']);self.assertTrue(all(r['seconds_observed'] is None for r in value['cost_records']))
    def test_negative_or_boolean_delta_rejected(self):
        for bad in (True,-1,1.5,'4'):
            with self.subTest(bad=bad):
                p=proof('bad',bad,0)
                with self.assertRaises(ValueError): C.validate_disclosure(p)
    def test_false_terminal_or_exact_time_rejected(self):
        for key,value in [('final_elapsed_seconds',0),('unflushed_tail_counters',{}),('graceful_terminal_flush',True),('extra_work_not_added_again',False)]:
            p=proof('bad',0,0);p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):C.validate_disclosure(p)
    def test_duplicate_primary_closure_rejected(self):
        self.base['cost_records'].append(copy.deepcopy(self.base['cost_records'][0]))
        self.assertEqual(len(self.run_case()['input_read_errors']),1)
    def test_duplicate_alias_does_not_duplicate_primary_time_or_counters(self):
        alias=copy.deepcopy(self.base['cost_records'][0]);alias['aggregation_role']='duplicate_invocation_artifact';self.base['cost_records'].append(alias)
        value=self.run_case();self.assertFalse(value['input_read_errors'])
        self.assertEqual(value['category_observations']['formal_training_attempt']['seconds_unknown_records'],2)
        self.assertEqual(len(value['inputs']),8)
    def test_real_zero_other_record_retained(self):
        self.base['cost_records'].append({'artifact':descriptor('other.json'),'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','seconds_observed':0})
        self.assertEqual(self.run_case()['category_observations']['formal_training_attempt']['known_recorded_scope_seconds_subtotal'],0)
    def test_reaugmentation_rejected(self):
        for bad in ({'schema_version':6},{'schema_version':7},{'schema_version':8},{'schema_version':9},{'source_bound_lost_f3_attempts':[]}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.run_case({**self.base,**bad})
    def test_completed_attempt_keeps_original_elapsed_inclusive_scope(self):
        for r in self.records.values():r['status']='complete'
        value=self.run_case()
        self.assertEqual(value['cost_records'],self.base['cost_records'])
    def test_actual_new_source_not_claimed_as_old_v6(self):
        value=self.run_case()
        self.assertEqual(value['schema_version'],9)
        self.assertEqual(value['source'],C.descriptor(C.SOURCE))
        self.assertEqual(value['previous_cost_source_v6']['sha256'],C.DELEGATE_SHA)
        self.assertEqual(value['lost_f3_closure_verifier_source'],descriptor('L3.py'))
    def test_three_distinct_losses_preserve_event_deltas_without_extra_addition(self):
        path = 'out/finetune_v2_attempts/LaBraM-native4s/third/attempt.json'
        self.records[path] = {'implementation': {'version': 3}, 'condition': 'LaBraM-native4s',
                              'seed': 33, 'fold': 1, 'status': 'running'}
        self.proofs[path] = proof('third', 4, 27)
        self.proofs[list(self.proofs)[1]].update(known_extra_optimizer_returns_outside_checkpoint=4,
                                              known_extra_backward_returns_outside_checkpoint=26)
        self.base['cost_records'].append({'artifact': descriptor(path), 'category': 'formal_training_attempt',
            'aggregation_role': 'primary_observed_invocation', 'status': 'running', 'seconds_observed': 100000,
            'observed_counters': {'optimizer_returned': 220, 'backward_returned': 1485}})
        before = copy.deepcopy(self.base)
        value = self.run_case()
        self.assertEqual([(v['known_extra_optimizer_returns_outside_checkpoint'],
                           v['known_extra_backward_returns_outside_checkpoint'])
                          for v in value['source_bound_lost_f3_attempts']], [(4,27),(4,26),(4,27)])
        self.assertEqual([v['observed_counters'] for v in value['cost_records']],
                         [v['observed_counters'] for v in before['cost_records']])
        self.assertTrue(all(v['seconds_observed'] is None for v in value['cost_records']))
        self.assertEqual(value['source_bound_stopped_attempts'], before['source_bound_stopped_attempts'])
        self.assertEqual(value['legacy_resume_events'], before['legacy_resume_events'])

    def add_parent(self):
        parent=descriptor('out/continuation_queue_v5/attempts/LUNA/status.json')
        self.records[parent['path']]={'status':'running','seconds':None}
        self.base['cost_records'].append({'artifact':parent,'category':'dispatcher_elapsed','aggregation_role':'supplemental_nested_or_unmeasured',
                                         'status':'running','seconds_observed':99999})
        self.proofs[list(self.proofs)[-1]].update(original_queue_attempt=parent,queue_attempt_final_elapsed_seconds=None)
        return parent
    def test_lost_parent_unknown_and_supplemental(self):
        self.add_parent();value=self.run_case();row=value['cost_records'][-1]
        self.assertIsNone(row['seconds_observed']);self.assertFalse(row['parent_elapsed_added_to_formal_or_primary_work'])
        self.assertEqual(len(value['source_bound_lost_parent_observer_scopes']),1)
        self.assertFalse(value['input_read_errors'])
    def test_lost_parent_cannot_fabricate_elapsed(self):
        self.add_parent();self.proofs[list(self.proofs)[-1]]['queue_attempt_final_elapsed_seconds']=0
        value=self.run_case();self.assertTrue(value['input_read_errors']);self.assertFalse(value['lost_parent_observer_verification']['passed_for_observed_closures'])
    def test_lost_parent_missing_or_primary_rejected(self):
        self.add_parent();self.base['cost_records'][-1]['aggregation_role']='primary_observed_invocation'
        self.assertTrue(self.run_case()['input_read_errors'])


if __name__=='__main__':unittest.main(verbosity=2)
