"""Stdlib source/JSON fixtures; no recovery verifier, collect, model or GPU."""
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

REPORT=Path(__file__).resolve().parent

def load(name):
    spec=importlib.util.spec_from_file_location('_post6_'+name,REPORT/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

F=load('finalize_full_plan_cost_v6'); C=load('cost_aggregation_v4')
N=load('supplemental_head_resource_measurement_v6')
# Only these pure gate functions are compiled. The project verifier is a fake.
resource_path=REPORT/'supplemental_head_resource_measurement_v6.py'
tree=ast.parse(resource_path.read_text(encoding='utf-8'))
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('verify_recovery_terminal','verify_cpu_shape')]
M=types.ModuleType('_resource6_pure');M.require=F.require;M.OUT_REL='out';M.L=types.SimpleNamespace()
exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(resource_path),'exec'),M.__dict__)

def d(name):return {'path':name,'sha256':'a'*64}

class RecoveryGateTests(unittest.TestCase):
    def test_measurement_math_delegated_to_original_v2(self):
        self.assertIs(N.measure,N.D.measure)
        self.assertEqual(N.measure.__code__.co_filename,str(N.V2_SOURCE))
        self.assertEqual(N.D.SOURCE,N.SOURCE)
    def test_resource6_provenance_actual_source(self):
        value={'subtype':'supplemental_head_resource_measurement','measurement_scope':{'seconds':'E2 prerequisite'}}
        with patch.object(N,'write_new') as write:N.write_measurement(Path('fixture.json'),value)
        result=write.call_args.args[1]
        self.assertEqual(result['resource_measurement_version'],6)
        self.assertEqual(result['delegated_resource_measurement_source'],N.delegated_source())
    def setUp(self):
        self.expected={k:d(k+'.json') for k in ('recovery_registration','recovery_execution_source','recovery_closure','previous_recovery_registration')}
        self.expected['previous_recovery_closure']={'path':'out/process_loss_closures/lost_f3_1791597116209422300.json','sha256':'e56bbb201c067bf0bab31c0d405520fb2860deb12796f088072bb30bdd73de65'}
        self.terminal={**self.expected,'recovery_execution_version':2}
        self.lineage={'registration':self.expected['recovery_registration'],'source':self.expected['recovery_execution_source'],
            'value':{'previous_recovery_registration':self.expected['previous_recovery_registration']}}
        self.closure={'original_attempt':d('attempt.json'),'condition':'LUNA','seed':33,'fold':3}
        self.proof={'passed':True,'closure':self.expected['recovery_closure'],'recovery_registration':self.expected['recovery_registration'],'closure_source':self.expected['recovery_execution_source']}
        M.L=types.SimpleNamespace(verify_recovery_registration=lambda *_:self.lineage,read_json=lambda *_:self.closure,verify_lost_f3_attempt=lambda *_:self.proof)
    def test_actual_two_recovery_chain(self):self.assertEqual(self.expected,M.verify_recovery_terminal(self.terminal))
    def test_missing_or_old_recovery_terminal_rejected(self):
        for key in self.terminal:
            value=deepcopy(self.terminal);value.pop(key)
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError)):M.verify_recovery_terminal(value)
    def test_wrong_new_or_previous_binding_rejected(self):
        for key in self.expected:
            value={**self.terminal,key:d('wrong.json')}
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_recovery_terminal(value)
    def test_version_not_boolean_or_old(self):
        for version in (True,1,3,'2'):
            with self.subTest(version=version),self.assertRaises(ValueError):M.verify_recovery_terminal({**self.terminal,'recovery_execution_version':version})
    def test_failed_or_different_new_closure_rejected(self):
        for key,val in [('passed',False),('closure',d('other')),('recovery_registration',d('old')),('closure_source',d('old.py'))]:
            old=self.proof[key];self.proof[key]=val
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_recovery_terminal(self.terminal)
            self.proof[key]=old
    def test_cpu6_requires_unique135_actual405(self):
        reg={'fits':[{'fit_id':str(i),'stage':'3'} for i in range(135)]};binding=d('registry.json')
        deep={'cpu_tensors_verified':True,'checkpoint_loads':3,'only_map_location_cpu':True,'gpu_operations':0,'model_constructions':0}
        value={'schema_version':6,'registration':binding,'registered_total_fits':360,'registered_finetune_fits':135,
          'training_complete_finetune_observed':135,'verified_completed_finetune_fits':135,'logical_completed_fits_counted':135,'checkpoint_loads':405,
          'status':'all_finetune_verified','all_135_finetune_checkpoints_verified':True,'partial_snapshot':False,'pending_fits':[],'deferred_completed_fits':[],'errors':[],
          'completed_rows':[{'fit_id':str(i),'passed':True,'cpu_checkpoint_audit':deepcopy(deep)} for i in range(135)],
          'observed_auditor_operations_including_failed_checks':{'cpu_checkpoint_loads_started':405,'cpu_checkpoint_loads_returned':405,'model_construction_attempts_blocked':0,'gpu_initialization_attempts_blocked':0}}
        self.assertEqual(135,len(M.verify_cpu_shape(value,reg,binding)))
        for key,bad in [('schema_version',5),('checkpoint_loads',402),('verified_completed_finetune_fits',True),('partial_snapshot',True),('deferred_completed_fits',[{}])]:
            with self.subTest(key=key),self.assertRaises(ValueError):M.verify_cpu_shape({**value,key:bad},reg,binding)
        bad=deepcopy(value);bad['completed_rows'][1]=bad['completed_rows'][0]
        with self.assertRaises(ValueError):M.verify_cpu_shape(bad,reg,binding)

class RecoveryCostTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=REPORT,prefix='_post6_');self.root=Path(self.temp.name).resolve();self.out=self.root/'out'
        (self.out/'process_loss_closures').mkdir(parents=True)
        self.rows=[];self.records=[];self.regs=[];self.closures=[];self.cpu=[]
        for event,condition,fold,epoch,updates,backs,du,db,seconds in [('old','CSBrain',4,8,32,216,4,27,14.4206806),('new','LUNA',3,10,40,270,2,15,None)]:
            reg=self.save(event+'_reg.json',{'source':d(event+'.py')});self.regs.append(reg)
            closure=self.save('out/process_loss_closures/lost_f3_'+event+'.json',{});self.closures.append(closure)
            original={'condition':condition,'seed':33,'fold':fold,'status':'running','seconds':None,'optimizer_step_returned':updates+du,'backward_returned':backs+db,'forward_returned':900}
            attempt=self.save(event+'_attempt.json',original)
            cpu=self.save(event+'_cpu.json',{'passed':True,'checkpoint_loads':2,'seconds':seconds,**{k:0 for k in ('model_constructions','GPU_operations','raw_arrays_loaded','test_probability_reads','optimizer_updates','backward_calls')}});self.cpu.append(cpu)
            self.rows.append({'passed':True,'closure':closure,'recovery_registration':reg,'closure_source':d(event+'.py'),
              'aggregation_role':'primary_observed_invocation','artifact':attempt,'known_extra_optimizer_returns_outside_checkpoint':du,'known_extra_backward_returns_outside_checkpoint':db,
              'durable_epoch':epoch,'durable_optimizer_updates':updates,'durable_backward_calls':backs,'unflushed_tail_unknown':True,'unflushed_tail_counters':None,'final_elapsed_seconds':None,'graceful_terminal_flush':False,'extra_work_not_added_again':True,'checkpoint_cpu_resume_proof':cpu})
            self.records.append({'artifact':attempt,'category':'formal_training_attempt','aggregation_role':'primary_observed_invocation','status':'running','seconds_observed':None,**C.normalized(original)})
        self.parent=d('parent.json');self.rows[1]['original_queue_attempt']=self.parent
        self.recovery={'previous_recovery_registration':self.regs[0],'previous_recovery_closure':self.closures[0],
          'recovery_registration':self.regs[1],'recovery_closure':self.closures[1],'recovery_execution_source':self.save('new.py',{})}
        # Actual source descriptor in manifests is deliberately a real fixture.
        regvalue=json.loads(F.checked(self.regs[1]).read_text());regvalue['source']=self.recovery['recovery_execution_source']
        self.regs[1]=self.save('new_reg.json',regvalue);self.recovery['recovery_registration']=self.regs[1]
        self.rows[1]['recovery_registration']=self.regs[1];self.rows[1]['closure_source']=self.recovery['recovery_execution_source']
        self.snapshot={'lost_f3_closure_verification':{'passed_for_observed_closures':True,'errors':[]},
          'lost_parent_observer_verification':{'passed_for_observed_closures':True,'errors':[]},
          'source_bound_lost_f3_attempts':self.rows,'source_bound_lost_parent_observer_scopes':[{'artifact':self.parent,'closure':self.closures[1],'final_elapsed_seconds':None,'added_again':False}],
          'cost_records':self.records,'inputs':[],'category_observations':{}}
    def save(self,name,value):
        p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value),encoding='utf-8');return F.desc(p)
    def tearDown(self):self.temp.cleanup()
    def bind(self):
        with patch.object(F,'OUT',self.out):return F.bind_lost_f3_recovery_cost(self.snapshot,self.recovery,C)
    def test_both_boundaries_once_physical_not_added(self):
        before=deepcopy(self.records);inventory=self.bind()
        self.assertEqual(before,self.snapshot['cost_records'][:2]);self.assertEqual(4,len(inventory))
        self.assertEqual(2,self.snapshot['category_observations']['lost_f3_recovery_evidence']['supplemental_records'])
        self.assertTrue(all(r['counts_added_again'] is False for r in self.snapshot['cost_records'][2:4]))
        cpu=self.snapshot['category_observations']['recovery_cpu_preflight']
        self.assertEqual(14.4206806,cpu['known_recorded_scope_seconds_subtotal']);self.assertEqual(1,cpu['seconds_unknown_records'])
    def test_new_delta_not_forced_to_old4_27(self):self.assertEqual(4,len(self.bind()))
    def test_wrong_durable_delta_rejected(self):
        self.rows[1]['known_extra_optimizer_returns_outside_checkpoint']=4
        with self.assertRaisesRegex(ValueError,'checkpoint boundary'):self.bind()
    def test_bool_delta_rejected(self):
        self.rows[1]['known_extra_backward_returns_outside_checkpoint']=True
        with self.assertRaises(ValueError):self.bind()
    def test_source_or_manifest_swapping_rejected(self):
        self.rows[1]['recovery_registration']=self.regs[0]
        with self.assertRaises(ValueError):self.bind()
    def test_unknown_tail_time_cannot_be_zero(self):
        for key,bad in [('unflushed_tail_counters',{}),('final_elapsed_seconds',0),('graceful_terminal_flush',True)]:
            old=self.rows[1][key];self.rows[1][key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):self.bind()
            self.rows[1][key]=old
    def test_counter_record_must_match_original_json(self):
        self.records[1]['observed_counters']['optimizer_returned']+=1
        with self.assertRaisesRegex(ValueError,'physical counters'):self.bind()
    def test_duplicate_primary_rejected(self):
        self.records.append(deepcopy(self.records[1]))
        with self.assertRaises(ValueError):self.bind()
    def test_alias_does_not_add_another_closure(self):
        alias={**deepcopy(self.rows[1]),'aggregation_role':'duplicate_invocation_artifact'};self.rows.append(alias)
        self.assertEqual(4,len(self.bind()))
    def test_unaccounted_loss_event_rejected(self):
        self.save('out/process_loss_closures/lost_f3_extra.json',{})
        with self.assertRaisesRegex(ValueError,'omitted or ambiguous'):self.bind()
    def test_parent_cannot_invent_finished_time(self):
        self.snapshot['source_bound_lost_parent_observer_scopes'][0]['final_elapsed_seconds']=0
        with self.assertRaises(ValueError):self.bind()
    def test_previously_classified_cpu_not_readded(self):
        self.records.append({'artifact':self.cpu[0]})
        with self.assertRaisesRegex(ValueError,'already classified'):self.bind()
    def test_no_known_cpu_timer_is_null_and_endpoint_not_subtracted(self):
        for i,event in enumerate(('old','new')):
            value=json.loads(F.checked(self.cpu[i]).read_text());value.pop('seconds');value.update(started_at='2026-10-10T00:00:00',finished_at='2026-10-11T00:00:00')
            self.rows[i]['checkpoint_cpu_resume_proof']=self.save(event+'_cpu.json',value)
        self.bind();self.assertIsNone(self.snapshot['category_observations']['recovery_cpu_preflight']['known_recorded_scope_seconds_subtotal'])
    def test_real_cpu_zero_timer_retained(self):
        for i,event in enumerate(('old','new')):
            value=json.loads(F.checked(self.cpu[i]).read_text());value['seconds']=0
            self.rows[i]['checkpoint_cpu_resume_proof']=self.save(event+'_cpu.json',value)
        self.bind();self.assertEqual(0,self.snapshot['category_observations']['recovery_cpu_preflight']['known_recorded_scope_seconds_subtotal'])
    def test_unrecorded_cpu_counter_fields_remain_unknown(self):
        value=json.loads(F.checked(self.cpu[1]).read_text())
        for key in ('seconds','optimizer_updates','backward_calls'):value.pop(key)
        self.rows[1]['checkpoint_cpu_resume_proof']=self.save('new_cpu.json',value)
        self.bind();row=self.snapshot['cost_records'][-1]
        self.assertIsNone(row['seconds_observed'])
        self.assertIsNone(row['observed_counters']['optimizer_returned'])
        self.assertIsNone(row['observed_counters']['backward_returned'])
    def test_repeated_binding_rejected(self):
        self.bind()
        with self.assertRaises(ValueError):self.bind()
    def test_pending_before_verifier_or_collect(self):
        fits=[{'fit_id':f'C{c}/seed_{s}/fold_{f}','condition':f'C{c}','objective':'self_supervised' if c==23 else 'supervised'} for c in range(24) for s in (11,22,33) for f in range(1,6)]
        self.save('out/registration.json',{'fits':fits})
        with patch.object(F,'OUT',self.out),patch.object(F,'verify_existing_evaluation',side_effect=AssertionError('E3 forbidden')),patch.object(F,'collect_cost',side_effect=AssertionError('collect forbidden')),patch.object(F,'recovery_execution_proof',side_effect=AssertionError('recovery forbidden')):
            result=F.finalize()
        self.assertEqual('pending',result['status']);self.assertFalse(result['outer_prediction_artifacts_read'])
    def test_preserved_math_helpers_ast_identical(self):
        def funcs(path):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(Path(path).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
        old=funcs(REPORT/'finalize_full_plan_cost_v5.py');new=funcs(F.__file__)
        for name in ('measurement_plan','head_resource_gate','bind_mamba_recheck_evidence','append_optional_qualification_v2_cost','append_boundary_control_evidence','verify_phase_coverage','render_report','verify_existing_evaluation'):
            self.assertEqual(old[name],new[name],name)

if __name__=='__main__':unittest.main(verbosity=2)
