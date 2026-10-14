"""Independent final-companion compatibility review; source/JSON + synthetic costs only."""
from pathlib import Path
import ast,copy,hashlib,json,sys,time,types
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3];OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
checks=[]
def require(v,m):
 if not v:raise ValueError(m)
def ok(m,v=True):require(v,m);checks.append(m)
def reject(m,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):checks.append(m);return
 raise AssertionError('Unsafe acceptance: '+m)
def desc(p):
 p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8-sig'))
def node(p,n):return next(x for x in tree(p).body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(n):return ast.dump(n,include_attributes=False)
def load(p):
 m=types.ModuleType('_independent_'+p.stem);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m

def main():
 names=['pause_memory_resume_cost_companion_v2.py','sync_boundary_v3_cost_companion_v2.py','audit_memory_pause_final_delivery_v2.py','system_standby_cost_companion_v2.py','audit_system_standby_final_delivery_v2.py']
 hashes=['fe5a5d0102090aaac9480770035ce45de1ac86468b7412cc1c9be2a57d3ea460','c44dd96bef288da461e9e7b5841f31fc476087c836a3d71952887096c6cdc369','95e2716952cb2b1bbf7d351cbddbbbc29e6566da0e72f4d5b37b0aa52f120a2e','09b54a5d23a7a499fc9b982eeb4d6abb3a5ba7fcd5902bf6a4204256be9932e4','bc3d3b61144d838290a5c4ba04d0739c027deb9a329e865579196583446a685c']
 files=[A/n for n in names];sources=[desc(p) for p in files]
 ok('exact five frozen sources', [d['sha256'] for d in sources]==hashes)
 contract=A/'final_companion_v2_stdlib_contract_1791830527150841700.json';cp=read(contract)
 strict=A/'final_companion_v2_strict_cli_regression_1791830603502522900.json';sp=read(strict)
 ok('author36 proof binds same source bytes',desc(contract)['sha256']=='168bf2f9035131409a387a6cb7adcc417850e2df38adbef6a6fe1596fd28ba5d' and cp['passed'] is True and len(cp['checks'])==36 and cp['sources']==sources)
 ok('strict CLI regression evidence exact',desc(strict)['sha256']=='9a8cc6682ae6b9f9dbf5c0b823c3d16d92cbd3f38cc9ff6cfd0313df9f033c20' and sp.get('passed') is True)
 registration=OUT/'continuation_queue_v14/registration.json';reg=read(registration)
 ok('actual194 sealed training sources unchanged',len(reg['sources'])==194 and all(desc(WORK/d['path'])==d for d in reg['sources']))
 prior=A/'independent_final_postprocessing_v7_source_review_1791829456847964300.json';pr=read(prior)
 ok('previous independent v7 review and all three sources preserved',desc(prior)['sha256']=='9e7c8f634d6d4048d42f129cc3abcc1bfa8a3700b0a6d592c6614a7d18f09baa' and pr['passed'] is True and all(desc(WORK/d['path'])==d for d in pr['reviewed_sources']))
 before=set(sys.modules);h,g,mem,stand,gate=[load(p) for p in files]
 ok('five imports stay standard-library and zero Torch',not any(x=='torch' or x.startswith('torch.') or x.startswith('tridep_ccml') for x in set(sys.modules)-before))
 # Actual immutable closure/source/JSON facts combined with a synthetic cost shell.
 # This is an interface/guard test, never a final cost collector or certificate.
 reader=h.Reader();cs=[reader.checked(d) for d in h.LOSSES];primary=[];parents=[];records=[]
 for d,c in zip(h.LOSSES,cs):
  dur=c['durable'];extra=c['known_extra_lower_bounds']
  primary.append({'closure':d,'artifact':c['original_attempt'],'aggregation_role':'primary_observed_invocation','passed':True,'recovery_registration':c['recovery_registration'],'closure_source':c['closure_source'],'checkpoint_cpu_resume_proof':c['checkpoint_cpu_resume_proof'],'final_elapsed_seconds':None,'unflushed_tail_unknown':True,'extra_work_not_added_again':True,'durable_epoch':dur['epoch'],'durable_optimizer_updates':dur['optimizer_updates'],'durable_backward_calls':dur['backward_calls'],'known_extra_optimizer_returns_outside_checkpoint':extra['optimizer_updates'],'known_extra_backward_returns_outside_checkpoint':extra['backward_calls']})
  records.append({'artifact':c['original_attempt'],'aggregation_role':'primary_observed_invocation','category':'formal_training_attempt','status':'running','seconds_observed':None})
  if 'old_queue_attempt_original' in c:
   parents.append({'closure':d,'artifact':c['old_queue_attempt_original'],'final_elapsed_seconds':None,'added_again':False})
   records.append({'artifact':c['old_queue_attempt_original'],'aggregation_role':'supplemental_nested_or_unmeasured','category':'dispatcher_elapsed','seconds_observed':None})
 proof=dict(zip(('original_recovery_closure','previous_recovery_closure','recovery_closure'),h.LOSSES));proof.update(original_recovery_registration=cs[0]['recovery_registration'],previous_recovery_registration=cs[1]['recovery_registration'],recovery_registration=cs[2]['recovery_registration'])
 cost={'completion_gate':{'recovery_execution_proof':proof},'source_bound_lost_f3_attempts':primary,'source_bound_lost_parent_observer_scopes':parents,'cost_records':records}
 actual=h.verify_loss_chain(reader.checked,cost)
 ok('three genuine closure JSON interfaces pass synthetic cost shell',actual['passed'] is True and len(actual['three_loss_events'])==3 and len(actual['lost_parent_scopes'])==2)
 ok('actual three lower-bound deltas remain separate and not added',[(c['known_extra_lower_bounds']['optimizer_updates'],c['known_extra_lower_bounds']['backward_calls']) for c in cs]==[(4,27),(4,26),(4,27)] and all(x['counts_added_again'] is False and x['producer_terminal'] is False for x in actual['three_loss_events']))
 ok('actual lost parent producer bytes running and elapsed absent-or-null',all(reader.checked(x['artifact'])['status']=='running' and reader.checked(x['artifact']).get('seconds') is None and reader.checked(x['artifact']).get('returncode') is None for x in actual['lost_parent_scopes']))
 def changed(fn):
  c=copy.deepcopy(cost);fn(c);return h.verify_loss_chain(reader.checked,c)
 reject('missing actual loss rejected',lambda:changed(lambda c:c['source_bound_lost_f3_attempts'].pop()))
 reject('duplicate loss primary rejected',lambda:changed(lambda c:c['source_bound_lost_f3_attempts'].append(copy.deepcopy(c['source_bound_lost_f3_attempts'][0]))))
 reject('lost parent incorrectly promoted to primary rejected',lambda:changed(lambda c:next(r for r in c['cost_records'] if r['category']=='dispatcher_elapsed').update(aggregation_role='primary_observed_invocation')))
 reject('lost parent numeric zero filling rejected',lambda:changed(lambda c:next(r for r in c['cost_records'] if r['category']=='dispatcher_elapsed').update(seconds_observed=0)))
 reject('extra returns added twice rejected',lambda:changed(lambda c:c['source_bound_lost_f3_attempts'][2].update(extra_work_not_added_again=False)))
 reject('loss order/ancestor mismatch rejected',lambda:changed(lambda c:c['completion_gate']['recovery_execution_proof'].update(previous_recovery_closure=h.LOSSES[0])))
 # G3 and noon reuse remain tied to actual old evidence; no recollection entry.
 g3=read(A/'sync_boundary_v3_cost_companion_1791601316065099800.json')
 ok('old actual G3 zero-model deferral retained',g3['source']==g.OLD_G3_SOURCE and g3['passed'] is True and g3['status']=='verified_terminal_control_cost' and g3['primary_engineering_ledgers']==[])
 noon=reader.checked(h.NOON)
 ok('actual noon three timer terminal report preserved separately',noon['source']==h.NOON_SOURCE and noon['passed'] is True and noon['all_affected_producer_timers_terminal'] is True and len(noon['affected_timer_observations'])==3)
 ok('G3 parent exact completed timer semantics unchanged',dump(node(files[1],'final_parent_timer'))==dump(node(A/'sync_boundary_v3_cost_companion_v1.py','final_parent_timer')))
 ok('G3 no collect entry and fixed original ancestry binding',not hasattr(g,'collect') and 'original_recovery_registration' in dump(node(files[1],'build_bundle')) and 'original_recovery_closure' in dump(node(files[1],'build_bundle')))
 for current,old,n in [(files[2],A/'audit_memory_pause_final_delivery_v1.py','verify_resume'),(files[2],A/'audit_memory_pause_final_delivery_v1.py','recorded_formal_subset'),(files[4],A/'audit_system_standby_final_delivery_v1.py','verify_events'),(files[4],A/'audit_system_standby_final_delivery_v1.py','verify_native_observation'),(files[4],A/'audit_system_standby_final_delivery_v1.py','verify_scope_closure'),(files[4],A/'audit_system_standby_final_delivery_v1.py','formal_subset')]:
  ok(current.name+' '+n+' original exact AST retained',dump(node(current,n))==dump(node(old,n)))
 old=node(A/'audit_memory_pause_final_delivery_v1.py','verify_pause_timers')
 for x in ast.walk(old):
  if isinstance(x,ast.Constant) and x.value=='pause_memory_resume_cost_companion_v1':x.value='pause_memory_resume_cost_companion_v2'
 ok('original noon accounting guard only companion schema adapted',dump(old)==dump(node(files[2],'verify_pause_timers')))
 for p in (files[2],files[4]):
  fn=node(p,'main'); strict_if=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.BoolOp) and any(isinstance(x,ast.Attribute) and x.attr=='require_complete' for x in ast.walk(n.test))); prep_if=next(n for n in fn.body if isinstance(n,ast.If) and isinstance(n.test,ast.UnaryOp))
  ok(p.name+' strict-complete cannot default to preparation',strict_if.lineno<prep_if.lineno and any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='error' for n in ast.walk(strict_if)))
 scope=dump(node(files[4],'verify_scope_closure'))
 ok('full-range event pairing and additional-fit barrier intact',all(t in scope for t in ['event_range_snapshots','full_formal_run_event_range_reviewed','additional_affected_fit_ids','unresolved_scenarios','registration','captured_at','Gap in final event range coverage']))
 ok('lost LUNA standby branch requires exact loss and preserves unknowns',all(t in dump(node(files[4],'verify_standby')) for t in ['lost_event_closed_unknown_elapsed','loss_closure','producer_terminal','producer_status','recorded_elapsed_including_observed_low_power_window_seconds','eligible_for_active_known_subtotal','wall_or_residency_subtracted']))
 ok('same final semantic cost/wide/G3 six-input binding retained',set(mem.INPUT_NAMES)=={'final_cost','wide','semantic_review','g3_bundle','pause_companion','resume_review'} and all(t in dump(node(files[2],'verify_base')) for t in ['SEMANTIC_IDS','audit_source','cost_snapshot','all_plan_wall_seconds_exact','R%02d']))
 reader.recheck()
 result={'status':'final_companion_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':sources,'training_registration':desc(registration),'postprocessing_source_review':desc(prior),'author_contract':desc(contract),'strict_cli_regression':desc(strict),'checks':checks,'check_count':len(checks),'findings':[],'fixed_before_review':['Both --require-complete entrypoints now reject missing --run instead of returning preparation success.'],'scope':'Independent source/stdlib review only. Actual immutable closure/noon/G3 JSON fields were checked against synthetic cost records; this is not collection or verification of an actual final cost, bundle, companion, or delivery certificate.','operations':{'existing_JSON_and_source_reads':True,'checkpoint_byte_reads':0,'Tensor_loads':0,'model_calls':0,'GPU_queries':0,'native_process_queries':0,'test_probability_reads':0,'actual_collect_calls':0,'actual_audit_run_calls':0,'sealed_source_edits':0},'actual_final_delivery_complete':False,'limitations':['Actual cost17/wide7/semantic/E11/CPU16 completion still required.','Final full-range Windows events and independent scope review remain required; any gap, unresolved pair or additional affected fit continues to block this version.','Lost producers retain original running/unknown elapsed and no numeric terminal is fabricated.','Exact active time remains null; parent and nested timers are not added as formal primary work.','Final delivery manifest must bind this independent source review.'],'read_evidence':list(reader.inputs.values())}
 target=A/('independent_final_companion_v2_source_review_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'review':desc(target),'checks':len(checks),'passed':True}))
if __name__=='__main__':main()
