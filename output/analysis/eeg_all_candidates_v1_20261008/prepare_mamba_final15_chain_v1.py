"""Prepare unregistered Mamba15 sources only. No model/process/GPU execution."""
from pathlib import Path
import ast, hashlib, json, re, time
WORK=Path(__file__).resolve().parents[3]; A=Path(__file__).resolve().parent
M=WORK/'tridep_ccml/eeg_all_candidates'; OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
MAP={'deterministic_repair_execution_v8':'deterministic_repair_execution_v9','finetune_v11':'finetune_v12','repair_observation_v8':'repair_observation_v9','repair_runtime_owner_v8':'repair_runtime_owner_v9','repair_owned_worker_v8':'repair_owned_worker_v9','continuation_queue_v13':'continuation_queue_v14','audit_completed_finetune_checkpoints_v15':'audit_completed_finetune_checkpoints_v16','cost_aggregation_v16':'cost_aggregation_v17','evaluate_v10':'evaluate_v11','review_actual_repair_engineering_v7':'review_actual_repair_engineering_v8','repair_postprocessing_gate_v7':'repair_postprocessing_gate_v8'}
LABEL={'observer11':'observer12','Observer11':'Observer12','Actual11':'Actual12','actual11':'actual12','F11':'F12','Q13':'Q14','CPU15':'CPU16','Cost16':'Cost17','cost16':'cost17','eeg_finetune_deterministic_repair_execution_v8':'eeg_finetune_deterministic_repair_execution_v9','eeg_deterministic_repair_queue_v13':'eeg_deterministic_repair_queue_v14'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):return {'path':Path(p).relative_to(WORK).as_posix(),'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def once(s,a,b):
 assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def fun(s,name,body):
 node=next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name)
 return once(s,ast.get_source_segment(s,node),body)
def getfun(s,name):return ast.get_source_segment(s,next(x for x in ast.parse(s).body if isinstance(x,ast.FunctionDef) and x.name==name))
def migrate(s):
 keys={**MAP,**LABEL,'qualify_deterministic_repairs_v9':'qualify_deterministic_repairs_v10'}
 return re.sub('|'.join(re.escape(x) for x in sorted(keys,key=len,reverse=True)),lambda m:keys[m[0]],s)
def pin(s,name,value):
 out,n=re.subn(rf"(?m)^({re.escape(name)} = )'[^']*'",lambda m:m[1]+repr(value),s);assert n==1;return out

def main():
 prior=read(OUT/'continuation_queue_v13/registration.json');assert len(prior['sources'])==175
 originals={x['path']:x['sha256'] for x in prior['sources']}
 assert all(sha(WORK/p)==h for p,h in originals.items())
 texts={}; paths={}
 for old,new in MAP.items():
  folder=M if (M/(old+'.py')).exists() else A
  paths[new]=folder/(new+'.py');assert not paths[new].exists()
  texts[new]=migrate((folder/(old+'.py')).read_text(encoding='utf-8'))
 reg=desc(OUT/'continuation_queue_v13/registration.json'); terminal=desc(OUT/'continuation_queue_v13/status.json')
 source_review=desc(A/'independent_codebrain_only_partial_queue_source_review_1791784413249755100.json')
 reduction_review=desc(A/'independent_mamba_backward_reductions_qualification_review_1791824525780271400.json')
 d=texts['deterministic_repair_execution_v9']
 d=pin(d,'QUALIFIER_SHA',sha(M/'qualify_deterministic_repairs_v10.py'))
 d=d.replace("NEW_ENTRY_NAMES = ('finetune_v12.py', 'continuation_queue_v14.py')","NEW_ENTRY_NAMES = ('finetune_v12.py', 'continuation_queue_v14.py', 'evaluate_v11.py')")
 d=d.replace("'cost_aggregation_v17.py')","'cost_aggregation_v17.py', 'review_actual_repair_engineering_v8.py', 'repair_postprocessing_gate_v8.py')")
 d=d.replace("condition == REPAIR_CONDITIONS[0], 'This partial execution admits only the genuinely qualified CodeBrain15'","condition == REPAIR_CONDITIONS[1], 'Only original Mamba15 uses this new execution'")
 d=d.replace("condition == REPAIR_CONDITIONS[0], 'Mamba has no qualified partial execution; evaluation remains prohibited'","condition == REPAIR_CONDITIONS[1], 'Only original Mamba15 uses actual F12'")
 d=d.replace("expected_mamba = '.adapters.eegmamba_v4'", "expected_mamba = '.adapters.eegmamba_v5'")
 d=d.replace("    provider.check_boundary()", "    previous_codebrain_completion()")
 d=d.replace("value.get('version') == 8","value.get('version') == 9").replace("'version': 11","'version': 12").replace("'observer_version': 11","'observer_version': 12").replace("'new_actual_observer_version': 11","'new_actual_observer_version': 12")
 start=d.index('    prior_binding = ',d.index('def implementation_sources():'));end=d.index('    for name in NEW_ENTRY_NAMES:',start)
 d=d[:start]+"    prior_binding = "+repr(reg)+"\n    prior = read_bound(prior_binding)\n    require(prior.get('version') == 13 and len(prior['sources']) == 175, 'Actual Q13 original175 sources required')\n    records = {x['path']: x for x in prior['sources']}\n    records[prior_binding['path']] = prior_binding\n    for binding in "+repr([source_review,reduction_review])+":\n        review = read_bound(binding)\n        require(review.get('passed') is True, 'Preserved independent source review required')\n        records[binding['path']] = binding\n    diagnostic = S.desc(SOURCE.with_name('diagnose_mamba_repeat_v1.py'))\n    records[diagnostic['path']] = diagnostic\n"+d[end:]
 previous_fn='''def previous_codebrain_completion():
    """Validate released true Q13/CPU15 without requiring its old pending rows to stay blocked."""
    PREVIOUS_Q.check_boundary()
    root = S.OUT / 'continuation_queue_v13'
    require(not (root / 'owner.json').exists(), 'Actual CodeBrain owner must remain released')
    reg_binding = REG_BINDING
    terminal_binding = TERMINAL_BINDING
    reg = read_bound(reg_binding)
    state = read_bound(terminal_binding)
    require(reg['version'] == 13 and len(reg['sources']) == 175 and reg['registered_conditions'] == [REPAIR_CONDITIONS[0]], 'Original CodeBrain-only registration required')
    require(state['status'] == 'codebrain15_complete_mamba15_blocked_no_evaluation'
            and state['source'] == S.desc(SOURCE.with_name('continuation_queue_v13.py'))
            and state['registration'] == reg_binding and state['execution_version'] == 13
            and state['test_evaluation_frozen'] is True and state['all_plan_complete'] is False
            and set(state['outcomes']) == {REPAIR_CONDITIONS[0]}, 'True historical Q13 terminal required')
    child = read_bound(state['outcomes'][REPAIR_CONDITIONS[0]])
    require(child['status'] == 'complete' and type(child['returncode']) is int and child['returncode'] == 0, 'CodeBrain child must have actual exit0')
    cpu = read_bound(state['partial_CPU_audit'])
    require(state['partial_CPU_audit']['sha256'] == 'b2e46f49da6cd79f09de15077641e15b4124fe9545d813b6ef45ea6c6834ab1e'
            and cpu['schema_version'] == 15 and cpu['verified_completed_finetune_fits'] == 120
            and cpu['checkpoint_loads'] == 360 and cpu['errors'] == [] and cpu['deferred_completed_fits'] == []
            and cpu['all_135_finetune_checkpoints_verified'] is False, 'Actual prior CPU15 proof required')
    from . import registry as R
    fits = [x for x in R.load_registration()['fits'] if x['condition'] == REPAIR_CONDITIONS[0]]
    require(len(fits) == 15, 'Original CodeBrain15 identities required')
    for fit in fits:
        status = R.fit_status(fit['condition'], fit['seed'], fit['fold'])
        require(status['status'] == 'training_complete', 'CodeBrain completion changed')
        R._validate_complete(status['evidence'], R.fit_directory(fit['condition'], fit['seed'], fit['fold']))
    return {'registration': reg_binding, 'terminal': terminal_binding, 'CPU_audit': state['partial_CPU_audit'], 'child': state['outcomes'][REPAIR_CONDITIONS[0]], 'formal_fits_verified': 15}
'''.replace('REG_BINDING',repr(reg)).replace('TERMINAL_BINDING',repr(terminal))
 d=once(d,'def qualification(binding, condition):',previous_fn+'\ndef qualification(binding, condition):')
 d=once(d,"    contract = validate_report_data(value, condition, fit)","    contract = validate_report_data(value, condition, fit)\n    require(value.get('eligible_for_formal_registration') is True and value.get('diagnostic_only') is False, 'Actual Qual10 success required, not diagnostic or source preparation')")
 texts['deterministic_repair_execution_v9']=d
 f=texts['finetune_v12'].replace("'version': 11","'version': 12").replace("['observer_version'] == 11","['observer_version'] == 12").replace("'actual_observer_version': 11","'actual_observer_version': 12")
 f=f.replace("condition == D.REPAIR_CONDITIONS[0], 'Only original CodeBrain15 is registered in this partial runner'","condition == D.REPAIR_CONDITIONS[1], 'Only original Mamba15 is registered in this runner'")
 f=once(f,"        self.record['implementation'] = provenance(self.record['execution_acceleration'])","        if getattr(self, '_native_snapshot', None) is not None:\n            self.record['native_backward_supplementary'] = native_delta(self._native_before, self._native_snapshot())\n        self.record['implementation'] = provenance(self.record['execution_acceleration'])")
 native_fn='''def native_delta(before, after):
    require(len(before) == len(after) == 12, 'Exact twelve-layer Mamba snapshot required')
    rows = []
    for left, right in zip(before, after):
        require(left['layer_index'] == right['layer_index'] == len(rows) and set(left['counts']) == set(right['counts']), 'Native snapshot layer/counter scope changed')
        delta = {}
        for key, value in right['counts'].items():
            initial = left['counts'][key]
            require(type(value) is int and type(initial) is int and value >= initial >= 0, 'Native snapshot counters must be monotonic integers')
            delta[key] = value - initial
        rows.append({'layer_index': left['layer_index'], 'counts': delta})
    return {'before': before, 'after': after, 'delta': rows, 'scope': 'native Python backward functions nested in model backward; CUDA/autotuner internal launches unknown', 'supplementary_only': True, 'added_to_model_backward_or_seconds': False}

'''
 f=once(f,'class RepairAttempt(',native_fn+'class RepairAttempt(')
 f=once(f,"                model.forward = trace.observe('forward', model.forward)","                trace._native_snapshot = lambda: model.adapter.snapshot_native_backward_counts(model.encoder)\n                trace._native_before = trace._native_snapshot()\n                model.forward = trace.observe('forward', model.forward)")
 f=once(f,"                    value['backward_repair_qualification'] = execution['qualification']","                    value['backward_repair_qualification'] = execution['qualification']\n                    value['native_backward_supplementary'] = native_delta(trace._native_before, trace._native_snapshot())")
 texts['finetune_v12']=f
 v=texts['repair_observation_v9'].replace('== 11','== 12').replace('[11]','[12]').replace("'observer_version': 11","'observer_version': 12")
 texts['repair_observation_v9']=v
 o=texts['repair_runtime_owner_v9'].replace("get('execution_version') == 13","get('execution_version') == 14").replace("registration['version'] == 13","registration['version'] == 14")
 o=o.replace("condition == 'CodeBrain-lastblock-finetune' and sys.platform == 'win32', 'Only the qualified CodeBrain Windows worker is admitted'","condition == 'EEGMamba-lastblock-finetune' and sys.platform == 'linux', 'Only the qualified Mamba Linux worker is admitted'")
 o=o.replace("require(not (S.OUT / 'continuation_queue_v5/owner.json').exists(),", "require(not (S.OUT / 'continuation_queue_v5/owner.json').exists() and not (S.OUT / 'continuation_queue_v13/owner.json').exists(),")
 texts['repair_runtime_owner_v9']=o
 w=texts['repair_owned_worker_v9'].replace("condition == 'CodeBrain-lastblock-finetune' and sys.platform == 'win32', 'Mamba/evaluation are excluded from this partial queue'","condition == 'EEGMamba-lastblock-finetune' and sys.platform == 'linux', 'Only Mamba15 is admitted; CodeBrain/evaluation excluded'")
 w=once(w,"'actual_queue_owner': owner}","'actual_queue_owner': owner, 'native_backward_supplementary': result.get('native_backward_supplementary'), 'native_counts_not_added_to_model_totals': True}")
 texts['repair_owned_worker_v9']=w
 q=texts['continuation_queue_v14']
 q=once(q,ast.get_source_segment(q,ast.parse(q).body[0]),'"""Final original Mamba15 queue, preserving completed CodeBrain/Q13 provenance.\nDefault zero operation. No test evaluation before true360/CPU16 gate.\n"""')
 q=q.replace("CONDITIONS = ('CodeBrain-lastblock-finetune',)","CONDITIONS = ('EEGMamba-lastblock-finetune',)")
 q=q.replace("    Q.check_boundary()","    D.previous_codebrain_completion()")
 q=q.replace("x['condition'] not in D.REPAIR_CONDITIONS", "x['condition'] not in CONDITIONS").replace("len(others) == 330", "len(others) == 345").replace('Original non-repair330 identity scope differs','Original completed345 identity scope differs')
 q=fun(q,'pending_mamba',getfun(q,'pending_mamba').replace('pending_mamba','unstarted_mamba'))
 q=q.replace('blocked = pending_mamba()', 'blocked = unstarted_mamba()')
 q=q.replace("Only the original qualified CodeBrain15 group may be registered","Only original qualified Mamba15 may be registered")
 q=q.replace("'version': 8","'version': 9").replace("'version': 13","'version': 14").replace("['version'] == 13","['version'] == 14").replace("'execution_version': 13","'execution_version': 14").replace("'new_actual_observer_version': 11","'new_actual_observer_version': 12")
 q=q.replace("'original_other330_fit_ids': others","'original_other345_fit_ids': others, 'previous_CodeBrain_completion': D.previous_codebrain_completion()")
 q=q.replace("'mamba_qualification_passed': False, 'evaluation_permanently_disabled': True","'mamba_qualification_passed': True, 'evaluation_in_separate_owned_entry_after_CPU_gate': True")
 q=q.replace("'partial_CPU_audit_expected': {'fits': 120, 'loads': 360, 'version': 15}","'CPU_gate_required': {'fits': 135, 'loads': 405, 'version': 16}")
 q=q.replace("and value['evaluation_permanently_disabled'] is True, 'Exact CodeBrain-only registration required'","and value['evaluation_in_separate_owned_entry_after_CPU_gate'] is True, 'Exact Mamba-only registration required'")
 q=once(q,"    require(value['pending_mamba_statuses'] == pending_mamba(), 'Blocked Mamba state changed')","    require(value['previous_CodeBrain_completion'] == D.previous_codebrain_completion(), 'Historical CodeBrain provenance changed')")
 q=q.replace("'This queue cannot start Mamba or an evaluator'","'This queue starts only original Mamba15'")
 gate=getfun((M/'continuation_queue_v12.py').read_text(encoding='utf-8'),'cpu_gate').replace('audit_completed_finetune_checkpoints_v14','audit_completed_finetune_checkpoints_v16').replace('CPU14','CPU16')
 gate=gate.replace("report['auditor_source'] ==", "report['schema_version'] == 16 and report['auditor_source'] ==")
 gate=gate.replace("'CPU_audit': output['artifact']", "'CPU_audit': output['artifact'], 'previous_CodeBrain_completion': D.previous_codebrain_completion()")
 q=fun(q,'cpu_partial_audit',gate)
 q=once(q,"        partial = cpu_partial_audit()\n        require(value['pending_mamba_statuses'] == pending_mamba(), 'Blocked Mamba bytes changed during CodeBrain work')\n        record.update(status='codebrain15_complete_mamba15_blocked_no_evaluation', partial_CPU_audit=partial, finished_at=S.now(), test_evaluation_frozen=True, evaluation_permanently_disabled=True, full360_CPU_gate_passed=False, all_plan_complete=False)","        gate = cpu_gate()\n        record.update(status='all_training_CPU_gate_complete_evaluation_pending', CPU_gate=gate, previous_CodeBrain_completion=D.previous_codebrain_completion(), finished_at=S.now(), test_evaluation_frozen=True, full360_CPU_gate_passed=True, all_plan_complete=False)")
 q=q.replace('codebrain-report','mamba-report').replace('codebrain-review','mamba-review')
 texts['continuation_queue_v14']=q
 c=texts['audit_completed_finetune_checkpoints_v16']
 c=once(c,"from tridep_ccml.eeg_all_candidates import repair_observation_v9 as V","from tridep_ccml.eeg_all_candidates import repair_observation_v9 as V\nfrom tridep_ccml.eeg_all_candidates import repair_observation_v8 as OLD_V")
 c=c.replace('schema_version=15','schema_version=16').replace('completed_finetune_checkpoint_audit_v15_', 'completed_finetune_checkpoint_audit_v16_')
 c=once(c,"        if 11 not in versions:","        if versions == {11}:\n            require(fit['condition'] == 'CodeBrain-lastblock-finetune', 'Only actual CodeBrain uses F11')\n            return OLD_V.trace_boundary(base.legacy_trace_boundary, fit, metadata, inputs)\n        if 12 not in versions:")
 c=c.replace("versions == {11} and fit['condition'] in V.D.REPAIR_CONDITIONS", "versions == {12} and fit['condition'] == 'EEGMamba-lastblock-finetune'")
 c=c.replace("        V.source_check()", "        V.source_check()\n        OLD_V.source_check()")
 c=c.replace("'actual_CPU16': S.desc(SOURCE)","'actual_CPU16': S.desc(SOURCE), 'actual_F11_observer_verifier': S.desc(OLD_V.SOURCE)")
 c=c.replace('actual_observer11_sources_verified', 'actual_observer11_and12_sources_verified').replace('actual_observer11_never', 'actual_observer11_and12_never')
 texts['audit_completed_finetune_checkpoints_v16']=c
 cost=texts['cost_aggregation_v17']
 cost=once(cost,'from tridep_ccml.eeg_all_candidates import repair_observation_v9 as V', 'from tridep_ccml.eeg_all_candidates import repair_observation_v9 as V\nfrom tridep_ccml.eeg_all_candidates import repair_observation_v8 as OLD_V\nfrom tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v9 as PREVIOUS_REPEAT_Q\nfrom tridep_ccml.eeg_all_candidates import diagnose_mamba_repeat_v1 as DIAG')
 cost=once(cost,"    if data.get('implementation', {}).get('version') != 11:\n        return old(data, artifact)","    version = data.get('implementation', {}).get('version')\n    if version not in (11, 12):\n        return old(data, artifact)\n    provider = OLD_V if version == 11 else V\n    require(data['condition'] == ('CodeBrain-lastblock-finetune' if version == 11 else 'EEGMamba-lastblock-finetune'), 'Actual repair observer/condition mismatch')")
 cost=cost.replace('proof = V.verify_observer(data, artifact, fit)', 'proof = provider.verify_observer(data, artifact, fit)')
 cost=cost.replace(" else Q\n    require(any", " else PREVIOUS_REPEAT_Q if '/deterministic_repair_qualification_v9/' in artifact['path'] else DIAG if '/mamba_repeat_diagnosis_v1/' in artifact['path'] else Q\n    require(any")
 cost=cost.replace('for v in (4, 5, 6, 7, 8, 9))),','for v in (4, 5, 6, 7, 8, 9, 10))) or \'/mamba_repeat_diagnosis_v1/\' in artifact[\'path\'],')
 cost=once(cost,"    for pattern in patterns:","    patterns += ('deterministic_repair_qualification_v9/*/*/attempt.json', 'deterministic_repair_qualification_v9/*/*/controller.json', 'mamba_repeat_diagnosis_v1/*/*/attempt.json', 'mamba_repeat_diagnosis_v1/*/*/controller.json', 'continuation_queue_v13/attempts/*/status.json', 'continuation_queue_v13/attempts/*/worker.json')\n    require(len(set(patterns)) == len(patterns), 'Duplicate engineering ledger patterns')\n    for pattern in patterns:")
 cost=cost.replace("('continuation_queue_v14/attempts/' in artifact['path'])", "any(prefix in artifact['path'] for prefix in ('continuation_queue_v13/attempts/', 'continuation_queue_v14/attempts/'))")
 cost=cost.replace('schema_version=16','schema_version=17').replace('actual_cost_v16_', 'actual_cost_v17_')
 cost=cost.replace('actual12_versions_not_relabelled=True','actual11_12_versions_not_relabelled=True, prior_CodeBrain_observer_source=S.desc(OLD_V.SOURCE)')
 cost=once(cost,"                added.append(row)","                if value.get('native_backward_function_counts') is not None or value.get('native_backward_supplementary') is not None:\n                    row['native_backward_supplementary'] = value.get('native_backward_function_counts', value.get('native_backward_supplementary'))\n                    row['native_calls_not_added_to_model_backward_or_seconds'] = True\n                added.append(row)")
 texts['cost_aggregation_v17']=cost
 e=texts['evaluate_v11']
 e=e.replace('repair_observation_v7 as V','repair_observation_v9 as V\nfrom . import repair_observation_v8 as OLD_V\nfrom . import deterministic_repair_execution_v9 as D')
 e=e.replace('continuation_queue_v12','continuation_queue_v14').replace('CPU14','CPU16').replace('audit_completed_finetune_checkpoints_v14','audit_completed_finetune_checkpoints_v16').replace('E10','E11').replace('Q12','Q14').replace('evaluation_v10_control','evaluation_v11_control')
 e=e.replace("'version': 10", "'version': 11").replace("get('schema_version') == 14", "get('schema_version') == 16").replace("queue['version'] == 12", "queue['version'] == 14").replace('eeg_deterministic_repair_queue_v12','eeg_deterministic_repair_queue_v14')
 e=e.replace("'new_repair_execution_verifier_source': S.desc(V.SOURCE)","'new_repair_execution_verifier_source': S.desc(V.SOURCE), 'CodeBrain_F11_verifier_source': S.desc(OLD_V.SOURCE)").replace('old_and_repair10_training_provenance_verified','old_and_F11_F12_training_provenance_verified')
 e=once(e,"        return V.verify_execution(metadata, condition=metadata['condition'], seed=metadata['seed'], fold=metadata['fold'])","        version = metadata['actual_deterministic_repair_execution'].get('observer_version')\n        require((version, metadata['condition']) in ((11, 'CodeBrain-lastblock-finetune'), (12, 'EEGMamba-lastblock-finetune')), 'Actual mixed F11/F12 condition/source scope differs')\n        provider = OLD_V if version == 11 else V\n        return provider.verify_execution(metadata, condition=metadata['condition'], seed=metadata['seed'], fold=metadata['fold'])")
 e=once(e,"    gate = S.read(path)","    gate = S.read(path)\n    require(gate.get('previous_CodeBrain_completion') == D.previous_codebrain_completion(), 'True completed Q13 lineage required')\n    terminal = S.read(S.OUT / 'continuation_queue_v14/status.json')\n    require(terminal.get('status') == 'all_training_CPU_gate_complete_evaluation_pending' and terminal.get('source') == S.desc(SOURCE.with_name('continuation_queue_v14.py')) and terminal.get('CPU_gate') == binding and terminal.get('execution_version') == 14, 'Actual Q14 terminal and native CPU gate required')")
 texts['evaluate_v11']=e
 eng=texts['review_actual_repair_engineering_v8'].replace('deterministic_repair_execution_v7 as D','deterministic_repair_execution_v9 as D').replace('    Q.check_boundary()','    D.previous_codebrain_completion()')
 texts['review_actual_repair_engineering_v8']=eng
 post=texts['repair_postprocessing_gate_v8'].replace('continuation_queue_v12','continuation_queue_v14').replace('audit_completed_finetune_checkpoints_v14','audit_completed_finetune_checkpoints_v16').replace('cost_aggregation_v15','cost_aggregation_v17').replace('qualify_deterministic_repairs_v9','qualify_deterministic_repairs_v10').replace('E10','E11').replace('Q12','Q14').replace('CPU14','CPU16').replace('evaluation_v10_control','evaluation_v11_control').replace('deterministic_repair_qualification_v9','deterministic_repair_qualification_v10').replace("get('schema_version') == 14","get('schema_version') == 16").replace("'q9_registration'","'q14_registration'").replace("'q9_terminal'","'q14_terminal'")
 post=post.replace("'continuation_queue_v14/owner.json'", "'continuation_queue_v13/owner.json', 'continuation_queue_v14/owner.json'")
 texts['repair_postprocessing_gate_v8']=post
 order=['finetune_v12','repair_runtime_owner_v9','repair_owned_worker_v9','continuation_queue_v14','deterministic_repair_execution_v9','repair_observation_v9','audit_completed_finetune_checkpoints_v16','cost_aggregation_v17','evaluate_v11','review_actual_repair_engineering_v8','repair_postprocessing_gate_v8']
 for name in order:
  text=texts[name]
  if name=='repair_observation_v9':text=pin(pin(text,'F4_SHA',sha(paths['finetune_v12'])),'D_SHA',sha(paths['deterministic_repair_execution_v9']))
  if name=='repair_postprocessing_gate_v8':
   tree=ast.parse(text);node=next(x for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in x.targets))
   names=['continuation_queue_v14','evaluate_v11','audit_completed_finetune_checkpoints_v16','cost_aggregation_v17']
   expr='PINS = {'+', '.join('S.WORK / '+repr(desc(paths[n])['path'])+': '+repr(sha(paths[n])) for n in names)+'}'
   text=once(text,ast.get_source_segment(text,node),expr)
  compile(text,str(paths[name]),'exec')
  with paths[name].open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
 assert all(sha(WORK/p)==h for p,h in originals.items())
 proof={'schema':'mamba_final15_source_preparation_v1','source':desc(Path(__file__)),'prior_Q13_registration':reg,'prior175_sources_preserved':True,'new_sources':[desc(paths[n]) for n in order],'actual_qualification10':desc(OUT/'deterministic_repair_qualification_v10/EEGMamba-lastblock-finetune/1791824574948282100/report.json'),'formal_scope':'original Mamba15 only; completed CodeBrain15 preserved','model_calls':0,'GPU_queries':0,'process_queries':0,'actual_registration_created':False,'independent_review_passed':False}
 target=A/('mamba_final15_source_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
 print(json.dumps({'artifact':desc(target),'new_sources':proof['new_sources']}))
if __name__=='__main__':main()
