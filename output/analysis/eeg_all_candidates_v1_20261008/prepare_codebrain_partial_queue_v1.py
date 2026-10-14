"""Prepare only a CodeBrain15 queue; no runtime, registration or evaluation."""
from pathlib import Path
import ast,hashlib,json,re,time
WORK=Path(__file__).resolve().parents[3];M=WORK/'tridep_ccml/eeg_all_candidates';A=Path(__file__).resolve().parent
PRIOR=A/'independent_full_module_compile_repair_chain_source_review_1791783895805780100.json'
PRIOR_SHA='40a39257cf551b776c90843f8ed3778e13a7184dd3a106f1f0fe38d220ee9ea9'
MAP={'deterministic_repair_execution_v7':'deterministic_repair_execution_v8','finetune_v10':'finetune_v11','repair_observation_v7':'repair_observation_v8','repair_runtime_owner_v7':'repair_runtime_owner_v8','repair_owned_worker_v7':'repair_owned_worker_v8','continuation_queue_v12':'continuation_queue_v13','audit_completed_finetune_checkpoints_v14':'audit_completed_finetune_checkpoints_v15','cost_aggregation_v15':'cost_aggregation_v16'}
LABELS={'observer10':'observer11','Observer10':'Observer11','observer_version10':'observer_version11','Actual10':'Actual11','actual10':'actual11','actual_F10':'actual_F11','repair10':'repair11','F10':'F11','Q12':'Q13','CPU14':'CPU15','Cost15':'Cost16','cost15':'cost16','eeg_finetune_deterministic_repair_execution_v7':'eeg_finetune_deterministic_repair_execution_v8','eeg_deterministic_repair_queue_v12':'eeg_deterministic_repair_queue_v13'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return {'path':Path(path).relative_to(WORK).as_posix(),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def once(text,old,new):
 assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)
def migrate(text):
 values={**MAP,**LABELS};return re.sub('|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True)),lambda m:values[m.group()],text)
def pin(text,name,value):
 result,n=re.subn(rf"(?m)^({re.escape(name)} = )'[^']*'",lambda m:m[1]+repr(value),text);assert n==1;return result
def replace_function(text,name,body):
 node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)
 return once(text,ast.get_source_segment(text,node),body)
def main():
 assert sha(PRIOR)==PRIOR_SHA;prior=read(PRIOR);assert prior['passed'] and len(prior['reviewed_sources'])==166
 original={x['path']:sha(WORK/x['path']) for x in prior['reviewed_sources']};assert all(original[x['path']]==x['sha256'] for x in prior['reviewed_sources'])
 texts={};locations={}
 for old,new in MAP.items():
  folder=M if (M/(old+'.py')).exists() else A;locations[new]=folder/(new+'.py');assert not locations[new].exists()
  texts[new]=migrate((folder/(old+'.py')).read_text(encoding='utf-8'))
 d=texts['deterministic_repair_execution_v8']
 d=d.replace("NEW_ENTRY_NAMES = ('finetune_v11.py', 'continuation_queue_v13.py', 'evaluate_v10.py')","NEW_ENTRY_NAMES = ('finetune_v11.py', 'continuation_queue_v13.py')")
 d=d.replace("value.get('version') == 7","value.get('version') == 8").replace("'version': 10","'version': 11").replace("'observer_version': 10","'observer_version': 11").replace("'new_actual_observer_version': 10","'new_actual_observer_version': 11")
 d=once(d,'def qualification(binding, condition):','def qualification(binding, condition):\n    require(condition == REPAIR_CONDITIONS[0], \'This partial execution admits only the genuinely qualified CodeBrain15\')')
 d=once(d,'def load_execution(binding, *, condition, seed, fold):','def load_execution(binding, *, condition, seed, fold):\n    require(condition == REPAIR_CONDITIONS[0], \'Mamba has no qualified partial execution; evaluation remains prohibited\')')
 start=d.index('    prior_binding = ',d.index('def implementation_sources():'));end=d.index('    for name in NEW_ENTRY_NAMES:',start)
 d=d[:start]+"    prior_binding = "+repr(desc(PRIOR))+"\n    prior = read_bound(prior_binding)\n    require(prior.get('passed') is True and len(prior['reviewed_sources']) == 166, 'Previous166 reviewed source lineage required')\n    records = {x['path']: x for x in prior['reviewed_sources']}\n    records[prior_binding['path']] = prior_binding\n"+d[end:]
 texts['deterministic_repair_execution_v8']=d
 f=texts['finetune_v11'].replace("'version': 10","'version': 11").replace("['observer_version'] == 10","['observer_version'] == 11").replace("'actual_observer_version': 10","'actual_observer_version': 11")
 f=once(f,'def run(condition, *, repair_registration, resume=False, limit=None):','def run(condition, *, repair_registration, resume=False, limit=None):\n    require(condition == D.REPAIR_CONDITIONS[0], \'Only original CodeBrain15 is registered in this partial runner\')')
 texts['finetune_v11']=f
 texts['repair_observation_v8']=texts['repair_observation_v8'].replace('== 10','== 11').replace('[10]','[11]').replace("'observer_version': 10","'observer_version': 11")
 o=texts['repair_runtime_owner_v8'].replace("get('execution_version') == 12","get('execution_version') == 13").replace("registration['version'] == 12","registration['version'] == 13")
 o=once(o,'def enter(binding, token, *, condition, repair_registration):','def enter(binding, token, *, condition, repair_registration):\n    require(condition == \'CodeBrain-lastblock-finetune\' and sys.platform == \'win32\', \'Only the qualified CodeBrain Windows worker is admitted\')')
 texts['repair_runtime_owner_v8']=o
 w=texts['repair_owned_worker_v8'];w=once(w,'def run(*, condition, repair_registration, admission, token, resume=False):','def run(*, condition, repair_registration, admission, token, resume=False):\n    require(condition == \'CodeBrain-lastblock-finetune\' and sys.platform == \'win32\', \'Mamba/evaluation are excluded from this partial queue\')');texts['repair_owned_worker_v8']=w
 texts['audit_completed_finetune_checkpoints_v15']=texts['audit_completed_finetune_checkpoints_v15'].replace('if 10 not in versions:','if 11 not in versions:').replace('versions == {10}','versions == {11}').replace('schema_version=14','schema_version=15').replace('completed_finetune_checkpoint_audit_v14_','completed_finetune_checkpoint_audit_v15_')
 texts['cost_aggregation_v16']=texts['cost_aggregation_v16'].replace('!= 10','!= 11').replace('schema_version=15','schema_version=16').replace('actual_cost_v15_','actual_cost_v16_')
 q=texts['continuation_queue_v13']
 doc=ast.get_source_segment(q,ast.parse(q).body[0]);q=once(q,doc,'"""CodeBrain-only original15 continuation; Mamba remains blocked; no evaluation.\n\nThis partial queue reuses actual Qual5 CodeBrain evidence and never requires or\nclaims Mamba qualification. Old sources/statuses and all original360 identities\nremain unchanged. Default performs no runtime or registration operation.\n"""')
 q=once(q,"SCHEMA = 'eeg_deterministic_repair_queue_v13'","SCHEMA = 'eeg_deterministic_repair_queue_v13'\nCONDITIONS = ('CodeBrain-lastblock-finetune',)\nMAMBA = 'EEGMamba-lastblock-finetune'")
 q=once(q,'def register(qualifications, reviews, source_review):',"""def pending_mamba():
    fits = [fit for fit in R.load_registration()['fits'] if fit['condition'] == MAMBA]
    require(len(fits) == 15, 'Original Mamba15 identities required')
    values = []
    for fit in fits:
        folder = S.OUT / 'runs' / fit['fit_id']
        status = S.read(folder / 'status.json')
        require(status['status'] == 'blocked' and not (folder / 'started.json').exists()
                and not (folder / 'complete.json').exists(), 'Unqualified Mamba trajectory changed')
        values.append(S.desc(folder / 'status.json'))
    return values

def register(qualifications, reviews, source_review):""")
 q=q.replace('set(D.REPAIR_CONDITIONS)', 'set(CONDITIONS)').replace('Both original repair15 groups required','Only the original qualified CodeBrain15 group may be registered')
 q=q.replace('for condition in D.REPAIR_CONDITIONS:', 'for condition in CONDITIONS:').replace('for condition in D.REPAIR_CONDITIONS}', 'for condition in CONDITIONS}').replace('for c in D.REPAIR_CONDITIONS for x', 'for c in CONDITIONS for x')
 q=once(q,'    others = original_others_complete()','    others = original_others_complete()\n    blocked = pending_mamba()')
 for old,new in [("'version': 7","'version': 8"),("'version': 12","'version': 13"),("['version'] == 12","['version'] == 13"),("'execution_version': 12","'execution_version': 13"),("'new_actual_observer_version': 10","'new_actual_observer_version': 11")]:q=q.replace(old,new)
 q=q.replace("'formal_fits_to_run': 30","'formal_fits_to_run': 15, 'registered_conditions': list(CONDITIONS), 'pending_mamba_statuses': blocked, 'mamba_qualification_passed': False, 'evaluation_permanently_disabled': True")
 q=q.replace("'CPU_gate_required': {'fits': 135, 'loads': 405, 'version': 14}","'partial_CPU_audit_expected': {'fits': 120, 'loads': 360, 'version': 15}, 'full360_CPU_gate_passed': False")
 q=once(q,"    require(not (ROOT / 'owner.json').exists(), 'Actual Q13 owner already present; do not duplicate queue')","    require(value['formal_fits_to_run'] == 15 and value['registered_conditions'] == list(CONDITIONS)\n            and set(value['repair_registrations']) == set(CONDITIONS)\n            and value['evaluation_permanently_disabled'] is True, 'Exact CodeBrain-only registration required')\n    require(value['pending_mamba_statuses'] == pending_mamba(), 'Blocked Mamba state changed')\n    require(not (ROOT / 'owner.json').exists(), 'Actual Q13 owner already present; do not duplicate queue')")
 q=once(q,'def child(condition, repair_registration, owner, resume):','def child(condition, repair_registration, owner, resume):\n    require(condition in CONDITIONS, \'This queue cannot start Mamba or an evaluator\')')
 q=replace_function(q,'cpu_gate',"""def cpu_partial_audit():
    for item in S.read(ROOT / 'registration.json')['sources']:
        S.checked(item)
    require(R.status_summary()['statuses'].get('training_complete') == 345,
            'Only original330 plus CodeBrain15 may be complete here')
    path = S.REPORT / 'audit_completed_finetune_checkpoints_v15.py'
    result = subprocess.run([str(S.WORK / 'S00/.venv/Scripts/python.exe'), '-B', '-u', str(path), '--run'],
                            cwd=S.WORK, capture_output=True, text=True, check=True)
    output = json.loads(result.stdout.strip().splitlines()[-1])
    report = S.read(S.checked(output['artifact']))
    require(report['schema_version'] == 15 and report['auditor_source'] == S.desc(path)
            and report['verified_completed_finetune_fits'] == 120
            and report['checkpoint_loads'] == 360 and report['errors'] == []
            and report['deferred_completed_fits'] == [] and len(report['pending_fits']) == 15
            and report['all_135_finetune_checkpoints_verified'] is False,
            'Actual partial CPU15 audit differs; this is never a full135/405 test gate')
    return output['artifact']""")
 q=once(q,"        gate = cpu_gate()\n        record.update(status='all_training_CPU_gate_complete_evaluation_pending', CPU_gate=gate, finished_at=S.now(), test_evaluation_frozen=True, all_plan_complete=False)","        partial = cpu_partial_audit()\n        require(value['pending_mamba_statuses'] == pending_mamba(), 'Blocked Mamba bytes changed during CodeBrain work')\n        record.update(status='codebrain15_complete_mamba15_blocked_no_evaluation', partial_CPU_audit=partial, finished_at=S.now(), test_evaluation_frozen=True, evaluation_permanently_disabled=True, full360_CPU_gate_passed=False, all_plan_complete=False)")
 q=q.replace("('codebrain-report', 'mamba-report', 'codebrain-review', 'mamba-review', 'source-review')","('codebrain-report', 'codebrain-review', 'source-review')")
 q=once(q,"return register({D.REPAIR_CONDITIONS[0]: binding('codebrain-report'), D.REPAIR_CONDITIONS[1]: binding('mamba-report')}, {D.REPAIR_CONDITIONS[0]: binding('codebrain-review'), D.REPAIR_CONDITIONS[1]: binding('mamba-review')}, binding('source-review'))","return register({CONDITIONS[0]: binding('codebrain-report')}, {CONDITIONS[0]: binding('codebrain-review')}, binding('source-review'))")
 texts['continuation_queue_v13']=q
 order=['finetune_v11','repair_runtime_owner_v8','repair_owned_worker_v8','continuation_queue_v13','deterministic_repair_execution_v8','repair_observation_v8','audit_completed_finetune_checkpoints_v15','cost_aggregation_v16']
 for name in order:
  text=texts[name]
  if name=='repair_observation_v8':text=pin(pin(text,'F4_SHA',sha(locations['finetune_v11'])),'D_SHA',sha(locations['deterministic_repair_execution_v8']))
  compile(text,str(locations[name]),'exec')
  with locations[name].open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
 assert all(sha(WORK/path)==digest for path,digest in original.items())
 proof={'schema':'codebrain_only_source_preparation_v1','previous166_review':desc(PRIOR),'new_sources':[desc(locations[n]) for n in order],'old166_sources_preserved':True,'formal_fit_scope':'original CodeBrain15 only','Mamba15_kept_blocked':True,'evaluation_permanently_disabled':True,'CodeBrain_original_Qual5_reuse_required':True,'process_queries':0,'model_calls':0,'GPU_calls':0,'actual_registration_created':False,'independent_review_passed':False,'source':desc(Path(__file__))}
 target=A/('codebrain_only_source_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
 print(json.dumps({'proof':desc(target),'new_sources':proof['new_sources']}))
if __name__=='__main__':main()
