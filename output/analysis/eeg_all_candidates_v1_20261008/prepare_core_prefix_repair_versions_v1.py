"""Prepare only new source generation; never run qualification or training."""
from pathlib import Path
import ast, hashlib, json, re, time
WORK=Path(__file__).resolve().parents[3]
M=WORK/'tridep_ccml/eeg_all_candidates'
A=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
PRIOR=A/'independent_native_launcher_reuse_chain_source_review_1791782167310882100.json'
PRIOR_SHA='8433357e8b1248129eb05bc1c41f15254f9b054094d3c9e3ec099a5c861aa09e'
MAP={
 'qualify_deterministic_repairs_v7':'qualify_deterministic_repairs_v8',
 'deterministic_repair_execution_v5':'deterministic_repair_execution_v6',
 'finetune_v8':'finetune_v9', 'repair_observation_v5':'repair_observation_v6',
 'repair_runtime_owner_v5':'repair_runtime_owner_v6', 'repair_owned_worker_v5':'repair_owned_worker_v6',
 'continuation_queue_v10':'continuation_queue_v11', 'evaluate_v8':'evaluate_v9',
 'audit_completed_finetune_checkpoints_v12':'audit_completed_finetune_checkpoints_v13',
 'cost_aggregation_v13':'cost_aggregation_v14',
 'review_actual_repair_engineering_v5':'review_actual_repair_engineering_v6',
 'repair_postprocessing_gate_v5':'repair_postprocessing_gate_v6'}
LABELS={'observer8':'observer9','Observer8':'Observer9','observer_version8':'observer_version9',
 'Actual8':'Actual9','actual8':'actual9','actual_F8':'actual_F9','repair8':'repair9',
 'F8':'F9','Q10':'Q11','E8':'E9','CPU12':'CPU13','Cost13':'Cost14','cost13':'cost14',
 'evaluation_v8_control':'evaluation_v9_control',
 'deterministic_repair_qualification_v7':'deterministic_repair_qualification_v8',
 'eeg_finetune_deterministic_repair_execution_v5':'eeg_finetune_deterministic_repair_execution_v6',
 'eeg_deterministic_repair_queue_v10':'eeg_deterministic_repair_queue_v11'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return {'path':Path(path).relative_to(WORK).as_posix(),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def once(text,old,new):
 assert text.count(old)==1,(old,text.count(old))
 return text.replace(old,new)
def migrate(text):
 values={**MAP,**LABELS}
 return re.sub('|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True)),lambda m:values[m.group()],text)
def pin(text,name,value):
 result,n=re.subn(rf"(?m)^({re.escape(name)} = )'[^']*'",lambda m:m[1]+repr(value),text)
 assert n==1,name
 return result
def main():
 assert sha(PRIOR)==PRIOR_SHA
 prior=read(PRIOR)
 assert prior['passed'] is True and len(prior['reviewed_sources'])==139
 original={x['path']:sha(WORK/x['path']) for x in prior['reviewed_sources']}
 assert all(original[x['path']]==x['sha256'] for x in prior['reviewed_sources'])
 adapters=[M/'mamba_cumsum_cpu_fp32_v2.py',M/'mamba_private_backward_graph_v2.py',M/'adapters/eegmamba_v3.py']
 for path in adapters:compile(path.read_text(encoding='utf-8'),str(path),'exec')
 locations={};texts={}
 for old,new in MAP.items():
  folder=M if (M/(old+'.py')).exists() else A
  locations[new]=folder/(new+'.py')
  assert not locations[new].exists(), 'Do not overwrite any prepared source'
  texts[new]=migrate((folder/(old+'.py')).read_text(encoding='utf-8'))
 q=texts['qualify_deterministic_repairs_v8']
 q=q.replace("version in (5, 6)","version in (5, 6, 7)")
 q=q.replace("'.adapters.eegmamba': '.adapters.eegmamba_v2'","'.adapters.eegmamba': '.adapters.eegmamba_v3'")
 replacements={'adapters/eegmamba_v2.py':'adapters/eegmamba_v3.py','mamba_private_backward_graph_v1.py':'mamba_private_backward_graph_v2.py','mamba_cumsum_cpu_fp32_v1.py':'mamba_cumsum_cpu_fp32_v2.py'}
 for old,new in replacements.items():
  q=once(q,repr(old)+': '+repr(sha(M/old)),repr(new)+': '+repr(sha(M/new)))
 texts['qualify_deterministic_repairs_v8']=q
 o=texts['repair_runtime_owner_v6'].replace("get('execution_version') == 10","get('execution_version') == 11").replace("registration['version'] == 10","registration['version'] == 11")
 texts['repair_runtime_owner_v6']=o
 queue=texts['continuation_queue_v11']
 for old,new in [("'version': 5","'version': 6"),("'version': 10","'version': 11"),("['version'] == 10","['version'] == 11"),("'execution_version': 10","'execution_version': 11"),("'new_actual_observer_version': 8","'new_actual_observer_version': 9"),("'version': 12","'version': 13")]:queue=queue.replace(old,new)
 texts['continuation_queue_v11']=queue
 e=texts['evaluate_v9'].replace("'version': 8","'version': 9").replace("get('schema_version') == 12","get('schema_version') == 13").replace("queue['version'] == 10","queue['version'] == 11")
 texts['evaluate_v9']=e
 texts['finetune_v9']=texts['finetune_v9'].replace("'version': 8","'version': 9").replace("['observer_version'] == 8","['observer_version'] == 9").replace("'actual_observer_version': 8","'actual_observer_version': 9")
 texts['repair_observation_v6']=texts['repair_observation_v6'].replace('== 8','== 9').replace('[8]','[9]').replace("'observer_version': 8","'observer_version': 9")
 texts['audit_completed_finetune_checkpoints_v13']=texts['audit_completed_finetune_checkpoints_v13'].replace('if 8 not in versions:','if 9 not in versions:').replace('versions == {8}','versions == {9}').replace('schema_version=12','schema_version=13')
 d=texts['deterministic_repair_execution_v6']
 d=d.replace("value.get('version') == 5","value.get('version') == 6").replace("'version': 8","'version': 9").replace("'observer_version': 8","'observer_version': 9").replace("'new_actual_observer_version': 8","'new_actual_observer_version': 9")
 start=d.index('    prior_binding = ',d.index('def implementation_sources():'))
 end=d.index('    for name in NEW_ENTRY_NAMES:',start)
 d=d[:start]+"    prior_binding = "+repr(desc(PRIOR))+"\n    prior = read_bound(prior_binding)\n    require(prior.get('passed') is True and len(prior['reviewed_sources']) == 139, 'Previous139 reviewed source lineage required')\n    records = {x['path']: x for x in prior['reviewed_sources']}\n    records[prior_binding['path']] = prior_binding\n"+d[end:]
 d=once(d,"        return ast.dump(function, include_attributes=False)","""        # Only Mamba's private adapter generation changes. CodeBrain mapping,
        # dispatch and builder bytecode must remain the qualified Qual5 path.
        mapping = next((node.value for node in function.body if isinstance(node, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'mapping' for t in node.targets)), None)
        require(isinstance(mapping, ast.Dict), 'Exact private builder mapping required')
        values = ast.literal_eval(mapping)
        expected_mamba = '.adapters.eegmamba_v3' if module is Q else '.adapters.eegmamba_v2'
        require(values == {'.adapters.codebrain_v2': '.adapters.codebrain_v3',
                           '.adapters.eegmamba': expected_mamba}, 'Unknown condition routing change')
        if module is Q:
            found = [value for key, value in zip(mapping.keys, mapping.values)
                     if isinstance(key, ast.Constant) and key.value == '.adapters.eegmamba']
            require(len(found) == 1 and isinstance(found[0], ast.Constant), 'One Mamba route required')
            found[0].value = '.adapters.eegmamba_v2'
        return ast.dump(function, include_attributes=False)""")
 d=d.replace('Numeric builder AST changed; successful old qualification cannot be reused','CodeBrain builder AST or unexpected routing changed; old qualification cannot be reused')
 texts['deterministic_repair_execution_v6']=d
 cost=texts['cost_aggregation_v14']
 cost=cost.replace('from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v6 as PREVIOUS_IDENTITY_Q','from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v6 as PREVIOUS_IDENTITY_Q\nfrom tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v7 as PREVIOUS_TREE_Q')
 cost=cost.replace("!= 8","!= 9").replace('schema_version=13','schema_version=14').replace('actual_cost_v13_','actual_cost_v14_')
 cost=once(cost,"else PREVIOUS_IDENTITY_Q if '/deterministic_repair_qualification_v6/' in artifact['path'] else Q","else PREVIOUS_IDENTITY_Q if '/deterministic_repair_qualification_v6/' in artifact['path'] else PREVIOUS_TREE_Q if '/deterministic_repair_qualification_v7/' in artifact['path'] else Q")
 cost=cost.replace('for v in (4, 5, 6, 7)','for v in (4, 5, 6, 7, 8)')
 cost=cost.replace("patterns = (","patterns = ('deterministic_repair_qualification_v7/*/*/attempt.json', 'deterministic_repair_qualification_v7/*/*/controller.json', ",1)
 texts['cost_aggregation_v14']=cost
 texts['repair_postprocessing_gate_v6']=texts['repair_postprocessing_gate_v6'].replace("get('schema_version') == 12","get('schema_version') == 13")
 order=['qualify_deterministic_repairs_v8','finetune_v9','repair_runtime_owner_v6','repair_owned_worker_v6','continuation_queue_v11','evaluate_v9','deterministic_repair_execution_v6','repair_observation_v6','audit_completed_finetune_checkpoints_v13','cost_aggregation_v14','review_actual_repair_engineering_v6','repair_postprocessing_gate_v6']
 for name in order:
  text=texts[name]
  if name=='deterministic_repair_execution_v6':text=pin(text,'QUALIFIER_SHA',sha(locations['qualify_deterministic_repairs_v8']))
  if name=='repair_observation_v6':
   text=pin(text,'F4_SHA',sha(locations['finetune_v9']))
   text=pin(text,'D_SHA',sha(locations['deterministic_repair_execution_v6']))
  if name=='repair_postprocessing_gate_v6':
   node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
   pins={locations[n].relative_to(WORK).as_posix():sha(locations[n]) for n in ('continuation_queue_v11','evaluate_v9','audit_completed_finetune_checkpoints_v13','cost_aggregation_v14')}
   text=once(text,ast.get_source_segment(text,node),'PINS = {'+', '.join('S.WORK / '+repr(k)+': '+repr(v) for k,v in pins.items())+'}')
  compile(text,str(locations[name]),'exec')
  with locations[name].open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
 assert all(sha(WORK/path)==digest for path,digest in original.items())
 failure=OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/1791782206453969000'
 actual=read(failure/'attempt.json')
 assert actual['status']=='failed' and actual['forward_started']==actual['forward_returned']==5 and actual['backward_started']==3 and actual['backward_returned']==2 and actual['optimizer_step_started']==actual['optimizer_step_returned']==0
 proof={'schema':'core_prefix_repair_source_preparation_v1','previous139_review':desc(PRIOR),'new_sources':[desc(locations[n]) for n in order]+[desc(p) for p in adapters],'old139_sources_preserved':True,'one_Mamba_core_prefix_call_changed':True,'numeric_order_change_disclosed':True,'CodeBrain_original_Qual5_reuse_required':True,'failed_qualifier7_artifacts':[desc(failure/name) for name in ('attempt.json','controller.json','process.log','released_owner.json')],'failed_primary_seconds':actual['primary_engineering_seconds'],'failed_unreturned_call_internal_work_unknown':actual['unreturned_call_internal_work_unknown'],'process_queries':0,'model_calls':0,'GPU_calls':0,'actual_qualification_started':False,'actual_registration_created':False,'independent_review_passed':False,'source':desc(Path(__file__))}
 target=A/('core_prefix_repair_source_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
 print(json.dumps({'proof':desc(target),'new_sources':proof['new_sources']}))
if __name__=='__main__':main()
