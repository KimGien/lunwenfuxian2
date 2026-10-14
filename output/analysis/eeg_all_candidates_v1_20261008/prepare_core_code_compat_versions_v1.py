"""Prepare next immutable source generation only; no runtime/registration."""
from pathlib import Path
import ast,hashlib,json,re,time
WORK=Path(__file__).resolve().parents[3];M=WORK/'tridep_ccml/eeg_all_candidates';A=Path(__file__).resolve().parent
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
PRIOR=A/'independent_core_prefix_repair_chain_source_review_1791783174737274500.json'
PRIOR_SHA='8a7337276c7eac3a74ea4580566668374f655c9c0cacbe1c666870e02dbc53b9'
MAP={'qualify_deterministic_repairs_v8':'qualify_deterministic_repairs_v9','deterministic_repair_execution_v6':'deterministic_repair_execution_v7','finetune_v9':'finetune_v10','repair_observation_v6':'repair_observation_v7','repair_runtime_owner_v6':'repair_runtime_owner_v7','repair_owned_worker_v6':'repair_owned_worker_v7','continuation_queue_v11':'continuation_queue_v12','evaluate_v9':'evaluate_v10','audit_completed_finetune_checkpoints_v13':'audit_completed_finetune_checkpoints_v14','cost_aggregation_v14':'cost_aggregation_v15','review_actual_repair_engineering_v6':'review_actual_repair_engineering_v7','repair_postprocessing_gate_v6':'repair_postprocessing_gate_v7'}
LABELS={'observer9':'observer10','Observer9':'Observer10','observer_version9':'observer_version10','Actual9':'Actual10','actual9':'actual10','actual_F9':'actual_F10','repair9':'repair10','F9':'F10','Q11':'Q12','E9':'E10','CPU13':'CPU14','Cost14':'Cost15','cost14':'cost15','evaluation_v9_control':'evaluation_v10_control','deterministic_repair_qualification_v8':'deterministic_repair_qualification_v9','eeg_finetune_deterministic_repair_execution_v6':'eeg_finetune_deterministic_repair_execution_v7','eeg_deterministic_repair_queue_v11':'eeg_deterministic_repair_queue_v12'}
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def desc(path):return {'path':Path(path).relative_to(WORK).as_posix(),'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def once(text,old,new):
 assert text.count(old)==1,(old,text.count(old))
 return text.replace(old,new)
def migrate(text):
 values={**MAP,**LABELS};return re.sub('|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True)),lambda m:values[m.group()],text)
def pin(text,name,value):
 result,n=re.subn(rf"(?m)^({re.escape(name)} = )'[^']*'",lambda m:m[1]+repr(value),text);assert n==1,name;return result
def main():
 assert sha(PRIOR)==PRIOR_SHA
 prior=read(PRIOR);assert prior['passed'] is True and len(prior['reviewed_sources'])==153
 original={x['path']:sha(WORK/x['path']) for x in prior['reviewed_sources']}
 assert all(original[x['path']]==x['sha256'] for x in prior['reviewed_sources'])
 adapters=[M/'mamba_private_backward_graph_v3.py',M/'adapters/eegmamba_v4.py']
 for path in adapters:compile(path.read_text(encoding='utf-8'),str(path),'exec')
 locations={};texts={}
 for old,new in MAP.items():
  folder=M if (M/(old+'.py')).exists() else A;locations[new]=folder/(new+'.py')
  assert not locations[new].exists(),'Never overwrite prepared or sealed source'
  texts[new]=migrate((folder/(old+'.py')).read_text(encoding='utf-8'))
 q=texts['qualify_deterministic_repairs_v9'].replace('version in (5, 6, 7)','version in (5, 6, 7, 8)').replace("'.adapters.eegmamba': '.adapters.eegmamba_v3'","'.adapters.eegmamba': '.adapters.eegmamba_v4'").replace('Actual108','Actual88')
 for old,new in {'adapters/eegmamba_v3.py':'adapters/eegmamba_v4.py','mamba_private_backward_graph_v2.py':'mamba_private_backward_graph_v3.py'}.items():q=once(q,repr(old)+': '+repr(sha(M/old)),repr(new)+': '+repr(sha(M/new)))
 texts['qualify_deterministic_repairs_v9']=q
 texts['repair_runtime_owner_v7']=texts['repair_runtime_owner_v7'].replace("get('execution_version') == 11","get('execution_version') == 12").replace("registration['version'] == 11","registration['version'] == 12")
 queue=texts['continuation_queue_v12']
 for old,new in [("'version': 6","'version': 7"),("'version': 11","'version': 12"),("['version'] == 11","['version'] == 12"),("'execution_version': 11","'execution_version': 12"),("'new_actual_observer_version': 9","'new_actual_observer_version': 10"),("'version': 13","'version': 14")]:queue=queue.replace(old,new)
 texts['continuation_queue_v12']=queue
 texts['evaluate_v10']=texts['evaluate_v10'].replace("'version': 9","'version': 10").replace("get('schema_version') == 13","get('schema_version') == 14").replace("queue['version'] == 11","queue['version'] == 12")
 texts['finetune_v10']=texts['finetune_v10'].replace("'version': 9","'version': 10").replace("['observer_version'] == 9","['observer_version'] == 10").replace("'actual_observer_version': 9","'actual_observer_version': 10")
 texts['repair_observation_v7']=texts['repair_observation_v7'].replace('== 9','== 10').replace('[9]','[10]').replace("'observer_version': 9","'observer_version': 10")
 texts['audit_completed_finetune_checkpoints_v14']=texts['audit_completed_finetune_checkpoints_v14'].replace('if 9 not in versions:','if 10 not in versions:').replace('versions == {9}','versions == {10}').replace('schema_version=13','schema_version=14').replace('completed_finetune_checkpoint_audit_v13_','completed_finetune_checkpoint_audit_v14_')
 d=texts['deterministic_repair_execution_v7'].replace("value.get('version') == 6","value.get('version') == 7").replace("'version': 9","'version': 10").replace("'observer_version': 9","'observer_version': 10").replace("'new_actual_observer_version': 9","'new_actual_observer_version': 10").replace("expected_mamba = '.adapters.eegmamba_v3'","expected_mamba = '.adapters.eegmamba_v4'")
 start=d.index('    prior_binding = ',d.index('def implementation_sources():'));end=d.index('    for name in NEW_ENTRY_NAMES:',start)
 d=d[:start]+"    prior_binding = "+repr(desc(PRIOR))+"\n    prior = read_bound(prior_binding)\n    require(prior.get('passed') is True and len(prior['reviewed_sources']) == 153, 'Previous153 reviewed source lineage required')\n    records = {x['path']: x for x in prior['reviewed_sources']}\n    records[prior_binding['path']] = prior_binding\n"+d[end:]
 texts['deterministic_repair_execution_v7']=d
 cost=texts['cost_aggregation_v15'].replace('from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v7 as PREVIOUS_TREE_Q','from tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v7 as PREVIOUS_TREE_Q\nfrom tridep_ccml.eeg_all_candidates import qualify_deterministic_repairs_v8 as PREVIOUS_CORE_Q')
 cost=cost.replace('!= 9','!= 10').replace('schema_version=14','schema_version=15').replace('actual_cost_v14_','actual_cost_v15_')
 cost=once(cost,"else PREVIOUS_TREE_Q if '/deterministic_repair_qualification_v7/' in artifact['path'] else Q","else PREVIOUS_TREE_Q if '/deterministic_repair_qualification_v7/' in artifact['path'] else PREVIOUS_CORE_Q if '/deterministic_repair_qualification_v8/' in artifact['path'] else Q")
 cost=cost.replace('for v in (4, 5, 6, 7, 8)','for v in (4, 5, 6, 7, 8, 9)')
 cost=cost.replace('patterns = (',"patterns = ('deterministic_repair_qualification_v8/*/*/attempt.json', 'deterministic_repair_qualification_v8/*/*/controller.json', ",1)
 texts['cost_aggregation_v15']=cost
 texts['repair_postprocessing_gate_v7']=texts['repair_postprocessing_gate_v7'].replace("get('schema_version') == 13","get('schema_version') == 14")
 order=['qualify_deterministic_repairs_v9','finetune_v10','repair_runtime_owner_v7','repair_owned_worker_v7','continuation_queue_v12','evaluate_v10','deterministic_repair_execution_v7','repair_observation_v7','audit_completed_finetune_checkpoints_v14','cost_aggregation_v15','review_actual_repair_engineering_v7','repair_postprocessing_gate_v7']
 for name in order:
  text=texts[name]
  if name=='deterministic_repair_execution_v7':text=pin(text,'QUALIFIER_SHA',sha(locations['qualify_deterministic_repairs_v9']))
  if name=='repair_observation_v7':text=pin(pin(text,'F4_SHA',sha(locations['finetune_v10'])),'D_SHA',sha(locations['deterministic_repair_execution_v7']))
  if name=='repair_postprocessing_gate_v7':
   node=next(n for n in ast.parse(text).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
   pins={locations[n].relative_to(WORK).as_posix():sha(locations[n]) for n in ('continuation_queue_v12','evaluate_v10','audit_completed_finetune_checkpoints_v14','cost_aggregation_v15')}
   text=once(text,ast.get_source_segment(text,node),'PINS = {'+', '.join('S.WORK / '+repr(k)+': '+repr(v) for k,v in pins.items())+'}')
  compile(text,str(locations[name]),'exec')
  with locations[name].open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
 assert all(sha(WORK/path)==digest for path,digest in original.items())
 failure=OUT/'deterministic_repair_qualification_v8/EEGMamba-lastblock-finetune/1791783245785244500';actual=read(failure/'attempt.json')
 assert actual['status']=='failed' and actual['forward_started']==actual['forward_returned']==0 and actual['backward_started']==actual['backward_returned']==0 and actual['optimizer_step_started']==actual['optimizer_step_returned']==0
 proof={'schema':'core_code_compat_source_preparation_v1','previous153_review':desc(PRIOR),'new_sources':[desc(locations[n]) for n in order]+[desc(p) for p in adapters],'old153_sources_preserved':True,'CodeBrain_original_Qual5_reuse_required':True,'failed_qualifier8_artifacts':[desc(failure/name) for name in ('attempt.json','controller.json','process.log','released_owner.json')],'failed_primary_seconds':actual['primary_engineering_seconds'],'prior_model_construction_occurred':True,'failed_unreturned_call_internal_work_unknown':actual['unreturned_call_internal_work_unknown'],'process_queries':0,'model_calls':0,'GPU_calls':0,'actual_qualification_started':False,'actual_registration_created':False,'independent_review_passed':False,'source':desc(Path(__file__))}
 target=A/('core_code_compat_source_preparation_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
 print(json.dumps({'proof':desc(target),'new_sources':proof['new_sources']}))
if __name__=='__main__':main()
