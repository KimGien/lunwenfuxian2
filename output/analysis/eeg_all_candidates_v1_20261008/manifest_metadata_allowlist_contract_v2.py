"""Pure-stdlib candidate checks: no collect/audit/model execution."""
import ast, copy, hashlib, json, sys, time, types
from pathlib import Path
WORK=Path(__file__).resolve().parents[3]
A=WORK/'output/analysis/eeg_all_candidates_v1_20261008'
PAIRS=[('pause_memory_resume_cost_companion_v4.py','pause_memory_resume_cost_companion_v5.py'),('audit_memory_pause_final_delivery_v4.py','audit_memory_pause_final_delivery_v5.py'),('system_standby_cost_companion_v4.py','system_standby_cost_companion_v5.py'),('audit_system_standby_final_delivery_v5.py','audit_system_standby_final_delivery_v6.py')]
checks=[]
def desc(p):return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def ok(name,value):
 if not value:raise AssertionError(name)
 checks.append(name)
def rejected(name,fn):
 try:fn()
 except ValueError:checks.append(name)
 else:raise AssertionError(name)
def load(name):
 p=A/name; ns={'__file__':str(p),'__name__':'_contract_candidate'}
 exec(compile(p.read_bytes(),str(p),'exec'),ns)
 return ns

def run():
 namespaces={new:load(new) for _,new in PAIRS}
 hashes={old:desc(A/old) for old,_ in PAIRS}
 manifest=next(iter(namespaces.values()))['MAMBA_MANIFEST']
 raw=(WORK/manifest['path']).read_bytes();actual=json.loads(raw)
 ok('actual fixed E12 metadata SHA',hashlib.sha256(raw).hexdigest()==manifest['sha256'])
 for old,new in PAIRS:
  ns=namespaces[new];r=ns['Reader']();method=r.read if hasattr(r,'read') else r.checked
  ok(new+': actual metadata only read',method(manifest)==actual)
  denied=[manifest['path'].replace('EEGMamba-lastblock-finetune','CodeBrain-lastblock-finetune'),manifest['path'].replace('predictions_manifest.json','predictions.json'),manifest['path'].replace('predictions_manifest.json','oof_data.json'),manifest['path'].replace('predictions_manifest.json','probabilities.json'),'../outside.json']
  for i,path in enumerate(denied):
   rejected(new+': blocked different path '+str(i),lambda path=path:method({'path':path,'sha256':manifest['sha256']}))
  ok(new+': same physical canonical path permitted',ns['fixed_manifest_path']((WORK/manifest['path']).parent/'..'/'EEGMamba-lastblock-finetune'/'predictions_manifest.json'))
  ok(new+': different physical root denied',not ns['fixed_manifest_path'](WORK.parent/manifest['path']))
  rejected(new+': changed bytes rejected by fixed SHA',lambda:ns['validate_fixed_manifest_metadata'](WORK/manifest['path'],raw+b' '))
  rejected(new+': caller wrong descriptor SHA rejected',lambda:method({'path':manifest['path'],'sha256':'0'*64}))
  # Only isolated field-validation unit cases bypass the hash to reach schema guards.
  original_hashlib=ns['hashlib'];ns['hashlib']=types.SimpleNamespace(sha256=lambda _:types.SimpleNamespace(hexdigest=lambda:manifest['sha256']))
  changes=[('payload field',lambda d:d.update(probabilities=[])),('boolean count',lambda d:d.update(distinct_subjects=True)),('condition',lambda d:d.update(condition='Other')),('missing completion',lambda d:d['evaluation_fit_artifacts'].pop()),('non-descriptor completion',lambda d:d['evaluation_fit_artifacts'][0].update(pMDD=0.5)),('wrong E version',lambda d:d['evaluator_execution'].update(version=13)),('wrong source',lambda d:d['evaluator_source'].update(sha256='0'*64))]
  for label,mutate in changes:
   value=copy.deepcopy(actual);mutate(value)
   rejected(new+': schema rejects '+label,lambda value=value:ns['validate_fixed_manifest_metadata'](WORK/manifest['path'],json.dumps(value).encode()))
  ns['hashlib']=original_hashlib
  ok(new+': all new exception helpers identical',ast.dump(next(n for n in ast.parse((A/new).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='validate_fixed_manifest_metadata'))==ast.dump(next(n for n in ast.parse((A/PAIRS[0][1]).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='validate_fixed_manifest_metadata')))
 # Preserve original scientific, cost, loss, and complete-event functions modulo schema labels only.
 mapping={'pause_memory_resume_cost_companion_v5':'pause_memory_resume_cost_companion_v4','memory_pause_final_delivery_gate_v5':'memory_pause_final_delivery_gate_v4','system_standby_cost_companion_v5':'system_standby_cost_companion_v4','system_standby_final_delivery_gate_v6':'system_standby_final_delivery_gate_v5'}
 class Back(ast.NodeTransformer):
  def visit_Constant(self,n):
   if isinstance(n.value,str) and n.value in mapping:n.value=mapping[n.value]
   return n
 for old,new in PAIRS:
  a=ast.parse((A/old).read_text(encoding='utf-8'));b=Back().visit(ast.parse((A/new).read_text(encoding='utf-8')))
  oldfunc={n.name:n for n in a.body if isinstance(n,ast.FunctionDef)}
  newfunc={n.name:n for n in b.body if isinstance(n,ast.FunctionDef)}
  compared=[]
  for name in oldfunc:
   if name in ('main','collect'):continue
   ok(new+': unchanged protected function '+name,ast.dump(oldfunc[name])==ast.dump(newfunc[name]));compared.append(name)
  ok(new+': protected functions covered',len(compared)>2)
  ok(new+': original bytes unchanged',desc(A/old)==hashes[old])
 # Check directly bound sources from candidate module constants; no imported project code.
 for new,ns in namespaces.items():
  pins=[]
  for key in ('PINNED','PAUSE_SOURCE','MEMORY_SOURCE','STANDBY_SOURCE','FULL_SCOPE_SOURCE','FINALIZER','WIDE','COST'):
   val=ns.get(key)
   if isinstance(val,dict):pins.extend([val] if set(val)=={'path','sha256'} else [v for v in val.values() if isinstance(v,dict) and set(v)=={'path','sha256'}])
  ok(new+': exact current source pins',all(desc(WORK/d['path'])==d for d in pins))
  if 'PINNED' in ns:ok(new+': unchanged Cost19 wide9 G3v4',ns['PINNED']['cost']['path'].endswith('cost_aggregation_v19.py') and ns['PINNED']['wide']['path'].endswith('audit_full_plan_acceptance_v9.py') and ns['PINNED'].get('g3',{}).get('path','').endswith('sync_boundary_v3_cost_companion_v4.py') if 'g3' in ns['PINNED'] else True)
  if 'audit_' in new:
   body=ast.unparse(next(n for n in ast.parse((A/new).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main'))
   ok(new+': require-complete cannot silently prepare',"if args.require_complete and (not args.run):" in body and "parser.error('--require-complete requires --run')" in body)
 ok('full scope helper retained',namespaces[PAIRS[-1][1]]['FULL_SCOPE_SOURCE']=={'path':'output/analysis/eeg_all_candidates_v1_20261008/full_system_power_timer_scope_v1.py','sha256':'c28bdd912c3a97510cd17d725ce7542c078640ec3e77f2c48340d6c0e923caec'})
 failure={'path':'output/analysis/eeg_all_candidates_v1_20261008/actual_pause_v4_manifest_guard_failure_1791881597248839900.json','sha256':'4b68f357d86da043b19926f486fdfd8bab8948c5243d4b339fb602331010521a'}
 ok('original actual failure artifact preserved',desc(WORK/failure['path'])==failure)
 return {'schema':'manifest_metadata_allowlist_stdlib_contract_v1','passed':True,'status':'prepared_not_executed','sources':[desc(A/new) for _,new in PAIRS],'original_sources':[hashes[old] for old,_ in PAIRS],'source':desc(Path(__file__).resolve()),'actual_metadata_manifest':manifest,'original_failure':failure,'checks':checks,'checks_count':len(checks),'operations':{'actual_metadata_reads_only':True,'probability_payload_reads':0,'Tensor_model_GPU_native':0,'collect_audit_or_final_gate_calls':0,'scientific_or_cost_reexecution':0,'existing_artifact_modifications':0,'schema_tests_used_isolated_fake_hash_only':True}}
if __name__=='__main__':
 value=run();p=A/('manifest_metadata_allowlist_stdlib_contract_'+str(time.time_ns())+'.json')
 with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
 print(json.dumps({'report':desc(p),'passed':value['passed'],'checks_count':value['checks_count'],'sources':value['sources']}))
