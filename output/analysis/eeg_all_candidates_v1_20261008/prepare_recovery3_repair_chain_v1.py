"""Prepare new, unregistered recovery3-compatible sources; never start work."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

WORK = Path('C:/Users/jin  qiyan/Desktop/lunwen2')
MOD = WORK/'tridep_ccml/eeg_all_candidates'
A = WORK/'output/analysis/eeg_all_candidates_v1_20261008'
OUT = WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
MAP = {
    'qualify_deterministic_repairs_v3':'qualify_deterministic_repairs_v4',
    'deterministic_repair_execution_v1':'deterministic_repair_execution_v2',
    'finetune_v4':'finetune_v5',
    'repair_observation_v1':'repair_observation_v2',
    'repair_runtime_owner_v1':'repair_runtime_owner_v2',
    'repair_owned_worker_v1':'repair_owned_worker_v2',
    'continuation_queue_v6':'continuation_queue_v7',
    'evaluate_v4':'evaluate_v5',
    'evaluation_v4_control':'evaluation_v5_control',
    'deterministic_repair_qualification_v3':'deterministic_repair_qualification_v4',
    'audit_completed_finetune_checkpoints_v7':'audit_completed_finetune_checkpoints_v9',
    'cost_aggregation_v8':'cost_aggregation_v10',
    'repair_postprocessing_gate_v1':'repair_postprocessing_gate_v2',
    'review_actual_repair_engineering_v1':'review_actual_repair_engineering_v2',
}
OLD_NAMES = [x for x in MAP if (MOD/(x+'.py')).exists() or (A/(x+'.py')).exists()]
PINNED = {
    'qualify_deterministic_repairs_v3':'880a51f30b229674e7b3ed6d810669d80b114a48cacb91678adf2a60c98ae01b',
    'deterministic_repair_execution_v1':'aa2dce2cdb42a81a6025245a2db8b062e02d5c10d1d1deeed21cce9ce8968618',
    'finetune_v4':'8928371053e96511d7f4823221d3b6e670b1d61bdcc81c0108e3f1364455204d',
    'continuation_queue_v6':'836a80443b8e007d212833f2ec5235da50d267c100e7a1ef74f144196052fe8e',
    'audit_completed_finetune_checkpoints_v7':'0ce9a1d9eb7e97c4b0786a553b706bc0e2650bfde1cfebc787da796efa0477e3',
    'cost_aggregation_v8':'64030d81f55e16179dec258aadfa3c0bf7873a1882ed559478d2318a3fa2d2bf',
    'evaluate_v4':'69a954240a6d2c0b9eb5c6765affced06110aff6a8e20d8c172e123386c1f387',
    'repair_postprocessing_gate_v1':'93df313551fa00a1a0c1444ce920b7ad83fdfe5ccf39d45aad6079735dbbd22b',
}

def require(ok, msg):
    if not ok: raise ValueError(msg)

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def desc(path): return {'path':path.relative_to(WORK).as_posix(),'sha256':digest(path)}
def locate(name):
    candidates=[p for p in (MOD/(name+'.py'), A/(name+'.py')) if p.exists()]
    require(len(candidates)==1, 'Ambiguous/missing original '+name)
    return candidates[0]

old_paths={x:locate(x) for x in OLD_NAMES}
old_bytes={x:p.read_bytes() for x,p in old_paths.items()}
for x,p in old_paths.items():
    if x in PINNED: require(digest(p)==PINNED[x], 'Previous prepared source changed '+x)
new_paths={x:p.with_name(MAP[x]+'.py') for x,p in old_paths.items()}
require(all(not p.exists() for p in new_paths.values()), 'New draft already exists; inspect before another transaction')
manifest=OUT/'continuation_queue_v5/recovery_registration_v3.json'
require(digest(manifest)=='653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4','Actual recovery3 manifest differs')
require(len(json.loads(manifest.read_text())['sources'])==88,'Actual recovery3 source scope differs')

def rename(s):
    for old,new in sorted(MAP.items(), key=lambda x:-len(x[0])): s=s.replace(old,new)
    s=s.replace('recovery_registration_v2.json','recovery_registration_v3.json')
    for old,new in [('Q6','Q7'),('CPU7','CPU9'),('Cost8','Cost10'),('Cost7','Cost9'),('E4','E5'),('F4','F5'),('observer4','observer5'),('actual4','actual5'),('Actual4','Actual5')]:
        s=s.replace(old,new)
    return s

class Bindings(ast.NodeTransformer):
    def __init__(self, name): self.name=name
    def visit_ImportFrom(self,node):
        if node.module: node.module=rename(node.module)
        for alias in node.names: alias.name=rename(alias.name)
        return node
    def visit_Constant(self,node):
        if isinstance(node.value,str): node.value=rename(node.value)
        return node
    def version(self, value, key='version'):
        n=self.name
        if key in ('observer_version','new_actual_observer_version') and value==4: return 5
        if key=='schema_version':
            return {'audit_completed_finetune_checkpoints_v7':{7:9},'cost_aggregation_v8':{8:10},'repair_postprocessing_gate_v1':{7:9},'evaluate_v4':{7:9}}.get(n,{}).get(value,value)
        if key=='recovery_execution_version' and n=='qualify_deterministic_repairs_v3' and value==2: return 3
        if key=='execution_version' and n in ('continuation_queue_v6','repair_runtime_owner_v1') and value==6: return 7
        if key=='version':
            if n=='deterministic_repair_execution_v1': return {1:2,4:5}.get(value,value)
            if n in ('finetune_v4','repair_observation_v1'): return 5 if value==4 else value
            if n=='continuation_queue_v6': return {1:2,6:7,7:9}.get(value,value)
            if n=='repair_runtime_owner_v1': return 7 if value==6 else value
            if n=='evaluate_v4': return {4:5,6:7}.get(value,value)
        return value
    def visit_Dict(self,node):
        self.generic_visit(node)
        for key,val in zip(node.keys,node.values):
            if isinstance(key,ast.Constant) and isinstance(val,ast.Constant) and type(val.value) is int:
                val.value=self.version(val.value,key.value)
        return node
    def visit_keyword(self,node):
        self.generic_visit(node)
        if isinstance(node.value,ast.Constant) and type(node.value.value) is int:
            node.value.value=self.version(node.value.value,node.arg)
        if node.arg=='actual_observer_versions' and isinstance(node.value,ast.List):
            for val in node.value.elts:
                if isinstance(val,ast.Constant) and val.value==4:val.value=5
        return node
    def visit_Compare(self,node):
        self.generic_visit(node)
        left=ast.unparse(node.left)
        keys=('schema_version','recovery_execution_version','execution_version','observer_version','version')
        for key in keys:
            if re.search(r"['\"]"+key+r"['\"]",left):
                for val in node.comparators:
                    if isinstance(val,ast.Constant) and type(val.value) is int:val.value=self.version(val.value,key)
                break
        if self.name=='audit_completed_finetune_checkpoints_v7':
            if isinstance(node.left,ast.Constant) and node.left.value==4 and any(isinstance(c,ast.Name) and c.id=='versions' for c in node.comparators):node.left.value=5
            if left=='versions':
                for c in node.comparators:
                    if isinstance(c,ast.Set):
                        for e in c.elts:
                            if isinstance(e,ast.Constant) and e.value==4:e.value=5
        return node

raw={}
for name,path in old_paths.items():
    tree=Bindings(name).visit(ast.parse(old_bytes[name].decode('utf-8-sig')))
    raw[name]=ast.unparse(ast.fix_missing_locations(tree))+'\n'

# Qualification is bound to the true recovery3 terminal and original nonrepair330,
# without querying the GPU or controlling the currently running experiment.
name='qualify_deterministic_repairs_v3';s=raw[name]
s=s.replace("MANIFEST_SHA = '525f6ed6e838fe267956c54d90d4e6d13d7da3bc710863aca90c35af466782cd'", "MANIFEST_SHA = '653df2183b1a9cdcd74755f69629d2cf89ca3de8a6b7e9bf26b7df0bfcabbaf4'")
s=s.replace('len(sources) == 84','len(sources) == 88').replace('Actual84-source','Actual88-source').replace('recoveryV2','recoveryV3')
needle='    return status\n\ndef source_check():'
replacement="""    require(status.get('recovery_registration') == desc(queue / 'recovery_registration_v3.json'), 'Canonical actual recovery3 registration required')
    require(status.get('recovery_execution_source') == desc(WORK / 'tridep_ccml/eeg_all_candidates/recover_q5_after_process_loss_v3.py'), 'Actual recovery3 execution source differs')
    closure = status.get('recovery_closure')
    require(type(closure) is dict and read(checked(closure)).get('source') == status['recovery_execution_source'], 'Actual loss3 closure/source binding required')
    registry = read(OUT / 'registration.json')
    others = [f for f in registry['fits'] if f['condition'] not in CONDITIONS]
    require(len(others) == 330, 'Original nonrepair330 identities differ')
    for fit in others:
        value = read(OUT / 'runs' / fit['fit_id'] / 'status.json')
        require(value.get('status') == 'training_complete' and value.get('fit_id') == fit['fit_id'], 'Original nonrepair330 has incomplete or aliased status')
    return status

def source_check():"""
require(needle in s,'Qualifier boundary insertion absent')
raw[name]=s.replace(needle,replacement)

# Mixed CPU audit delegates CPU8, thus preserving all three real loss events.
name='audit_completed_finetune_checkpoints_v7';s=raw[name]
s=s.replace("audit_completed_finetune_checkpoints_v6.py", "audit_completed_finetune_checkpoints_v8.py")
s=s.replace('1f671d110c58d5a29fd33bbfa1c1c685daa51dc0a5f3dbc5d539fa9006e55128','43ab48da0bd862399d13aed718ad87500a718e795e9742663d78ba97c958252f')
s=s.replace('CPU6','CPU8').replace('actual_observer4','actual_observer5')
raw[name]=s

# Cost9 augmentation runs exactly once after the original V5 inventory, then
# the source-bound qualification and repair-parent ledgers are appended.
name='cost_aggregation_v8';s=raw[name]
s=s.replace('lost_f3_closure_v2','lost_f3_closure_v3').replace('cost_aggregation_v7.py','cost_aggregation_v9.py')
s=s.replace('ee232c5e231cc665e0173db27c5a9890b5b421167410ada1b6198f99af1f46a1','4d0863abf0a4dbfe30e8e2bd853ca7d7bb8242212fce65683be36d9cd70811e0')
s=s.replace('actual4_versions','actual5_versions').replace('actual_Cost_v8','actual_Cost_v10')
raw[name]=s

# Build hashes in dependency order; never change any previous source bytes.
order=['qualify_deterministic_repairs_v3','deterministic_repair_execution_v1','finetune_v4',
       'repair_observation_v1','repair_runtime_owner_v1','repair_owned_worker_v1',
       'audit_completed_finetune_checkpoints_v7','cost_aggregation_v8',
       'continuation_queue_v6','evaluate_v4','review_actual_repair_engineering_v1','repair_postprocessing_gate_v1']
new_hashes={}
for name in order:
    s=raw[name]
    if name=='deterministic_repair_execution_v1':s=s.replace(PINNED['qualify_deterministic_repairs_v3'],new_hashes['qualify_deterministic_repairs_v3'])
    if name=='repair_observation_v1':
        s=s.replace(PINNED['finetune_v4'],new_hashes['finetune_v4']).replace(PINNED['deterministic_repair_execution_v1'],new_hashes['deterministic_repair_execution_v1'])
    if name=='repair_postprocessing_gate_v1':
        for dependency in ('continuation_queue_v6','evaluate_v4','audit_completed_finetune_checkpoints_v7','cost_aggregation_v8'):
            s=s.replace(PINNED[dependency],new_hashes[dependency])
    compile(s,str(new_paths[name]),'exec')
    raw[name]=s
    new_hashes[name]=hashlib.sha256(s.encode()).hexdigest()
require(set(order)==set(OLD_NAMES),'Prepared-file scope differs')
for name in order:
    with new_paths[name].open('x',encoding='utf-8',newline='\n') as stream:stream.write(raw[name])
for name,path in old_paths.items(): require(path.read_bytes()==old_bytes[name],'Old source bytes changed '+name)
report={'schema':'recovery3_repair_chain_preparation_v1','status':'source_preparation_only','captured_ns':time.time_ns(),
        'actual_recovery_registration':desc(manifest),'inherited_frozen_sources':88,
        'previous_sources':[desc(p) for p in old_paths.values()], 'new_sources':[desc(p) for p in new_paths.values()],
        'model_calls':0,'GPU_queries':0,'checkpoint_tensor_loads':0,'formal_fits':0,'new_registration_created':False,
        'actual_qualification_complete':False,'independent_current_chain_review_complete':False,
        'semantic_contract_tests_complete':False,'full_final_postprocessing_compatibility_complete':False,
        'all_plan_complete':False}
target=A/('recovery3_repair_chain_preparation_'+str(report['captured_ns'])+'.json')
with target.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False);f.write('\n')
print(json.dumps({'artifact':desc(target),'prepared_sources':len(new_paths),'all_plan_complete':False}))
