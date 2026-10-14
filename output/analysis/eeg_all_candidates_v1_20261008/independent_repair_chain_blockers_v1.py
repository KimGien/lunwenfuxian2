"""Independent pure-AST reproductions of unregistered repair-chain blockers."""
import ast
import hashlib
import json
from pathlib import Path
import time
import types
WORK=Path(__file__).resolve().parents[3]
A=Path(__file__).resolve().parent
MOD=WORK/'tridep_ccml/eeg_all_candidates'
def desc(p):return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def require(ok,msg):
    if not ok:raise ValueError(msg)
def extract(path,name,ns):
    raw=path.read_bytes();tree=ast.parse(raw);node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
    return ns[name]
old_sources={
 'qualify_deterministic_repairs_v4.py':'d95bbaa4216202885c55f56a7ad4e731449539b13cd834891fbf4ee5857880a4',
 'repair_runtime_owner_v2.py':'840fcd5f0f237ff7c2eca6cb5d4c482c135da76ddeedbe5d6d0dc08cffc6d714',
 'evaluate_v5.py':'14505fa2adf745bf2d56bf6732ff664aaf5d8fbbb4df80222dc7806a88231279'}
for name,digest in old_sources.items():require(desc(MOD/name)['sha256']==digest,'Reproducer is bound to reviewed draft '+name)
parent={'pid':10,'parent':1,'birth':'100','command':'queue'}
module='tridep_ccml.eeg_all_candidates.repair_owned_worker_v2'
launch={'pid':20,'parent':10,'birth':'200','command':'wsl.exe '+module+' --worker token123'}
owner={'id':'owner','execution_version':7,'child_token':'token123','child_job':'mamba','native_roles':[parent],'child_pid':20,'child_native_identity':launch,'pid':10,'source':{'path':'E5','sha256':'fake'}}
admission={'queue_owner_id':'owner','token':'token123','job':'mamba'}
f=extract(MOD/'repair_runtime_owner_v2.py','validate_native',{'require':require,'WORKER_MODULE':module})
returned=[f([parent,launch],owner,admission,worker_pid=p,platform='linux') for p in (777,888)]
require(returned[0]==returned[1]==[parent,launch],'Linux worker identity bypass no longer reproduced')
e_owner={**owner,'child_identity':launch}
e=extract(MOD/'evaluate_v5.py','admit_worker',{'require':require,'S':types.SimpleNamespace(desc=lambda p:owner['source']),'SOURCE':'E5','Q':types.SimpleNamespace(admit_host=lambda items,roles:None)})
e_returned=[e([parent,launch],e_owner,'token123',p,'linux') for p in (777,888)]
require(e_returned[0]==e_returned[1]==[parent,launch],'E5 Linux worker identity bypass no longer reproduced')

# Fake paths avoid any filesystem process/fit read inside the copied function.
class FakePath:
    def __init__(self,text):self.text=text
    def __truediv__(self,x):return FakePath(self.text+'/'+str(x))
    def exists(self):return False
queue=FakePath('queue');out=FakePath('out');work=FakePath('work')
manifest={'path':'manifest3','sha256':'x'};source={'path':'source3','sha256':'s'};closure={'path':'closure3','sha256':'c'}
status={'status':'failed','execution_version':5,'recovery_execution_version':3,'recovery_registration':manifest,'recovery_execution_source':source,'recovery_closure':closure,'source':{'path':'wrongQ3','sha256':'bad'},'execution_registration':{'path':'wrongBase','sha256':'bad'},'outcomes':{}}
fitlist=[{'fit_id':'f'+str(i),'condition':'original'} for i in range(330)]
def fake_read(path):
    key=path.text if isinstance(path,FakePath) else path
    if key=='queue/status.json':return status
    if key=='out/registration.json':return {'fits':fitlist}
    if key=='closure_bound':return {'recovery_registration':manifest,'closure_source':source}
    if key.startswith('out/runs/') and key.endswith('/status.json'):return {'status':'training_complete','fit_id':key.split('/')[-2]}
    raise AssertionError('Unexpected read '+key)
def fake_desc(path):
    name=path.text.split('/')[-1]
    if name=='recovery_registration_v3.json':return manifest
    if name=='recover_q5_after_process_loss_v3.py':return source
    if name=='lost_f3_v3_1791767849813215500.json':return closure
    raise AssertionError('Unexpected binding '+name)
ns={'QUEUE':queue,'OUT':out,'WORK':work,'CONDITIONS':('repair_a','repair_b'),'MANIFEST_SHA':'x','require':require,'read':fake_read,'desc':fake_desc,'checked':lambda x:'closure_bound'}
b=extract(MOD/'qualify_deterministic_repairs_v4.py','check_boundary',ns)
require(b() is status,'Unsafe incomplete natural-handoff evidence no longer accepted')

stamp=str(time.time_ns());folder=A/'new_repair_source_preparation_history'/('independent_blocked_review_'+stamp);folder.mkdir(parents=True,exist_ok=False)
for name in old_sources:
    raw=(MOD/name).read_bytes()
    with (folder/name).open('xb') as h:h.write(raw)
report={'schema':'independent_recovery3_repair_chain_blocking_review_v1','status':'blocked_repair_chain_source_review','passed':False,'reviewer':'/root/loss3_independent_review','reviewer_source':desc(Path(__file__).resolve()),'reviewed_source_subset':[desc(MOD/n) for n in old_sources], 'preserved_sources':[desc(folder/n) for n in old_sources],
 'reproduced_blockers':[
 {'id':'natural_boundary_unverified','source':'qualify_deterministic_repairs_v4.py','lines':[89,104],'observed':'failed old queue with wrong source/base, outcomes empty and330 status-only rows accepted; completion bytes and original child exit0 not required'},
 {'id':'linux_training_worker_unbound','source':'repair_runtime_owner_v2.py','lines':[45,75],'observed':'Linux worker_pid777 and888 both accepted for same Windows WSL launcher; no exact Linux parent/birth/command nonce gate. S.save immutable uses exists then replace, not atomic claim.'},
 {'id':'linux_evaluator_worker_unbound','source':'evaluate_v5.py','function':'admit_worker','observed':'Linux worker_pid777 and888 both accepted for same Windows WSL launcher; exclusive owner write limits duplicates but not rightful Linux identity.'}],
 'model_calls':0,'GPU_queries':0,'process_queries':0,'process_actions':0,'Tensor_reads':0,'outer_test_probability_reads':0,'actual_registration_or_qualification_created':False,
 'scope':'Pure AST extracted functions with synthetic mappings and paths; no imports of execution modules. Failed review cannot be consumed as Q7 source approval.'}
p=A/('independent_recovery3_repair_chain_blocking_review_'+stamp+'.json')
with p.open('x',encoding='utf-8') as h:json.dump(report,h,indent=2);h.write('\n')
print(json.dumps({'report':desc(p),'passed':False,'blockers':3}))