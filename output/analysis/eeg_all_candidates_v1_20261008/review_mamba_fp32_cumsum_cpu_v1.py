"""Independent explicit IEEE-float32 sequential-prefix reference, CPU only."""
from pathlib import Path
import ast,hashlib,json,struct,sys,time
from unittest.mock import patch
W=Path(__file__).resolve().parents[3];sys.path.insert(0,str(W));SOURCE=Path(__file__).resolve()
import torch
from tridep_ccml.eeg_all_candidates import mamba_cumsum_cpu_fp32_v1 as M
def d(p):
 p=Path(p).resolve()
 return {'path':p.relative_to(W).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def main():
 producer=SOURCE.parent/'mamba_fixed_fp32_cumsum_CPU_preparation_1791713923374887400.json'
 p=json.loads(producer.read_text(encoding='utf-8-sig'))
 assert p['passed'] and p['source']==d(M.__file__) and p['production_integration_applied']==False
 text=Path(M.__file__).read_text(encoding='utf-8')
 tree=ast.parse(text);calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert 'torch.cumsum' not in calls and not any('use_deterministic_algorithms' in c or 'load_encoder' in c or 'torch.load' in c for c in calls)
 assert 'running=running+host[...,i]' in text and 'value.data_ptr()==out.data_ptr()' in text
 original=torch.cumsum;started=time.perf_counter();cases=[]
 with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('No GPU allowed')):
  generator=torch.Generator(device='cpu').manual_seed(911)
  for n in (1,2,7,32,256):
   x=torch.randn(3,2,n,generator=generator)
   values=x.reshape(-1,n).tolist();reference=[]
   for row in values:
    acc=0.;r=[]
    for v in row:acc=f32(acc+v);r.append(acc)
    reference.append(r)
   expected=torch.tensor(reference,dtype=torch.float32).reshape_as(x)
   actual=M.cumsum(x,dim=-1,out=x)
   assert torch.equal(actual,expected)
   cases.append({'shape':[3,2,n],'IEEE_single_precision_scalar_reference_bitwise_equal':True})
  assert torch.cumsum is original
 chunk=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_chunk_state.py'
 combined=chunk.with_name('ssd_combined.py')
 ct=chunk.read_text(encoding='utf-8');assert 'torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)' in ct
 assert 'torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])' in ct
 assert 'dtype=torch.float32' in ct
 result={'schema':'independent_mamba_sequential_fp32_cumsum_CPU_review_v1','passed':True,
  'review_source':d(SOURCE),'prototype_source':d(M.__file__),'producer_tests':d(producer),
  'installed_source_bindings':[d(chunk),d(combined)],'cases':cases,
  'measured_CPU_review_seconds':time.perf_counter()-started,
  'accumulation_order_change_explicit':True,'native_CPU_CUDA_numerical_identity_not_claimed':True,
  'scope':'CPU scalar IEEE-FP32 reference and source review only; installed Mamba not imported or modified',
  'model_constructions':0,'checkpoint_loads':0,'GPU_operations':0,'optimizer_steps':0,
  'production_integration_applied':False,'GPU_qualification_complete':False,'final_plan_certificate':False,
  'remaining':['Private Linux backward graph wiring','Strict real GPU/model gradient and state engineering',
               'Numerical/order disclosure and transfer-cost measurement','New formal execution and complete final gate source binding']}
 target=SOURCE.parent/('independent_mamba_fp32_cumsum_CPU_review_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':True,'cases':len(cases),'artifact':d(target),'prototype':d(M.__file__)}))
if __name__=='__main__':main()
