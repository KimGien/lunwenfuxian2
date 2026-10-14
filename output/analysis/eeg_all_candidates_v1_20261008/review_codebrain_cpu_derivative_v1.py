"""Independent source/derivative check of unintegrated CodeBrain CPU bridge."""
from pathlib import Path
import ast,hashlib,io,json,math,sys,time
from unittest.mock import patch
WORK=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(WORK))
SOURCE=Path(__file__).resolve()
PROTOTYPE=WORK/'tridep_ccml/eeg_all_candidates/codebrain_linear_backward_cpu_v1.py'
PIN='7e98dfbfaa8f85ef062f6dcee45e698beaf224ceaab65fda0c8e01da5527881d'
def desc(p):
 p=Path(p).resolve()
 if not p.is_relative_to(WORK): raise ValueError('Outside workspace')
 return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def require(v,m):
 if not v: raise ValueError(m)

def static_check():
 require(desc(PROTOTYPE)['sha256']==PIN,'Prototype changed')
 tree=ast.parse(PROTOTYPE.read_text(encoding='utf-8'))
 calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
 require(not any(any(x in c for x in ('use_deterministic','autocast','cuda','manual_seed','load_encoder','torch.load')) for c in calls),
         'Prototype cannot alter flags, initialize GPU, or load a model/checkpoint')
 require('torch.ops.aten.upsample_linear1d_backward.default' in calls,'Native derivative absent')
 cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
 backward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='backward')
 forward=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
 require(sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='_NATIVE_INTERPOLATE' for n in ast.walk(forward))==1,
         'Native original forward must be called once')
 require(any(isinstance(n,ast.keyword) and n.arg=='device' and isinstance(n.value,ast.Constant) and n.value.value=='cpu'
             for n in ast.walk(backward)),'Derivative must explicitly use CPU')
 sssm=WORK/'tridep_ccml/eeg_all_candidates/vendor/codebrain/Models/SSSM.py'
 sg=WORK/'tridep_ccml/eeg_all_candidates/vendor/codebrain/Models/SGConv.py'
 adapter=WORK/'tridep_ccml/eeg_all_candidates/adapters/codebrain.py'
 native=WORK/'S00/.venv/Lib/site-packages/torch/include/ATen/native/UpSample.h'
 texts=[p.read_text(encoding='utf-8') for p in (sssm,sg,adapter,native)]
 require('d_model=2 * self.res_channels' in texts[0] and 'channels=4' in texts[0]
         and 'res_channels=200' in texts[2] and 's4_lmax=570' in texts[2] and 's4_bidirectional=True' in texts[2],
         'Expected author kernel dimensions changed')
 require('channels *= 2' in texts[1] and "'kernel_dim', 2" in texts[1] and "'init_scale', 0" in texts[1],
         'Author kernel default/bi-directionality changed')
 require('max(0, i-1)+self.init_scale' in texts[1] and 'compute_source_index_and_lambda' in texts[3]
         and 'input_index1 = input_index0 + offset' in texts[3], 'Interpolation contract differs')
 return [desc(p) for p in (sssm,sg,adapter,native)]

def main():
 bound=static_check()
 import torch
 import torch.nn.functional as F
 from tridep_ccml.eeg_all_candidates import codebrain_linear_backward_cpu_v1 as M
 previous=torch.are_deterministic_algorithms_enabled()
 warn=torch.is_deterministic_algorithms_warn_only_enabled()
 cases=[]; started=time.perf_counter()
 original=F.interpolate
 try:
  torch.use_deterministic_algorithms(True)
  with patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU-only reviewer')):
   # Author config: four channels doubled by bidirectionality; 2*200 hidden,
   # width2; ceil(log2(570/2))+1=10 scales, cat factors1,1,2,...,256.
   factors=[2**max(0,i-1) for i in range(10)]
   generator=torch.Generator(device='cpu').manual_seed(771)
   for factor in factors:
    x=torch.randn(8,400,2,generator=generator).requires_grad_()
    y=M.interpolate(x,scale_factor=factor,mode='linear')
    reference=original(x.detach(),scale_factor=factor,mode='linear',align_corners=False)
    require(torch.equal(y.detach(),reference),'CPU original forward differs')
    # An independent explicit source-coordinate Jacobian-transpose calculation.
    # Binary-exact bounded upstream gradients avoid spurious association errors.
    gradient=torch.randint(-8,9,y.shape,generator=generator).float()/8
    y.backward(gradient)
    expected=torch.zeros_like(x)
    for out_index in range(y.shape[-1]):
     if factor==1: left=right=out_index; w1=0.0
     else:
      coordinate=max((out_index+0.5)/factor-0.5,0.0)
      left=min(math.floor(coordinate),x.shape[-1]-1)
      right=left+(1 if left<x.shape[-1]-1 else 0)
      w1=min(max(coordinate-left,0.0),1.0)
     expected[...,left].add_(gradient[...,out_index]*(1-w1))
     expected[...,right].add_(gradient[...,out_index]*w1)
    require(torch.equal(x.grad,expected),'Explicit CPU derivative differs')
    require(torch.are_deterministic_algorithms_enabled()
            and not torch.is_deterministic_algorithms_warn_only_enabled(),'Strict flags changed')
    cases.append({'shape':[8,400,2],'factor':factor,'native_CPU_forward_bitwise_equal':True,
                  'explicit_coordinate_derivative_bitwise_equal':True})
   namespace=M.functional_namespace()
   require(F.interpolate is original and namespace.interpolate is M.interpolate
           and namespace.pad is F.pad,'Global torch functional was modified')
 finally: torch.use_deterministic_algorithms(previous,warn_only=warn)
 value={'schema':'independent_codebrain_cpu_bridge_source_and_derivative_review_v1','passed':True,
        'review_source':desc(SOURCE),'prototype_source':desc(PROTOTYPE),'author_and_native_sources':bound,
        'cases':cases,'measured_CPU_review_seconds':time.perf_counter()-started,
        'scope':'Independent AST/source review and CPU explicit-coordinate derivative; no author model or weights',
        'GPU_operations':0,'model_constructions':0,'checkpoint_loads':0,'optimizer_steps':0,
        'current_queue_or_sources_modified':False,'GPU_qualification_complete':False,
        'formal_integration_applied':False,'higher_order_derivatives_qualified':False,
        'full_model_cross_device_numerical_equivalence_established':False,'final_plan_certificate':False,
        'remaining':['GPU native-forward equality and real warm/backbone engineering',
                     'CPU transfer/sync cost measurement','Fresh execution/audit/evaluation/cost/final source binding']}
 target=SOURCE.parent/('independent_codebrain_cpu_bridge_review_'+str(time.time_ns())+'.json')
 with target.open('x',encoding='utf-8') as f: json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'passed':True,'cases':len(cases),'artifact':desc(target)}))
if __name__=='__main__': main()
