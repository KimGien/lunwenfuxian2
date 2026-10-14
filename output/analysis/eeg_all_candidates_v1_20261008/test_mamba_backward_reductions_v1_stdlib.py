"""Stdlib fake-tensor guards only; no Torch/model/GPU imports or operations."""
from pathlib import Path
from array import array
import ast,hashlib,itertools,json,math,struct,time,types,unittest
W=Path(__file__).resolve().parents[3];M=W/'tridep_ccml/eeg_all_candidates';A=W/'output/analysis/eeg_all_candidates_v1_20261008'
R=types.ModuleType('reduction_fixture');src=M/'mamba_backward_reductions_v1.py';exec(compile(src.read_bytes(),str(src),'exec'),R.__dict__)
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def strides(shape):
    out=[];acc=1
    for n in reversed(shape):out.append(acc);acc*=n
    return tuple(reversed(out))
class Tensor:
    def __init__(self,shape,value=0,device='cuda:0',dtype='float32',stride=None,base=None,offset=0):
        self.shape=tuple(shape);self.ndim=len(shape);self.device=device;self.dtype=dtype;self.is_cuda=str(device).startswith('cuda')
        self._stride=tuple(stride or strides(shape));self.offset=offset
        self.base=base if base is not None else array('f',[value])*(1+sum((n-1)*s for n,s in zip(shape,self._stride)))
    def __getitem__(self,key):
        assert isinstance(key,slice) and key.step is None
        lo=key.start or 0;hi=key.stop or self.shape[0]
        return Tensor((hi-lo,*self.shape[1:]),device=self.device,dtype=self.dtype,stride=self._stride,base=self.base,offset=self.offset+lo*self._stride[0])
    def positions(self):
        return (self.offset+sum(i*s for i,s in zip(indices,self._stride)) for indices in itertools.product(*[range(n) for n in self.shape]))
    def vals(self):return [self.base[i] for i in self.positions()]
    def stride(self,k=None):return self._stride if k is None else self._stride[k]
    def data_ptr(self):return (id(self.base),self.offset)
    def detach(self):return self
    def contiguous(self):
        if self._stride==strides(self.shape):return self
        out=Tensor(self.shape,device=self.device,dtype=self.dtype);out.copy_(self);return out
    def to(self,device=None,dtype=None):
        out=Tensor(self.shape,device=device or self.device,dtype=dtype or self.dtype);out.copy_(self);return out
    def copy_(self,other):
        assert self.shape==other.shape
        for to,fr in zip(self.positions(),other.positions()):self.base[to]=other.base[fr]
        return self
    def fill(self,value):
        for p in self.positions():self.base[p]=value
        return self
    def __add__(self,other):
        assert self.shape==other.shape and self.device==other.device and self.dtype==other.dtype
        out=Tensor(self.shape,device=self.device,dtype=self.dtype)
        for p,a,b in zip(out.positions(),self.positions(),other.positions()):out.base[p]=f32(self.base[a]+other.base[b])
        return out
class Torch:
    Tensor=Tensor;float32='float32';grad=False
    @staticmethod
    def is_grad_enabled():return Torch.grad
    @staticmethod
    def empty_like(x,**kw):return Tensor(x.shape,device=kw.get('device',x.device),dtype=kw.get('dtype',x.dtype),stride=x.stride())
    zeros_like=empty_like
class Contracts(unittest.TestCase):
    def setUp(self):Torch.grad=False;self.chunk_calls=[];self.conv_calls=[]
    def native_chunk(self,ddA,ddt_out,dt,A,dt_bias=None,dt_softplus=False,dt_limit=None,ddt=None):
        i=len(self.chunk_calls);self.chunk_calls.append((ddA.shape,dt.shape,ddt.stride()))
        self.assertEqual(ddA.shape,(1,8,1,256));self.assertEqual(dt.shape,(1,95,8))
        ddt.fill(101+i)
        return ddt,Tensor((8,),[2**24,1,-2**24][i]),None if dt_bias is None else Tensor((8,),i+1)
    def native_conv(self,x,w,b,dy,seq,initial,final,dx,ret,act):
        i=len(self.conv_calls);self.conv_calls.append((x.shape,x.stride(),dy.stride(),dx.stride()))
        self.assertEqual(x.shape,(1,528,95));self.assertEqual(x.stride(),(528*95,95,1));self.assertEqual(dy.stride(),x.stride());self.assertEqual(dx.stride(),x.stride())
        return dx.fill(201+i),Tensor(w.shape,[2**24,1,-2**24][i]),None if b is None else Tensor(b.shape,i+1),None
    def build(self,chunk=None,conv=None):
        module=types.SimpleNamespace(causal_conv1d_fwd=lambda *args:None,causal_conv1d_bwd=conv or self.native_conv)
        c,p,proof=R.build_private_reductions(chunk or self.native_chunk,module,torch_module=Torch)
        self.assertIs(p.causal_conv1d_fwd,module.causal_conv1d_fwd)
        return c,p,proof
    def chunk_args(self,b=3,bias=True):return [Tensor((b,8,1,256)),Tensor((b,8,1,256)),Tensor((b,95,8)),Tensor((8,)),Tensor((8,)) if bias else None]
    def conv_args(self,b=3,bias=True):
        return [Tensor((b,528,95),stride=(50160,1,528)),Tensor((528,4)),Tensor((528,)) if bias else None,Tensor((b,528,95),stride=(50160,1,528)),None,None,None,Tensor((b,528,95),stride=(100320,1,1056)),False,True]
    def test_chunk_preserves_noncontiguous_ddt_and_fixed_fp32_order(self):
        c,_,p=self.build();args=self.chunk_args();dest=Tensor((3,95,8),stride=(1520,16,1))
        ddt,da,db=c(*args,dt_softplus=True,ddt=dest)
        self.assertIs(ddt,dest);self.assertEqual(da.vals(),[0.0]*8);self.assertEqual(db.vals(),[6.0]*8)
        for i in range(3):self.assertTrue(all(v==101+i for v in dest[i:i+1].vals()))
        self.assertEqual(R.snapshot_native_call_counts(c._native_call_counts)['chunk_native_calls_returned'],3)
    def test_causal_channels_first_and_copy_back_original_view(self):
        _,p,proof=self.build();args=self.conv_args();before=args[0].vals();dest=args[7];savedstride=dest.stride();out,dw,db,tail=p.causal_conv1d_bwd(*args)
        self.assertIs(out,dest);self.assertEqual(out.stride(),savedstride);self.assertIsNone(tail)
        self.assertEqual(dw.vals(),[0.0]*(528*4));self.assertEqual(db.vals(),[6.0]*528);self.assertEqual(args[0].vals(),before)
        for i in range(3):self.assertTrue(all(v==201+i for v in out[i:i+1].vals()))
        self.assertEqual(R.snapshot_native_call_counts(p.causal_conv1d_bwd._native_call_counts)['causal_native_calls_started'],3)
    def test_optional_bias_and_output_allocation(self):
        c,p,_=self.build();ddt,da,db=c(*self.chunk_args(1,False),dt_softplus=True);self.assertIsNone(db);self.assertEqual(ddt.shape,(1,95,8))
        args=self.conv_args(1,False);args[7]=None;out,dw,db,tail=p.causal_conv1d_bwd(*args);self.assertIsNone(db);self.assertIsNone(tail);self.assertEqual(out.shape,(1,528,95))
    def test_chunk_shape_dtype_device_and_policy_guards_before_native(self):
        c,_,_=self.build()
        cases=[(0,Tensor((3,8,2,256))),(2,Tensor((3,96,8))),(2,Tensor((9,95,8))),(3,Tensor((8,),dtype='float16')),(3,Tensor((8,),device='cuda:1'))]
        for idx,value in cases:
            args=self.chunk_args();args[idx]=value
            with self.assertRaises(ValueError):c(*args,dt_softplus=True)
        with self.assertRaises(ValueError):c(*self.chunk_args(),dt_softplus=False)
        self.assertEqual(self.chunk_calls,[])
    def test_causal_state_activation_and_shape_guards_before_native(self):
        _,p,_=self.build()
        for idx,value in [(4,object()),(5,object()),(6,object()),(8,True),(9,False),(1,Tensor((528,3))),(0,Tensor((3,528,96))),(2,Tensor((528,),device='cuda:1'))]:
            args=self.conv_args();args[idx]=value
            with self.assertRaises(ValueError):p.causal_conv1d_bwd(*args)
        self.assertEqual(self.conv_calls,[])
    def test_higher_order_rejected_without_calls(self):
        c,p,_=self.build();Torch.grad=True
        with self.assertRaises(ValueError):c(*self.chunk_args(),dt_softplus=True)
        with self.assertRaises(ValueError):p.causal_conv1d_bwd(*self.conv_args())
        self.assertEqual(self.chunk_calls+self.conv_calls,[])
    def test_native_view_substitution_rejected_and_actual_returns_counted(self):
        def bad(*args,**kw):return Tensor((1,95,8)),Tensor((8,)),Tensor((8,))
        c,_,proof=self.build(chunk=bad)
        with self.assertRaisesRegex(ValueError,'supplied ddt view'):c(*self.chunk_args(),dt_softplus=True)
        self.assertEqual(R.snapshot_native_call_counts(c._native_call_counts)['chunk_native_calls_started'],1);self.assertEqual(R.snapshot_native_call_counts(c._native_call_counts)['chunk_native_calls_returned'],1)
    def test_native_conv_scratch_substitution_rejected(self):
        def bad(x,w,b,dy,seq,ini,final,dx,ret,act):return Tensor(x.shape),Tensor(w.shape),Tensor(b.shape),None
        _,p,proof=self.build(conv=bad)
        with self.assertRaisesRegex(ValueError,'channels-first scratch'):p.causal_conv1d_bwd(*self.conv_args())
        self.assertEqual(R.snapshot_native_call_counts(p.causal_conv1d_bwd._native_call_counts)['causal_native_calls_returned'],1)
    def test_original_G3_source_functions_preserved(self):
        def funcs(name):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse((M/name).read_text(encoding='utf-8-sig')).body if isinstance(x,ast.FunctionDef)}
        a=funcs('mamba_private_backward_graph_v3.py');b=funcs('mamba_private_backward_graph_v4.py')
        for key in a:
            if key!='build_private_split_scan':self.assertEqual(a[key],b[key],key)
        self.assertEqual(hashlib.sha256((M/'mamba_private_backward_graph_v3.py').read_bytes()).hexdigest(),'4ae48821d2350ebf1eaf5172e94ef20392efa931a7ff8f5afd73ba1bbdaa9ee9')
        self.assertEqual(hashlib.sha256((M/'adapters/eegmamba_v4.py').read_bytes()).hexdigest(),'82198ac0025be47199b743162952f30953342370c7a05845f727879f8a38d0c2')
    def test_live_counts_never_change_static_contract(self):
        c,p,proof=self.build();before=json.dumps(proof,sort_keys=True)
        snapshot=R.snapshot_native_call_counts(c._native_call_counts)
        c(*self.chunk_args(1),dt_softplus=True)
        p.causal_conv1d_bwd(*self.conv_args(1))
        self.assertEqual(json.dumps(proof,sort_keys=True),before)
        self.assertEqual(snapshot['chunk_native_calls_returned'],0)
        self.assertEqual(R.snapshot_native_call_counts(c._native_call_counts)['chunk_native_calls_returned'],1)
        self.assertEqual(R.snapshot_native_call_counts(c._native_call_counts)['causal_native_calls_returned'],1)
        with self.assertRaises(ValueError):R.snapshot_native_call_counts({'unexpected':0})
    def test_adapter_source_pins_and_no_global_mutation(self):
        text=(M/'adapters/eegmamba_v5.py').read_text();ast.parse(text)
        for name in ['mamba_backward_reductions_v1.py','mamba_private_backward_graph_v4.py']:
            self.assertIn(hashlib.sha256((M/name).read_bytes()).hexdigest(),text)
        self.assertIn('ac8ce3550175e716b150c091588a0e973062bd5ff2e695effff89fbe19ad9a84',text)
        self.assertIn('7286f93996561532dac8452de1e75bd44a31baf5acbb3ad4fd48b87fbc7fdc94',text)
        self.assertFalse(R.build_private_reductions.__globals__.get('torch'))
r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
names=['mamba_backward_reductions_v1.py','mamba_private_backward_graph_v4.py','adapters/eegmamba_v5.py']
report={'schema':'mamba_backward_reductions_stdlib_contract_v1','passed':r.wasSuccessful(),'tests':r.testsRun,'failures':[str(t) for t,_ in r.failures],'errors':[str(t) for t,_ in r.errors],
 'sources':[{'path':(M/n).relative_to(W).as_posix(),'sha256':hashlib.sha256((M/n).read_bytes()).hexdigest()} for n in names],
 'test_source':{'path':Path(__file__).resolve().relative_to(W).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'scope':'stdlib array-f32 fake tensors only; native functions replaced by fixtures','model':0,'GPU':0,'torch_tensor':0,'process_control':0,'actual_qualification':False,
 'official_native_reference':['https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.5.0/csrc/causal_conv1d.cpp','https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.5.0/csrc/causal_conv1d_bwd.cu']}
out=A/('mamba_backward_reductions_stdlib_contract_'+str(time.time_ns())+'.json')
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'report':out.relative_to(W).as_posix(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'passed':r.wasSuccessful(),'tests':r.testsRun}))
raise SystemExit(0 if r.wasSuccessful() else 1)
