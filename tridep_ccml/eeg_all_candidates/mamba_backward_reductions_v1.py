"""Private, shape-restricted first-order Mamba backward reductions.

No global/native package is patched. Forward dispatch remains the original
function. Native backward batch items are run separately and small parameter
gradients are accumulated left-to-right in CPU FP32. This changes numerical
order and native-call count; CUDA equivalence/speed have not been established.
"""
from types import SimpleNamespace

SEQ=95
CHUNK=256
HEADS=8
CONV_DIM=528
CONV_WIDTH=4
MAX_BATCH=8
COUNT_KEYS=('chunk_wrappers_started','chunk_native_calls_started','chunk_native_calls_returned',
            'causal_wrappers_started','causal_native_calls_started','causal_native_calls_returned')


def snapshot_native_call_counts(counts):
    require(isinstance(counts,dict) and set(counts)==set(COUNT_KEYS),
            'Exact private native call counters required')
    require(all(type(value) is int and value>=0 for value in counts.values()),
            'Nonnegative integer native call counters required')
    return {key:counts[key] for key in COUNT_KEYS}


def require(ok,message):
    if not ok:raise ValueError(message)


def build_private_reductions(native_chunk,native_conv,*,torch_module):
    T=torch_module
    require(callable(native_chunk) and callable(getattr(native_conv,'causal_conv1d_fwd',None))
            and callable(getattr(native_conv,'causal_conv1d_bwd',None)), 'Exact native backward functions required')
    counts={'chunk_wrappers_started':0,'chunk_native_calls_started':0,'chunk_native_calls_returned':0,
            'causal_wrappers_started':0,'causal_native_calls_started':0,'causal_native_calls_returned':0}
    def tensor(value,shape,device=None):
        require(isinstance(value,T.Tensor) and tuple(value.shape)==tuple(shape)
                and value.dtype==T.float32 and value.is_cuda, 'Pinned CUDA FP32 tensor shape required')
        require(device is None or value.device==device,'One CUDA device required')
    def first_order():
        require(not T.is_grad_enabled(),'Higher-order backward is unsupported')
    def add_cpu(total,value):
        # Each add is a separate FP32 operation in fixed increasing batch order.
        return total+value.detach().to(device='cpu',dtype=T.float32).contiguous()
    def chunk(ddA,ddt_out,dt,A,dt_bias=None,dt_softplus=False,
              dt_limit=(0.0,float('inf')),ddt=None):
        first_order()
        require(isinstance(dt,T.Tensor) and dt.ndim==3,'Pinned dt rank required')
        batch,seqlen,nheads=dt.shape
        require(1<=batch<=MAX_BATCH and seqlen==SEQ and nheads==HEADS,
                'Only registered 1..8 windows with 95 tokens and 8 heads supported')
        tensor(dt,(batch,SEQ,HEADS));device=dt.device
        tensor(ddA,(batch,HEADS,1,CHUNK),device)
        tensor(ddt_out,(batch,HEADS,1,CHUNK),device);tensor(A,(HEADS,),device)
        require(type(dt_softplus) is bool and dt_softplus is True
                and tuple(dt_limit)==(0.0,float('inf')), 'Pinned softplus/dt range required')
        if dt_bias is not None:tensor(dt_bias,(HEADS,),device)
        if ddt is None:ddt=T.empty_like(dt)
        tensor(ddt,(batch,SEQ,HEADS),device)
        dA_cpu=T.zeros_like(A,device='cpu',dtype=T.float32)
        db_cpu=None if dt_bias is None else T.zeros_like(dt_bias,device='cpu',dtype=T.float32)
        counts['chunk_wrappers_started']+=1
        for index in range(batch):
            target=ddt[index:index+1]
            counts['chunk_native_calls_started']+=1
            local_dt,local_A,local_bias=native_chunk(ddA[index:index+1],ddt_out[index:index+1],
                dt[index:index+1],A,dt_bias=dt_bias,dt_softplus=dt_softplus,dt_limit=dt_limit,ddt=target)
            counts['chunk_native_calls_returned']+=1
            tensor(local_dt,(1,SEQ,HEADS),device);tensor(local_A,(HEADS,),device)
            require(local_dt.data_ptr()==target.data_ptr() and local_dt.stride()==target.stride(),
                    'Native chunk must retain the supplied ddt view')
            dA_cpu=add_cpu(dA_cpu,local_A)
            if dt_bias is None:require(local_bias is None,'Unexpected native dt-bias gradient')
            else:
                tensor(local_bias,(HEADS,),device);db_cpu=add_cpu(db_cpu,local_bias)
        return ddt,dA_cpu.to(device=device),None if db_cpu is None else db_cpu.to(device=device)
    def causal(x,weight,bias,dout,seq_idx,initial_states,dfinal_states,dx,
               return_dinitial_states,activation):
        first_order()
        require(seq_idx is None and initial_states is None and dfinal_states is None
                and return_dinitial_states is False and activation is True,
                'Only registered stateless SiLU causal backward supported')
        require(isinstance(x,T.Tensor) and x.ndim==3,'Pinned causal rank required')
        batch,dim,seqlen=x.shape
        require(1<=batch<=MAX_BATCH and dim==CONV_DIM and seqlen==SEQ,'Pinned causal shape required')
        tensor(x,(batch,CONV_DIM,SEQ));device=x.device
        tensor(dout,x.shape,device);tensor(weight,(CONV_DIM,CONV_WIDTH),device)
        require(weight.stride(-1)==1,'Contiguous native convolution weights required')
        if bias is not None:tensor(bias,(CONV_DIM,),device)
        if dx is None:dx=T.empty_like(x)
        tensor(dx,x.shape,device)
        dw_cpu=T.zeros_like(weight,device='cpu',dtype=T.float32)
        db_cpu=None if bias is None else T.zeros_like(bias,device='cpu',dtype=T.float32)
        counts['causal_wrappers_started']+=1
        for index in range(batch):
            # Shape [1,D,L], row-major: native channels-first grid is (1,D).
            xb=x[index:index+1].detach().contiguous()
            dyb=dout[index:index+1].detach().contiguous()
            scratch=T.empty_like(xb)
            require(xb.stride(-1)==dyb.stride(-1)==scratch.stride(-1)==1
                    and xb.stride(1)==dyb.stride(1)==scratch.stride(1)==SEQ,
                    'Channels-first native backward buffers required')
            counts['causal_native_calls_started']+=1
            result=native_conv.causal_conv1d_bwd(xb,weight,bias,dyb,None,None,None,scratch,False,True)
            counts['causal_native_calls_returned']+=1
            require(isinstance(result,(tuple,list)) and len(result)==4 and result[3] is None,
                    'Exact stateless native causal result required')
            local_dx,local_dw,local_db,_=result
            tensor(local_dx,(1,CONV_DIM,SEQ),device);tensor(local_dw,weight.shape,device)
            require(local_dx.data_ptr()==scratch.data_ptr() and local_dx.stride()==scratch.stride(),
                    'Native causal must retain channels-first scratch')
            dx[index:index+1].copy_(local_dx)
            dw_cpu=add_cpu(dw_cpu,local_dw)
            if bias is None:require(local_db is None,'Unexpected native convolution bias gradient')
            else:
                tensor(local_db,bias.shape,device);db_cpu=add_cpu(db_cpu,local_db)
        return dx,dw_cpu.to(device=device),None if db_cpu is None else db_cpu.to(device=device),None
    chunk._native_call_counts=counts
    causal._native_call_counts=counts
    proxy=SimpleNamespace(causal_conv1d_fwd=native_conv.causal_conv1d_fwd,causal_conv1d_bwd=causal)
    proof={'shape_contract':{'tokens':SEQ,'chunk_size':CHUNK,'nchunks':1,'heads':HEADS,
                            'conv_channels':CONV_DIM,'conv_width':CONV_WIDTH,'batch_range':[1,MAX_BATCH]},
           'backward_only_batch_decomposition':True,'native_causal_backward_layout_changed':True,'forward_function_identity_retained':True,
           'native_backward_calls_per_wrapper':'B instead of 1 for each affected native function',
           'native_function_call_counts':'separate private Python instrumentation; excluded from static contract','actual_native_kernel_launch_count':None,
           'native_autotuner_internal_launches_included_in_counts':False,
           'parameter_reduction':'CPU FP32 sequential increasing batch index',
           'accumulation_order_changed':True,'original_CUDA_gradient_equivalence_established':False,
           'transfer_and_extra_native_call_time_unmeasured':True,'GPU_qualification_complete':False}
    return chunk,proxy,proof
