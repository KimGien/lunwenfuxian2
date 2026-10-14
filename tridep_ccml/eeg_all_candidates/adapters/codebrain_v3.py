"""Unintegrated CodeBrain adapter: private original GConv forward globals.

Keep pinned V2 official initialization/cache/trainability. Only per-instance
GConv.forward receives a private F namespace for the CPU derivative bridge.
No global torch functional or author class is patched. GPU qualification and
fresh formal execution/auditor/evaluator/cost bindings are still mandatory.
"""
import copy
import hashlib
import inspect
from pathlib import Path
import types

from . import codebrain_v2 as V2
from .. import codebrain_linear_backward_cpu_v1 as M
from ._author_support import WORK, require

SOURCE=Path(__file__).resolve()
PREVIOUS_SHA='06c783ff917461d8b25362f38104f8445c7147180872f0f9cd0a433fb69210ac'
BRIDGE_SHA='7e98dfbfaa8f85ef062f6dcee45e698beaf224ceaab65fda0c8e01da5527881d'
AUTHOR=WORK/'tridep_ccml/eeg_all_candidates/vendor/codebrain/Models/SGConv.py'
AUTHOR_SHA='94f12f897ab4e8784a45d64fb65664184c872e6928159a6759fda2920c488be8'
FEATURE_DIM=V2.FEATURE_DIM
INPUT_CONTRACT=copy.deepcopy(V2.INPUT_CONTRACT)
INPUT_CONTRACT.update(adapter_version=3,linear_interpolation_backward_backend='native ATen CPU FP32 with original-device gradient return')

def desc(path):
    p=Path(path).resolve()
    require(p.is_relative_to(WORK),'Source outside workspace')
    return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}

def check_sources():
    for path,digest in ((Path(V2.__file__),PREVIOUS_SHA),(Path(M.__file__),BRIDGE_SHA),(AUTHOR,AUTHOR_SHA)):
        require(desc(path)['sha256']==digest,'Pinned source changed: '+str(path))

def clone_forward(original, namespace):
    """Pure function clone: same bytecode/defaults/closure, independent globals."""
    require(isinstance(original,types.FunctionType),'Python author function required')
    context=dict(original.__globals__)
    context['F']=namespace
    result=types.FunctionType(original.__code__,context,original.__name__,
                              original.__defaults__,original.__closure__)
    result.__kwdefaults__=copy.copy(original.__kwdefaults__)
    result.__annotations__=copy.copy(original.__annotations__)
    result.__qualname__=original.__qualname__
    result.__module__=original.__module__
    result.__wrapped__=original
    return result

def bind_author_block(gconv):
    check_sources()
    require(not getattr(gconv,'_codebrain_cpu_bridge_bound',False),'No double binding')
    original=gconv.forward.__func__
    require(gconv.__class__.__name__=='GConv' and Path(inspect.getsourcefile(original)).resolve()==AUTHOR.resolve()
            and original.__name__=='forward','Only pinned author GConv may be adapted')
    require(original.__globals__['F'].interpolate is M._NATIVE_INTERPOLATE,
            'Author interpolation was already patched')
    require(bool(gconv.kernel_norm_initialized),'V2 cache initialization must precede adaptation')
    function=clone_forward(original,M.functional_namespace())
    gconv.forward=types.MethodType(function,gconv)
    gconv._codebrain_cpu_bridge_bound=True
    require(gconv.forward.__func__.__code__ is original.__code__
            and original.__globals__['F'].interpolate is M._NATIVE_INTERPOLATE,
            'Original class/global function changed')
    return {'author_source':desc(AUTHOR),'bytecode_and_defaults_retained':True,
            'global_torch_functional_or_author_class_modified':False}

def load_encoder(device='cuda',trainable=False):
    check_sources()
    model,audit=V2.load_encoder(device=device,trainable=trainable)
    before=[n for n,_ in model.named_parameters()]
    selected=[n for n,p in model.named_parameters() if p.requires_grad]
    require(len(model.residual_layer.residual_blocks)==8,'Pinned8 residual blocks required')
    bindings=[bind_author_block(b.S41) for b in model.residual_layer.residual_blocks]
    require(before==[n for n,_ in model.named_parameters()]
            and selected==[n for n,p in model.named_parameters() if p.requires_grad],
            'Parameter identities or trainability changed')
    audit.update(adapter_version=3,input_contract=copy.deepcopy(INPUT_CONTRACT),
        runtime_gradient_backend_adaptation={'scope':'Eight private GConv instance forward namespaces',
            'native_forward_retained':True,'linear_backward_cpu_bridge':desc(Path(M.__file__)),
            'host_transfer_sync_cost_unmeasured':True,'performance_improvement_claimed':False,
            'first_order_derivative_only':True,'bindings':bindings},
        GPU_qualification_complete=False,new_formal_execution_registration_required=True)
    audit['implementation_sources']=list(audit.get('implementation_sources',[]))+[desc(SOURCE),desc(Path(M.__file__))]
    return model,audit

def exact_finetune_parameter_names(model): return V2.exact_finetune_parameter_names(model)
def adapt_raw(raw): return V2.adapt_raw(raw)
def encode_windows(model,tensor): return V2.encode_windows(model,tensor)
