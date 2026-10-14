"""Pure private function-graph construction; not a qualified runtime entry.

Caller must source-verify the installed split-scan function and provide a
reviewed private torch namespace and autograd class factory. This helper neither
imports Mamba/torch nor patches a global/package/class. Formal execution needs
actual Linux GPU/model qualification and new complete provenance bindings.
"""
import copy
import inspect
import types

CLASS_KEY='MambaSplitConv1dScanCombinedFn'
CORE_KEY='_mamba_chunk_scan_combined_bwd'
HELPERS=('_chunk_state_bwd_db','_chunk_state_bwd_ddAcs_stable')

def require(ok,message):
    if not ok:raise ValueError(message)

def clone(function,overrides=None,closure=None):
    require(isinstance(function,types.FunctionType),'Python function required')
    namespace={**function.__globals__,**(overrides or {})}
    result=types.FunctionType(function.__code__,namespace,function.__name__,
                              function.__defaults__,function.__closure__ if closure is None else closure)
    result.__kwdefaults__=copy.copy(function.__kwdefaults__)
    result.__annotations__=copy.copy(function.__annotations__)
    result.__dict__.update(function.__dict__)
    result.__qualname__=function.__qualname__;result.__module__=function.__module__
    return result

def cell(value):
    return (lambda:value).__closure__[0]

def replace_wrapped_leaf(wrapper,leaf,replacement,depth=0):
    require(depth<=8,'Bounded decorator chain required')
    if wrapper is leaf:return replacement
    child=getattr(wrapper,'__wrapped__',None)
    require(isinstance(child,types.FunctionType) and wrapper.__closure__ is not None,
            'Known source-bound wrapper closure required')
    matches=[i for i,c in enumerate(wrapper.__closure__) if c.cell_contents is child]
    require(len(matches)==1,'Exactly one captured wrapped function required')
    new_child=replace_wrapped_leaf(child,leaf,replacement,depth+1)
    closure=list(wrapper.__closure__);closure[matches[0]]=cell(new_child)
    result=clone(wrapper,closure=tuple(closure))
    result.__wrapped__=new_child
    return result

def build_private_split_scan(split_function,*,torch_namespace,class_factory):
    require(isinstance(split_function,types.FunctionType) and callable(class_factory),
            'Reviewed split function and class factory required')
    require(callable(getattr(torch_namespace,'cumsum',None)),'Explicit private cumsum required')
    original_class=split_function.__globals__[CLASS_KEY]
    wrapped=original_class.backward
    inner=inspect.unwrap(wrapped)
    require(isinstance(inner,types.FunctionType),'Source-bound Python backward required')
    core=inner.__globals__[CORE_KEY]
    changes={}
    for key in HELPERS:
        function=core.__globals__.get(key)
        require(isinstance(function,types.FunctionType),'Known SSD helper absent: '+key)
        changes[key]=clone(function,{'torch':torch_namespace})
    new_core=clone(core,changes)
    new_inner=clone(inner,{CORE_KEY:new_core})
    new_backward=replace_wrapped_leaf(wrapped,inner,new_inner)
    new_class=class_factory(original_class,original_class.forward,new_backward)
    require(new_class is not original_class and new_class.forward is original_class.forward
            and new_class.backward is new_backward,'Class factory did not retain exact native forward')
    result=clone(split_function,{CLASS_KEY:new_class})
    require(split_function.__globals__[CLASS_KEY] is original_class
            and original_class.backward is wrapped and inner.__globals__[CORE_KEY] is core,
            'An original global/class was modified')
    return result,{'private_helpers':list(HELPERS),'original_forward_function_reused':True,
                   'original_function_bytecode_preserved':True,'global_or_package_patch_applied':False,
                   'GPU_qualification_complete':False,'new_formal_execution_binding_required':True}
