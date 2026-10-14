"""Pure private function-graph construction; not a qualified runtime entry.

Caller must source-verify the installed split-scan function and provide a
reviewed private torch namespace and autograd class factory. Source-core code
objects retain the complete module compiler context without module execution. This helper neither
imports Mamba/torch nor patches a global/package/class. Formal execution needs
actual Linux GPU/model qualification and new complete provenance bindings.
"""
import ast
import copy
import hashlib
from pathlib import Path
import inspect
import types

CLASS_KEY='MambaSplitConv1dScanCombinedFn'
CORE_KEY='_mamba_chunk_scan_combined_bwd'
HELPERS=('_chunk_state_bwd_db','_chunk_state_bwd_ddAcs_stable')
COMBINED_SHA='3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba'
PRIVATE_CUMSUM='_eeg_private_fixed_fp32_cumsum'
TARGET_EXPRESSION='ddA_cumsum_prev.flip([-1]).cumsum(dim=-1)'


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

def rewrite_core_ast(node):
    # The sole changed call is the source-locked Tensor.cumsum within ddA_prev.
    node=copy.deepcopy(node)
    require(isinstance(node,ast.FunctionDef) and node.name==CORE_KEY
            and not node.decorator_list, 'Exact undecorated SSD backward function required')
    wanted=ast.dump(ast.parse(TARGET_EXPRESSION,mode='eval').body,include_attributes=False)
    replacements=0
    class Replace(ast.NodeTransformer):
        def visit_Call(self,current):
            nonlocal replacements
            if ast.dump(current,include_attributes=False)==wanted:
                replacements+=1
                return ast.copy_location(ast.Call(func=ast.Name(id=PRIVATE_CUMSUM,ctx=ast.Load()),
                    args=[copy.deepcopy(current.func.value)],
                    keywords=[ast.keyword(arg='dim',value=ast.Constant(value=-1))]),current)
            return self.generic_visit(current)
    changed=Replace().visit(node)
    require(replacements==1,'Exactly one pinned core reverse-cumsum call required')
    require(not any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                    and x.func.attr=='cumsum' for x in ast.walk(changed)),
            'Unreviewed remaining core Tensor.cumsum call')
    return ast.fix_missing_locations(changed)

def _code_for(module,filename):
    # Full-module symbol-table context is required by Python 3.12's compiler.
    # compile() builds code objects only; no installed module/import is executed.
    require(isinstance(module,ast.Module),'Complete pinned module AST required')
    code=compile(ast.fix_missing_locations(module),filename,'exec',dont_inherit=True)
    values=[x for x in code.co_consts if isinstance(x,types.CodeType) and x.co_name==CORE_KEY]
    require(len(values)==1,'Exactly one compiled source core required')
    return values[0]

def private_core(core,changes,cumsum_helper):
    require(isinstance(core,types.FunctionType) and callable(cumsum_helper),
            'Exact Python core and explicit private helper required')
    path=Path(inspect.getsourcefile(core)).resolve()
    raw=path.read_bytes()
    require(path.name=='ssd_combined.py' and hashlib.sha256(raw).hexdigest()==COMBINED_SHA,
            'Installed SSD combined source changed')
    tree=ast.parse(raw.decode('utf-8'),filename=str(path))
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name==CORE_KEY]
    require(len(nodes)==1 and not core.__code__.co_freevars, 'Exact top-level SSD core required')
    original=_code_for(tree,str(path))
    require(all(getattr(original,key)==getattr(core.__code__,key) for key in
            ('co_code','co_consts','co_names','co_varnames','co_argcount','co_posonlyargcount',
             'co_kwonlyargcount','co_freevars','co_cellvars','co_flags')),
            'Loaded SSD core differs from pinned source function')
    require(PRIVATE_CUMSUM not in core.__globals__ and PRIVATE_CUMSUM not in changes,
            'Private core helper name collision')
    changed_tree=copy.deepcopy(tree)
    indices=[i for i,node in enumerate(changed_tree.body)
             if isinstance(node,ast.FunctionDef) and node.name==CORE_KEY]
    require(len(indices)==1,'Exactly one complete-module core replacement required')
    changed_tree.body[indices[0]]=rewrite_core_ast(changed_tree.body[indices[0]])
    code=_code_for(changed_tree,str(path))
    namespace={**core.__globals__,**changes,PRIVATE_CUMSUM:cumsum_helper}
    result=types.FunctionType(code,namespace,core.__name__,core.__defaults__,core.__closure__)
    result.__kwdefaults__=copy.copy(core.__kwdefaults__)
    result.__annotations__=copy.copy(core.__annotations__)
    result.__dict__.update(core.__dict__)
    result.__qualname__=core.__qualname__;result.__module__=core.__module__
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
    new_core=private_core(core,changes,torch_namespace.cumsum)
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
                   'original_function_bytecode_preserved':False,
                   'split_wrapper_and_native_forward_bytecode_preserved':True,
                   'core_source_sha256':COMBINED_SHA,'core_AST_single_cumsum_call_replaced':True,
                   'core_compilation_scope':'complete_pinned_module_AST_without_execution',
                   'core_private_cumsum_out_none':True,'accumulation_order_changed':True,'global_or_package_patch_applied':False,
                   'GPU_qualification_complete':False,'new_formal_execution_binding_required':True}
