"""Read source/pyc code objects under actual Linux Python; no module execution."""
from pathlib import Path
import ast, hashlib, importlib.util, json, marshal, struct, sys, time, types
WORK=Path(__file__).resolve().parents[3]
REL='tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton'
SOURCE=WORK/REL/'ssd_combined.py'
CACHE=SOURCE.parent/'__pycache__/ssd_combined.cpython-312.pyc'
KEY='_mamba_chunk_scan_combined_bwd'
PINSHA='3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba'
raw=SOURCE.read_bytes();cache=CACHE.read_bytes()
assert hashlib.sha256(raw).hexdigest()==PINSHA
assert sys.version_info[:2]==(3,12)
assert cache[:4]==importlib.util.MAGIC_NUMBER
cached=marshal.loads(cache[16:])
tree=ast.parse(raw.decode('utf-8'),filename=str(SOURCE))
node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==KEY]
single=compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(SOURCE),'exec',dont_inherit=True)
full=compile(raw,str(SOURCE),'exec',dont_inherit=True)
def core(module):
 values=[n for n in module.co_consts if isinstance(n,types.CodeType) and n.co_name==KEY]
 assert len(values)==1
 return values[0]
codes={k:core(v) for k,v in [('cached',cached),('full_compile',full),('single_function_compile',single)]}
FIELDS=('co_code','co_consts','co_names','co_varnames','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_freevars','co_cellvars','co_flags',
        'co_nlocals','co_stacksize','co_firstlineno','co_filename','co_name','co_qualname','co_linetable','co_exceptiontable')
def summary(x):
 if isinstance(x,bytes):return {'type':'bytes','length':len(x),'sha256':hashlib.sha256(x).hexdigest(),'hex':x.hex()}
 if isinstance(x,types.CodeType):return {'type':'code','name':x.co_name,'sha256_marshal':hashlib.sha256(marshal.dumps(x)).hexdigest()}
 if isinstance(x,tuple):return [summary(v) for v in x]
 return x
comparisons=[]
for left,right in [('cached','full_compile'),('cached','single_function_compile'),('full_compile','single_function_compile')]:
 differences={key:{left:summary(getattr(codes[left],key)),right:summary(getattr(codes[right],key))}
              for key in FIELDS if getattr(codes[left],key)!=getattr(codes[right],key)}
 comparisons.append({'left':left,'right':right,'equal_fields':[key for key in FIELDS if key not in differences],
                     'different_fields':list(differences),'differences':differences})
record={'schema':'mamba_core_codeobject_compile_scope_diagnosis_v1','python':sys.version,'executable':sys.executable,
        'source':{'path':SOURCE.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(raw).hexdigest()},
        'cache':{'path':CACHE.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(cache).hexdigest(),
                 'header_hex':cache[:16].hex(),'header_flags':int.from_bytes(cache[4:8],'little')},
        'diagnostic_source':{'path':Path(__file__).resolve().relative_to(WORK).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'comparisons':comparisons,'no_installed_module_executed':True,'model_constructions':0,'tensor_operations':0,'GPU_operations':0,'process_controls':0,
        'actual_repair_qualification':False}
out=WORK/'output/analysis/eeg_all_candidates_v1_20261008'/('mamba_core_compile_scope_diagnosis_'+str(time.time_ns())+'.json')
with out.open('x',encoding='utf-8') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'artifact':{'path':out.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},
                  'comparisons':[{k:v for k,v in c.items() if k!='differences'} for c in comparisons]}))

