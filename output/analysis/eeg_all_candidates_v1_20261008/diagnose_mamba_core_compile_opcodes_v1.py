from pathlib import Path
import ast, dis, hashlib, importlib.util, json, marshal, sys, time, types
W=Path(__file__).resolve().parents[3]
P=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py'
raw=P.read_bytes(); assert hashlib.sha256(raw).hexdigest()=='3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba'
assert sys.version_info[:2]==(3,12)
K='_mamba_chunk_scan_combined_bwd'
tree=ast.parse(raw.decode(),filename=str(P));node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==K]
def core(module):
    cs=[c for c in module.co_consts if isinstance(c,types.CodeType) and c.co_name==K];assert len(cs)==1;return cs[0]
full=core(compile(raw,str(P),'exec',dont_inherit=True));isolated=core(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(P),'exec',dont_inherit=True))
cache=P.parent/'__pycache__/ssd_combined.cpython-312.pyc';b=cache.read_bytes();assert b[:4]==importlib.util.MAGIC_NUMBER
cached=core(marshal.loads(b[16:]));assert full.co_code==cached.co_code
fs={i.offset:i for i in dis.get_instructions(full,show_caches=True)};ss={i.offset:i for i in dis.get_instructions(isolated,show_caches=True)}
def val(i):return {'offset':i.offset,'opname':i.opname,'arg':i.arg,'argrepr':i.argrepr,'line':i.positions.lineno,'column':i.positions.col_offset}
rows=[{'full':val(fs[o]),'isolated':val(ss[o])} for o in sorted(set(fs)|set(ss)) if (fs[o].opcode,fs[o].arg)!=(ss[o].opcode,ss[o].arg)]
out=W/'output/analysis/eeg_all_candidates_v1_20261008'/('mamba_core_compile_opcode_difference_'+str(time.time_ns())+'.json')
r={'schema':'mamba_core_compile_opcode_difference_v1','python':sys.version,'source_sha256':hashlib.sha256(raw).hexdigest(),'cached_equals_full_bytecode':True,'differing_instructions':rows,'source':{'path':Path(__file__).resolve().relative_to(W).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'model':0,'GPU':0,'Tensor':0,'module_execution':0,'process_controls':0}
with out.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({'path':out.relative_to(W).as_posix(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'differences':rows}))
