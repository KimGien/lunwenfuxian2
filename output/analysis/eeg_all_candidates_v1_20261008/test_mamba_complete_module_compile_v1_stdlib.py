"""Actual Python3.12 code-object contracts; installed model code is never executed."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,marshal,sys,time,types,unittest
W=Path(__file__).resolve().parents[3];M=W/'tridep_ccml/eeg_all_candidates';A=W/'output/analysis/eeg_all_candidates_v1_20261008'
P=W/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/assets/eegmamba_linux_env/lib/python3.12/site-packages/mamba_ssm/ops/triton/ssd_combined.py'
GFILE=M/'mamba_private_backward_graph_v3.py'
PINS={'mamba_cumsum_cpu_fp32_v2.py':'689eec011f11cdf4f54a4cf2bc2545ac115011dded729f4048bc84de8381cabe','mamba_private_backward_graph_v2.py':'8686a7f890675b7026f0949be3f66f2ac607426bd0f89b1f0fc38829f74b33d8','adapters/eegmamba_v3.py':'f8abccb63454a20316522aaad32dd58354e517837498ffba109ba927cd3e0557','mamba_private_backward_graph_v3.py':'4ae48821d2350ebf1eaf5172e94ef20392efa931a7ff8f5afd73ba1bbdaa9ee9','adapters/eegmamba_v4.py':'82198ac0025be47199b743162952f30953342370c7a05845f727879f8a38d0c2'}
assert sys.version_info[:2]==(3,12)
raw=P.read_bytes();assert hashlib.sha256(raw).hexdigest()=='3717ba118605d9a5c9271d9195e2168c55b82242adcd46b143d3d9918f7c72ba'
cache=P.parent/'__pycache__/ssd_combined.cpython-312.pyc';cb=cache.read_bytes();assert cb[:4]==importlib.util.MAGIC_NUMBER
G=types.ModuleType('private_graph_stdlib_contract');G.__file__=str(GFILE)
exec(compile(GFILE.read_bytes(),str(GFILE),'exec'),G.__dict__)
tree=ast.parse(raw.decode(),filename=str(P));K=G.CORE_KEY
cached,=[c for c in marshal.loads(cb[16:]).co_consts if isinstance(c,types.CodeType) and c.co_name==K]
F=('co_code','co_consts','co_names','co_varnames','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_freevars','co_cellvars','co_flags','co_nlocals','co_stacksize','co_firstlineno','co_filename','co_name','co_qualname','co_linetable','co_exceptiontable')
def fixture(code=cached,ns=None):return types.FunctionType(code,{'__name__':'never_executed_core',**(ns or {})})
def helper(*a,**k):raise AssertionError('Model helper must never execute in this contract')
class Contracts(unittest.TestCase):
    def test_full_module_matches_actual_cache_all18(self):
        compiled=G._code_for(copy.deepcopy(tree),str(P))
        for key in F:self.assertEqual(getattr(cached,key),getattr(compiled,key),key)
    def test_actual_cached_function_passes_and_original_unchanged(self):
        original=fixture();oldglobals=original.__globals__.copy();new=G.private_core(original,{},helper)
        self.assertIs(original.__code__,cached);self.assertEqual(original.__globals__,oldglobals)
        self.assertIs(new.__globals__[G.PRIVATE_CUMSUM],helper);self.assertIsNot(new,original)
        self.assertEqual(new.__defaults__,original.__defaults__);self.assertEqual(new.__code__.co_argcount,cached.co_argcount)
        self.assertNotEqual(new.__code__.co_code,cached.co_code)
    def test_bad_co_code_rejected(self):
        b=bytearray(cached.co_code);b[109]^=1
        with self.assertRaisesRegex(ValueError,'Loaded SSD core differs'):G.private_core(fixture(cached.replace(co_code=bytes(b))),{},helper)
    def test_bad_co_consts_rejected(self):
        with self.assertRaisesRegex(ValueError,'Loaded SSD core differs'):G.private_core(fixture(cached.replace(co_consts=cached.co_consts+('UNEXPECTED',))),{},helper)
    def test_isolated_AST_forbidden(self):
        node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==K]
        with self.assertRaisesRegex(ValueError,'Complete pinned module'):G._code_for(node,str(P))
    def test_private_name_collision_rejected(self):
        with self.assertRaisesRegex(ValueError,'name collision'):G.private_core(fixture(ns={G.PRIVATE_CUMSUM:helper}),{},helper)
    def test_single_call_AST_only_and_other_module_nodes_unchanged(self):
        original=copy.deepcopy(tree);modified=copy.deepcopy(tree)
        idx,=[i for i,n in enumerate(modified.body) if isinstance(n,ast.FunctionDef) and n.name==K]
        modified.body[idx]=G.rewrite_core_ast(modified.body[idx]);newcode=G._code_for(modified,str(P))
        actual=G.private_core(fixture(),{},helper)
        for key in F:self.assertEqual(getattr(newcode,key),getattr(actual.__code__,key),key)
        count=0
        class Undo(ast.NodeTransformer):
            def visit_Call(self,node):
                nonlocal count
                if isinstance(node.func,ast.Name) and node.func.id==G.PRIVATE_CUMSUM:
                    count+=1;return ast.parse(G.TARGET_EXPRESSION,mode='eval').body
                return self.generic_visit(node)
        restored=Undo().visit(modified);self.assertEqual(count,1)
        self.assertEqual(ast.dump(original,include_attributes=False),ast.dump(restored,include_attributes=False))
    def test_old_sources_and_new_pins_fixed(self):
        for rel,h in PINS.items():self.assertEqual(hashlib.sha256((M/rel).read_bytes()).hexdigest(),h,rel)
    def test_adapter_only_graph_pin_and_version_changes(self):
        old=(M/'adapters/eegmamba_v3.py').read_text(encoding='utf-8-sig');new=(M/'adapters/eegmamba_v4.py').read_text()
        expected=old.replace('mamba_private_backward_graph_v2 as G','mamba_private_backward_graph_v3 as G').replace(PINS['mamba_private_backward_graph_v2.py'],PINS['mamba_private_backward_graph_v3.py']).replace('adapter_version=3','adapter_version=4')
        self.assertEqual(expected,new)
suite=unittest.defaultTestLoader.loadTestsFromTestCase(Contracts);r=unittest.TextTestRunner(verbosity=1).run(suite)
report={'schema':'mamba_complete_module_compile_stdlib_contract_v1','python':sys.version,'passed':r.wasSuccessful(),'tests_run':r.testsRun,'failures':[str(t) for t,_ in r.failures],'errors':[str(t) for t,_ in r.errors],'reviewed_sources':[{'path':(M/rel).relative_to(W).as_posix(),'sha256':h} for rel,h in PINS.items()],'installed_source_sha256':hashlib.sha256(raw).hexdigest(),'actual_cache_sha256':hashlib.sha256(cb).hexdigest(),'test_source':{'path':Path(__file__).resolve().relative_to(W).as_posix(),'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},'executed_module_scope':'stdlib-only G3 helper; installed Mamba code objects never executed','model_constructions':0,'GPU_operations':0,'Tensor_operations':0,'process_controls':0,'actual_GPU_qualification':False}
out=A/('mamba_complete_module_compile_stdlib_contract_'+str(time.time_ns())+'.json')
with out.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'path':out.relative_to(W).as_posix(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'passed':r.wasSuccessful(),'tests_run':r.testsRun}));sys.exit(0 if r.wasSuccessful() else 1)
