"""Independent minimal source/stdlib review; no actual postprocessing or model work."""
from pathlib import Path
import ast,copy,contextlib,hashlib,importlib.util,io,json,sys,time,types
SOURCE=Path(__file__).resolve();A=SOURCE.parent;WORK=SOURCE.parents[3];OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
checks=[]
def require(v,m):
 if not v:raise ValueError(m)
def ok(m,v=True):require(v,m);checks.append(m)
def reject(m,fn):
 try:fn()
 except (ValueError,KeyError,AssertionError):checks.append(m);return
 raise AssertionError('Unsafe acceptance: '+m)
def desc(p):
 p=Path(p).resolve();return {'path':p.relative_to(WORK).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def tree(p):return ast.parse(Path(p).read_text(encoding='utf-8-sig'))
def node(p,n):return next(x for x in tree(p).body if isinstance(x,ast.FunctionDef) and x.name==n)
def dump(n):return ast.dump(n,include_attributes=False)
def extract(p,names,ns):
 nodes=[copy.deepcopy(node(p,n)) for n in names]
 for n in nodes:n.decorator_list=[]
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(p),'exec'),ns);return ns

def main():
 files=[A/n for n in ('supplemental_head_resource_measurement_v7.py','finalize_full_plan_cost_v7.py','audit_full_plan_acceptance_v7.py')]
 expected=['3b85f4c444fff69711a1299ec260b0695f5896b8caff48a631fcc0ec28f36711','ab5d4d892f3d6a0e75c2c82db5bd3bbcfdffd086b4be4f92858234612982a712','ad8ddc3b50fd8c4537d5aaffe02451196762f27a6d3ffaa33fdf282fee65e795']
 sources=[desc(p) for p in files]
 ok('exact final three source hashes', [x['sha256'] for x in sources]==expected)
 proof_path=A/'final_postprocessing_v7_stdlib_contract_1791829328338536400.json';proof=read(proof_path)
 ok('author actual 36 contracts bind same final bytes',desc(proof_path)['sha256']=='6251217611e5069704d3be5a77a2297c81244e0524648bc4f0fc210e399550e1' and proof['passed'] is True and proof['check_count']==36 and proof['sources']==sources)
 registration=OUT/'continuation_queue_v14/registration.json';reg=read(registration)
 ok('all actual 194 sources preserved',len(reg['sources'])==194 and all(desc(WORK/x['path'])==x for x in reg['sources']) and proof['training_registration']==desc(registration))
 before=set(sys.modules);mods=[]
 for p in files:
  m=types.ModuleType('_independent_final7_'+p.stem);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);mods.append(m)
 res,fin,wide=mods
 ok('review imports remain without Torch',not any(x=='torch' or x.startswith('torch.') for x in set(sys.modules)-before))
 ok('resource delegate exact pinned and private method globals',res.measure.__globals__ is res.D.__dict__ and res.measure.__globals__['assert_terminal_still_bound'] is res.assert_terminal_still_bound and res.D.write_new is res.write_measurement and res.write_new is not res.D.write_new and desc(res.V2_SOURCE)['sha256']==res.V2_SHA)
 # Model-free mutation of a synthetic in-memory trace exercises the real write adapter.
 captured=[]
 ns=extract(files[0],['write_measurement'],{'delegated_source':lambda:{'path':'delegate','sha256':'hash'},'write_new':lambda p,v:captured.append(copy.deepcopy(v))})
 record={'subtype':'supplemental_head_resource_measurement','measurement_scope':{'seconds':'includes IO; excludes E2 prerequisite verification.'}}
 ns['write_measurement']('unused',record)
 ok('all persisted resource updates keep new provenance in same trace',record==captured[0] and record['resource_measurement_version']==7 and 'E11 prerequisite' in record['measurement_scope']['seconds'])
 # A new pause must stop before ownership/source/checkpoint access in each delegated boundary.
 calls=[]
 def pause():calls.append('pause');raise ValueError('paused')
 ns=extract(files[0],['assert_terminal_still_bound'],{'P':types.SimpleNamespace(Q=types.SimpleNamespace(check_pause=pause)),'require':require,'Path':Path,'OUT':OUT})
 reject('new pause rejects at first resource boundary',lambda:ns['assert_terminal_still_bound']({},OUT))
 ok('pause checked before all other resource boundary reads',calls==['pause'])
 # The review binding itself rejects changed source order, registration, and inherited inventory.
 review={'passed':True,'status':'final_postprocessing_sources_reviewed','reviewed_sources':sources,'training_registration':desc(registration),'reviewer_source':{'path':'reviewer','sha256':'r'}}
 binding={'path':'review','sha256':'x'};readmap={'review':review,str(registration):reg};checked_reads=[]
 def ck(v):checked_reads.append(v);return str(registration) if v==desc(registration) else v['path']
 ns=extract(files[0],['source_review'],{'require':require,'read':lambda p:readmap[p],'checked':ck,'descriptor':desc,'REPORT':A,'OUT':OUT})
 ok('exact ordered source review and real registration accepted',ns['source_review'](binding)==binding)
 review['reviewed_sources']=sources[::-1]
 reject('reordered source review rejected',lambda:ns['source_review'](binding));review['reviewed_sources']=sources
 review['training_registration']={'path':'other','sha256':'x'}
 reject('foreign training registration rejected',lambda:ns['source_review'](binding));review['training_registration']=desc(registration)
 old=reg['sources'];reg['sources']=old[:-1]
 reject('incomplete training source inventory rejected',lambda:ns['source_review'](binding));reg['sources']=old
 # CLI path resolution: no filesystem output, deeper work intercepted as pending.
 for p in files:
  seen=[]
  def cli_desc(path):seen.append(Path(path));return {'path':'canonical/review.json','sha256':'wanted'}
  dummy={'status':'pending','formal_fits_verified':0,'full_plan_completed':False,'requirements':[]}
  ns={'argparse':__import__('argparse'),'json':json,'__doc__':'test','WORK':WORK,'require':require,'descriptor':cli_desc,'desc':cli_desc,'S':types.SimpleNamespace(desc=cli_desc),'P':types.SimpleNamespace(inspect=lambda:dummy),'inspect_gate':lambda **kw:dummy,'preflight':lambda:dummy,'finalize':lambda:dummy}
  if p==files[2]:continue # wide output path is intentionally real-only; normalized descriptor AST checked below.
  extract(p,['main'],ns)
  with contextlib.redirect_stdout(io.StringIO()):ret=ns['main'](['--run','--source-review',str(A/'review.json'),'--source-review-sha256','wanted'])
  ok(p.name+' absolute CLI canonicalization precedes pending',seen==[A/'review.json'] and ret==2)
  reject(p.name+' wrong explicit review SHA rejected',lambda:ns['main'](['--run','--source-review',str(A/'review.json'),'--source-review-sha256','wrong']))
 wide_main=dump(node(files[2],'main'))
 ok('wide absolute CLI uses same canonical descriptor',"attr='desc'" in wide_main and "id='WORK'" in wide_main and 'Explicit source-review SHA differs' in wide_main)
 # Source-locked prefix stops before outer report but preserves complete lifecycle and file checks.
 post=res.POST_SOURCE;v=node(post,'verify_existing');edges=[i for i,n in enumerate(v.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='manifest' for t in n.targets)]
 prefix=ast.Module(body=v.body[:edges[0]],type_ignores=[])
 ok('pre-report lifecycle retains full byte gate and both exit0 stages',len(edges)==1 and all(x in dump(prefix) for x in ['assert_test_unlocked','validate_CPU_shape','returncode','mamba_linux','all_windows','released_owner.json']) and 'completed_report_verified' not in dump(prefix))
 # Isolate added CPU same-byte suffix using synthetic records; never open a checkpoint.
 n=copy.deepcopy(node(files[0],'lifecycle_evidence'));cut=next(i for i,x in enumerate(n.body) if isinstance(x,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='cpu' for z in x.targets));n.name='same_bytes';n.args=ast.arguments(posonlyargs=[],args=[ast.arg(arg='result')],kwonlyargs=[],kw_defaults=[],defaults=[]);n.body=n.body[cut:]
 metadata={'path':'meta','sha256':'m'};selected={'path':'best','sha256':'b'};row={'completion':'completion','status_artifact':'status','metadata':metadata,'cpu_checkpoint_audit':{'checkpoints':['initial','latest',selected]}};completion={'metadata':metadata,'checkpoint':selected};cpu={'completed_rows':[row]};mapping={'audit':cpu,'completion':completion,'status':{'evidence':completion}}
 seen=[]
 def checked(v):seen.append(v);return v
 ns={'read':lambda p:mapping[p],'checked':checked,'require':require};exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),str(files[0]),'exec'),ns)
 ok('CPU same-byte suffix checks each exact checkpoint',ns['same_bytes']({'CPU_audit':'audit'})['CPU_checkpoint_same_bytes_verified']==405 and seen[-3:]==['initial','latest',selected])
 def tampered(v):
  if v==selected:raise ValueError('changed bytes')
  return v
 ns['checked']=tampered;reject('changed selected checkpoint bytes rejected before resource start',lambda:ns['same_bytes']({'CPU_audit':'audit'}))
 # Original 30 evaluator, retention, exported metrics, and cost semantics remain; only bound providers change.
 original=A/'audit_full_plan_acceptance_v4.py'
 ok('all thirty original requirement engine delegated unchanged',wide.D.audit.__code__.co_filename==str(original) and wide.D.audit.__globals__['check_fit'] is wide.check_fit and wide.D.audit.__globals__['deep_checkpoint'] is wide.deep_checkpoint)
 oldeval=node(original,'evaluation_audit');astgate=node(files[2],'bind_original_consumers')
 ok('exact scoped original evaluation and cost substitution guards',"changed == 1" in ast.unparse(astgate) and "counts.values()" in ast.unparse(astgate) and 'evaluate_v11' in ast.unparse(astgate) and 'cost_aggregation_v17.py' in ast.unparse(astgate))
 ok('FT deep audit reuses actual same-byte CPU16 and nonFT keeps original',wide._original_deep.__code__.co_filename==str(original) and all(x in dump(node(files[2],'deep_checkpoint')) for x in ['_original_deep','f2_physical_durable_boundary','checkpoints','only_map_location_cpu','new_checkpoint_loads']))
 ok('strict finalizer three losses preserve unknown and no double counting',all(x in dump(node(files[1],'bind_lost_f3_recovery_cost')) for x in ['original_recovery_closure','previous_recovery_closure','recovery_closure','unflushed_tail_unknown','final_elapsed_seconds','extra_work_not_added_again','supplemental_checkpoint_boundary_already_in_formal_attempt']))
 ok('cost collectors and all qualifier plus diagnosis families explicit',fin.COST_SHA256==desc(A/'cost_aggregation_v17.py')['sha256'] and all('repair_qualification_'+str(v) in fin.PHASE_PATTERNS for v in range(4,11)) and 'mamba_repeat_diagnosis' in fin.PHASE_PATTERNS)
 ok('resource pinned before import',next(n.lineno for n in tree(files[1]).body if isinstance(n,ast.If) and 'RESOURCE_RUNNER.read_bytes' in ast.unparse(n))<next(n.lineno for n in tree(files[1]).body if isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id=='_spec' for z in n.targets)))
 ok('later separate delivery gates not represented as passed', 'separate_G3_pause_standby_delivery_gates_still_required=True' in ast.unparse(node(files[2],'audit')))
 result={'status':'final_postprocessing_sources_reviewed','passed':True,'reviewer_source':desc(SOURCE),'reviewed_sources':sources,'training_registration':desc(registration),'author_contract':desc(proof_path),'checks':checks,'check_count':len(checks),'scope':'Independent source and standard-library fake review of three postprocessing entrypoints only; not actual resource measurement, final cost, semantic review, or final acceptance. Existing 194 training sources unchanged.','findings':[],'fixed_before_review':['resource source hash validated before import','absolute/relative source-review CLI paths canonicalized','same-byte CPU16 405-checkpoint evidence validated before resource probes','new user pause checked at every resource boundary'],'limitations':['Actual all360 / CPU135405 / E11 completion still required.','Twelve resource probes and actual final cost/30 requirements not executed by this review.','G3, pause-memory, system-standby delivery gates remain independently required.','Prior CPU16 Tensor proofs reused only with current metadata/trace/checkpoint byte evidence; no new FT Tensor loads implied.'],'operations':{'source_and_existing_JSON_reads':True,'checkpoint_byte_reads':0,'Tensor_loads':0,'model_calls':0,'GPU_queries':0,'native_process_queries':0,'test_probability_reads':0,'training_source_edits':0,'actual_postprocessing_calls':0},'all_training_source_bytes_unchanged':True}
 stamp=str(time.time_ns());target=A/('independent_final_postprocessing_v7_source_review_'+stamp+'.json')
 with target.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'review':desc(target),'checks':len(checks),'passed':True},ensure_ascii=True))
if __name__=='__main__':main()
