"""Create a new standard-library source contract; does not execute runtimes."""
from pathlib import Path
import ast,re
A=Path(__file__).resolve().parent
helper=A/'prepare_core_prefix_repair_versions_v1.py'
tree=ast.parse(helper.read_text())
env={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('MAP','LABELS') for t in node.targets):
  exec(compile(ast.Module(body=[node],type_ignores=[]),'migration_map','exec'),env)
values={**env['MAP'],**env['LABELS']}
text=(A/'test_native_launcher_tree_chain_v1.py').read_text()
text=re.sub('|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True)),lambda m:values[m.group()],text)
text=text.replace("'execution_version':10","'execution_version':11").replace("'wrong_version',{'execution_version':9}","'wrong_version',{'execution_version':10}")
text=text.replace("'schema_version':12","'schema_version':13").replace("('schema_version',11)","('schema_version',12)")
text=text.replace("'checkpoint_audit_v12_'","'checkpoint_audit_v13_'").replace('completed_finetune_checkpoint_audit_v12_','completed_finetune_checkpoint_audit_v13_')
old="check('qualifier_numeric_builder_unchanged',lambda:eq(ast.dump(ast.parse(fsource(M/'qualify_deterministic_repairs_v8.py','isolated_builder'))),ast.dump(ast.parse(fsource(M/'qualify_deterministic_repairs_v6.py','isolated_builder')))))"
assert text.count(old)==1
new="""def normalized_builder(source):
    tree = ast.parse(source)
    mapping = next(n.value for n in tree.body[0].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='mapping' for t in n.targets))
    values = ast.literal_eval(mapping)
    eq(values,{'.adapters.codebrain_v2':'.adapters.codebrain_v3','.adapters.eegmamba':'.adapters.eegmamba_v3'})
    for key,value in zip(mapping.keys,mapping.values):
        if key.value=='.adapters.eegmamba': value.value='.adapters.eegmamba_v2'
    return ast.dump(tree)
check('builder_exact_only_Mamba_route_changed',lambda:eq(normalized_builder(fsource(M/'qualify_deterministic_repairs_v8.py','isolated_builder')),ast.dump(ast.parse(fsource(M/'qualify_deterministic_repairs_v7.py','isolated_builder')))))"""
text=text.replace(old,new)
text=text.replace("prep=read(A/'native_launcher_tree_source_preparation_1791781685869294300.json')","prep=read(sorted(A.glob('core_prefix_repair_source_preparation_*.json'))[-1])")
text=text.replace('previous127_review','previous139_review').replace('prior127','prior139').replace('exact127','exact139').replace('new13','new15')
text=text.replace('source_chain139','source_chain153').replace('(139,139)','(153,153)')
text=text.replace("'PREVIOUS_IDENTITY_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v6.py')","'PREVIOUS_IDENTITY_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v6.py'),'PREVIOUS_TREE_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v7.py')")
text=text.replace('for v in (4,5,6,7):','for v in (4,5,6,7,8):').replace('for v in (4,5,6):','for v in (4,5,6,7):')
text=text.replace("report={'schema':'native_launcher_tree_chain_stdlib_contract_v1'", "report={'schema':'core_prefix_repair_chain_stdlib_contract_v1'")
text=text.replace("target=A/('native_launcher_tree_chain_contracts_'","target=A/('core_prefix_repair_chain_contracts_'")
marker="report={'schema':'core_prefix_repair_chain_stdlib_contract_v1'"
extra="""# Actual preserved qualifier7 failure is a primary row once, with unknown internal tail.
failure=OUT/'deterministic_repair_qualification_v7/EEGMamba-lastblock-finetune/1791782206453969000/attempt.json'
failed=read(failure)
def normalize_actual(v):
    return {'observed_counters':{key:v[key] for key in ('forward_returned','backward_started','backward_returned','encoder_windows_returned','optimizer_step_returned')}}
actual_row=C['repair_worker_record'](desc(failure),failed,normalize_actual)
check('actual_Qual7_failed_primary_48_540251404',lambda:eq(actual_row['seconds_observed'],48.540251404))
check('actual_Qual7_unreturned_internal_work_unknown',lambda:eq(actual_row['unreturned_call_internal_work_unknown'],True))
check('actual_Qual7_observed_counts_preserved',lambda:eq(actual_row['observed_counters'],{'forward_returned':5,'backward_started':3,'backward_returned':2,'encoder_windows_returned':40,'optimizer_step_returned':0}))
check('actual_Qual7_failure_status_not_relabelled',lambda:eq(actual_row['status'],'failed'))
check('all_C2_G2_M3_bound_in_chain',lambda:require(all(desc(M/name) in chain for name in ('mamba_cumsum_cpu_fp32_v2.py','mamba_private_backward_graph_v2.py','adapters/eegmamba_v3.py')),'Mamba private sources missing'))
# Execute the actual nested D6 normalizer on source objects, never adapters.
nested=next(n for n in ast.walk(ast.parse(Dsource.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='builder_ast')
qmod=NS(__file__=str(qsource));oldqmod=NS(__file__=str(M/'qualify_deterministic_repairs_v5.py'))
ns={'ast':ast,'Path':Path,'require':require,'Q':qmod}
exec(compile(ast.Module(body=[nested],type_ignores=[]),'D6_nested_source_normalizer','exec'),ns)
check('actual_D6_Qual5_CB_builder_reuse_gate',lambda:eq(ns['builder_ast'](qmod),ns['builder_ast'](oldqmod)))

"""
assert text.count(marker)==1
text=text.replace(marker,extra+marker)
target=A/'test_core_prefix_repair_chain_v1.py'
compile(text,str(target),'exec')
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print(str(target))
