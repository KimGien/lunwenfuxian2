"""Create stdlib/AST-only contracts for the next unregistered generation."""
from pathlib import Path
import ast,re
A=Path(__file__).resolve().parent
tree=ast.parse((A/'prepare_core_code_compat_versions_v1.py').read_text())
env={}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('MAP','LABELS') for t in node.targets):exec(compile(ast.Module(body=[node],type_ignores=[]),'migration_map','exec'),env)
values={**env['MAP'],**env['LABELS']}
text=(A/'test_core_prefix_repair_chain_v1.py').read_text()
text=re.sub('|'.join(re.escape(x) for x in sorted(values,key=len,reverse=True)),lambda m:values[m.group()],text)
text=text.replace("'execution_version':11","'execution_version':12").replace("'wrong_version',{'execution_version':10}","'wrong_version',{'execution_version':11}")
text=text.replace("'schema_version':13","'schema_version':14").replace("('schema_version',12)","('schema_version',13)")
text=text.replace('completed_finetune_checkpoint_audit_v13_','completed_finetune_checkpoint_audit_v14_')
text=text.replace("'.adapters.eegmamba_v3'","'.adapters.eegmamba_v4'")
text=text.replace("core_prefix_repair_source_preparation_*.json","core_code_compat_source_preparation_*.json")
text=text.replace('previous139_review','previous153_review').replace('prior139','prior153').replace('exact139','exact153').replace('new15','new14')
text=text.replace('source_chain153','source_chain166').replace('(153,153)','(166,166)')
text=text.replace("'PREVIOUS_TREE_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v7.py')","'PREVIOUS_TREE_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v7.py'),'PREVIOUS_CORE_Q':NS(SOURCE=M/'qualify_deterministic_repairs_v8.py')")
text=text.replace('for v in (4,5,6,7,8):','for v in (4,5,6,7,8,9):').replace('for v in (4,5,6,7):','for v in (4,5,6,7,8):')
text=text.replace("'mamba_private_backward_graph_v2.py','adapters/eegmamba_v3.py'","'mamba_private_backward_graph_v3.py','adapters/eegmamba_v4.py'")
text=text.replace("report={'schema':'core_prefix_repair_chain_stdlib_contract_v1'","report={'schema':'core_code_compat_chain_stdlib_contract_v1'")
text=text.replace("target=A/('core_prefix_repair_chain_contracts_'","target=A/('core_code_compat_chain_contracts_'")
marker="report={'schema':'core_code_compat_chain_stdlib_contract_v1'"
extra="""# The actual Qual8 constructor failure is retained as positive elapsed cost,
# although it returned no forward/backward call; no zero-model claim is made.
failure8=OUT/'deterministic_repair_qualification_v8/EEGMamba-lastblock-finetune/1791783245785244500/attempt.json'
failed8=read(failure8)
actual8=C['repair_worker_record'](desc(failure8),failed8,normalize_actual)
check('actual_Qual8_failed_primary_6_125486071',lambda:eq(actual8['seconds_observed'],6.125486071))
check('actual_Qual8_counts_zero_but_model_constructor_ran',lambda:eq(actual8['observed_counters'],{'forward_returned':0,'backward_started':0,'backward_returned':0,'encoder_windows_returned':0,'optimizer_step_returned':0}))
check('actual_Qual8_original_status_and_failure_preserved',lambda:require(actual8['status']=='failed' and 'Loaded SSD core differs from pinned source function' in failed8['failure'],'Constructor failure changed'))

"""
assert text.count(marker)==1
text=text.replace(marker,extra+marker)
target=A/'test_core_code_compat_chain_v1.py'
compile(text,str(target),'exec')
with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
print(str(target))
