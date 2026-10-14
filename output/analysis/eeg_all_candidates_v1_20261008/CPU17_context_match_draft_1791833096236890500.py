"""CPU17: strict original-Linux LR expectations for actual Mamba F12 only.
The private CPU16 delegate retains all three CPU checkpoint checks for all135.
Default is preparation only; no original training/source/history is changed.
"""
from __future__ import annotations
import argparse, ast, copy, importlib.util, json, math, sys, time
from pathlib import Path
SOURCE=Path(__file__).resolve()
WORK=SOURCE.parents[3]
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S
DELEGATE=SOURCE.with_name('audit_completed_finetune_checkpoints_v16.py')
DELEGATE_SHA='108c7cb87fa2e8e64e7882fa0365fa431c55e2e67a34cfbe95310ee8386fa9e3'
MAMBA='EEGMamba-lastblock-finetune'
LINUX_REFERENCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/actual_platform_learning_rate_reference_linux_1791832760707973329.json', 'sha256': 'c7109d9ced13c853ecb6de99606b98ab1c2f2b9978535a0a2c614049fb682463'}
WINDOWS_REFERENCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/actual_platform_learning_rate_reference_win32_1791832754389436100.json', 'sha256': '0ce44e1bbfb87d9e0d3b300e4960bdcafdc726b851b8c76b473d90f8e8ce8b1f'}
CAPTURE_SOURCE={'path': 'output/analysis/eeg_all_candidates_v1_20261008/capture_actual_platform_lr_reference_v1.py', 'sha256': '98b87c81db48b8089d910dde39d338b0c1d6117f473d948dd85fc4c1e5ec05a6'}
CHILD_COMPLETE={'path': 'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008/continuation_queue_v14/attempts/f_1791825911463169900_7c008897/status.json', 'sha256': 'd6178fbe5c5206baf8a1da21f1616fff99cb5bf9d6f34edc28e1778ad14ff980'}

def require(ok,message):
    if not ok: raise ValueError(message)

def reference_verified():
    linux=S.read(S.checked(LINUX_REFERENCE)); windows=S.read(S.checked(WINDOWS_REFERENCE))
    require(linux['schema']=='actual_platform_learning_rate_reference_v1' and linux['source']==CAPTURE_SOURCE and windows['source']==CAPTURE_SOURCE,'Exact platform capture source required')
    S.checked(CAPTURE_SOURCE)
    for key in ('protocol','Q14_registration','Q14_failed_terminal','failed_CPU16'):
        require(windows[key]==linux[key], 'Platform captures must bind the same original inputs')
        S.checked(linux[key])
    require(linux['runtime']['platform']=='linux' and linux['runtime']['python'].startswith('3.12.3 ') and linux['runtime']['executable'].endswith('/assets/eegmamba_linux_env/bin/python'),'Actual original Linux runtime required')
    require(windows['runtime']['platform']=='win32','Actual Windows comparison required')
    require(linux['all_15_histories_strict_equal_to_this_platform'] is True and linux['differences_total']==0 and linux['zero_tolerance'] is True,'Independent Linux table must match all actual histories exactly')
    require(windows['differences_total']==15 and windows['all_15_histories_strict_equal_to_this_platform'] is False,'Historical cross-platform discrepancy proof required')
    table=linux['reference_table']
    require(len(table)==200 and [x['epoch'] for x in table]==list(range(1,201)),'Whole original 200 epoch table required')
    for row in table:
        for group in ('head','backbone'):
            for suffix in ('multiplier','lr'):
                key=group+'_'+suffix; value=row[key]
                require(type(value) is float and math.isfinite(value) and value.hex()==row[key+'_hex'] and float.fromhex(row[key+'_hex'])==value,'Exact finite float/hex reference required')
    expected={f'{MAMBA}/seed_{seed}/fold_{fold}' for seed in (11,22,33) for fold in range(1,6)}
    require(len(linux['fits'])==15 and {x['fit_id'] for x in linux['fits']}==expected,'Exact original Mamba15 reference scope required')
    require([x['history'] for x in windows['fits']]==[x['history'] for x in linux['fits']],'Both platforms must compare the same history bytes')
    child=S.read(S.checked(CHILD_COMPLETE))
    require(child['status']=='complete' and type(child['returncode']) is int and child['returncode']==0 and child['condition']==MAMBA,'Actual Mamba child exit0 required')
    require(child['linux_worker_identity']['identity']['argv'][0]==linux['runtime']['executable'] and child['linux_worker_identity']['identity']['executable']=='/usr/bin/python3.12','Reference must use the actual original training interpreter')
    require('tridep_ccml.eeg_all_candidates.repair_owned_worker_v9' in child['linux_worker_identity']['identity']['argv'],'Actual original F12 worker provenance required')
    queue=S.read(S.checked(linux['Q14_registration']))
    require(len(queue['sources'])==194 and linux['protocol'] in queue['sources'],'Original194 protocol binding required')
    for item in linux['fits']:
        require(item['strict_equal'] is True and item['differences']==[],'Reference history comparison failed')
        status=S.read(S.checked(item['status'])); S.checked(item['history'])
        require(status['fit_id']==item['fit_id'] and status['status']=='training_complete','Reference completed fit identity changed')
        metadata=S.read(S.checked(status['evidence']['metadata']))
        actual=metadata['actual_deterministic_repair_execution']
        require(actual['observer_version']==actual['version']==12 and actual['condition']==MAMBA and actual['source'] in queue['sources'] and actual['source']['path'].endswith('/finetune_v12.py'),'Linux reference permitted only for actual registered F12 Mamba')
        require(actual['registration']==child['repair_registration'],'Mamba reference belongs to another execution')
    return linux

# Only these original expected-value expressions change. Non-Mamba else AST
# remains exactly the old multiplication and strict == statements remain intact.
EXPRESSIONS=(
 ('check_fit', 'P.POLICIES[policy].head_lr * P.learning_rate_multiplier(policy, epoch)', 'epoch', "'head'"),
 ('check_fit', 'P.POLICIES[policy].backbone_lr * P.learning_rate_multiplier(policy, epoch, parameter_group="backbone")', 'epoch', "'backbone'"),
 ('deep_checkpoint', 'base_lr * P.learning_rate_multiplier(policy, row["epochs"], parameter_group=name)', 'row["epochs"]', 'name'),
)

def patch_wide(wide, reference):
    table={row['epoch']:row for row in reference['reference_table']}
    fit_ids={x['fit_id'] for x in reference['fits']}
    def linux_expected(fit,policy,epoch,group):
        require(fit['condition']==MAMBA and fit['fit_id'] in fit_ids and policy=='finetune','Linux LR scope must be exact actual Mamba F12')
        require(type(epoch) is int and epoch in table and group in ('head','backbone'),'Original LR epoch/group required')
        return table[epoch][group+'_lr']
    wide.__dict__['_CPU17_linux_expected']=linux_expected
    original=ast.parse(Path(wide.__file__).read_text(encoding='utf-8'))
    names={x[0] for x in EXPRESSIONS}
    for name in names:
        function=copy.deepcopy(next(x for x in original.body if isinstance(x,ast.FunctionDef) and x.name==name))
        specifications=[x for x in EXPRESSIONS if x[0]==name]
        counts=[0]*len(specifications)
        class Replace(ast.NodeTransformer):
            def visit_BinOp(self,node):
                for i,(_,expression,epoch,group) in enumerate(specifications):
                    if ast.dump(node,include_attributes=False)==ast.dump(ast.parse(expression,mode='eval').body,include_attributes=False):
                        counts[i]+=1
                        replacement=ast.parse(f"_CPU17_linux_expected(fit, policy, {epoch}, {group}) if fit['condition']=={MAMBA!r} else 0",mode='eval').body
                        replacement.orelse=copy.deepcopy(node)
                        return ast.copy_location(replacement,node)
                return self.generic_visit(node)
        changed=Replace().visit(function)
        require(counts==[1]*len(specifications),'Exact three old LR expected expressions required')
        module=ast.fix_missing_locations(ast.Module(body=[changed],type_ignores=[]))
        exec(compile(module,wide.__file__,'exec'),wide.__dict__)
    return wide

def load_delegate():
    require(S.digest(DELEGATE)==DELEGATE_SHA,'Original CPU16 source changed')
    spec=importlib.util.spec_from_file_location('_CPU17_private_original_CPU16',DELEGATE)
    delegate=importlib.util.module_from_spec(spec);spec.loader.exec_module(delegate)
    base=delegate.B; base.SOURCE=SOURCE
    original_load=base.load_wide; original_sources=base.delegated_sources
    def load(): return patch_wide(original_load(),reference_verified())
    def sources():
        require(S.digest(DELEGATE)==DELEGATE_SHA,'Original CPU16 source changed')
        return {**original_sources(),'previous_failed_CPU16_source':S.desc(DELEGATE),'actual_CPU17':S.desc(SOURCE),'original_Linux_LR_reference':LINUX_REFERENCE,'actual_Windows_LR_discrepancy':WINDOWS_REFERENCE,'platform_capture_source':CAPTURE_SOURCE}
    base.load_wide=load; base.delegated_sources=sources
    return base

def audit():
    before=S.desc(SOURCE);reference_verified();base=load_delegate();result=base.audit()
    require(result['schema_version']==5 and result['auditor_source']==before and S.desc(SOURCE)==before,'Exact inherited CPU audit identity required')
    result.update(schema_version=17,auditor_source=before,delegated_CPU16_source=S.desc(DELEGATE),strict_Mamba_original_Linux_LR_reference=LINUX_REFERENCE,Windows_cross_platform_discrepancy=WINDOWS_REFERENCE,learning_rate_tolerance_used=False,three_expected_LR_expressions_only=True,other_checks_unchanged=True,failed_CPU16_preserved_not_relabelled=True)
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true');args=parser.parse_args(argv)
    if not args.run:
        print(json.dumps({'status':'preparation_only','Tensor_loads':0,'model_calls':0,'GPU_calls':0,'writes':0}));return 0
    result=audit(); target=S.REPORT/('completed_finetune_checkpoint_audit_v17_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as stream:
        json.dump(result,stream,indent=2,ensure_ascii=False,allow_nan=False);stream.write('\n')
    print(json.dumps({'artifact':S.desc(target),'status':result['status'],'verified':result['verified_completed_finetune_fits'],'checkpoint_loads':result['checkpoint_loads'],'errors':result['errors']}))
    return 1 if result['errors'] else 0
if __name__=='__main__': raise SystemExit(main())
