"""Review the complete currently prepared repair99 source chain, not GPU work.

This is the source-review input only. Q6 still requires actual GPU reports,
their independent reviews, old330 completion and released owners before it can
register. No qualification, registration, checkpoint or probability is produced.
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time
SOURCE=Path(__file__).resolve();WORK=SOURCE.parents[3]
sys.path.insert(0,str(WORK))
from tridep_ccml.eeg_all_candidates import storage as S,continuation_queue_v6 as Q6,deterministic_repair_execution_v1 as D

def require(ok,message):
    if not ok:raise ValueError(message)

def inspect():
    records=Q6.sources();checks={}
    def check(name,value):
        require(value,'Complete source review failed: '+name);checks[name]=True
    prior=S.read(S.OUT/'continuation_queue_v5/recovery_registration_v2.json')['sources']
    check('all_original84_descriptors_inherited',len(prior)==84 and all(x in records for x in prior))
    check('all_source_records_unique',len(records)==len({x['path'] for x in records}))
    for item in records:
        path=S.checked(item)
        if path.suffix=='.py':
            raw=path.read_text(encoding='utf-8-sig');compile(raw,str(path),'exec')
    check('all_bound_python_sources_compile',True)
    by_name={Path(x['path']).name:x for x in records}
    for name in ('finetune_v4.py','continuation_queue_v6.py','evaluate_v4.py','repair_observation_v1.py',
                 'repair_runtime_owner_v1.py','repair_owned_worker_v1.py','deterministic_repair_execution_v1.py',
                 'audit_completed_finetune_checkpoints_v7.py','cost_aggregation_v8.py'):
        check('actual_source_bound:'+name,name in by_name)
    def text(name):return S.checked(by_name[name]).read_text(encoding='utf-8')
    def function(name,fn):
        raw=text(name);tree=ast.parse(raw);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn)
        return ast.get_source_segment(raw,node)
    check('registration_requires_actual_qualifications',
          'D.qualification(' in function('continuation_queue_v6.py','register')
          and 'review_engineering(' in function('continuation_queue_v6.py','register'))
    check('register_actual_original330_and_blocked30_only',
          'original_others_complete()' in function('continuation_queue_v6.py','register')
          and "status.json')['status']=='blocked'" in function('continuation_queue_v6.py','register'))
    check('registration_exclusive_not_before_review',
          function('continuation_queue_v6.py','register').index("new_repair_chain_sources_reviewed")<
          function('continuation_queue_v6.py','register').index("registration_intent.json',"))
    check('full_binding_rechecked_before_CPU7',
          "S.read(ROOT/'registration.json')['sources']" in function('continuation_queue_v6.py','cpu_gate'))
    check('actual_CPU7_explicit_run',
          "str(path),'--run']" in function('continuation_queue_v6.py','cpu_gate'))
    check('true4_observer_before_first_flush',
          "self.record['implementation']=provenance(" in text('finetune_v4.py')
          and "'version':4" in function('finetune_v4.py','provenance'))
    check('exact_old_F1_training_call',
          'v2._FIT(condition,seed,fold,resume=resume,micro_subjects=1,window_chunk=8)' in function('finetune_v4.py','fit'))
    check('captured_original_builder_before_runtime_wrapper',
          function('finetune_v4.py','activate').index('v2._BUILD')<
          function('finetune_v4.py','activate').index('with TEMPLATE.activate('))
    check('owned_worker_pre_tensor_live_guard',
          function('repair_owned_worker_v1.py','run').index('O.enter(')<
          function('repair_owned_worker_v1.py','run').index("importlib.import_module('.finetune_v4'"))
    check('CPU7_true4_branch_and_unchanged_old_rows',
          'old_boundary(fit,metadata,inputs)' in text('audit_completed_finetune_checkpoints_v7.py')
          and 'V.trace_boundary(base.legacy_trace_boundary' in text('audit_completed_finetune_checkpoints_v7.py'))
    check('native_inference_original_metric_pipeline',
          'previous_materialize(*args,**kwargs)' in function('evaluate_v4.py','isolated_driver')
          and 'cpu.get(\'checkpoint_loads\')==405' in function('evaluate_v4.py','validate_gate_data'))
    check('evaluation_and_training_never_overlap',
          "continuation_queue_v6/owner.json" in function('evaluate_v4.py','verify_gate')
          and 'process.wait()' in function('evaluate_v4.py','controller'))
    check('no_actual_gradient_equivalence_or_speed_claim',
          'original_CUDA_backward_equivalence_established' in text('finetune_v4.py')
          and 'performance_improvement_claimed' in text('finetune_v4.py'))
    check('old_loss_closures_once_and_nested_parent_not_primary',
          function('cost_aggregation_v8.py','collect').count('previous.augment(')==1
          and "x['aggregation_role']=='primary_observed_invocation'" in function('cost_aggregation_v8.py','category_totals'))
    check('no_unknown_gpu_closure_or_elapsed_fabrication',
          'all_plan_wall_seconds_exact=None' in function('cost_aggregation_v8.py','collect')
          and "current.get('child_pid') is None" in function('continuation_queue_v6.py','run'))
    return {'status':'new_repair_chain_sources_reviewed','passed':True,'reviewer_source':S.desc(SOURCE),
            'reviewed_sources':records,'source_count':len(records),'checks':checks,
            'scope':'Current source syntax/provenance/control and declared delegation only; actual qualification separately required.',
            'actual_GPU_qualification_complete':False,'actual_new_formal_registration_created':False,
            'model_calls':0,'GPU_queries':0,'checkpoint_tensor_loads':0,'outer_probability_reads':0,
            'source_byte_hashing_performed':True,'actual_total_plan_complete':False,
            'limitations':['Real Windows/WSL model/gradient/owner/drain behavior is not proved by source inspection.',
                           'Q6 requires both real engineering reports and independent real-report review.',
                           'Full postprocessing/final/companion compatibility and actual final semantic acceptance remain pending.']}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run:
        print(json.dumps({'status':'preparation_only','model_calls':0,'GPU_queries':0,'actual_source_review_started':False}));return
    value=inspect()
    target=SOURCE.parent/('independent_complete_repair_chain_source_review_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps(S.desc(target)))

if __name__=='__main__':main()
