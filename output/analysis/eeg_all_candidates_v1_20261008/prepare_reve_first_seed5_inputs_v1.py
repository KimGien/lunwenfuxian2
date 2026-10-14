"""Observe completed seed11 JSON inputs while waiting for genuine CPU35 proof."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

SOURCE=Path(__file__).resolve()
HELPER=SOURCE.with_name('review_reve_first_seed5_v1.py')
if hashlib.sha256(HELPER.read_bytes()).hexdigest() != 'b5b32889a76bef07b96d43d347f48e6b6a705a768f187c42a56d5fab5a9cfdcc':
    raise ValueError('Prepared review helper changed')
spec=importlib.util.spec_from_file_location('_reve_firstseed_input_helper',HELPER)
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


def main():
    reader=M.Reader();registry,binding,fits,statuses=M.gate(reader)
    M.require(all(row['status']=='training_complete' for row in statuses),'Actual five completions still pending')
    formal,engineering=M.inventory(reader)
    rows=[];totals=Counter();seconds=0.;aliases=[];seen=set()
    for fit in fits:
        folder=M.OUT/'runs'/fit['fit_id']
        complete,completion=reader.snapshot(folder/'complete.json')
        metadata=reader.checked(complete['metadata'])
        history=reader.checked(complete['train_history']);validation=reader.checked(complete['validation_history'])
        M.require(len(history)==len(validation)==metadata['epochs'],'Actual JSON curves incomplete')
        attempts=[]
        for data,desc in formal[fit['fit_id']]:
            if desc['sha256'] in seen:
                aliases.append(desc);continue
            seen.add(desc['sha256'])
            counters={k+s:data[k+s] for k in M.H.COUNTERS for s in ('_started','_returned')}
            totals.update(counters);seconds+=data['seconds']
            attempts.append({'artifact':desc,'status':data['status'],'observer_version':data['implementation']['version'],
                'seconds':data['seconds'],'counts':counters,'zero_call_recheck':not any(counters.values())})
        rows.append({'fit_id':fit['fit_id'],'completion':completion,'metadata':complete['metadata'],
            'train_history':complete['train_history'],'validation_history':complete['validation_history'],
            'condition_source_lock':metadata['condition_source_lock'],'execution_acceleration':metadata['execution_acceleration'],
            'epochs':metadata['epochs'],'best_epoch':metadata['best_epoch'],'stop':metadata['stop_reason'],
            'extended':metadata['extended'],'selected_phase':metadata['selected_checkpoint_phase'],
            'durable_updates':metadata['optimizer_updates'],'head_group_updates':metadata['head_optimizer_updates'],
            'backbone_group_updates':metadata['backbone_optimizer_updates'],'durable_backwards':metadata['backward_calls'],
            'nested_metadata_seconds_not_added':metadata['seconds_this_attempt'],'attempts':attempts})
    value={'schema':'REVE_first_seed5_actual_JSON_inputs_pending_CPU_v1','status':'pending_actual_CPU35_proof','passed':False,
        'source':reader.snapshot(SOURCE)[1],'review_helper':reader.snapshot(HELPER)[1],'registration':binding,
        'actual_CPU_report_read':False,'real_completed_statuses_observed':5,'stage_passed_claimed':False,
        'rows':rows,'epochs_observed':[r['epochs'] for r in rows],'best_epochs_observed':[r['best_epoch'] for r in rows],
        'durable_updates_observed':sum(r['durable_updates'] for r in rows),'durable_backwards_observed':sum(r['durable_backwards'] for r in rows),
        'physical_recorded_subtotals':dict(totals),'unique_formal_attempts':len(seen),'content_aliases_not_added':aliases,
        'recorded_formal_attempt_seconds':seconds,'nested_metadata_seconds_not_added':sum(r['nested_metadata_seconds_not_added'] for r in rows),
        'engineering_separate':[{'artifact':desc,'seconds':data['seconds'],'backward_returned':data['backward_returned'],
            'optimizer_step_returned':data['optimizer_step_returned']} for data,desc in engineering],
        'all_historical_exact_costs':None,'operations':{'JSON_python_inputs_only':True,'binary_tensor_model_GPU_process_raw_test':0},
        'full_REVE15_or_FT135_or_plan360_completion_claimed':False,'input_bindings':list(reader.inputs.values())}
    reader.recheck()
    target=SOURCE.parent/('REVE_first_seed5_actual_inputs_pending_CPU_'+str(time.time_ns())+'.json')
    with target.open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'artifact':reader.snapshot(target)[1],'status':value['status'],'epochs':value['epochs_observed'],
        'updates':value['durable_updates_observed'],'backwards':value['durable_backwards_observed']}))


if __name__=='__main__':main()
