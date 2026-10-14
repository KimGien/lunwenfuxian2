"""Final paired-data, event-scope, cost and artifact delivery gate. Zero fitting."""
from __future__ import annotations
import argparse
import base64
import csv
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET
from . import common as C

SOURCE=Path(__file__).resolve()
CONTROL=C.OUT/'delivery_v3'
KEYS=('macro_f1','balanced_accuracy','mdd_recall','auc','nll','brier')

def prepare():
    earliest=min(p.stat().st_birthtime for p in C.CODE.rglob('*.py'))
    value={'schema':'fusion_delivery_v3','source':C.desc(SOURCE),'scope_start':datetime.fromtimestamp(earliest,timezone.utc).isoformat(),
      'scope_start_basis':'earliest owned implementation/vendor source creation; conservative envelope includes engineering',
      'requires_core_audit':'final_audit_v1/complete.json','requires_models':1200,
      'power_event_sources':'all System events in fixed envelope, full raw XML, native provider event descriptions',
      'relevant_event_candidates':{'Microsoft-Windows-Kernel-Power':[41,42,107,506,507,566],
                                 'Microsoft-Windows-Power-Troubleshooter':[1],
                                 'Microsoft-Windows-Kernel-General':[12,13],
                                 'Microsoft-Windows-Eventlog':[104]},
      'event_influence':'conservative sequential fit envelopes; no event duration subtraction from perf_counter',
      'cost_unknowns':'retain null; no fabricated active compute total or complete engineering/failure cost',
      'screen_off_is_not_proof_of_no_compute':True,
      'documentation':['https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/modern-standby-states',
                       'https://learn.microsoft.com/en-us/windows/win32/power/system-power-states'],
      'unknown_events_or_source_gaps':'preserve evidence and require narrow review; do not issue completion',
      'outer_data_not_read_until_core_audit':True,
      'resource_observations':'exact-birth-bound actual process snapshots; process peak-so-far; whole-GPU diagnostic distinct from exclusive allocations',
      'transfer_archive_counters':'source-derived successful fit calls; explicitly distinct from measured OS I/O counters',
      'evaluation_recovery':'bind original validation-only failure and reviewed evaluation-only owner; no formal training recovery',
      'cached_evaluation_provenance':'17 actual successful v1 evaluations reused; remaining v2; full source disclosure'}
    C.save(CONTROL/'protocol.json',value,immutable=True)
    return value

def read_gate():
    protocol=prepare();C.check_pause()
    C.require(not (C.OUT/'control/owner.json').exists() and not (C.OUT/'control/queue.lock').exists(),'Queue has not released')
    active=[r for r in C.inventory() if 'fusion_eeg_grid_v2.run' in r['command'] and '--run' in r['command']]
    C.require(not active,'Native training process remains')
    audit=C.read(C.OUT/'final_audit_v1/complete.json')
    C.require(audit['passed'] and audit['models']==1200 and audit['conditions']==80 and
              audit['training_and_selection_reconstructed'] and audit['independent_metrics_recomputed'],
              'Independent core audit must truly pass')
    for key in ('source','protocol','results','report','cost'):C.checked(audit[key])
    return protocol,audit

def capture_events(protocol,end):
    path=CONTROL/'system_events.json'
    if path.exists():
        value=C.read(path);C.require(value['scope_start']==protocol['scope_start'] and value['scope_end']==end,
                                    'Existing event capture has different scope')
        return value
    start=protocol['scope_start']
    # Only generated ISO dates are interpolated into fixed PowerShell code.
    C.require(all(ch in '0123456789-:.+TZ' for ch in start+end),'Unexpected date syntax')
    script="""$ErrorActionPreference='Stop'; [Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false);
$taskStart=[DateTimeOffset]::Parse('START').LocalDateTime; $taskEnd=[DateTimeOffset]::Parse('END').LocalDateTime;
$taskOldest=Get-WinEvent -LogName System -Oldest -MaxEvents 1;
$taskNewest=Get-WinEvent -LogName System -MaxEvents 1;
$taskRows=@(); try { $taskEvents=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$taskStart;EndTime=$taskEnd} -ErrorAction Stop) } catch { if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*') { $taskEvents=@() } else { throw } };
foreach($taskEvent in $taskEvents) { $taskRows += [pscustomobject]@{record_id=$taskEvent.RecordId;provider=$taskEvent.ProviderName;id=$taskEvent.Id;time=$taskEvent.TimeCreated.ToUniversalTime().ToString('o');xml=$taskEvent.ToXml()} };
$taskDefinitions=@(); foreach($taskName in @('Microsoft-Windows-Kernel-Power','Microsoft-Windows-Power-Troubleshooter','Microsoft-Windows-Kernel-General')) { $taskProvider=Get-WinEvent -ListProvider $taskName; foreach($taskDefinition in $taskProvider.Events) { if($taskDefinition.Id -in @(1,12,13,41,42,107,506,507,566)) { $taskDefinitions += [pscustomobject]@{provider=$taskName;id=$taskDefinition.Id;version=$taskDefinition.Version;description=$taskDefinition.Description} } } };
[pscustomobject]@{scope_start='START';scope_end='END';oldest_system_event_time=$taskOldest.TimeCreated.ToUniversalTime().ToString('o');oldest_record_id=$taskOldest.RecordId;newest_record_id=$taskNewest.RecordId;events=$taskRows;provider_definitions=$taskDefinitions;query_complete=$true;no_max_events_limit=$true} | ConvertTo-Json -Depth 8 -Compress""".replace('START',start).replace('END',end)
    result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-EncodedCommand',
       base64.b64encode(script.encode('utf-16le')).decode()],capture_output=True,text=True,encoding='utf-8',
       check=True,timeout=60)
    value=json.loads(result.stdout.lstrip('\ufeff'));C.save(path,value,immutable=True)
    return value

def power_scope(events,protocol):
    C.require(events['query_complete'] and events['no_max_events_limit'],'Incomplete System event capture')
    dt=lambda text:datetime.fromisoformat(text.replace('Z','+00:00'))
    C.require(dt(events['oldest_system_event_time'])<=dt(protocol['scope_start']),'System log coverage starts after experiment scope')
    relevant=[r for r in events['events'] if r['id'] in protocol['relevant_event_candidates'].get(r['provider'],[])]
    clear=[r for r in relevant if r['provider']=='Microsoft-Windows-Eventlog' and r['id']==104]
    C.require(not clear,'System event log was cleared inside experiment envelope')
    rows=sorted(relevant,key=lambda r:(r['time'],r['record_id']))
    windows=[];pending={};unresolved=[]
    for event in rows:
        provider,eid=event['provider'],event['id']
        if provider=='Microsoft-Windows-Kernel-Power' and eid in (42,506):
            kind='sleep' if eid==42 else 'connected_standby'
            if kind in pending:unresolved.append({'reason':'duplicate entry without exit','event':event})
            pending[kind]=event
        elif provider=='Microsoft-Windows-Kernel-Power' and eid in (107,507):
            kind='sleep' if eid==107 else 'connected_standby'
            if kind not in pending:unresolved.append({'reason':'exit without paired entry in scope','event':event})
            else:
                entry=pending.pop(kind);windows.append({'kind':kind,'start':entry['time'],'end':event['time'],
                    'entry_record':entry['record_id'],'exit_record':event['record_id'],'active_compute_seconds':None})
        elif provider=='Microsoft-Windows-Power-Troubleshooter' and eid==1:
            tree=ET.fromstring(event['xml']);data={x.get('Name'):x.text for x in tree.findall('.//{*}EventData/{*}Data')}
            if data.get('SleepTime') and data.get('WakeTime'):
                windows.append({'kind':'sleep_resume_event_data','start':data['SleepTime'],'end':data['WakeTime'],
                                'record':event['record_id'],'active_compute_seconds':None})
            else:unresolved.append({'reason':'resume event missing SleepTime/WakeTime','event':event})
        else:
            # 566 session transitions and boot/shutdown/crash events require their
            # native schema and timeline review, rather than assumed sleep time.
            unresolved.append({'reason':'power/session/boot event needs narrow native-schema review','event':event})
    unresolved.extend({'reason':'entry has no exit','event':e} for e in pending.values())
    releases=[C.read(p) for p in (C.OUT/'control').glob('released_*.json')]
    training_releases=[r for r in releases if r.get('module')=='tridep_ccml.fusion_eeg_grid_v2.run_v3']
    evaluation_releases=[r for r in releases if r.get('module')=='tridep_ccml.fusion_eeg_grid_v2.evaluation_only_v1']
    C.require(len(training_releases)==len(evaluation_releases)==1 and len(releases)==2,
              'Owned training/evaluation release-chain inventory differs')
    C.require(evaluation_releases[0]['new_optimizer_updates']==0,'Evaluation recovery performed optimizer updates')
    queue_start=dt(training_releases[0]['started_at']);feature_gate=C.read(C.OUT/'feature_gate.json')
    prior=dt(feature_gate['at']);envelopes=[]
    for identity in C.identities():
        completed=C.read(C.fitdir(identity)/'complete.json');end=dt(completed['finished_at'])
        C.require(end>=prior,'Sequential completion timestamps are not monotone')
        envelopes.append({'identity':identity,'start':prior.isoformat(),'end':end.isoformat(),
                          'scope':'conservative fit plus admission/initialization; not active optimizer interval'})
        prior=end
    for w in windows:
        begin,end=dt(w['start']),dt(w['end']);C.require(end>=begin,'Negative event interval')
        w['possible_affected_fits']=[e['identity'] for e in envelopes if dt(e['start'])<=end and dt(e['end'])>=begin]
        w['during_preparation_or_export']=begin<dt(feature_gate['at'])
        w['during_queue_envelope']=end>=queue_start
        w['known_training_pause_seconds']=None
    value={'passed':not unresolved,'raw_events':C.desc(CONTROL/'system_events.json'),'relevant_event_count':len(relevant),
      'raw_system_event_count':len(events['events']),'scope_start':protocol['scope_start'],'scope_end':events['scope_end'],
      'windows':windows,'unresolved':unresolved,'fit_envelopes':envelopes,
      'no_uninterrupted_active_compute_claim':True,'elapsed_categories_not_corrected_by_wall_subtraction':True,
      'documentation':protocol['documentation']}
    C.save(CONTROL/'power_scope.json',value,immutable=True)
    return value

def paired_review(results):
    from .final_audit_v1 import independent_metrics
    import numpy as np
    prediction=C.rows(C.REPORT/'predictions.jsonl')
    grouped={}
    for r in prediction:
        if r['role']=='test':grouped.setdefault((r['eeg'],r['method'],r['seed'],r['fold']),[]).append(r)
    C.require(len(grouped)==1200 and sum(map(len,grouped.values()))==8880,'Final test prediction identity coverage differs')
    original=C.rows(C.STAGE0/'branch_probabilities.jsonl')
    baseline={}
    for r in original:
        if r['split']=='test':
            key=(r['seed'],r.get('fold',r.get('outer_fold')))
            baseline.setdefault(key,[]).append({'subject_id':r['subject_id'],'label':r['label'],
              'p_mdd':.05*r['p_eeg']+.35*r['p_speech']+.60*r['p_text']})
    bm={key:independent_metrics(value,np) for key,value in baseline.items()}
    primary={key:sum(bm[(seed,fold)][key] for seed in C.SEEDS for fold in range(1,6))/15 for key in KEYS}
    C.require(abs(primary['macro_f1']-.7255026455026455)<1e-12,'Local fixed-weight baseline does not reproduce')
    with (C.REPORT/'paired_differences.csv').open(encoding='utf-8-sig',newline='') as f:paired=list(csv.DictReader(f))
    C.require(len(paired)==3375,'All75EST x15fold/seed x3 paired references required')
    for row in paired:
        eeg,method,seed,fold=row['eeg'],row['method'],int(row['seed']),int(row['fold'])
        current=grouped[(eeg,method,seed,fold)]
        ref=baseline[(seed,fold)] if row['reference']=='baseline' else grouped[(row['reference'],method,seed,fold)]
        actual=independent_metrics(current,np);previous=independent_metrics(ref,np)
        for key in KEYS:C.require(abs(actual[key]-previous[key]-float(row[key+'_delta']))<1e-12,'Paired metric delta differs')
        rp={r['subject_id']:r for r in ref};corrected=[];new_errors=[]
        for r in current:
            old=rp[r['subject_id']];C.require(old['label']==r['label'],'Paired label mismatch')
            was=int(old['p_mdd']>.5)==r['label'];isnow=int(r['p_mdd']>.5)==r['label']
            if isnow and not was:corrected.append(r['subject_id'])
            if was and not isnow:new_errors.append(r['subject_id'])
        C.require(json.loads(row['corrected'])==corrected and json.loads(row['new_errors'])==new_errors,
                  'Correction/new-error subject list differs')
    summaries={(r['eeg'],r['method']):r for r in results['summary']}
    for eeg in C.EEG:
        for method in C.METHODS:
            r=summaries[(eeg,method)]
            for reference in ('E00','ST'):
                C.require(abs(100*(r['macro_f1']-summaries[(reference,method)]['macro_f1'])-r['delta_'+reference+'_pp'])<1e-12,
                          'Internal paired summary differs')
    return {'passed':True,'paired_rows_verified':3375,'baseline_recomputed':primary,'test_rows':8880}

def run():
    protocol,audit=read_gate();results=C.read(C.REPORT/'results.json')
    paired=paired_review(results)
    cost=C.read(C.REPORT/'real_cost_v1.json')
    C.require(cost['formal_fits']==1200 and cost['combined_total_seconds'] is None and
              cost['all_process_active_training_seconds'] is None,'Unknown active cost fabricated')
    C.require(all(r.get('cost_unknown_seconds',r.get('repeated_forward_cost_unknown_seconds')) is None
                  for r in cost['failed_startup_attempts']),'Failed engineering cost not retained as unknown')
    C.require(not cost['recovery_events'] and len(cost['failure_evidence'])==1,
              'Unexpected training loss/failure chain requires further review')
    recovery=C.read(C.OUT/'evaluation_recovery_v1/complete.json')
    C.require(recovery['passed'] and recovery['new_optimizer_updates']==0 and recovery['cached_successes_reused']==17,
              'Actual evaluation-only recovery missing')
    review=C.read(C.checked(recovery['review']))
    for source in review['sources']:C.checked(source)
    for cached in review['cached_successful_evaluations']:C.checked(cached)
    failure=C.read(C.checked(cost['failure_evidence'][0]))
    C.require(failure['error']=="ValueError('Selected validation probability replay differs')" and
              'evaluation.py' in failure['traceback'] and 'Selected validation probability replay differs' in failure['traceback'],
              'Known evaluation validation failure identity differs')
    C.require(failure['owner']['module']=='tridep_ccml.fusion_eeg_grid_v2.run_v3',
              'Original failure has unknown owner')
    C.require(all(C.read(C.fitdir(i)/'complete.json')['finished_at']<=failure['at'] for i in C.identities()),
              'Original failure occurred before all formal fits completed')
    C.save(CONTROL/'evaluation_recovery_scope.json',{'passed':True,'original_failure':cost['failure_evidence'][0],
      'actual_recovery':C.desc(C.OUT/'evaluation_recovery_v1/complete.json'),'review':recovery['review'],
      'optimizer_updates_added':0,'formal_fits_retrained':0,'cached_successes_preserved':17,
      'extra_validation_diagnosis_forwards':4,'test_used_for_fix':False,
      'original_failed_validation_attempt_seconds':None,'diagnosis_elapsed_seconds':None,
      'unknown_exit_or_uncommitted_tail_seconds':None,
      'source_disclosure':'v1 17 cached evaluations, v2 1183 evaluations; preserved actual source bytes and inference_mode no-gradient semantics'},immutable=True)
    end=audit['at'];events=capture_events(protocol,end);scope=power_scope(events,protocol)
    C.require(scope['passed'],'System event scope contains unresolved records; inspect power_scope.json')
    engineering=C.read(C.OUT/'engineering/complete.json')
    C.require(engineering['passed'] and len(engineering['author_parity'])==10 and
              all(len(v['recovery'])==2 and all(q['model_optimizer_rng_history_bitwise'] for q in v['recovery'])
                  for v in engineering['methods'].values()),'Engineering source/restore evidence incomplete')
    for source in engineering['sources']:C.checked(source)
    observations=[(p,C.read(p)) for p in sorted((C.OUT/'control').glob('resource_observation_*.json'))]
    C.require(observations,'Actual runtime resource observation is missing')
    release=next(C.read(p) for p in (C.OUT/'control').glob('released_*.json')
                 if C.read(p).get('module')=='tridep_ccml.fusion_eeg_grid_v2.run_v3')
    for path,value in observations:
        C.require(value['owner_token']==release['token'] and value['no_model_forward'] and value['no_optimizer_update'],
                  'Resource observation is not bound to the actual queue')
        C.require(all(r['identity'] in release['native_tree'] for r in value['process_observations']),
                  'Resource process birth identity differs from released queue')
    resource={'passed':True,'observations':[C.desc(p) for p,_ in observations],
      'process_peak_scope':'true native peak working set as of each sample; not a guaranteed post-exit lifetime peak or per-fit peak',
      'highest_observed_worker_peak_bytes':max(v['process_observations'][0]['peak_working_set_bytes'] for _,v in observations),
      'exclusive_extraction_cuda_peak_bytes':None,'formal_head_device':{m:v['device'] for m,v in engineering['methods'].items()},
      'GPU_memory_scope':'whole-GPU samples, not exclusive experiment allocations',
      'source_derived_successful_fit_input_load_calls':1200,'source_derived_feature_archive_opens':2325,
      'source_derived_training_hot_loop_archive_opens':0,
      'formal_CPU_head_feature_H2D_transfers':0 if all(v['device']=='cpu' for v in engineering['methods'].values()) else None,
      'counter_basis':'fixed registered training.load_inputs called once per fit in actual successful single queue; not measured OS physical-read/transfer counters',
      'source':C.desc(C.CODE/'training.py'),'all_unknown_peaks_and_active_times_remain_null':True}
    C.save(CONTROL/'resource_scope.json',resource,immutable=True)
    report=C.REPORT/'最终全组合实验报告.md';text=report.read_text(encoding='utf-8')
    first=text.split('| 条件 |',1)[1].split('| EEG |',1)[0]
    C.require(first.count('本地 TRI-DEP 复现基线')==1 and sum(first.count('| '+eeg+' ') for eeg in C.EEG)==75,
              'Primary table must contain baseline and all75EST rows')
    C.require(all(word in text for word in ('二阶段开发','val3','111','临床','null','QMF','ECML')),
              'Required scope/known-adaptation disclosure missing')
    annex=['# 最终成本与运行范围附页','',
      '训练、验证、表示导出、写盘等已记录类别分列。未测的工程恢复、来源审计、失败重试和完整active总耗时保留null，不以零替代，也不与队列wall包络相加。',
      '',f"系统事件读取覆盖 {scope['scope_start']} 至 {scope['scope_end']}，保留全部 {scope['raw_system_event_count']} 条原始System事件，相关事件 {scope['relevant_event_count']} 条。",
      '','屏幕关闭不能当作所有计算停止的证据。Modern Standby内部仍区分Screen Off与Sleep状态。[微软说明](https://learn.microsoft.com/en-us/windows-hardware/design/device-experiences/modern-standby-states)。',
      '','事件窗口及可能受影响的fit列于power_scope.json，仅为保守执行包络，不冒称准确暂停/active时长；没有用wall时钟减perf_counter。',
      '','本轮没有正式训练队列丢失或训练恢复。全部1,200拟合结束后，评价因LMF参数标志切换改变matmul数值路径而失败；独立评价恢复没有增加optimizer更新，原17个成功评价保持字节不变。四次val诊断、原失败验证尝试、退出尾和额外成本未知部分仍null，完整链见evaluation_recovery_scope.json。',
      '',f"实际完整出生身份绑定的worker内存采样，最高观察到的截至采样时峰值工作集为 {resource['highest_observed_worker_peak_bytes']} 字节；不冒称退出后最终峰值或单fit峰值。GPU样本为整卡诊断，独占编码器显存峰值仍null。",
      '','五种小头均锁定CPU。源码与实际成功单队列推出1,200次输入装载、2,325次特征archive打开、训练热循环archive打开为0；这属于源码导出计数，不能冒称OS物理读取测量。完整资源证据见resource_scope.json。']
    # Avoid rewriting prior core report/cost evidence: final scope is an annex.
    C.atomic(C.REPORT/'最终成本与运行范围附页.md',('\n'.join(annex)+'\n').encode())
    artifacts=['最终全组合实验报告.md','最终成本与运行范围附页.md','results.json','EST_summary.csv','ST_summary.csv',
               'fit_metrics.csv','paired_differences.csv','OOF.csv','predictions.jsonl','real_cost_v1.json']
    files=[C.desc(C.REPORT/name) for name in artifacts]
    files.extend([C.desc(C.OUT/'registration.json'),C.desc(C.OUT/'source_registration.json'),C.desc(C.OUT/'feature_gate.json'),
      C.desc(C.OUT/'training_gate.json'),C.desc(C.OUT/'evaluation_complete.json'),C.desc(C.OUT/'engineering/complete.json'),
      C.desc(C.OUT/'final_audit_v1/complete.json'),C.desc(CONTROL/'protocol.json'),C.desc(CONTROL/'power_scope.json'),
      C.desc(CONTROL/'system_events.json'),C.desc(CONTROL/'resource_scope.json'),
      C.desc(CONTROL/'evaluation_recovery_scope.json'),C.desc(C.OUT/'evaluation_recovery_v1/complete.json'),
      recovery['review'],cost['failure_evidence'][0],C.desc(SOURCE)])
    files.extend(C.desc(p) for p,_ in observations)
    C.require(len(files)==26+len(observations),'Delivery binding inventory differs')
    for file in files:C.checked(file)
    value={'schema':'fusion_EEG15_Methods5_final_delivery_v3','passed':True,'all_plan_complete':True,
      'formal_fits':1200,'EST':1125,'ST':75,'conditions':80,'OOF_rows':8880,'paired_review':paired,
      'core_audit':C.desc(C.OUT/'final_audit_v1/complete.json'),'system_scope':C.desc(CONTROL/'power_scope.json'),
      'artifacts':files,'cost_unknowns_disclosed':True,'no_new_optimizer_updates':True,'no_new_model_forwards':True,
      'at':C.now()}
    C.save(CONTROL/'complete.json',value,immutable=True)
    C.save(C.OUT/'final_delivery.json',value,immutable=True)
    C.save(C.OUT/'status.json',{'phase':'complete','formal_fits_completed':1200,'all_plan_complete':True,
                              'final_delivery':C.desc(C.OUT/'final_delivery.json'),'at':C.now()})
    print(json.dumps({'complete':True,'models':1200,'conditions':80,'artifacts':len(files)}))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if a.run:run()
    else:print(json.dumps(prepare(),ensure_ascii=False))

if __name__=='__main__':main()
