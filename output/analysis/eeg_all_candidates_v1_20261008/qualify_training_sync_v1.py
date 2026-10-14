"""Explicit-run, zero-optimizer-step training synchronization qualification.

Without --run only JSON/source pending evidence is written. Actual execution
requires the root guardian's exact natural-exit boundary proof and the shared
exclusive qualification lease. This is engineering, never a formal fit.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import uuid

SOURCE=Path(__file__).resolve()
REPORT=SOURCE.parent
WORK=SOURCE.parents[3]
MOD=WORK/'tridep_ccml/eeg_all_candidates'
OUT=WORK/'tridep_ccml/experiments/EEG-AllCandidates-v1-37-20261008'
HELPER_SOURCE=REPORT/'qualify_acceleration_20261009.py'
HELPER_SHA='878fda0f97d0a6ec801501baf1bad084d77d2d319897609a2a39600e01568a5e'
GATE_SOURCE=MOD/'sync_qualification_gate_v1.py'
GATE_SHA='0fdb296e7b77e0d1ece1613de66e91884a508cc81126da361cb406edeb8f3b8e'
PROTOTYPE=MOD/'sync_subject_update_v1.py'
PROTOTYPE_SHA='4eb7caf0f7fabf6d72381e3d54967ed05afc5bac18a766d356cf3fa61a4feeae'
ORIGINAL_SHA='cda19474654233c95072acae475174d9011bb0739be650a597194f04c4b0aa24'
F2_SHA='fa53183ea4927daed5dcec0204e8a28ab09940f589742debda884402581c1a8b'
CONDITION='CSBrain-lastblock-finetune'
OWNER=OUT/'execution_acceleration_v1/sync_qualification_owner.json'
FIXED_COUNTS={'forward':96,'encoder_forward':96,'encoder_windows':768,'backward':96,
              'optimizer_step':0,'optimizer_step_stub':12}


def require(value,message):
    if not value:raise ValueError(message)


def descriptor(path):
    path=Path(path).resolve()
    require(path.is_relative_to(WORK),'Evidence outside workspace')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'path':path.relative_to(WORK).as_posix(),'sha256':digest}


def load_pinned(path,digest,name):
    require(descriptor(path)['sha256']==digest,'Reviewed source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def helpers():return load_pinned(HELPER_SOURCE,HELPER_SHA,'_training_sync_io')
def gate_module():return load_pinned(GATE_SOURCE,GATE_SHA,'_training_sync_gate')


def matrix_plan():
    """Exactly 1 warm pair and 2 measured pairs per phase; no hidden calls."""
    result=[]
    for phase in (1,11):
        for repetition in range(3):
            order=('original','candidate') if (repetition+(phase==11))%2==0 else ('candidate','original')
            for variant in order:
                result.append({'epoch_phase':phase,'repetition':repetition,'warmup':repetition==0,'variant':variant})
    return result


def validate_fixed_identity(proof,registry,fit,history):
    require(proof.get('schema')=='sync_qualification_safe_boundary_v1' and proof.get('passed') is True,
            'Natural-exit guardian proof required')
    require(fit['condition']==proof['condition']==CONDITION and type(fit['seed']) is int and fit['seed']==proof['seed']==22
            and type(fit['fold']) is int and fit['fold']==proof['fold']==5 and fit['stage']=='3',
            'Training qualification is fixed to completed CSBrain seed22/fold5')
    people=fit['train_ids'][:8]
    require(len(people)==len(set(people))==8 and not set(people)&set(fit['val_ids']+fit['test_ids']),
            'Exactly eight registered training people required')
    split=registry['folds']['fold_5']
    require(all(fit[k+'_ids']==split[k] for k in ('train','val','test')),'Registered fold roles differ')
    checkpoint=(WORK/proof['checkpoint']['path']).resolve()
    require(checkpoint==OUT/'runs'/fit['fit_id']/'latest.pt' and history
            and all(type(row['epoch']) is int and row['epoch']==i for i,row in enumerate(history,1)),
            'Completed exact latest checkpoint and complete history required')
    return people,split


def fingerprint(value,torch):
    """Exact typed bytes, including signed zero; used outside timed regions."""
    digest=hashlib.sha256()
    def add(item):
        if isinstance(item,torch.Tensor):
            tensor=item.detach().contiguous().cpu()
            digest.update(json.dumps(['tensor',str(tensor.dtype),list(tensor.shape)]).encode())
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif hasattr(item,'dtype') and hasattr(item,'shape') and hasattr(item,'tobytes'):
            digest.update(json.dumps(['array',str(item.dtype),list(item.shape)]).encode());digest.update(item.tobytes())
        elif isinstance(item,dict):
            digest.update(b'dict{')
            for key in sorted(item,key=lambda k:(type(k).__name__,repr(k))):add(key);add(item[key])
            digest.update(b'}')
        elif isinstance(item,(list,tuple)):
            digest.update(type(item).__name__.encode()+b'[')
            for child in item:add(child)
            digest.update(b']')
        else:digest.update(json.dumps([type(item).__name__,item],allow_nan=False,separators=(',',':')).encode()+b'\0')
    add(value);return digest.hexdigest()


@contextmanager
def forbid_optimizer_steps(torch,trace):
    """Deny every exported optimizer class; only the separate stub is allowed."""
    originals=[];seen=set()
    def forbidden(*args,**kwargs):
        trace['forbidden_optimizer_attempts']+=1
        raise RuntimeError('Actual optimizer.step forbidden in zero-update qualification')
    try:
        for cls in vars(torch.optim).values():
            if isinstance(cls,type) and issubclass(cls,torch.optim.Optimizer) and cls not in seen:
                seen.add(cls);owned='step' in cls.__dict__;original=cls.__dict__.get('step')
                originals.append((cls,owned,original));setattr(cls,'step',forbidden)
        yield
    finally:
        for cls,owned,original in reversed(originals):
            if owned:setattr(cls,'step',original)
            else:delattr(cls,'step')


class ZeroStepProxy:
    def __init__(self,optimizer,trace):self.optimizer,self.trace=optimizer,trace
    def zero_grad(self,*args,**kwargs):return self.optimizer.zero_grad(*args,**kwargs)
    def step(self):
        self.trace['optimizer_step_stub_started']+=1
        self.trace['optimizer_step_stub_returned']+=1


def compare_pair(original,candidate):
    fields=('ordered_logits_sha256','ordered_ce_sha256','result','pre_clip_gradients_sha256',
            'post_clip_gradients_sha256','clip_norm_sha256','post_rng_sha256','parameter_buffer_state_sha256',
            'optimizer_state_sha256','module_modes','active_parameter_names','operation_delta')
    flags={key:original[key]==candidate[key] for key in fields}
    return {'all_bitwise_and_exact':all(flags.values()),'checks':flags,
            'original_wall_seconds':original['synchronized_wall_seconds'],
            'candidate_wall_seconds':candidate['synchronized_wall_seconds']}


def summarize(measurements):
    require(len(measurements)==12,'Full registered qualification matrix missing')
    require(all(type(row['epoch_phase']) is int and type(row['repetition']) is int
                and type(row['warmup']) is bool and row['warmup']==(row['repetition']==0)
                and isinstance(row['synchronized_wall_seconds'],(int,float))
                and not isinstance(row['synchronized_wall_seconds'],bool)
                and math.isfinite(row['synchronized_wall_seconds']) and row['synchronized_wall_seconds']>0
                for row in measurements),'Qualification timing/identity fields are invalid')
    by_key={(row['epoch_phase'],row['repetition'],row['variant']):row for row in measurements}
    require(len(by_key)==12 and set(by_key)=={(r['epoch_phase'],r['repetition'],r['variant']) for r in matrix_plan()},
            'Duplicate/missing qualification identity')
    comparisons=[];phases=[]
    for phase in (1,11):
        for rep in range(3):
            result=compare_pair(by_key[phase,rep,'original'],by_key[phase,rep,'candidate'])
            comparisons.append({'epoch_phase':phase,'repetition':rep,'warmup':rep==0,**result})
        medians={variant:statistics.median(by_key[phase,rep,variant]['synchronized_wall_seconds'] for rep in (1,2))
                 for variant in ('original','candidate')}
        phases.append({'epoch_phase':phase,'baseline_median_seconds':medians['original'],
                       'candidate_median_seconds':medians['candidate'],
                       'candidate_strictly_faster':medians['candidate']<medians['original']})
    exact=all(row['all_bitwise_and_exact'] for row in comparisons)
    faster=all(row['candidate_strictly_faster'] for row in phases)
    return {'passed':exact and faster,'bitwise_qualified':exact,'faster_in_both_measured_phases':faster,
            'comparisons':comparisons,'phases':phases,'selected_variant':'candidate' if exact and faster else None,
            'scope':'Two repetitions per phase on this completed checkpoint/train-only batch; no universal speed or numerical equivalence claim'}


def run_matrix(torch,F,F2,C,proof,registry,fit,history,people,split,trace,attempt_path,Q):
    """Reached only behind the root's live safe boundary and shared lease."""
    checkpoint=Q.checked(proof['checkpoint'],binary=True)
    raw,raw_sources=F._raw_sources(split,people=people)
    require(set(raw)==set(people)==set(raw_sources),'Only the fixed training raw subset may be loaded')
    trace.update(raw_sources=raw_sources,probe_subject_ids=people,input_role='train')
    Q.atomic_json(attempt_path,trace)
    F.C.runtime(fit['seed'])
    trace['actual_runtime'].update(float32_matmul_precision=torch.get_float32_matmul_precision(),
        cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        deterministic_algorithms_enabled=torch.are_deterministic_algorithms_enabled())
    require(not F2._ACTIVE and F2._TRACE is None,'Qualification cannot reuse an active formal observer context')
    external_rng=F.rng_state('cuda')
    model=None;optimizer=None;measurements=[]
    trace['model_constructions_started']+=1;Q.atomic_json(attempt_path,trace)
    try:
        model,adapter_audit=F2.build_model(fit['condition'],seed=fit['seed'],device='cuda',window_chunk=8)
        trace['model_constructions_returned']+=1
        trace['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        lock=F2._check_v2_lock(fit['condition'])
        trace['latest_checkpoint_load_started']+=1;Q.atomic_json(attempt_path,trace)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        trace['latest_checkpoint_load_returned']+=1
        require((saved['condition'],saved['seed'],saved['fold'])==(CONDITION,22,5)
                and saved['registry']==proof['original_training_registration'] and saved['condition_source_lock']==lock
                and saved['history']==history and saved['epoch']==len(history), 'Completed checkpoint identity/history differs')
        require(all(saved['raw_sources'].get(sid)==binding for sid,binding in raw_sources.items())
                and set(saved['raw_sources'])==set(split['train']+split['val']), 'Checkpoint raw-source roles differ')
        model.load_state_dict(saved['state_dict'],strict=True)
        optimizer=F.optimizer_for(model)
        candidate=C.make_subject_update(F.subject_update)
        candidate_provenance=candidate.sync_candidate_provenance
        trace.update(condition_source_lock=lock,adapter_audit=adapter_audit,candidate_provenance=candidate_provenance)
        Q.atomic_json(attempt_path,trace)
        original_forward,original_encode=model.forward,model.adapter.encode_windows
        original_backward=torch.Tensor.backward
        original_ce=torch.nn.functional.cross_entropy
        original_clip=torch.nn.utils.clip_grad_norm_
        captured={'logits':[],'ce':[],'pre':None,'post':None,'norm':None}
        def forward(raw_value):
            require(tuple(raw_value.shape)==(1,8,19,1000),'Probe forward must remain micro1 and eight windows')
            trace['forward_started']+=1
            value=original_forward(raw_value)
            trace['forward_returned']+=1;captured['logits'].append(value.detach())
            return value
        def encode(*args,**kwargs):
            n=len(args[1]);require(n==8,'Training encoder chunk changed')
            trace['encoder_forward_started']+=1;trace['encoder_windows_started']+=n
            value=original_encode(*args,**kwargs)
            trace['encoder_forward_returned']+=1;trace['encoder_windows_returned']+=n
            return value
        def backward(tensor,*args,**kwargs):
            trace['backward_started']+=1
            value=original_backward(tensor,*args,**kwargs)
            trace['backward_returned']+=1
            return value
        def ce(*args,**kwargs):
            value=original_ce(*args,**kwargs);captured['ce'].append(value.detach());return value
        def clip(parameters,*args,**kwargs):
            parameters=list(parameters)
            expected=[p for p in model.parameters() if p.requires_grad]
            require(len(parameters)==len(expected) and all(a is b for a,b in zip(parameters,expected)),
                    'Clipping parameter identity/order changed')
            require(args==(F.POLICY.gradient_clip,) and not kwargs,'Original gradient clipping arguments changed')
            captured['pre']=[p.grad.detach().clone() for p in parameters]
            value=original_clip(parameters,*args,**kwargs)
            captured['post']=[p.grad.detach().clone() for p in parameters];captured['norm']=value.detach().clone()
            return value
        model.forward=forward;model.adapter.encode_windows=encode
        torch.Tensor.backward=backward;torch.nn.functional.cross_entropy=ce;torch.nn.utils.clip_grad_norm_=clip
        proxy=ZeroStepProxy(optimizer,trace)
        try:
            with forbid_optimizer_steps(torch,trace):
                for plan in matrix_plan():
                    model.load_state_dict(saved['state_dict'],strict=True)
                    optimizer.load_state_dict(saved['optimizer_state_dict'])
                    F.set_train_phase(model,plan['epoch_phase']);F.set_learning_rates(optimizer,plan['epoch_phase'])
                    F.restore_rng(saved['rng'])
                    state=fingerprint(model.state_dict(),torch);optimizer_state=fingerprint(optimizer.state_dict(),torch)
                    modes=[(name,module.training) for name,module in model.named_modules()]
                    active=[name for name,p in model.named_parameters() if p.requires_grad]
                    frozen={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
                    require(len(active)==(6 if plan['epoch_phase']==1 else 26),'CSBrain phase parameter contract differs')
                    initial_rng=F.rng_state(model.device)
                    captured.update(logits=[],ce=[],pre=None,post=None,norm=None)
                    before={key:trace[key+'_returned'] for key in FIXED_COUNTS}
                    trace.update(current_measurement=plan,phase='logical_batch_started');Q.atomic_json(attempt_path,trace)
                    trace['peak_allocated_bytes']=max(trace.get('peak_allocated_bytes',0),torch.cuda.max_memory_allocated())
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    event0,event1=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    event0.record();start=time.perf_counter()
                    function=F.subject_update if plan['variant']=='original' else candidate
                    result=function(model,proxy,people,raw,registry['labels'],F.bag_indices(plan['epoch_phase']),micro_subjects=1)
                    event1.record();torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    peak=torch.cuda.max_memory_allocated()
                    post_rng=F.rng_state(model.device)
                    delta={key:trace[key+'_returned']-before[key] for key in FIXED_COUNTS}
                    require(delta=={'forward':8,'encoder_forward':8,'encoder_windows':64,'backward':8,
                                    'optimizer_step':0,'optimizer_step_stub':1},'Actual probe operations differ')
                    require(len(captured['logits'])==len(captured['ce'])==8 and captured['pre'] is not None
                            and captured['post'] is not None,'Actual logits/CE/clip evidence incomplete')
                    require(state==fingerprint(model.state_dict(),torch) and optimizer_state==fingerprint(optimizer.state_dict(),torch),
                            'Zero-step probe mutated model/buffers or optimizer state')
                    require(modes==[(name,module.training) for name,module in model.named_modules()]
                            and model.window_chunk==8,'Probe changed module modes/chunk')
                    require(all(torch.equal(value,dict(model.named_parameters())[name]) for name,value in frozen.items()),
                            'Frozen parameter changed')
                    row={**plan,'synchronized_wall_seconds':elapsed,'gpu_event_interval_ms_not_busy_time':event0.elapsed_time(event1),
                         'peak_allocated_bytes':peak,'ordered_logits_sha256':fingerprint(captured['logits'],torch),
                         'ordered_ce_sha256':fingerprint(captured['ce'],torch),'result':result,
                         'pre_clip_gradients_sha256':fingerprint(captured['pre'],torch),
                         'post_clip_gradients_sha256':fingerprint(captured['post'],torch),
                         'clip_norm_sha256':fingerprint(captured['norm'],torch),'post_rng_sha256':fingerprint(post_rng,torch),
                         'parameter_buffer_state_sha256':state,'optimizer_state_sha256':optimizer_state,
                         'module_modes':modes,'active_parameter_names':active,'operation_delta':delta,
                         'state_unchanged':True,'optimizer_state_unchanged':True,'modes_unchanged':True,'frozen_parameters_unchanged':True}
                    F.restore_rng(initial_rng)
                    require(F.rng_equal(initial_rng,F.rng_state(model.device)),'Probe failed to restore starting RNG')
                    row['starting_rng_restored_after_measurement']=True
                    measurements.append(row)
                    trace.update(phase='logical_batch_returned',measurements=measurements,
                                 peak_allocated_bytes=max(trace.get('peak_allocated_bytes',0),peak))
                    Q.atomic_json(attempt_path,trace)
        finally:
            model.forward=original_forward;model.adapter.encode_windows=original_encode
            torch.Tensor.backward=original_backward;torch.nn.functional.cross_entropy=original_ce
            torch.nn.utils.clip_grad_norm_=original_clip
        Q.checked(proof['checkpoint'],binary=True);Q.checked(proof['train_history'])
        return measurements
    finally:
        F.restore_rng(external_rng)
        require(F.rng_equal(external_rng,F.rng_state('cuda')),'Qualification failed to restore external RNG')
        trace['external_rng_restored']=True


def qualify(boundary_binding):
    Q=helpers();token=str(time.time_ns())
    attempt_path=OUT/'supervised_engineering'/CONDITION/('training_sync_attempt_'+token+'.json')
    counters={kind+'_'+state:0 for kind in FIXED_COUNTS for state in ('started','returned')}
    trace={'schema':'training_sync_qualification_attempt_v1','subtype':'training_sync_zero_optimizer_qualification',
           'kind':'engineering','status':'running','source':descriptor(SOURCE),'prototype_source':descriptor(PROTOTYPE),
           'boundary_proof':boundary_binding,'condition':CONDITION,'seed':22,'fold':5,
           'formal_fits':0,'optimizer_updates':0,'test_forward_count':0,'outer_test_forward_started':0,'outer_test_forward_returned':0,
           'started_at':Q.datetime.now(Q.timezone.utc).isoformat(),'seconds':None,'gpu_stage_entered':False,
           'unflushed_hard_crash_tail_exact':None,'forbidden_optimizer_attempts':0,
           'model_constructions_started':0,'model_constructions_returned':0,
           'latest_checkpoint_load_started':0,'latest_checkpoint_load_returned':0,**counters,
           'fixed_engineering_budget':FIXED_COUNTS,'measurements':[],
           'measurement_scope':'Whole attempt wall once; warmups and per-pair instrumented timers are nested and never added again. Clip gradient clone overhead is included equally in both timed variants; post-timer CPU byte hashes are outside pair timers.'}
    Q.atomic_json(attempt_path,trace,exclusive=True)
    began=time.perf_counter();owner=None;torch=None
    try:
        require(descriptor(PROTOTYPE)['sha256']==PROTOTYPE_SHA,'Frozen training prototype changed')
        for name,sha in (('finetune.py',ORIGINAL_SHA),('finetune_v2.py',F2_SHA)):
            require(descriptor(MOD/name)['sha256']==sha,'Sealed mathematical source changed')
        gate=gate_module()
        require(gate.OWNER.resolve()==OWNER.resolve(),'Qualification lease path differs')
        proof,registry,fit,history,idle=gate.preflight(boundary_binding)
        people,split=validate_fixed_identity(proof,registry,fit,history)
        experiment=proof.get('experimental_sources',[])
        require(all(descriptor(path) in experiment for path in (SOURCE,PROTOTYPE,GATE_SOURCE)),
                'Guardian did not bind this qualifier/prototype/shared gate')
        owner={'id':uuid.uuid4().hex,'pid':os.getpid(),'attempt':attempt_path.relative_to(WORK).as_posix(),
               'kind':'training_sync_zero_optimizer_qualification'}
        Q.atomic_json(OWNER,owner,exclusive=True)
        proof2,registry2,fit2,history2,idle2=gate.preflight(boundary_binding)
        require((proof,registry,fit,history)==(proof2,registry2,fit2,history2),'Boundary changed while acquiring exclusive lease')
        source_bindings=[descriptor(p) for p in (SOURCE,PROTOTYPE,GATE_SOURCE,HELPER_SOURCE,MOD/'finetune.py',MOD/'finetune_v2.py')]
        trace.update(registration=proof['original_training_registration'],queue_registration=proof['registration'],
                     checkpoint=proof['checkpoint'],train_history=proof['train_history'],
                     gate_source=descriptor(GATE_SOURCE),initial_guard_evidence=idle,current_guard_evidence=idle2,
                     source_bindings=source_bindings)
        Q.atomic_json(attempt_path,trace)
        sys.path.insert(0,str(WORK))
        import torch
        from tridep_ccml.eeg_all_candidates import finetune as F,finetune_v2 as F2,sync_subject_update_v1 as C
        require(torch.cuda.is_available(),'Actual CUDA required')
        trace['gpu_stage_entered']=True
        torch.cuda.reset_peak_memory_stats()
        trace['actual_runtime']={'torch':torch.__version__,'cuda':torch.version.cuda,'gpu_name':torch.cuda.get_device_name(),
            'float32_matmul_precision':torch.get_float32_matmul_precision(),
            'cuda_matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
            'deterministic_algorithms_enabled':torch.are_deterministic_algorithms_enabled()}
        Q.atomic_json(attempt_path,trace)
        measurements=run_matrix(torch,F,F2,C,proof,registry,fit,history,people,split,trace,attempt_path,Q)
        require(all(type(trace[kind+'_'+state]) is int and trace[kind+'_'+state]==expected
                    for kind,expected in FIXED_COUNTS.items() for state in ('started','returned')),
                'Actual engineering budget or completed observations differ')
        require(trace['forbidden_optimizer_attempts']==0,'An actual optimizer call was attempted')
        outcome=summarize(measurements)
        Q.checked(boundary_binding)
        for binding in source_bindings:Q.checked(binding)
        trace.update(status='complete',passed=outcome['passed'],finished_at=Q.datetime.now(Q.timezone.utc).isoformat(),
                     seconds=time.perf_counter()-began,graceful_terminal_snapshot=True,
                     engineering_backward_calls=trace['backward_returned'],external_rng_restored=True)
        Q.atomic_json(attempt_path,trace)
        report={'schema':'training_sync_qualification_v1',**outcome,'source':descriptor(SOURCE),
                'prototype_source':descriptor(PROTOTYPE),'gate_source':descriptor(GATE_SOURCE),
                'original_f1_source':descriptor(MOD/'finetune.py'),'delegated_f2_source':descriptor(MOD/'finetune_v2.py'),
                'boundary_proof':boundary_binding,'registration':proof['original_training_registration'],
                'queue_registration':proof['registration'],'checkpoint':proof['checkpoint'],
                'train_history':proof['train_history'],'condition':CONDITION,'seed':22,'fold':5,
                'probe_subject_ids':people,'input_role':'train','training_window_chunk':8,'micro_subjects':1,
                'matrix_plan':matrix_plan(),'measurements':measurements,'actual_operation_counts':FIXED_COUNTS,
                'optimizer_updates':0,'external_rng_restored':True,'attempt':descriptor(attempt_path),
                'not_formal_training_authorization':True,'runtime_installed':False,
                'timing_scope':trace['measurement_scope']}
        path=REPORT/('training_sync_qualification_'+token+'.json');Q.atomic_json(path,report,exclusive=True)
        return {'artifact':descriptor(path),'attempt':descriptor(attempt_path),'passed':outcome['passed'],'selected_variant':outcome['selected_variant']}
    except BaseException as error:
        trace.update(status='failed',finished_at=Q.datetime.now(Q.timezone.utc).isoformat(),seconds=time.perf_counter()-began,
                     error_type=type(error).__name__,error=str(error),graceful_terminal_snapshot=True)
        if torch is not None and trace['gpu_stage_entered']:
            try:
                peak=torch.cuda.max_memory_allocated();trace['peak_allocated_bytes_observed_at_failure']=peak
                trace['peak_allocated_bytes']=max(peak,trace.get('peak_allocated_bytes',0))
            except BaseException as resource_error:trace['resource_read_error']=str(resource_error)
        Q.atomic_json(attempt_path,trace)
        raise
    finally:
        if owner is not None and OWNER.exists() and Q.json_read(OWNER).get('id')==owner['id']:OWNER.unlink()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true');parser.add_argument('--boundary-proof',type=Path)
    parser.add_argument('--boundary-sha')
    args=parser.parse_args(argv)
    if not args.run:
        value={'schema':'training_sync_qualification_pending_v1','source':descriptor(SOURCE),
               'prototype_source':descriptor(PROTOTYPE),'status':'pending_root_natural_exit_boundary_and_explicit_run',
               'passed':False,'GPU':0,'torch_imports':0,'checkpoint_reads':0,'raw_reads':0,'model_constructions':0,
               'formal_optimizer_updates':0,'planned_engineering_budget':FIXED_COUNTS}
        target=REPORT/('training_sync_qualification_pending_'+str(time.time_ns())+'.json')
        helpers().atomic_json(target,value,exclusive=True)
        print(json.dumps({'artifact':descriptor(target),'status':value['status']}));return 0
    require(args.boundary_proof and args.boundary_sha,'Exact root boundary proof path/SHA required')
    binding=descriptor(args.boundary_proof)
    require(binding['sha256']==args.boundary_sha,'Boundary proof SHA differs')
    result=qualify(binding);print(json.dumps(result));return 0 if result['passed'] else 2


if __name__=='__main__':raise SystemExit(main())
