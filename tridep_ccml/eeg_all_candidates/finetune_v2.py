"""Versioned engineering/cost evidence; original fine-tuning math delegated.

LUNA preserves author fixed rotary frequencies. Independent engineering probes
never reuse an attempt's immutable checkpoint. Selected versus terminal encoder
changes are disclosed separately, including a validation best from head warmup.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import importlib
from pathlib import Path
import time
import types
import uuid

from . import finetune as V1, storage as S

torch, np = V1.torch, V1.np
CONDITIONS = V1.CONDITIONS
SOURCE = Path(__file__).resolve()
_BUILD = V1.build_model
_FIT = V1.fit
_SOURCES = V1._implementation_sources
_SAVE = S.save
_IMPORT = importlib.import_module
_ACTIVE = False
_TRACE = None


def provenance():
    return {'version':2, 'wrapper_source':S.desc(SOURCE), 'delegated_runner_source':S.desc(Path(V1.__file__)),
            'luna_adapter_source':S.desc(SOURCE.parent/'adapters/luna_v2.py'),
            'repairs':['retain author fixed rotary frequencies', 'durable per-call attempt cost and independent engineering probe',
                       'separate terminal training updates from validation-selected checkpoint changes'],
            'training_math_sampler_warmup_optimizer_stop_and_resume_unchanged':True,
            'no_validation_or_test_score_configuration_selection':True}


class Attempt:
    def __init__(self, condition, kind, seed=None, fold=None):
        # Keep atomic-write UUID suffixes below Windows' path-length boundary.
        stem=('e_' if kind=='engineering' else 'f_')+str(time.time_ns())+'_'+uuid.uuid4().hex[:8]
        self.folder = S.OUT/'finetune_v2_attempts'/condition/stem
        self.path = self.folder/'attempt.json'
        self.started = time.monotonic()
        self.adapters = []
        self.record = {'condition':condition,'kind':kind,'seed':seed,'fold':fold,'started_at':S.now(),'status':'running',
                       'implementation':provenance(), 'forward_started':0,'forward_returned':0,
                       'encoder_forward_started':0,'encoder_forward_returned':0,'encoder_windows_started':0,'encoder_windows_returned':0,
                       'encode_started':0,'encode_returned':0,
                       'backward_started':0,'backward_returned':0,'optimizer_step_started':0,'optimizer_step_returned':0}
        _SAVE(self.path,self.record,immutable=True)

    def observe(self, kind, function):
        def wrapped(*args, **kwargs):
            self.record[kind+'_started'] += 1
            if kind == 'encoder_forward':
                self.record['encoder_windows_started'] += len(args[1])
                self.record['encode_started'] += 1
            _SAVE(self.path,self.record)
            result = function(*args,**kwargs)
            self.record[kind+'_returned'] += 1
            if kind == 'encoder_forward':
                self.record['encoder_windows_returned'] += len(args[1])
                self.record['encode_returned'] += 1
            _SAVE(self.path,self.record)
            return result
        return wrapped

    def bind_adapter(self,adapter):
        if any(saved[0] is adapter for saved in self.adapters):return
        original=adapter.encode_windows
        self.adapters.append((adapter,original))
        adapter.encode_windows=self.observe('encoder_forward',original)

    def restore(self):
        for adapter,original in self.adapters:adapter.encode_windows=original
        self.adapters=[]

    def finish(self,error=None):
        self.record.update(status='failed' if error else 'complete',finished_at=S.now(),seconds=time.monotonic()-self.started)
        if error:
            self.record.update(failure_type=type(error).__name__,failure=str(error))
        _SAVE(self.path,self.record)


def _route_import(name, package=None):
    return _IMPORT('.adapters.luna_v2',package) if name=='.adapters.luna' and package==V1.__package__ else _IMPORT(name,package)


def build_model(condition, *, seed=11, device='cuda', window_chunk=8):
    previous = V1.importlib
    V1.importlib = types.SimpleNamespace(import_module=_route_import)
    try:
        model,audit = _BUILD(condition,seed=seed,device=device,window_chunk=window_chunk)
    finally:
        V1.importlib = previous
    audit['versioned_finetune_implementation'] = provenance()
    if _TRACE is not None:
        model.forward = _TRACE.observe('forward',model.forward)
        _TRACE.bind_adapter(model.adapter)
    return model,audit


def implementation_sources(condition,model,audit_sources):
    records = _SOURCES(condition,model,audit_sources)
    by_path = {r['path']:r for r in records}
    for record in (provenance()['wrapper_source'],provenance()['delegated_runner_source'],provenance()['luna_adapter_source']):
        by_path[record['path']] = record
    if model.adapter.__name__.endswith('luna_v2'):
        record = S.desc(SOURCE.parent/'adapters/luna.py')
        by_path[record['path']] = record
    return list(by_path.values())


def selected_checkpoint_disclosure(metadata, initial, selected, terminal=None):
    names = [name for name in initial if name.startswith('encoder.')]
    changed = [name for name in names if not torch.equal(initial[name],selected[name])]
    head_changed=[n for n in initial if n.startswith('head.') and not torch.equal(initial[n],selected[n])]
    terminal_encoder=[n for n,v in metadata['parameter_changed_after_training'].items() if n.startswith('encoder.') and v]
    terminal_head=[n for n,v in metadata['parameter_changed_after_training'].items() if n.startswith('head.') and v]
    return {'selected_checkpoint_phase':'head_only_warmup' if metadata['best_epoch']<=10 else 'lastblock_finetune',
            'selected_encoder_parameters_changed':bool(changed), 'selected_encoder_parameter_change_count':len(changed),
            'selected_encoder_changed_parameter_count':len(changed), 'selected_head_changed_parameter_count':len(head_changed),
            'selected_head_changed_parameter_names':head_changed,
            'terminal_encoder_changed_parameter_count':len(terminal_encoder), 'terminal_head_changed_parameter_count':len(terminal_head),
            'selected_encoder_changed_parameter_names':changed,
            'terminal_encoder_parameter_change_count':sum(bool(v) for n,v in metadata['parameter_changed_after_training'].items() if n.startswith('encoder.')),
            'selected_encoder_initial_state_sha256':V1.state_digest({n:initial[n] for n in names}),
            'selected_encoder_state_sha256':V1.state_digest({n:selected[n] for n in names}),
            'terminal_encoder_state_sha256':V1.state_digest({n:terminal[n] for n in names}) if terminal is not None else None,
            'terminal_encoder_changed_parameter_names':terminal_encoder,
            'terminal_head_changed_parameter_names':terminal_head,
            'terminal_updates_are_not_claimed_as_selected_encoder_updates':True}


def _save_provenance(path,value,*,immutable=False):
    target = Path(path).resolve()
    if target.name=='metadata.json' and target.is_relative_to((S.OUT/'runs').resolve()) and isinstance(value,dict) and value.get('condition') in CONDITIONS:
        initial = torch.load(target.parent/'initial.pt',map_location='cpu',weights_only=False)['state_dict']
        selected = torch.load(target.parent/'best.pt',map_location='cpu',weights_only=False)['state_dict']
        terminal = torch.load(target.parent/'latest.pt',map_location='cpu',weights_only=False)['state_dict']
        value.update(selected_checkpoint_disclosure(value,initial,selected,terminal),versioned_finetune_implementation=provenance())
    return _SAVE(path,value,immutable=immutable)


@contextmanager
def activate():
    global _ACTIVE
    if _ACTIVE:
        yield
        return
    previous = (V1.build_model,V1._implementation_sources,S.save,torch.Tensor.backward,torch.optim.AdamW.step)
    V1.build_model=build_model; V1._implementation_sources=implementation_sources; S.save=_save_provenance
    original_backward,original_step = previous[3:]
    def backward(tensor,*args,**kwargs):
        fn = original_backward if _TRACE is None else _TRACE.observe('backward',original_backward)
        return fn(tensor,*args,**kwargs)
    def step(optimizer,*args,**kwargs):
        fn = original_step if _TRACE is None else _TRACE.observe('optimizer_step',original_step)
        return fn(optimizer,*args,**kwargs)
    torch.Tensor.backward=backward; torch.optim.AdamW.step=step; _ACTIVE=True
    try:
        yield
    finally:
        V1.build_model,V1._implementation_sources,S.save,torch.Tensor.backward,torch.optim.AdamW.step=previous
        _ACTIVE=False


def _check_v2_lock(condition):
    descriptor=V1.R._condition_lock(condition)
    contract=S.read(S.checked(descriptor))['contract']
    V1.require(contract.get('versioned_finetune_implementation')==provenance(),'Condition source lock lacks this v2 engineering/parameter contract')
    return descriptor


def prepare_condition(condition,*,micro_subjects=1,window_chunk=8):
    global _TRACE
    V1.require(condition in CONDITIONS and micro_subjects in (1,2) and window_chunk>0,'Unregistered engineering condition or batch')
    trace=Attempt(condition,'engineering')
    model=optimizer=raw=None
    try:
        _TRACE=trace
        with activate():
            V1.require(torch.cuda.is_available(),'Versioned engineering still requires declared real CUDA environment')
            V1.R.load_registration()
            path=S.OUT/'engineering'/(condition+'.json')
            if path.exists():
                _check_v2_lock(condition)
                for seed in V1.P.SEEDS:
                    for fold in V1.P.FOLDS:
                        if V1.R.fit_status(condition,seed,fold)['status'] in ('engineering_pending','blocked'):
                            V1.R.record_fit_status(condition,seed,fold,'ready')
                trace.finish()
                return S.read(path)
            _,folds,labels=V1.C.context();split=folds['fold_1'];chosen=split['train'][:micro_subjects]
            raw,sources=V1._raw_sources(split,people=chosen)
            model,audit=build_model(condition,device='cuda',window_chunk=window_chunk)
            V1.C.runtime(11);initial=V1.state_digest(model.state_dict());optimizer=V1.optimizer_for(model)
            baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();checks=[]
            for epoch in (1,11):
                V1.set_train_phase(model,epoch);optimizer.zero_grad(set_to_none=True)
                target=torch.tensor([labels[s] for s in chosen],device=model.device)
                loss=torch.nn.functional.cross_entropy(model(np.stack([raw[s][V1.bag_indices(epoch)] for s in chosen])),target)
                V1.require(torch.isfinite(loss),'Engineering loss nonfinite');loss.backward()
                V1.require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad),'Engineering learnable parameter disconnected')
                V1.require(all(p.grad is None for p in model.parameters() if not p.requires_grad),'Frozen engineering gradient changed')
                checks.append({'epoch_phase':epoch,'actual_trainable_parameter_names':[n for n,p in model.named_parameters() if p.requires_grad]})
            V1.require(trace.record['forward_started']==trace.record['forward_returned']==2
                       and trace.record['backward_started']==trace.record['backward_returned']==2,
                       'Engineering observed forward/backward counts differ from actual two phases')
            V1.require(trace.record['encoder_windows_started']==trace.record['encoder_windows_returned']==len(chosen)*16,
                       'Engineering encoder-window cost coverage differs')
            V1.require(V1.state_digest(model.state_dict())==initial,'Zero-update engineering changed model state')
            state=V1.rng_state(model.device)
            probe=trace.folder/'resume_probe.pt'
            value={'state_dict':V1.cpu_state(model.state_dict()),'optimizer_state_dict':V1.cpu_state(optimizer.state_dict()),
                   'scheduler_state_dict':{'policy':'finetune','completed_epoch':0,'extended':False},'rng':state,
                   'sampler_position':{'epoch_completed':0,'next_batch':0}}
            S.save_torch(probe,value,immutable=True)
            restored=torch.load(probe,map_location='cpu',weights_only=False)
            model.load_state_dict(restored['state_dict'],strict=True);optimizer.load_state_dict(restored['optimizer_state_dict']);V1.restore_rng(restored['rng'])
            V1.require(V1.state_digest(model.state_dict())==initial and V1.rng_equal(state,V1.rng_state(model.device)),'Engineering full-state recovery differs')
            def descriptors(v):
                if isinstance(v,dict):
                    if 'path' in v and 'sha256' in v:yield {k:v[k] for k in ('path','sha256')}
                    else:
                        for child in v.values():yield from descriptors(child)
                elif isinstance(v,(list,tuple)):
                    for child in v:yield from descriptors(child)
            records={r['path']:r for r in descriptors(audit)}
            for record in records.values():S.checked(record)
            contract={'input_contract':{'adapter':audit,'windows_per_training_subject':8,'all_window_validation':60,'coverage_cycle_epochs':8,
                      'micro_subjects':micro_subjects,'window_chunk':window_chunk,'effective_subjects':8,'precision':'float32'},
                      'trainable_parameter_names':list(model.update_names),
                      'frozen_parameter_names':[n for n,_ in model.named_parameters() if n not in model.update_names],
                      'implementation_sources':implementation_sources(condition,model,[r for r in records.values() if r['path'].endswith('.py')]),
                      'weight_sources':list(records.values()),'forward_checked':True,'gradient_checked':True,'resume_checked':True,
                      'training_only_engineering_checked':True,'probe_subject_ids':chosen,'raw_sources':sources,
                      'engineering_optimizer_updates':0,'engineering_backward_calls':2,'gradient_checks':checks,
                      'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'baseline_allocated_bytes':baseline,
                      'initial_state_sha256':initial,'resume_probe':S.desc(probe),'versioned_finetune_implementation':provenance(),
                      'engineering_attempt_path':trace.path.relative_to(S.WORK).as_posix(),
                      'engineering_cost_counts':{k:v for k,v in trace.record.items() if k.endswith(('_started','_returned'))}}
            V1.require(trace.record['optimizer_step_started']==trace.record['optimizer_step_returned']==0,'Engineering unexpectedly executed an optimizer step')
            V1.R.lock_condition(condition,contract)
            for seed in V1.P.SEEDS:
                for fold in V1.P.FOLDS:V1.R.record_fit_status(condition,seed,fold,'ready')
            trace.finish()
            return contract
    except BaseException as exc:
        trace.finish(exc)
        raise
    finally:
        _TRACE=None
        trace.restore()
        del model,optimizer,raw;gc.collect();torch.cuda.empty_cache()


def fit(condition,seed,fold,*,resume=False,micro_subjects=1,window_chunk=8):
    global _TRACE
    trace=Attempt(condition,'formal',seed,fold)
    try:
        _TRACE=trace
        with activate():
            _check_v2_lock(condition)
            result=_FIT(condition,seed,fold,resume=resume,micro_subjects=micro_subjects,window_chunk=window_chunk)
        trace.finish()
        return result
    except BaseException as exc:
        trace.finish(exc)
        raise
    finally:
        trace.restore()
        _TRACE=None


def run(conditions=CONDITIONS,*,resume=False,limit=None,micro_subjects=1,window_chunk=8,prepare=False):
    V1.require(limit is None or limit>0,'Limit must be positive')
    completed=0
    for condition in conditions:
        if prepare:
            prepare_condition(condition,micro_subjects=micro_subjects,window_chunk=window_chunk);continue
        for seed in V1.P.SEEDS:
            for fold in V1.P.FOLDS:
                fit(condition,seed,fold,resume=resume,micro_subjects=micro_subjects,window_chunk=window_chunk)
                completed+=1
                if limit is not None and completed>=limit:return


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS));parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--resume',action='store_true');parser.add_argument('--limit',type=int)
    parser.add_argument('--micro-subjects',type=int,choices=(1,2),default=1);parser.add_argument('--window-chunk',type=int,default=8)
    args=parser.parse_args();run(args.conditions,resume=args.resume,limit=args.limit,micro_subjects=args.micro_subjects,window_chunk=args.window_chunk,prepare=args.prepare)


if __name__=='__main__':main()
