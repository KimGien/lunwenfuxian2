"""Read-only CPU tensor/metadata audit; no models, signals or inference."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random
import sys
import uuid

WORK=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(WORK))
import numpy as np
import torch
from tridep_ccml.eeg_all_candidates import protocol as P,registry as R,storage as S

CONDITION='EEG2Rep-fold-local-ssl'
ONLINE=('core.InputEmbedding.','core.Norm.','core.Norm2.','core.contex_encoder.')


def require(value,message):
    if not value:raise ValueError(message)


def digest(state):
    value=hashlib.sha256()
    for name,tensor in sorted(state.items()):
        a=tensor.detach().cpu().contiguous();value.update(name.encode());value.update(str(a.dtype).encode())
        value.update(str(tuple(a.shape)).encode());value.update(a.numpy().tobytes())
    return value.hexdigest()


def online(state):return {n:v for n,v in state.items() if n.startswith(ONLINE)}


def changed(before,after):
    require(before.keys()==after.keys(),'Online parameter namespace differs')
    return [n for n in before if not torch.equal(before[n],after[n])]


def batches(length,size):
    result=[list(range(start,min(start+size,length))) for start in range(0,length,size)]
    if len(result[-1])==1:result[-1].insert(0,result[-2].pop())
    return result


def mask_indices(mask):
    require(mask['token_count']==500 and len(mask['chunk_starts'])==2,'SSP contract differs')
    starts=mask['chunk_starts'];require(all(isinstance(v,int) and 0<=v<=375 for v in starts),'SSP starts invalid')
    visible=[i for start in starts for i in range(start,start+125)]
    require(mask['visible_count']==250 and mask['masked_count']==500-len(set(visible)),'SSP overlap/population differs')
    return visible


def fixed_starts(seed,fold,group):
    raw=f'{P.EXPERIMENT_ID}|{CONDITION}|{seed}|{fold}|fixed_val_mask_{group}'.encode()
    number=int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(2**63-1)
    rng=random.Random(number)
    return [rng.randint(0,375),rng.randint(0,375)]


def check_optimizer(saved,expected):
    require(saved['optimizer_updates']==saved['ema_updates']==expected,'Checkpoint optimizer/EMA count differs')
    states=saved['optimizer_state_dict']['state']
    if expected==0:require(not states,'Initialization has AdamW state')
    else:
        require(states and all(float(v['step'])==expected for v in states.values()),'Actual AdamW moments have wrong step count')
        require(all(v['exp_avg'].dtype==v['exp_avg_sq'].dtype==torch.float32 for v in states.values()),'AdamW moments not FP32')


def audit_fit(seed,fold,registration,lock,settings):
    folder=R.fit_directory(CONDITION,seed,fold);split=registration['folds'][f'fold_{fold}'];status=R.fit_status(CONDITION,seed,fold)
    require(status['status']=='training_complete' and status['condition_source_lock']==lock,'Formal completion/source lock differs')
    evidence=status['evidence'];R._validate_complete(evidence,folder);require(evidence==S.read(folder/'complete.json'),'Completion artifact differs')
    meta=S.read(S.checked(evidence['metadata']));history=S.read(S.checked(evidence['train_history']));validation=S.read(S.checked(evidence['validation_history']))
    require(meta['train_ids']==meta['encoder_fit_subject_ids']==split['train'] and meta['val_ids']==split['val'],'Original fold roles differ')
    require(not set(meta['encoder_fit_subject_ids'])&set(split['val']+split['test']),'Nontraining person used for encoder fit')
    require(meta['test_forward_count']==meta['validation_updates']==0 and meta['sampler_rng_saved'],'Forbidden test/val update or missing sampler RNG')
    require(meta['precision']=='bfloat16' and meta['loss_precision']=='float32' and meta['loss_autocast_enabled'] is False,'Declared precision differs')
    repair=meta['ssl_precision_repair'];require(repair==settings['ssl_precision_repair'],'FP32 loss provenance differs from actual prepared settings')
    for key in ('wrapper_source','delegated_runner_source','unchanged_adapter_source'):S.checked(repair[key])
    require(meta['registry']==R.registration_descriptor() and meta['settings']==S.desc(S.OUT/'ssl/engineering_settings.json'),'Registry/settings source differs')
    size=meta['actual_logical_batch'];require(size==64,'Actual logical batch changed')
    groups=batches(len(split['train'])*60,size);steps=len(groups)
    require(len(history)==len(validation)==meta['epochs'],'Full curve coverage differs')
    losses=[];extended=False
    for epoch,row in enumerate(history,1):
        require(row['epoch']==epoch and row['updates_completed']==row['gradient_passes_completed']==row['ema_updates_completed']==steps*epoch,'Epoch logical update/EMA accounting differs')
        require(row['train_windows']==len(split['train'])*60 and row['logical_batch_sizes']==[len(g) for g in groups],'Full balanced windows/logical sizes differ')
        require(len(row['masks'])==steps and all(math.isfinite(row[k]) for k in ('train_ssl_loss','val_ssl_loss')),'Mask/loss record invalid')
        for mask in row['masks']:mask_indices(mask)
        expected={'epoch':epoch,'val_ssl_loss':row['val_ssl_loss'],**row['val_components']}
        require(validation[epoch-1]==expected,'Standalone validation curve differs')
        require(abs(row['lr']-1e-4*P.learning_rate_multiplier('ssl',epoch))<1e-15,'LR schedule differs')
        losses.append(row['val_ssl_loss']);decision=P.training_decision('ssl',losses,extended=extended)
        require(not decision['stop'] or epoch==len(history),'Training continued beyond declared stopping rule')
        if decision['extend']:extended=True
    require(decision['stop'] and decision['reason']==meta['stopping_reason']==evidence['stop_reason'] and decision['best_epoch']==meta['best_epoch'],'Terminal/best stopping rule differs')
    require(meta['updates']==meta['gradient_passes']==meta['ema_updates']==meta['epochs']*steps,'Terminal counters differ')
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=False)
    best=torch.load(S.checked(evidence['checkpoint']),map_location='cpu',weights_only=False)
    latest=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
    for saved in (initial,best,latest):
        require(all(bool(torch.isfinite(v).all()) for v in saved['state_dict'].values()),'Nonfinite state in completed checkpoints')
        require(saved['identity']['train_ids']==split['train'] and saved['identity']['val_ids']==split['val'] and saved['identity']['test_forward_count']==0,'Checkpoint role/test identity differs')
        require(saved['identity']['settings']==meta['settings'] and saved['identity']['registry']==meta['registry'],'Checkpoint source identity differs')
    check_optimizer(initial,0);check_optimizer(best,meta['best_epoch']*steps);check_optimizer(latest,meta['updates'])
    before=online(initial['state_dict']);selected=online(best['state_dict']);terminal=online(latest['state_dict'])
    selected_changes=changed(before,selected);terminal_changes=changed(before,terminal)
    require(digest(initial['state_dict'])==meta['initial_state_sha256'] and digest(best['state_dict'])==meta['selected_state_sha256']==best['state_sha256'],'Full checkpoint fingerprints differ')
    require(digest(before)==meta['initial_online_state_sha256'] and digest(selected)==meta['selected_online_state_sha256']==best['online_state_sha256']
            and digest(terminal)==meta['terminal_online_state_sha256']==latest['online_state_sha256'],'Online-only fingerprints differ')
    require(selected_changes==meta['online_changed_parameter_names'] and len(selected_changes)==meta['online_changed_parameter_count']>0
            and len(terminal_changes)==meta['terminal_online_changed_parameter_count']>0,'Online-only actual tensor changes differ')
    require(meta['online_backbone_update_verified'] and evidence['online_backbone_update_verified'],'Online update proof missing')
    require(latest['epoch']==meta['epochs'] and best['epoch']==meta['best_epoch'] and latest['history']==history,'Selected/terminal checkpoint epochs or histories differ')
    require(digest(latest['best']['state_dict'])==digest(best['state_dict']) and latest['best']['optimizer_updates']==best['optimizer_updates'],'Durable selected state differs inside latest')
    require(len(best['history'])==best['epoch'] and len(latest['order'])==latest['epoch'] and best['order']==latest['order'][:best['epoch']],'Selected history/order prefix differs')
    for epoch,order in enumerate(latest['order'],1):
        require(sorted(order)==list(range(len(split['train'])*60)),'Training order lost or repeated windows')
        require(all(sorted(i//60 for i in order[start:start+len(split['train'])])==list(range(len(split['train'])))
                    for start in range(0,len(order),len(split['train']))),'Per-person interleaved window balance differs')
        require(hashlib.sha256(np.asarray(order,dtype='<i8').tobytes()).hexdigest()==history[epoch-1]['training_order_sha256'],'Recorded per-epoch order fingerprint differs')
    require(hashlib.sha256(np.asarray(latest['order'],dtype='<i8').tobytes()).hexdigest()==meta['training_order_sha256'],'Full order fingerprint differs')
    require(latest['sampler_position']=={'epoch_completed':meta['epochs'],'next_epoch':meta['epochs']+1,'next_batch':0}
            and latest['scheduler_state_dict']=={'policy':'ssl','completed_epoch':meta['epochs'],'extended':meta['extended']},'Resume sampler/scheduler differs')
    require(set(latest['sampling'])=={'window_generator','mask_rng'} and len(latest['rng']['torch_cuda'])==1,'Complete RNG/sampler state missing')
    require(all(torch.equal(initial['state_dict'][n],best['state_dict'][n]) and torch.equal(initial['state_dict'][n],latest['state_dict'][n])
                for n in initial['state_dict'] if n.startswith('core.predict_head.')),'Unused original window head changed')
    for n,t in initial['state_dict'].items():
        if n.startswith('core.contex_encoder.'):
            target=n.replace('core.contex_encoder.','core.target_encoder.',1)
            require(torch.equal(t,initial['state_dict'][target]),'Independent EMA target initial values differ')
    masks=S.read(S.checked(meta['validation_mask_manifest']));require(masks['val_ids']==split['val'] and masks['registry']==meta['registry']
             and masks['validation_updates']==masks['validation_ema_updates']==0 and masks['actual_logical_batch']==64,'Validation mask role/update contract differs')
    val_groups=batches(len(split['val'])*60,64);require(len(masks['records'])==len(val_groups),'Validation batch coverage differs')
    for group,record in enumerate(masks['records']):
        require(record['window_indices']==val_groups[group] and record['batch']==group,'Validation windows differ')
        mask_indices(record['mask']);require(record['mask']['chunk_starts']==fixed_starts(seed,fold,group),'Validation masks are not the registered deterministic stream')
    require(evidence['checkpoint_reload_verified'],'Recorded training-only checkpoint reload evidence missing')
    snapshots=[]
    for epoch in (60,120,200,300,500):
        path=folder/f'epoch_{epoch}.pt'
        require(path.exists()==(epoch<=meta['epochs']),'Registered reached SSL snapshot coverage differs')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=False);require(saved['epoch']==epoch,'Snapshot header differs')
            check_optimizer(saved,epoch*steps);snapshots.append(S.desc(path))
    return {'seed':seed,'fold':fold,'passed':True,'epochs':meta['epochs'],'best_epoch':meta['best_epoch'],
            'terminal_optimizer_and_ema_updates':meta['updates'],'selected_optimizer_and_ema_updates':best['optimizer_updates'],
            'selected_online_changed_names':selected_changes,'terminal_online_changed_names':terminal_changes,
            'selected_online_state_sha256':digest(selected),'terminal_online_state_sha256':digest(terminal),
            'initial_online_state_sha256':digest(before),'fp32_loss_source_verified':True,'logical_batch64_verified':True,
            'fixed_validation_masks_only_val_people_verified':True,'all_original_fold_roles_verified':True,
            'test_forward_count':0,'validation_updates':0,'selected_and_terminal_counters_not_conflated':True,
            'recorded_checkpoint_reload_verified':True,'no_independent_forward_performed':True,
            'evidence':evidence,'initial_checkpoint':S.desc(folder/'initial.pt'),'latest_checkpoint':S.desc(folder/'latest.pt'),
            'reached_fixed_snapshots':snapshots,'prior_failure_files':[S.desc(p) for p in folder.glob('failure_*.json')]}


def main():
    torch.set_num_threads(2);registration=R.load_registration();lock=R._condition_lock(CONDITION)
    contract=S.read(S.checked(lock))['contract'];settings=S.read(S.OUT/'ssl/engineering_settings.json')
    require(contract['input_contract']==settings and settings['actual_logical_batch']==64 and settings['precision']=='bfloat16'
            and settings['loss_precision']=='float32' and settings['loss_autocast_enabled'] is False,'Actual sealed engineering settings differ')
    report={'captured_at_Asia_Shanghai':S.now(),'source':S.desc(Path(__file__)),'condition_source_lock':lock,
            'scope':'read-only completed SSL CPU tensors/metadata; no ranking of validation scores',
            'gpu_calls':0,'model_constructions':0,'forward_calls':0,'backward_calls':0,'optimizer_executions':0,
            'real_signal_reads':0,'outer_test_probability_reads':0,'status_snapshot':[],'completed_fit_audits':[]}
    for seed in P.SEEDS:
        for fold in P.FOLDS:report['status_snapshot'].append({'seed':seed,'fold':fold,'status':R.fit_status(CONDITION,seed,fold)['status']})
    for item in report['status_snapshot']:
        if item['status']=='training_complete':report['completed_fit_audits'].append(audit_fit(item['seed'],item['fold'],registration,lock,settings))
    report.update(passed=True,completed_ssl_fits_verified=len(report['completed_fit_audits']),planned_ssl_fits=15,
                  cumulative_terminal_updates=sum(v['terminal_optimizer_and_ema_updates'] for v in report['completed_fit_audits']),
                  cumulative_selected_checkpoint_updates=sum(v['selected_optimizer_and_ema_updates'] for v in report['completed_fit_audits']))
    path=S.REPORT/('ssl_cumulative_readonly_audit_'+str(report['completed_ssl_fits_verified'])+'fits_'+uuid.uuid4().hex[:12]+'.json')
    S.save(path,report,immutable=True)
    print(json.dumps({'passed':True,'report':path.relative_to(S.WORK).as_posix(),'completed_ssl_fits_verified':report['completed_ssl_fits_verified'],
                     'terminal_updates':report['cumulative_terminal_updates'],'selected_updates':report['cumulative_selected_checkpoint_updates'],
                     'gpu_calls':0,'models_constructed':0,'real_signal_reads':0},indent=2))


if __name__=='__main__':main()
