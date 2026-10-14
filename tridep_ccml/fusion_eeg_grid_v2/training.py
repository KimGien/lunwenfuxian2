from __future__ import annotations
import copy
import math
from pathlib import Path
import time
import numpy as np
import torch
from . import common as C
from .models import Fusion

def lr_for_epoch(epoch):
    if epoch <= 10:
        return 1e-3 * epoch / 10
    return 1e-3 * (.1 + .9 * (1+math.cos(math.pi*(epoch-10)/110))/2)

def nll(prob, y):
    return -prob[torch.arange(len(y), device=y.device), y].clamp_min(1e-15).double().log().mean()

def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) == type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b

def setup(method, views, ntrain, seed, device):
    C.runtime(seed, device)
    model = Fusion(method, views, ntrain).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3, foreach=False)
    return model, optimizer

def train_epoch(model, optimizer, x, y, epoch):
    model.train()
    rate = lr_for_epoch(epoch)
    for group in optimizer.param_groups:
        group['lr'] = rate
    order = torch.randperm(len(y), device=x.device)
    losses, grads, terms = [], [], {}
    start = time.perf_counter()
    for offset in range(0, len(order), 8):
        indices = order[offset:offset+8]
        optimizer.zero_grad(set_to_none=True)
        loss, components = model.objective(x[indices], y[indices], epoch, indices)
        C.require(bool(torch.isfinite(loss)), 'Nonfinite training loss')
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        # Keep loss and gradient finite checks; transfer diagnostics once per epoch.
        losses.append(loss.detach()*len(indices)); grads.append(grad.detach())
        for k, v in components.items():
            terms.setdefault(k, []).append(v*len(indices))
    parameters_finite = torch.stack([p.detach().isfinite().all() for p in model.parameters()]).all()
    C.require(bool(parameters_finite), 'Nonfinite parameter update')
    diagnostic = torch.stack([sum(losses)/len(y), torch.stack(grads).max(),
                              *[sum(terms[k])/len(y) for k in sorted(terms)]]).cpu().tolist()
    return {'epoch': epoch, 'lr': rate, 'loss': diagnostic[0], 'max_gradient_norm': diagnostic[1],
            'components': dict(zip(sorted(terms), diagnostic[2:])), 'updates': math.ceil(len(y)/8),
            'train_seconds': time.perf_counter()-start,
            'train_order': order.cpu().tolist(), 'batch_sizes': [min(8, len(y)-i) for i in range(0,len(y),8)]}

def history_state(model):
    return {'cumulative_ce': model.history.copy(), 'confidence': model.confidence_history.copy()} if model.history is not None else None

def restore_history(model, state):
    if model.history is not None:
        C.require(state is not None and state['cumulative_ce'].shape == model.history.shape, 'QMF train-history identity differs')
        model.history = state['cumulative_ce'].copy()
        model.confidence_history = state['confidence'].copy()
    else:
        C.require(state is None, 'Unexpected quality history')

def load_inputs(identity, device):
    seed, fold, eeg = identity['seed'], identity['fold'], identity['eeg']
    st = C.OUT / 'features/ST' / f'seed_{seed}' / f'fold_{fold}.npz'
    stmeta = C.read(st.with_suffix('.json'))
    C.checked(stmeta['cache'])
    with np.load(st, allow_pickle=False) as z:
        x, ids, labels, splits = (z[k].copy() for k in ('features','subject_ids','labels','splits'))
    sources = [C.desc(st), C.desc(st.with_suffix('.json'))]
    if eeg != 'ST':
        ep = C.OUT / 'features' / eeg / f'seed_{seed}' / f'fold_{fold}.npz'
        emeta = C.read(ep.with_suffix('.json')); C.checked(emeta['cache'])
        with np.load(ep, allow_pickle=False) as z:
            C.require(np.array_equal(ids,z['subject_ids']) and np.array_equal(labels,z['labels']) and np.array_equal(splits,z['splits']), 'E/ST people, labels or roles differ')
            x = np.concatenate([z['features'][:,None], x], axis=1)
        sources.extend([C.desc(ep), C.desc(ep.with_suffix('.json'))])
    roles = stmeta['roles']
    lookup = {sid:i for i,sid in enumerate(ids.tolist())}
    indexes = {role:np.asarray([lookup[sid] for sid in roles[role]], dtype=np.int64) for role in roles}
    C.require(len(set(ids.tolist())) == 37 and int(labels.sum()) == 16, 'Cohort differs')
    C.require(all(all(splits[j] == role for j in ix) for role,ix in indexes.items()), 'Role index differs')
    mean = x[indexes['train']].astype(np.float64).mean(0)
    std = x[indexes['train']].astype(np.float64).std(0)
    std = np.maximum(std, 1e-6)
    normalized = ((x.astype(np.float64)-mean)/std).astype(np.float32)
    C.require(np.isfinite(normalized).all(), 'Nonfinite standardized features')
    return torch.as_tensor(normalized,device=device), torch.as_tensor(labels,device=device), indexes, {
        'sources': sources, 'subject_ids': ids.tolist(), 'labels': labels.tolist(), 'splits': splits.tolist(),
        'roles': roles, 'normalization': {'mean':mean.tolist(),'std':std.tolist(),'floor':1e-6,
         'fit_people':roles['train'],'arithmetic':'float64 population mean/std; normalized output float32'}}

def complete_verified(identity):
    directory = C.fitdir(identity)
    p = directory / 'complete.json'
    if not p.exists():
        return None
    value = C.read(p)
    C.require(value['identity'] == identity and value['passed'] and 60 <= value['epochs'] <= 120,
              'Completed fit identity/budget invalid')
    for key in ('best','latest','metadata','trajectory'):
        C.checked(value[key])
    C.require(value['updates'] > 0 and value['parameter_change_verified'] and value['test_forwards']==0,
              'Completed fit lacks actual training/isolation evidence')
    return value

def fit(identity, device, registration):
    done = complete_verified(identity)
    if done:
        return done
    C.check_pause()
    directory = C.fitdir(identity); directory.mkdir(parents=True,exist_ok=True)
    x,y,ix,binding = load_inputs(identity,device)
    train = torch.as_tensor(ix['train'],device=device); val = torch.as_tensor(ix['val'],device=device)
    model,optimizer = setup(identity['method'],x.shape[1],len(train),identity['seed'],device)
    initial_sha = C.state_sha(model.state_dict())
    resume = directory / 'latest.pt'
    epoch, updates, best_nll, best_epoch, patience_ref, stale = 0,0,None,0,None,0
    ledger = []; best_state = None; best_prob = None; best_history = None
    if resume.exists():
        saved = torch.load(resume,map_location='cpu',weights_only=False)
        C.require(saved['identity']==identity and saved['registration']==registration and saved['input_binding']==binding
                  and saved['device']==device, 'Resume identity/source/backend changed')
        model.load_state_dict(saved['state_dict'],strict=True); optimizer.load_state_dict(saved['optimizer'])
        restore_history(model,saved['quality_history']); C.restore_rng(saved['rng'])
        epoch,updates,best_nll,best_epoch,patience_ref,stale = (saved[k] for k in
          ('epoch','updates','best_nll','best_epoch','patience_ref','stale'))
        ledger = saved['ledger']; initial_sha = saved['initial_sha']
        best_state,best_prob,best_history = (saved[k] for k in ('best_state','best_prob','best_history'))
        C.save(directory / f'recovery_epoch_{epoch}_{time.time_ns()}.json',
               {'at':C.now(),'latest':C.desc(resume),'restored_epoch':epoch,'restored_updates':updates,
                'unknown_uncommitted_tail_updates':None,'native_tree':C.own_tree()})

    def payload():
        return {'schema':'fusion_eeg_grid_v2_1','identity':identity,'registration':registration,
          'input_binding':binding,'device':device,'precision':'FP32','epoch':epoch,'updates':updates,
          'state_dict':C.cpu_state(model.state_dict()),'optimizer':C.cpu_state(optimizer.state_dict()),
          'scheduler':{'kind':'absolute_epoch_warmup_cosine','epoch':epoch,'next_lr':lr_for_epoch(min(120,epoch+1))},
          'rng':C.rng_state(),'quality_history':history_state(model),'best_nll':best_nll,
          'best_epoch':best_epoch,'patience_ref':patience_ref,'stale':stale,'initial_sha':initial_sha,
          'ledger':ledger,'best_state':best_state,'best_prob':best_prob,'best_history':best_history,
          'sampling':'whole epoch checkpoint; next randperm from restored device RNG'}

    if not resume.exists():
        C.save_torch(resume,payload())
    try:
        while epoch < 120 and not (epoch >= 60 and stale >= 60):
            C.check_pause()
            epoch += 1
            record = train_epoch(model,optimizer,x[train],y[train],epoch)
            updates += record['updates']
            t = time.perf_counter(); model.eval()
            with torch.inference_mode():
                probability = model(x[val])['p']
                C.require(bool(torch.isfinite(probability).all()), 'Nonfinite validation probability')
                vnll = float(nll(probability,y[val]))
                vp = probability.cpu().clone()
            record.update(val_nll=vnll,validation_seconds=time.perf_counter()-t, cumulative_updates=updates)
            if best_nll is None or vnll < best_nll:
                best_nll,best_epoch = vnll,epoch
                best_state = C.cpu_state(model.state_dict()); best_prob=vp; best_history=history_state(model)
            if patience_ref is None or vnll <= patience_ref-1e-4:
                patience_ref,stale = vnll,0
            else:
                stale += 1
            record.update(best_epoch=best_epoch,stale=stale)
            ledger.append(record)
            t = time.perf_counter()
            current = payload()
            C.save_torch(resume,current)
            if best_epoch == epoch:
                C.save_torch(directory/'best.pt',current)
            if epoch in (60,120):
                C.save_torch(directory/f'epoch_{epoch}.pt',current)
            record['write_seconds']=time.perf_counter()-t
            C.save(directory/'progress.json',{'identity':identity,'epoch':epoch,'updates':updates,
                      'best_epoch':best_epoch,'val_nll':vnll,'at':C.now(),'status':'training'})
        # Repair best-file transaction using the selected state embedded in latest.
        # Commit the measured last-epoch write time once before final acceptance.
        # On a hard interruption a missing final write duration remains unknown.
        C.save_torch(resume, payload())
        bestpath = directory/'best.pt'
        best = torch.load(bestpath,map_location='cpu',weights_only=False) if bestpath.exists() else None
        if best is None or best['epoch'] != best_epoch or not equal_state(best['state_dict'],best_state):
            recovered=payload(); recovered['state_dict']=best_state; recovered['epoch']=best_epoch
            recovered['quality_history']=best_history; recovered['recovered_selected_state_only']=True
            C.save_torch(bestpath,recovered)
        selected = Fusion(identity['method'],x.shape[1],len(train)).to(device)
        selected.load_state_dict(best_state,strict=True); restore_history(selected,best_history); selected.eval()
        with torch.inference_mode():
            restored_probability = selected(x[val])['p'].cpu()
        C.require(torch.equal(restored_probability,best_prob),'Best checkpoint validation replay differs')
        terminal_sha=C.state_sha(model.state_dict())
        C.require(terminal_sha!=initial_sha and updates==len(ledger)*math.ceil(len(train)/8), 'No parameter change or update mismatch')
        C.save(directory/'trajectory.json',ledger)
        metadata={'identity':identity,'input_binding':binding,'registration':registration,
          'device':device,'parameters':sum(p.numel() for p in model.parameters()),'epochs':epoch,'updates':updates,
          'best_epoch':best_epoch,'best_val_nll':best_nll,'initial_state_sha256':initial_sha,
          'terminal_state_sha256':terminal_sha,'best_state_sha256':C.state_sha(best_state),
          'train_seconds':sum(r['train_seconds'] for r in ledger),'validation_seconds':sum(r['validation_seconds'] for r in ledger),
          'durable_write_seconds':sum(r['write_seconds'] for r in ledger) if all('write_seconds' in r for r in ledger) else None,
          'stopping_reason':'validation_patience' if stale>=60 else 'max_120_epochs',
          'validation_replay_bitwise':True,'test_forwards':0,'batch8_last_kept':True,
          'precision':'FP32','resume_uncommitted_tail_is_unknown':True}
        C.save(directory/'metadata.json',metadata)
        value={'identity':identity,'passed':True,'epochs':epoch,'updates':updates,'best_epoch':best_epoch,
          'parameter_change_verified':True,'test_forwards':0,'validation_replay_bitwise':True,
          'metadata':C.desc(directory/'metadata.json'),'trajectory':C.desc(directory/'trajectory.json'),
          'best':C.desc(bestpath),'latest':C.desc(resume),'finished_at':C.now()}
        C.save(directory/'complete.json',value,immutable=True)
        C.save(directory/'progress.json',dict(identity=identity,status='complete',epoch=epoch,updates=updates,at=C.now()))
        return value
    except BaseException as error:
        # Latest is the only resume boundary. An interrupted active epoch is unknown,
        # never attributed to the last committed epoch as zero extra work.
        C.save(directory/f'event_{time.time_ns()}.json',{'at':C.now(),'error':repr(error),'epoch_attempt':epoch,
          'latest_durable':C.desc(resume),'unknown_uncommitted_tail_updates':None,'unknown_tail_seconds':None})
        raise
