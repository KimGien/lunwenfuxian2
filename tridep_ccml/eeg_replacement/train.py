"""Fixed 60-epoch CE/CCML training. No outer labels enter optimization."""
from __future__ import annotations
import argparse
import time
from .common import *
from .models import R1Model
from .data import FoldData
from tridep_ccml.ccml import ccml_loss
from torch.nn import functional as F


def orders(ids, seed, epochs=60):
    generator = torch.Generator().manual_seed(seed)
    return [[ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()] for _ in range(epochs)]


def step(model, optimizer, inputs, labels, epoch):
    require(model.training, 'Training requires train mode')
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = ccml_loss(output, labels, epoch, **LOSS) if model.joint else F.cross_entropy(output, labels)
    require(torch.isfinite(loss).item(), 'Nonfinite loss; pause entire paired stage')
    loss.backward()
    require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
            'Missing/nonfinite trainable gradient')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    model.constrain()
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite updated parameter')
    return float(loss.detach()), float(norm)


def train_one(condition, seed, fold, data):
    cfg = verify_lock()
    folder = run_dir(condition, seed, fold)
    done = folder/'trained.json'
    if done.exists():
        row = read_json(done)
        require(row['configuration'] == descriptor(OUT/'configuration.json'), 'Stale trained model')
        for key in ['checkpoint', 'history', 'batch_order']:
            verify_descriptor(row[key])
        print(f'REUSE {condition} seed={seed} fold={fold}', flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    require(not (folder/'checkpoint.pt').exists(), 'Unmarked checkpoint requires investigation before rerun')
    set_seed(seed)
    model = R1Model(condition, seed).cuda().train()
    initial = state_digest(model.state_dict())
    shared = state_digest(model.shared_state()) if model.joint else None
    optimizer = model.optimizer()
    ids = data.splits['train']
    epoch_orders = orders(ids, seed)
    write_new(folder/'batch_order.json', {'train_ids': ids, 'seed': seed, 'epochs':epoch_orders,
                                        'batch_size':8, 'drop_last':False})
    history, updates = [], 0
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch, order in enumerate(epoch_orders, 1):
        model.train()
        total, norms = 0., []
        epoch_start = time.perf_counter()
        for offset in range(0, len(order), 8):
            batch_ids = order[offset:offset+8]
            require(set(batch_ids) <= set(ids), 'Non-train subject in optimization')
            inputs, labels = data.batch(batch_ids, condition)
            loss, norm = step(model, optimizer, inputs, labels, epoch)
            total += loss*len(batch_ids)
            norms.append(norm)
            updates += 1
        torch.cuda.synchronize()
        row = {'epoch':epoch, 'loss':total/len(ids), 'max_grad_norm_before_clip':max(norms),
               'seconds':time.perf_counter()-epoch_start}
        if condition in ['U1','C1']:
            row['spatial_filter_max_norm'] = float(model.eeg.spatial.weight.flatten(1).norm(dim=1).max())
        if condition == 'U1':
            row['classifier_max_norm'] = float(model.classifier.weight.norm(dim=1).max())
        history.append(row)
        replace_json(OUT/'status.json', {'state':'training', 'condition':condition, 'seed':seed,
            'fold':fold, 'epoch':epoch, 'updates_this_run':updates, 'updated_utc':now(),
            'completed_runs':len(list((OUT/'runs').glob('*/seed_*/fold_*/trained.json'))),
            'planned_runs':120, 'augmentation':False, 'outer_metrics_released':False})
        if epoch in [1, 20, 40, 60]:
            print(f'TRAIN {condition} seed={seed} fold={fold} epoch={epoch}/60 loss={row["loss"]:.6f}', flush=True)
    require(updates == 240 and state_digest(model.state_dict()) != initial, 'Invalid training update count/state')
    model.eval()
    probe_ids = ids[:8]
    probe_inputs, _ = data.batch(probe_ids, condition)
    with torch.inference_mode():
        probe_output = model(probe_inputs).cpu()
        probe_features = model.representations(probe_inputs).cpu()
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    checkpoint = {'condition':condition, 'seed':seed, 'fold':fold, 'epoch':60, 'updates':updates,
        'model':state, 'optimizer':optimizer.state_dict(), 'train_ids':ids,
        'train_probe':{'subject_ids':probe_ids,'output':probe_output,'features':probe_features},
        'initial_state_sha256':initial, 'initial_shared_sha256':shared,
        'final_state_sha256':state_digest(state), 'configuration':descriptor(OUT/'configuration.json'),
        'standardizer':data.manifest['standardizers'][str(fold)] if condition in ['U1','C1','U2','C2'] else None,
        'rng_cpu':torch.get_rng_state(), 'rng_cuda':torch.cuda.get_rng_state(), 'augmentation':False}
    temp = folder/'checkpoint.pt.part'
    torch.save(checkpoint, temp)
    temp.replace(folder/'checkpoint.pt')
    write_new(folder/'history.json', history)
    write_new(done, {'condition':condition, 'seed':seed, 'fold':fold, 'epoch':60, 'updates':updates,
        'configuration':descriptor(OUT/'configuration.json'), 'checkpoint':descriptor(folder/'checkpoint.pt'),
        'history':descriptor(folder/'history.json'), 'batch_order':descriptor(folder/'batch_order.json'),
        'initial_state_sha256':initial, 'initial_shared_sha256':shared,
        'final_state_sha256':state_digest(state), 'seconds':time.perf_counter()-start,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'parameters':model.counts(), 'finished_utc':now()})
    print(f'DONE {condition} seed={seed} fold={fold} seconds={time.perf_counter()-start:.1f}', flush=True)
    del model, optimizer, checkpoint, state
    torch.cuda.empty_cache()


def train_all():
    verify_lock()
    # No outer inference or scoring here. All 120 fits are finished before release.
    for conditions in STAGES:
        for seed in SEEDS:
            for fold in FOLDS:
                data = FoldData(fold)
                for condition in conditions:
                    train_one(condition, seed, fold, data)
                del data
                torch.cuda.empty_cache()
    replace_json(OUT/'status.json', {'state':'trained_pending_verification', 'completed_runs':120,
        'planned_runs':120, 'augmentation':False, 'outer_metrics_released':False, 'updated_utc':now()})


if __name__ == '__main__':
    train_all()
