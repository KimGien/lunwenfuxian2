"""Train-only acceptance and immutable registration before any formal fit."""
from __future__ import annotations
import time
from torch.nn import functional as F
from .common import *
from .models import P1Model, R1Model, structure
from .health import audit, require_health, check_train_ids, gradient_norms
from tridep_ccml.eeg_replacement.data import FoldData


def tiny_fit(seed,fold,data,save=True):
    path = OUT/f'preflight/tiny_fit/seed_{seed}_fold_{fold}.json'
    if save and path.exists():
        result = read_json(path)
        require(result['encoder_source']==descriptor(CODE/'models.py') and result['gates']==GATES,
                'Stale tiny-fit result')
        require(result['passed'],'Previous tiny-fit failed; inspect before continuing')
        return result
    ids = sum([[s for s in data.splits['train'] if data.context.labels[s]==label][:2] for label in [0,1]],[])
    check_train_ids(data,ids)
    require(len(ids)==4,'Need 2 HC and 2 MDD training subjects')
    set_seed(seed)
    model = P1Model('U2',seed).cuda().train()
    initial = state_digest(model.state_dict())
    optimizer = torch.optim.Adamax(model.parameters(),lr=GATES['small_fit_lr'],weight_decay=0.)
    inputs,labels = data.batch(ids,'U2')
    # Fixed deterministic transform, no learned inputs and no augmentation.
    spectrum = model.eeg.spectrum(inputs['eeg']).detach()
    with torch.no_grad():
        torch.testing.assert_close(model.eeg.from_spectrum(spectrum),model.eeg(inputs['eeg']),rtol=0,atol=0)
    history = []
    start = time.perf_counter()
    for step in range(1,GATES['small_fit_steps']+1):
        optimizer.zero_grad(set_to_none=True)
        logits = model.classifier(model.eeg.from_spectrum(spectrum))
        loss = F.cross_entropy(logits,labels)
        require(torch.isfinite(loss).item(),'Tiny-fit nonfinite loss')
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters()),
                'Tiny-fit missing/nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        history.append({'step':step,'pre_update_ce':float(loss.detach())})
    model.eval()
    with torch.inference_mode():
        logits = model(inputs)
        ce = float(F.cross_entropy(logits,labels))
        accuracy = float((logits.argmax(-1)==labels).float().mean())
    health = audit(model,data,ids)
    result = {'seed':seed,'fold':fold,'subject_ids':ids,'labels':labels.tolist(),
        'encoder_source':descriptor(CODE/'models.py'),'gates':GATES,'updates':GATES['small_fit_steps'],
        'initial_state_sha256':initial,'final_ce':ce,'final_accuracy':accuracy,'health':health,
        'passed':ce<=GATES['small_fit_ce_max'] and accuracy>=GATES['small_fit_accuracy_min'] and health['passed'],
        'history':history,'seconds':time.perf_counter()-start,'augmentation':False,
        'diagnostic_only_model_discarded':True,'warmstart_formal_training':False}
    if save:
        write_new(path,result)
    print(f'TINY seed={seed} fold={fold} CE={ce:.6f} acc={accuracy:.3f} passed={result["passed"]}',flush=True)
    del model,optimizer,spectrum
    torch.cuda.empty_cache()
    require(result['passed'],'Tiny-fit acceptance failed: '+str((seed,fold,ce,accuracy,health['failures'])))
    return result


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    history_count = check_history()
    spec = structure()
    require(all(spec['parameters'][c]['eeg_including_head']<50000 for c in ['U2','C2']),'P1 EEG parameter budget')
    write_new(OUT/'structure_manifest.json',spec)
    if not (OUT/'registration.json').exists():
        write_new(OUT/'registration.json',{'plan':descriptor(PLAN),'structure':descriptor(OUT/'structure_manifest.json'),
        'gates':GATES,'conditions':CONDITIONS,'seeds':SEEDS,'folds':FOLDS,'augmentation':False,
        'formal_epochs':60,'formal_updates':14400,'diagnostic_updates':7500,'registered_utc':now()})
    tiny,initial,negative = [],[],[]
    for fold in FOLDS:
        data = FoldData(fold)
        set_seed(11)
        old = R1Model('C2',11).cuda().eval()
        cp = torch.load(PARENT/f'runs/C2/seed_11/fold_{fold}/checkpoint.pt',map_location='cpu',weights_only=True)
        old.load_state_dict(cp['model'],strict=True)
        result = audit(old,data,epoch=60)
        require(not result['passed'],'Old known-dead encoder was incorrectly accepted')
        result.update({'fold':fold,'checkpoint':descriptor(PARENT/f'runs/C2/seed_11/fold_{fold}/checkpoint.pt')})
        write_new(OUT/f'preflight/negative/fold_{fold}.json',result)
        negative.append(result)
        print(f'NEGATIVE fold={fold} correctly rejected ({len(result["failures"])} failures)',flush=True)
        del old,cp
        for seed in SEEDS:
            for condition in CONDITIONS:
                set_seed(seed)
                model = P1Model(condition,seed).cuda()
                result = audit(model,data)
                result.update({'condition':condition,'seed':seed,'fold':fold})
                write_new(OUT/f'preflight/initial/{condition}_seed_{seed}_fold_{fold}.json',result)
                require_health(result)
                initial.append(result)
                del model
            tiny.append(tiny_fit(seed,fold,data))
            replace_json(OUT/'status.json',{'state':'preflight','tiny_fits_complete':len(tiny),
                'initial_models_accepted':len(initial),'negative_models_rejected':len(negative),
                'outer_metrics_released':False,'updated_utc':now()})
        del data
        torch.cuda.empty_cache()
    import unittest
    suite = unittest.defaultTestLoader.loadTestsFromName('tridep_ccml.eeg_p1.test_contracts')
    tests = unittest.TextTestRunner(verbosity=2).run(suite)
    require(tests.wasSuccessful(),'P1 regression tests failed')
    write_new(OUT/'preflight.json',{'passed':True,'initial_models':len(initial),'tiny_fit_models':len(tiny),
        'negative_models_rejected':len(negative),'tiny_fit_updates':sum(t['updates'] for t in tiny),
        'tiny_fit_ce_max':max(t['final_ce'] for t in tiny),'tiny_fit_accuracy_min':min(t['final_accuracy'] for t in tiny),
        'tests_passed':tests.testsRun,'historical_files_unchanged':history_count,
        'details':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'preflight').rglob('*.json'))}})
    write_new(OUT/'configuration.json',{'experiment':'EEG-P1-37-fulltext','version':1,
        'conditions':CONDITIONS,'seeds':SEEDS,'folds':FOLDS,'epochs':60,'batch_size_subjects':8,
        'windows_per_subject':60,'augmentation':False,'gates':GATES,'formal_runs':60,'formal_updates':14400,
        'U_optimizer':{'name':'Adamax','lr':.001,'weight_decay':.0001},
        'C_optimizer':{'name':'Adamax','encoder_lr':.0001,'head_lr':.001,'weight_decay':0.},
        'loss':LOSS,'gradient_clip':1.,'threshold_MDD':'>0.5','checkpoint':'fixed epoch 60',
        'parent_configuration':descriptor(PARENT/'configuration.json'),
        'data_manifest':descriptor(PARENT/'data_manifest.json'),'preflight':descriptor(OUT/'preflight.json'),
        'structure_manifest':descriptor(OUT/'structure_manifest.json'),'sources':sources(),
        'cuda_device':torch.cuda.get_device_name(),'torch_version':str(torch.__version__),
        'registered_plan':descriptor(PLAN),'frozen_utc':now()})
    verify_lock()
    print('PREFLIGHT PASSED; configuration frozen; formal training may start.',flush=True)


if __name__=='__main__':
    prepare()
