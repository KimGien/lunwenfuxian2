"""v2 compatibility: audit revised historical plan separately; exact models unchanged."""
from __future__ import annotations
import gc
from pathlib import Path
import time
import numpy as np
import torch
from . import common as C

def source_registration():
    if (C.OUT/'source_registration.json').exists():
        result=C.read(C.OUT/'source_registration.json')
        for source in result['sources']:
            C.checked(source)
        return result
    sources=[]; units=[]
    from tridep_ccml.eeg_all_candidates import registry as R
    registration=R.load_registration(verify_sources=False)
    historical_documents=[]
    for key,source in registration['sources'].items():
        if key=='plan' and C.sha(C.ROOT/source['path'])!=source['sha256']:
            historical_documents.append({'role':key,'original_registration_binding':source,
              'current_document':C.desc(C.ROOT/source['path']),
              'reason':'human retrospective plan/retention revision after sealed training; no current model/input waiver'})
        else:
            C.checked(source);sources.append(source)
    C.require(C.read(C.OLD/'evaluation/complete.json')['passed'],'EEG source final evaluation incomplete')
    for seed in C.SEEDS:
        for fold in range(1,6):
            st=C.STAGE0/f'features/seed_{seed}/fold_{fold}.npz'
            sm=C.read(st.with_suffix('.json'))
            C.require(sm['seed']==seed and sm['fold']==fold and sm['frozen_eval'],'Stage0 source mismatch')
            sources.extend([C.desc(st),C.desc(st.with_suffix('.json'))])
            for item in sm['parent_checkpoints'].values():
                C.checked(item); sources.append(item)
            for eeg,condition in C.EEG.items():
                if eeg=='E00':
                    continue
                snap='epoch_300' if eeg=='E15' else 'best'
                path=C.OLD/'evaluation'/condition/f'seed_{seed}'/f'fold_{fold}'/snap/'complete.json'
                comp=C.read(path); b=comp['binding']
                C.require(comp['passed'] and b['condition']==condition and b['seed']==seed and b['fold']==fold,
                          'EEG source evaluation identity mismatch')
                C.require(str(b['snapshot'])==('300' if eeg=='E15' else 'best'),'EEG snapshot mismatch')
                for key in ('checkpoint','metadata','completion','condition_source_lock','registration'):
                    C.checked(b[key]); sources.append(b[key])
                sources.append(C.desc(path))
                meta=C.read(C.ROOT/b['metadata']['path'])
                C.require(meta['train_ids']==registration['folds'][f'fold_{fold}']['train'] and
                          meta['val_ids']==registration['folds'][f'fold_{fold}']['val'],'Source fold/roles mismatch')
                units.append({'eeg':eeg,'condition':condition,'seed':seed,'fold':fold,
                              'completion':C.desc(path),'checkpoint':b['checkpoint'],'metadata':b['metadata']})
    sources.extend([C.desc(C.OLD/'registration.json'),C.desc(C.OLD/'evaluation/complete.json'),
                    C.desc(C.STAGE0/'branch_probabilities.jsonl')])
    unique={r['path']:r for r in sources}
    result={'schema':'fusion_EEG15_sources_v2','EEG':C.EEG,'units':units,'sources':list(unique.values()),
            'seeds':list(C.SEEDS),'folds':registration['folds'],'base_models_new_fits':0,
            'historical_document_revisions':historical_documents,'compatibility_source':C.desc(__file__),
            'original_registration_byte_bound':C.desc(C.OLD/'registration.json'),'created_at':C.now()}
    C.save(C.OUT/'source_registration.json',result,immutable=True)
    return result

def roles_from_splits(ids,splits,folds,fold):
    roles=folds[f'fold_{fold}']
    C.require(set(roles)=={'train','val','test'},'Unexpected role keys')
    C.require(sum(map(len,roles.values()))==37 and len(set().union(*map(set,roles.values())))==37,
              'Source role overlaps or missing people')
    byid=dict(zip(ids.tolist(),splits.tolist()))
    C.require(all(byid[sid]==role for role,people in roles.items() for sid in people),'Stage0/EEG roles differ')
    return roles

def export_stage0(reg):
    branch={(r['seed'],r.get('fold',r.get('outer_fold')),r['subject_id']):r for r in
             C.rows(C.STAGE0/'branch_probabilities.jsonl')}
    count=0
    for seed in C.SEEDS:
        for fold in range(1,6):
            C.check_pause()
            src=C.STAGE0/f'features/seed_{seed}/fold_{fold}.npz'
            sm=C.read(src.with_suffix('.json'))
            with np.load(src,allow_pickle=False) as z:
                features,ids,labels,splits=(z[k].copy() for k in ('features','subject_ids','labels','splits'))
            roles=roles_from_splits(ids,splits,reg['folds'],fold)
            replay=[]
            for i,modality in enumerate(('eeg','speech','text')):
                saved=torch.load(C.checked(sm['parent_checkpoints'][modality]),map_location='cpu',weights_only=False)
                last=max((int(k.split('.')[1]),k) for k in saved['state_dict'] if k.startswith('classifier.') and k.endswith('.weight'))[1]
                prefix=last[:-6]; weight=saved['state_dict'][last]; bias=saved['state_dict'][prefix+'bias']
                C.require(weight.shape==(2,128),'Stage0 final Linear shape differs')
                h=torch.as_tensor(features[:,i]); logits=torch.nn.functional.linear(h,weight,bias)
                pmdd=logits.double().softmax(-1)[:,1].numpy()
                expected=np.asarray([branch[(seed,fold,sid)]['p_'+modality] for sid in ids.tolist()])
                error=float(np.max(np.abs(pmdd-expected)))
                C.require(error<=2e-6,'Stage0 final classifier replay exceeds preregistered tolerance')
                replay.append({'modality':modality,'max_probability_error':error,'atol':2e-6,
                               'checkpoint':sm['parent_checkpoints'][modality]})
            for target,array in [('ST',features[:,1:]),('E00',features[:,0])]:
                p=C.OUT/'features'/target/f'seed_{seed}'/f'fold_{fold}.npz'
                if p.with_suffix('.json').exists():
                    C.checked(C.read(p.with_suffix('.json'))['cache']);count+=1;continue
                C.save_npz(p,features=array,subject_ids=ids,labels=labels,splits=splits)
                C.save(p.with_suffix('.json'),{'eeg':target,'seed':seed,'fold':fold,'roles':roles,
                  'source':C.desc(src),'source_metadata':C.desc(src.with_suffix('.json')),'cache':C.desc(p),
                  'parent_checkpoints':sm['parent_checkpoints'],'original_128_cache_reused':True,
                  'final_linear_replay':replay,'base_model_updates':0,'created_at':C.now()},immutable=True)
                count+=1
    return count

def export_eeg(unit,reg):
    eeg,condition,seed,fold=(unit[k] for k in ('eeg','condition','seed','fold'))
    target=C.OUT/'features'/eeg/f'seed_{seed}'/f'fold_{fold}.npz'
    if target.with_suffix('.json').exists():
        meta=C.read(target.with_suffix('.json'));C.checked(meta['cache'])
        C.require(meta['source_unit']==unit and meta['replay_passed'],'Existing feature cache source differs')
        return meta
    C.check_pause()
    start=time.perf_counter()
    from tridep_ccml.eeg_all_candidates import evaluate as E
    from tridep_ccml.eeg_all_candidates import registry as R, storage as S
    from tridep_ccml.tridep_stage1 import common as PARENT
    comp=C.read(C.checked(unit['completion'])); b=comp['binding']
    metadata=C.read(C.checked(unit['metadata']));registration=R.load_registration(verify_sources=False)
    split=reg['folds'][f'fold_{fold}']
    plan=E._input_plan(condition,metadata,registration)
    # Source hash verification occurs at admission, not inside head-training loops.
    for source in plan['sources']:
        S.checked(source)
    if condition in ('EEGConformer-scratch','EEG2Rep-supervised'):
        model=materialize_supervised(condition,seed,fold,split,metadata,unit['checkpoint'])
    else:
        model=E._materialize_predictor(condition,seed,fold,split,metadata,unit['checkpoint'],plan,
                                        300 if eeg=='E15' else 'best')
    # The parent runtime is intentionally identical to the sealed inference source.
    C.require(PARENT.state_digest(model.state_dict())==b['selected_model_state_sha256'],
              'Loaded parent state differs from sealed evaluation')
    if hasattr(model,'head'):
        linear=model.head.classifier
    elif isinstance(getattr(model,'classifier',None),torch.nn.Linear):
        linear=model.classifier
    else:
        linear=model.classifier[-1]
    C.require(isinstance(linear,torch.nn.Linear) and linear.in_features==128 and linear.out_features==2,
              'Wrong final 128-dimensional endpoint')
    reference={}
    for batch in comp['batches']:
        for r in C.read(C.checked(batch))['rows']:
            C.require(r['subject_id'] not in reference,'Duplicate sealed EEG prediction')
            reference[r['subject_id']]=r
    ids=b['ordered_cohort'];captured=[]
    handle=linear.register_forward_pre_hook(lambda _m,args:captured.append(args[0].detach().cpu().clone()))
    features=[];max_error=0.;max_reconnect=0.;forwards=0
    try:
        with torch.inference_mode():
            for sid in ids:
                C.check_pause()
                raw=E._load_array(plan,sid)
                value=raw[None] if plan['kind']=='raw' else torch.as_tensor(raw[None],device='cuda')
                before=len(captured);logits=model(value)
                C.require(len(captured)==before+1 and captured[-1].shape==(1,128),'Endpoint capture differs')
                reconnect=linear(captured[-1].to('cuda'))
                error=float((logits.double().cpu()-torch.tensor(reference[sid]['logits'],dtype=torch.float64)).abs().max())
                reconnect_error=float((reconnect-logits).abs().max())
                C.require(error<=2e-5 and reconnect_error==0.,'Frozen EEG logit replay exceeds locked tolerance')
                max_error=max(max_error,error);max_reconnect=max(max_reconnect,reconnect_error)
                features.append(captured.pop().numpy()[0]);forwards+=1
    finally:
        handle.remove()
    C.require(PARENT.state_digest(model.state_dict())==b['selected_model_state_sha256'] and
              all(p.grad is None for p in model.parameters()),'Parent changed during feature export')
    original=C.OUT/'features/ST'/f'seed_{seed}'/f'fold_{fold}.npz'
    with np.load(original,allow_pickle=False) as z:
        ix={sid:i for i,sid in enumerate(ids)}
        features=np.asarray([features[ix[sid]] for sid in z['subject_ids'].tolist()],dtype=np.float32)
        C.require(features.shape==(37,128) and np.isfinite(features).all(),'EEG feature export invalid')
        C.save_npz(target,features=features,subject_ids=z['subject_ids'],labels=z['labels'],splits=z['splits'])
    meta={'eeg':eeg,'condition':condition,'seed':seed,'fold':fold,'source_unit':unit,'cache':C.desc(target),
      'roles':split,'input_sources':plan['sources'],'endpoint':'same complete model final Linear input',
      'pooling':'original parent forward; per-window learned nonlinear head before original all60 mean',
      'replay_passed':True,'replay_atol':2e-5,'max_logit_error':max_error,'max_reconnect_error':max_reconnect,
      'parent_state_unchanged':True,'base_model_updates':0,'subjects_forwarded':forwards,
      'selected_checkpoint_phase':metadata.get('selected_checkpoint_phase','not_applicable'),
      'selected_encoder_parameter_change_count':metadata.get('selected_encoder_parameter_change_count'),
      'device':'cuda','precision':'FP32','window_chunk':metadata.get('window_chunk'),
      'seconds':time.perf_counter()-start,'created_at':C.now(),
      'batch_optimization':'not enabled; original per-person and original window chunk retained'}
    C.save(target.with_suffix('.json'),meta,immutable=True)
    del model;gc.collect();torch.cuda.empty_cache()
    return meta

def materialize_supervised(condition,seed,fold,split,metadata,checkpoint):
    # Build the original complete architecture, then strictly load the sealed
    # supervised state. No SSL training or old fit-status gate is re-executed.
    from tridep_ccml.eeg_all_candidates import supervised as U, finetune as F, registry as R
    from tridep_ccml.eeg_all_candidates.adapters import eegconformer as EC, eeg2rep as ER
    from tridep_ccml.tridep_stage1 import common as PARENT
    PARENT.runtime(seed)
    saved=torch.load(C.checked(checkpoint),map_location='cpu',weights_only=False)
    C.require(saved['identity']=={k:metadata[k] for k in saved['identity']} and
              saved['best_epoch']==metadata['best_epoch'],'Supervised checkpoint identity differs')
    if condition=='EEGConformer-scratch':
        statistics=saved['statistics'];bound=C.read(C.checked(metadata['statistics']))
        C.require(bound['statistics']==statistics and statistics['fit_subject_ids']==split['train'] and
                  bound['validation_or_test_fit_people']==[],'Original Conformer training statistics differ')
        encoder,audit=EC.load_encoder(device='cuda',trainable=True);adapter=EC
    else:
        C.require(saved['statistics'] is None,'EEG2Rep unexpected fitted statistics')
        statistics=None;ssl=metadata['ssl_binding']
        C.require(ssl['seed']==seed and ssl['fold']==fold and ssl['train_ids']==split['train'] and
                  ssl['val_ids']==split['val'],'Same-fold EEG2Rep SSL provenance differs')
        C.checked(ssl['checkpoint']);C.checked(ssl['metadata']);C.checked(ssl['completion'])
        sslmeta=C.read(C.ROOT/ssl['metadata']['path'])
        C.require(sslmeta['online_backbone_update_verified'] and sslmeta['online_changed_parameter_count']>0,
                  'Sealed SSL encoder lacks real update evidence')
        encoder,audit=ER.build_model(device='cuda',stage='supervised');adapter=ER
    head=F.SubjectBagHead(audit['feature_dim']).to('cuda')
    selected=[name for name,p in encoder.named_parameters() if p.requires_grad]
    model=U.SupervisedModel(encoder,head,adapter,selected,statistics,metadata['window_chunk'])
    model.load_state_dict(saved['state_dict'],strict=True)
    C.require(PARENT.state_digest(model.state_dict())==metadata['selected_state_sha256'],
              'Strict supervised model state differs from source')
    return model.requires_grad_(False).eval()

def export_all():
    reg=source_registration();export_stage0(reg)
    for n,unit in enumerate(reg['units'],1):
        C.save(C.OUT/'status.json',{'phase':'feature_export','unit':unit,'completed_eeg_caches':n-1,
                                'total_eeg_caches':210,'formal_fits_completed':0,'at':C.now()})
        meta=export_eeg(unit,reg)
        print(f"Feature {n}/210 {unit['eeg']} seed{unit['seed']} fold{unit['fold']} replay={meta['max_logit_error']}",flush=True)
    files=[]
    for name in ('ST',*C.EEG):
        for seed in C.SEEDS:
            for fold in range(1,6):
                p=C.OUT/'features'/name/f'seed_{seed}'/f'fold_{fold}.json'
                m=C.read(p);C.checked(m['cache']);files.extend([C.desc(p),m['cache']])
    result={'passed':True,'bundles':240,'ST':15,'EEG':225,'sources':files,'base_model_updates':0,
            'source_registration':C.desc(C.OUT/'source_registration.json'),'at':C.now()}
    C.save(C.OUT/'feature_gate.json',result,immutable=True)
    return result
