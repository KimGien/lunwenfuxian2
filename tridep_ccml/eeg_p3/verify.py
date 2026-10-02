"""Independent post-run identity, arithmetic and frozen-state audit."""
from __future__ import annotations
import math
from .common import *
from .methods import validate_oof,apply_sigmoid

def verify():
    cfg=lock_check(); context=RecordContext()
    gate=read_json(OUT/'release_gate.json')
    require(gate['passed'],'Missing release gate')
    for name,sha in gate['fitted_artifacts'].items():
        require(digest(OUT/name)==sha,'Fitted artifact changed after release: '+name)
    healths=[]; training_seconds=0.; outer=0; inner=0
    for path in sorted((OUT/'runs').rglob('trained.json')):
        info=read_json(path); c=info['condition']; fold=info['fold']; seed=info['seed']
        allowed=context.folds[f'fold_{fold}']['train']; fit=info['fit_ids']
        expected=allowed if info['inner'] is None else cfg['inner_splits'][str(fold)][info['inner']-1]['fit']
        require(fit==expected and set(fit)<=set(allowed),'Wrong fitting identities')
        require(info['updates']==60*math.ceil(len(fit)/8),'Wrong training updates')
        cp=torch.load(verify_descriptor(info['checkpoint']),map_location='cpu',weights_only=True)
        require(cp['fit_ids']==fit and cp['state_sha256']==state_digest(cp['model']),'Checkpoint identity/state failed')
        order=read_json(path.parent/'batch_order.json')
        require(len(order['orders'])==60 and all(sorted(o)==sorted(fit) for o in order['orders']),'Invalid batch coverage')
        for name in ['initial_health.json','final_health.json']:
            h=read_json(path.parent/name)
            require(h['passed'] and h['train_ids']==fit,'Learning gate failed')
            healths.append(h)
        training_seconds+=info['seconds']
        outer+=info['inner'] is None;inner+=info['inner'] is not None
    derived=0
    for path in sorted((OUT/'runs').rglob('derived.json')):
        info=read_json(path)
        original=torch.load(verify_descriptor(info['source']),map_location='cpu',weights_only=True)['model']
        target=torch.load(verify_descriptor(info['checkpoint']),map_location='cpu',weights_only=True)['model']
        changed=[k for k in original if not torch.equal(original[k],target[k])]
        require(bool(changed) and all(k.endswith(('running_mean','running_var')) for k in changed),
                'BN derivation changed learned parameters or no statistics')
        bn=read_json(verify_descriptor(info['bn_fit']))
        require(len(bn['layers'])==3 and bn['learned_parameters_unchanged'],'Missing BN layers')
        for layer in bn['layers']:
            require(layer['count_per_channel']>0 and min(layer['population_variance'])>=0,'Invalid BN moments')
        derived+=1
    calibrators=0
    for path in sorted((OUT/'calibrators').rglob('fold_*.json')):
        info=read_json(path);fold=info['fold'];allowed=context.folds[f'fold_{fold}']['train']
        require(info['train_ids']==allowed and not info['outer_val_test_used'],'Calibration leakage')
        for endpoint in ['eeg','fusion']:
            rows=info['oof'][endpoint];validate_oof(rows,allowed)
            for r in rows:
                require(r['label']==context.labels[r['subject_id']],'Wrong calibration label')
                split=cfg['inner_splits'][str(fold)][r['inner']-1]
                require(r['fit_ids']==split['fit'] and r['subject_id'] in split['calibration'],'Inner split mismatch')
                verify_descriptor(r['checkpoint'])
            for method in ['sigmoid','threshold']:
                require(info['parameters'][endpoint][method]['fit_subject_ids']==allowed,'Parameter fit IDs mismatch')
            p=info['parameters'][endpoint]['sigmoid']
            require(.01<=p['a']<=10 and -10<=p['b']<=10,'Calibration bounds changed')
            require(np.all(np.diff(apply_sigmoid(np.linspace(.001,.999,1000),p))>=0),'Calibration is not monotone')
            raw=np.clip([r['p'] for r in rows],1e-7,1-1e-7)
            labels=np.array([r['label'] for r in rows])
            x=np.log(raw)-np.log1p(-raw);z=p['a']*x+p['b']
            q=1/(1+np.exp(-z))
            objective=np.mean(np.logaddexp(0,z)-labels*z)+.005*((p['a']-1)**2+p['b']**2)
            require(abs(objective-p['objective'])<1e-10,'Independent sigmoid objective mismatch')
            grad=np.array([np.mean((q-labels)*x)+.01*(p['a']-1),np.mean(q-labels)+.01*p['b']])
            for value,g,low,high in zip([p['a'],p['b']],grad,[.01,-10],[10,10]):
                residual=min(g,0) if value<=low+1e-8 else max(g,0) if value>=high-1e-8 else g
                require(abs(residual)<1e-5,'Calibration optimality check failed')
        calibrators+=1
    predictions=0
    for path in sorted((OUT/'predictions').rglob('fold_*.json')):
        info=read_json(path);f=info['fold']
        require(len(info['rows'])==37 and sorted(r['subject_id'] for r in info['rows'])==sorted(context.ids),
                'Incomplete outer predictions')
        require(info['checkpoint_reload_passed'] and info['batch_independence_passed'] and info['inference_state_unchanged'],
                'Inference/reload audit failed')
        old=None
        if info['arm']=='B':
            require(info['baseline_replay_passed'],'Original control replay failed')
            old={r['subject_id']:r for r in read_json(R1/f'runs/{info["condition"]}/seed_{info["seed"]}/fold_{f}/predictions.json')}
        for r in info['rows']:
            require(r['subject_id'] in context.folds[f'fold_{f}'][r['split']],'Split mapping wrong')
            require(r['label']==context.labels[r['subject_id']],'Prediction label wrong')
            require(all(0<v<1 and np.isfinite(v) for v in r['probabilities'].values()),'Invalid probability')
            if old is not None:
                require(all(int(p>.5)==old[r['subject_id']]['predictions'][e] for e,p in r['probabilities'].items()),
                        'Control hard predictions differ from frozen original')
        predictions+=1
    require((outer,inner,derived,calibrators,predictions)==(60,90,60,30,180),'Incomplete experiment')
    history=history_check()
    for name,sha in read_json(P2/'completion.json')['implementation_sources'].items():
        require(digest(WORK/name)==sha,'Frozen P2 source changed: '+name)
    result={'passed':True,'new_outer_models':outer,'new_inner_models':inner,'BN_derived_states':derived,
        'calibrator_sets':calibrators,'prediction_files':predictions,'health_checks':len(healths),
        'train_subject_isolation':True,'calibration_crossfit_identity_passed':True,
        'BN_only_running_mean_variance_changed':True,'positive_sigmoid_rank_preserved':True,
        'checkpoint_reload_and_control_replay_passed':True,'fit_artifacts_unchanged_after_release':True,
        'historical_artifacts_unchanged':history,'training_seconds_including_health_checks':training_seconds,
        'min_representation_std_rms':min(h['representation_std_rms'] for h in healths),
        'min_zero_input_relative_response':min(h['zero_input_relative_response'] for h in healths),
        'min_module_gradient_l2':min(v for h in healths for v in h['module_max_gradient_l2'].values()),
        'finished_utc':now()}
    write_new(REPORT/'verification.json',result)
    print(result)

if __name__=='__main__':verify()
