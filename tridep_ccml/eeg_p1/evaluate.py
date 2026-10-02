"""Condition-wide release gate, checkpoint replay, independent probability checks."""
from __future__ import annotations
import math
from .common import *
from .models import P1Model as R1Model
from tridep_ccml.eeg_replacement.data import FoldData
from tridep_ccml.ccml import evidence_dc, dirichlet_from_evidence
from tridep_ccml.metrics import binary_metrics

METRICS = ['macro_f1','balanced_accuracy','mdd_recall','nll','brier']


def independent_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities,dtype=np.float64)
    pred = (p > .5).astype(int)
    matrix = np.zeros((2,2),dtype=int)
    np.add.at(matrix,(y,pred),1)
    tn,fp,fn,tp = matrix.ravel()
    f1_hc = 2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    f1_mdd = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    recall = tp/(tp+fn)
    return {'macro_f1':float((f1_hc+f1_mdd)/2), 'balanced_accuracy':float((tn/(tn+fp)+recall)/2),
        'mdd_recall':float(recall), 'nll':float(-np.log(np.clip(np.where(y,p,1-p),1e-15,1-1e-15)).mean()),
        'brier':float(np.square(p-y).mean()), 'confusion_matrix_HC_MDD':matrix.tolist(),
        'predicted_HC':int((pred==0).sum()), 'predicted_MDD':int(pred.sum()),
        'single_class_predictions':bool(len(np.unique(pred))==1), 'n_subjects':len(y)}


def verify_probability_row(row):
    if row['condition'].startswith('U'):
        logits = np.asarray(row['logits'],dtype=float)
        exp = np.exp(logits-logits.max())
        np.testing.assert_allclose(exp/exp.sum(),row['probabilities']['eeg'],rtol=1e-5,atol=1e-6)
        require(row['evidence'] is None, 'CE must not fabricate evidence')
    else:
        e = np.asarray(row['evidence'],dtype=float)
        require(e.shape == (3,2) and np.isfinite(e).all() and (e>=0).all(), 'Bad evidence axes')
        ps = (e+1)/(e+1).sum(-1,keepdims=True)
        for i,m in enumerate(MODALITIES):
            np.testing.assert_allclose(ps[i],row['probabilities'][m],rtol=1e-5,atol=1e-6)
        common = e.min(0)
        complementary = (e-common).mean(0)
        if common.max() > 0:
            powers = (common/common.max())**1.25
            consistent = 3*common.sum()*powers/powers.sum()
        else:
            consistent = np.zeros(2)
        final = consistent+complementary
        for k,v in [('m',common),('e_con',consistent),('e_cmp',complementary),('e_final',final)]:
            np.testing.assert_allclose(v,row[k],rtol=1e-5,atol=1e-6)
        np.testing.assert_allclose((final+1)/(final+1).sum(),row['probabilities']['fusion'],rtol=1e-5,atol=1e-6)
    for endpoint,p in row['probabilities'].items():
        require(row['predictions'][endpoint] == int(p[1]>.5), 'Threshold/tie error')


def trained_gate():
    verify_lock()
    context = RecordContext()
    shared, batches = {}, {}
    counts = {}
    for condition in CONDITIONS:
        counts[condition] = 0
        for seed in SEEDS:
            for fold in FOLDS:
                folder = run_dir(condition,seed,fold)
                meta = read_json(folder/'trained.json')
                require(meta['condition']==condition and meta['seed']==seed and meta['fold']==fold and
                        meta['updates']==240 and meta['epoch']==60, 'Incomplete formal training')
                require(meta['configuration']==descriptor(OUT/'configuration.json'), 'Training config mismatch')
                for k in ['checkpoint','history','batch_order','initial_health','final_health']:
                    verify_descriptor(meta[k])
                require(read_json(folder/'initial_health.json')['passed'] and read_json(folder/'final_health.json')['passed'],'Health gate failed')
                if condition in ['U0','C0']:
                    require(read_json(folder/'control_replay.json')['passed'],'Control replay failed')
                batch = read_json(folder/'batch_order.json')
                require(batch['train_ids']==context.folds[f'fold_{fold}']['train'] and len(batch['epochs'])==60,
                        'Wrong training cohort/batches')
                key = (seed,fold)
                previous = batches.setdefault(key,meta['batch_order']['sha256'])
                require(previous==meta['batch_order']['sha256'], 'Paired subject orders differ')
                if condition.startswith('C'):
                    previous = shared.setdefault(key,meta['initial_shared_sha256'])
                    require(previous==meta['initial_shared_sha256'], 'Paired non-EEG weights differ')
                counts[condition] += 1
    require(all(n==15 for n in counts.values()), 'All 15 models per condition required before release')
    write_new(OUT/'training_gate.json', {'passed':True,'runs':60,'updates':14400,'condition_counts':counts,
        'paired_subject_orders':True,'paired_speech_text_and_heads':True,'configuration':descriptor(OUT/'configuration.json')})


def evaluate_one(condition, seed, fold, data):
    folder = run_dir(condition,seed,fold)
    if (folder/'verified.json').exists():
        meta = read_json(folder/'verified.json')
        for k in ['predictions','features','checkpoint']:
            verify_descriptor(meta[k])
        return
    checkpoint_path = folder/'checkpoint.pt'
    cp = torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    require(cp['condition']==condition and cp['seed']==seed and cp['fold']==fold and cp['epoch']==60 and
            cp['updates']==240 and not cp['augmentation'] and cp['train_ids']==data.splits['train'], 'Checkpoint metadata mismatch')
    require(cp['configuration']==descriptor(OUT/'configuration.json') and
            state_digest(cp['model'])==cp['final_state_sha256'], 'Checkpoint state/config mismatch')
    model = R1Model(condition,seed).cuda().eval()
    model.load_state_dict(cp['model'],strict=True)
    before = state_digest(model.state_dict())
    probe = cp['train_probe']
    require(probe['subject_ids']==data.splits['train'][:8], 'Probe contains outer subjects')
    x,_ = data.batch(probe['subject_ids'],condition)
    with torch.inference_mode():
        actual = model(x).cpu()
        features = model.representations(x).cpu()
    torch.testing.assert_close(actual,probe['output'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(features,probe['features'],rtol=1e-5,atol=1e-6)
    probe_error = float((actual-probe['output']).abs().max())
    ids = data.context.ids
    outputs, features = [], []
    with torch.inference_mode():
        for start in range(0,len(ids),8):
            x,_ = data.batch(ids[start:start+8],condition)
            outputs.append(model(x).cpu())
            features.append(model.representations(x).cpu())
        output, reps = torch.cat(outputs), torch.cat(features)
        if model.joint:
            view = dirichlet_from_evidence(output)
            fused = evidence_dc(output)
        else:
            probabilities = output.softmax(-1)
    require(before==state_digest(model.state_dict()), 'Evaluation changed parameters/BN')
    split_map = {s:k for k,v in data.splits.items() for s in v}
    rows = []
    for i,sid in enumerate(ids):
        row = {'condition':condition,'line':'C' if model.joint else 'U','seed':seed,'fold':fold,
               'subject_id':sid,'label':data.context.labels[sid],'split':split_map[sid],
               'train_ids':data.splits['train'],'checkpoint_sha256':digest(checkpoint_path)}
        if model.joint:
            row.update({'logits':None,'evidence':output[i].tolist(),
                'view_alpha':view['alpha'][i].tolist(),'view_u':view['u'][i].tolist(),
                'probabilities':{m:view['p'][i,j].tolist() for j,m in enumerate(MODALITIES)},
                **{k:fused[k][i].tolist() for k in ['m','e_con','e_cmp','e_final']},
                'u_final':float(fused['u'][i])})
            row['probabilities']['fusion'] = fused['p'][i].tolist()
        else:
            row.update({'logits':output[i].tolist(),'evidence':None,'probabilities':{'eeg':probabilities[i].tolist()}})
        row['predictions'] = {k:int(p[1]>.5) for k,p in row['probabilities'].items()}
        verify_probability_row(row)
        rows.append(row)
    write_new(folder/'predictions.json',rows)
    feat_path = folder/'features_128.npz'
    arrays = {'features':reps.numpy(),'subject_ids':np.array(ids),'labels':np.array([data.context.labels[s] for s in ids]),
              'splits':np.array([split_map[s] for s in ids]),'seed':np.array(seed),'fold':np.array(fold)}
    if feat_path.exists():
        with np.load(feat_path,allow_pickle=False) as old:
            require(all(np.array_equal(old[k],v) for k,v in arrays.items()),'Stored features changed')
    else:
        with feat_path.open('xb') as f:
            np.savez(f,**arrays)
    write_new(folder/'verified.json', {'passed':True,'checkpoint':descriptor(checkpoint_path),
        'predictions':descriptor(folder/'predictions.json'),'features':descriptor(feat_path),
        'train_probe_max_abs_error':probe_error,'reload_rtol':1e-5,'reload_atol':1e-6,
        'inference_parameters_and_BN_unchanged':True,'independent_evidence_recompute':True,
        'subject_count':37,'supervised_features_bound_to_seed_fold':True})
    del model,cp
    torch.cuda.empty_cache()


def replay_all():
    """Replay all 60 train probes before the first new outer inference."""
    rows = []
    for fold in FOLDS:
        data = FoldData(fold)
        for seed in SEEDS:
            for condition in CONDITIONS:
                folder = run_dir(condition,seed,fold)
                cp = torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=True)
                require(cp['train_ids']==data.splits['train'],'Reload training identity mismatch')
                require(cp['configuration']==descriptor(OUT/'configuration.json'),'Reload config mismatch')
                require(state_digest(cp['model'])==cp['final_state_sha256'],'Reload state digest mismatch')
                model = R1Model(condition,seed).cuda().eval()
                model.load_state_dict(cp['model'],strict=True)
                probe = cp['train_probe']
                require(probe['subject_ids']==data.splits['train'][:8],'Non-train replay probe')
                inputs,_ = data.batch(probe['subject_ids'],condition)
                with torch.inference_mode():
                    output = model(inputs).cpu()
                    features = model.representations(inputs).cpu()
                torch.testing.assert_close(output,probe['output'],rtol=1e-5,atol=1e-6)
                torch.testing.assert_close(features,probe['features'],rtol=1e-5,atol=1e-6)
                rows.append({'condition':condition,'seed':seed,'fold':fold,
                    'output_max_error':float((output-probe['output']).abs().max()),
                    'feature_max_error':float((features-probe['features']).abs().max())})
                del model,cp
        del data
        torch.cuda.empty_cache()
    write_new(OUT/'replay_gate.json',{'passed':True,'runs':len(rows),'train_only':True,'details':rows})


def evaluate_all():
    trained_gate()
    replay_all()
    for seed in SEEDS:
        for fold in FOLDS:
            data = FoldData(fold)
            for condition in CONDITIONS:
                evaluate_one(condition,seed,fold,data)
            print(f'RELOAD verified all 4 conditions seed={seed} fold={fold}',flush=True)
            del data
            torch.cuda.empty_cache()


def metric(rows,endpoint):
    labels = [r['label'] for r in rows]
    ps = [r['probabilities'][endpoint][1] for r in rows]
    primary = binary_metrics(labels,ps)
    independent = independent_metrics(labels,ps)
    for key in METRICS:
        require(abs(primary[key]-independent[key])<=1e-12,'Independent metric mismatch: '+key)
    return independent


def summarize(rows):
    result = {}
    for condition in CONDITIONS:
        subset = [r for r in rows if r['condition']==condition]
        require(len(subset)==15*37,'Prediction coverage wrong')
        endpoints = ['eeg'] if condition.startswith('U') else MODALITIES+['fusion']
        result[condition] = {}
        for endpoint in endpoints:
            splits = {}
            for split in ['train','val','test']:
                per_fold, per_seed = {}, {}
                for seed in SEEDS:
                    per_fold[str(seed)] = {}
                    for fold in FOLDS:
                        selected = [r for r in subset if r['seed']==seed and r['fold']==fold and r['split']==split]
                        per_fold[str(seed)][str(fold)] = metric(selected,endpoint)
                    per_seed[str(seed)] = {k:float(np.mean([v[k] for v in per_fold[str(seed)].values()])) for k in METRICS}
                splits[split] = {'folds':per_fold,'seed_fold_means':per_seed,
                    'primary':{k:float(np.mean([v[k] for v in per_seed.values()])) for k in METRICS},
                    'seed_std_ddof1':{k:float(np.std([v[k] for v in per_seed.values()],ddof=1)) for k in METRICS}}
            pooled = {}
            for seed in SEEDS:
                selected = [r for r in subset if r['seed']==seed and r['split']=='test']
                require(len(selected)==37 and len({r['subject_id'] for r in selected})==37,'Repeated/missing OOF subject')
                pooled[str(seed)] = metric(selected,endpoint)
            all_oof = [r for r in subset if r['split']=='test']
            result[condition][endpoint] = {'splits':splits,'OOF_per_seed':pooled,
                'OOF_mean_over_seed_metrics':{k:float(np.mean([v[k] for v in pooled.values()])) for k in METRICS},
                'OOF_111_prediction_rows_descriptive_only':metric(all_oof,endpoint),
                'independent_subjects':37,'train_minus_test':{k:splits['train']['primary'][k]-splits['test']['primary'][k] for k in METRICS},
                'single_class_test_folds':sum(v['single_class_predictions'] for fs in splits['test']['folds'].values() for v in fs.values())}
    return result


def comparisons(summary,rows):
    pairs = [('U2','U0',True),('C2','C0',True)]
    reports = []
    for candidate,control,direct in pairs:
        endpoints = ['eeg'] if candidate.startswith('U') else ['eeg','fusion']
        out = {'candidate':candidate,'control':control,'direct_predeclared_test':direct,'endpoints':{}}
        for endpoint in endpoints:
            a, b = summary[candidate][endpoint], summary[control][endpoint]
            at, bt = a['splits']['test'], b['splits']['test']
            delta = {k:at['primary'][k]-bt['primary'][k] for k in METRICS}
            wins = sum(at['seed_fold_means'][str(seed)]['macro_f1']>bt['seed_fold_means'][str(seed)]['macro_f1'] for seed in SEEDS)
            # Conservative usability gate: disclose and fail any single-class outer test fold.
            usable = at['primary']['balanced_accuracy']>.5 and at['primary']['mdd_recall']>=.5 and a['single_class_test_folds']==0
            out['endpoints'][endpoint] = {'delta':delta,'seeds_with_higher_fold_mean_F1':wins,
                'improvement_threshold_pass':delta['macro_f1']>=.03 and wins>=2 and delta['mdd_recall']>=-.05,
                'usability_pass':usable,'single_class_test_folds':a['single_class_test_folds'],
                'usability_definition':'primary BA>0.5, primary MDD recall>=0.5, zero single-class test folds (conservative interpretation)'}
            index = {(r['seed'],r['fold'],r['subject_id']):r for r in rows if r['condition']==control and r['split']=='test'}
            changed = []
            for r in rows:
                if r['condition']!=candidate or r['split']!='test':
                    continue
                old = index[(r['seed'],r['fold'],r['subject_id'])]
                if old['predictions'][endpoint]!=r['predictions'][endpoint]:
                    changed.append({'seed':r['seed'],'fold':r['fold'],'subject_id':r['subject_id'],'label':r['label'],
                        'control_p_MDD':old['probabilities'][endpoint][1],'candidate_p_MDD':r['probabilities'][endpoint][1],
                        'control_prediction':old['predictions'][endpoint],'candidate_prediction':r['predictions'][endpoint]})
            out['endpoints'][endpoint]['classification_changes'] = changed
        eeg = out['endpoints']['eeg']
        out['EEG_improved_and_usable'] = eeg['improvement_threshold_pass'] and eeg['usability_pass']
        if candidate.startswith('C'):
            d = out['endpoints']['fusion']['delta']
            out['fusion_preservation_pass'] = d['macro_f1']>=-.01 and d['mdd_recall']>=-.05
            out['CCML_acceptance_pass'] = out['EEG_improved_and_usable'] and out['fusion_preservation_pass'] and out['endpoints']['fusion']['usability_pass']
        reports.append(out)
    return reports


if __name__ == '__main__':
    evaluate_all()
