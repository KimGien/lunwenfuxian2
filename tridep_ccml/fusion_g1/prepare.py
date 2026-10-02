from .common import *
from .model import Model,OFFSETS
from tridep_ccml.eeg_p3.methods import subject_standardize

def main():
    assert not (OUT/'registration.json').exists(),'Already registered'
    OUT.mkdir(parents=True,exist_ok=True)
    save(OUT/'historical_before.json',history(),immutable=True)
    ctx=RecordContext(); r1=read(R1/'data_manifest.json'); ei=read(E2/'input_manifest.json')
    handoff=read(WORK/'output/analysis/eeg_e2v2_20260929/handoff.json')
    assert handoff['eeg_mainline_completed'] and handoff['input_engineering_passed']
    assert r1['subject_ids']==ctx.ids and not ei['augmentation'] and not ei['labels_used']
    records={r['subject_id']:r for r in ei['records']}
    raw=np.stack([np.load(verify(records[s]['new_cache'])) for s in ctx.ids])
    eeg,mean,std=subject_standardize(raw)
    stats=read(E2/'subject_standardizers.json')['new']
    assert stats['subject_ids']==ctx.ids
    assert np.array_equal(mean,np.array(stats['mean'])) and np.array_equal(std,np.array(stats['std']))
    (OUT/'cache').mkdir(exist_ok=True)
    np.save(OUT/'cache/eeg.npy',eeg,allow_pickle=False)
    arrays={'eeg':desc(OUT/'cache/eeg.npy')}
    for m in ['speech','text']:
        p=verify(r1['pooled_inputs']['arrays'][m]); a=np.load(p)
        assert a.shape==(37,768) and a.dtype==np.float32 and np.isfinite(a).all()
        arrays[m]=desc(p)
    for d in r1['pooled_inputs']['speech_source'].values():verify(d)
    verify(r1['pooled_inputs']['S_control_manifest'])
    save(OUT/'subject_standardizers.json',{'subject_ids':ctx.ids,'mean':mean.tolist(),'std':std.tolist(),
        'fit_subject_ids':[],'rule':'each own 300 s, each channel, ddof=0','matches_E2v2_exactly':True},immutable=True)
    save(OUT/'input_manifest.json',{'subject_ids':ctx.ids,'labels':ctx.labels,'folds':ctx.folds,
        'cohort':desc(ctx.cohort_path),'fold_file':desc(ctx.folds_path),'arrays':arrays,
        'EEG_source':desc(E2/'input_manifest.json'),'EEG_raw_sources':[records[s]['new_cache'] for s in ctx.ids],
        'speech_text_sources':r1['pooled_inputs'],'speech_questions':1073,'text_questions_per_person':29,
        'frozen_external_backbones':True,'supervised_initialization':False,'augmentation':False,
        'fit_subject_ids':[],'labels_used_in_preprocessing':False},immutable=True)
    (OUT/'initial').mkdir(exist_ok=True)
    initial=[]
    for seed in SEEDS:
        model=Model('G-CE-EST',seed)
        path=OUT/f'initial/seed_{seed}.pt';torch.save(model.state_dict(),path); initial.append(desc(path))
        for fold in FOLDS:save(OUT/f'orders/seed_{seed}/fold_{fold}.json',orders(ctx.folds[f'fold_{fold}']['train'],seed),immutable=True)
    files=[PLAN,APPENDIX,OUT/'input_manifest.json',OUT/'subject_standardizers.json',OUT/'historical_before.json',
           WORK/'tridep_ccml/ccml.py',WORK/'tridep_ccml/eeg_replacement/models.py',WORK/'tridep_ccml/eeg_p3/methods.py',
           WORK/'tridep_ccml/v2/common.py',WORK/'tridep_ccml/records.py']
    files+=sorted(CODE.glob('*.py'))+sorted((OUT/'orders').rglob('*.json'))
    save(OUT/'registration.json',{'registered_utc':now(),'conditions':CONDITIONS,'seeds':SEEDS,'folds':FOLDS,
        'epochs':60,'batch_size':8,'optimizer':'Adamax','encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':0,
        'global_clip_norm':1,'eeg_spatial_max_norm':1,'head_max_norm':None,'threshold':.5,'tie':'HC',
        'normalizer':'non-affine LayerNorm(128, eps=1e-5)','initialization_offsets':OFFSETS,
        'initialization':'fresh independent per modality, shared across six conditions; seed reused across folds',
        'model_initial_files':initial,'train_seed':'reset to base seed after model construction',
        'loss_sum_over_views':True,'rho':1.25,'delta':1,'gamma':1,'anneal_epochs':50,
        'checkpoint':'last epoch only','val_selection':False,'new_models':90,'expected_updates':21600,
        'primary_comparisons':[['G-CCML-EST','G-CE-EST'],['G-CCML-EST','G-CCML-ST']],
        'G2_registered_in':desc(APPENDIX),'G4_postprocess_enabled':False,'complete_inner_selection':False,
        'DFMU_in_scope':False,'independent_population_validation':False,
        'frozen_files':[desc(p) for p in files]},immutable=True)
    save(OUT/'status.json',{'stage':'registered','completed_models':0,'outer_results_released':False,'updated_utc':now()})
    print('Registered six conditions / 90 fits, no G4; input and historical hashes verified.',flush=True)

if __name__=='__main__':main()
