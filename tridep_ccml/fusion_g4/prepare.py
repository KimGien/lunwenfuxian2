from .common import *

def main():
    assert not OUT.exists(),'G4 already exists; inspect before preparing again'
    OUT.mkdir(parents=True);ctx=RecordContext();h=history()
    save(OUT/'historical_before.json',h,immutable=True)
    assert read(G1/'completion.json')['stage']=='G0_G3_complete'
    splits={str(f):inner_splits(ctx.folds[f'fold_{f}']['train'],ctx.labels,f) for f in FOLDS}
    budget=0
    for seed in SEEDS:
        for f in FOLDS:
            outer=ctx.folds[f'fold_{f}']
            for part in splits[str(f)]:
                assert set(part['fit']).isdisjoint(part['calibration'])
                assert set(part['fit'])|set(part['calibration'])==set(outer['train'])
                assert not (set(part['fit'])|set(part['calibration'])) & (set(outer['val'])|set(outer['test']))
                assert len(part['fit']) in [17,18]
                save(OUT/f'orders/seed_{seed}/fold_{f}/inner_{part["inner"]}.json',orders(part['fit'],seed),immutable=True)
                budget+=2*60*((len(part['fit'])+7)//8)
    assert budget==16200
    source=read(G1/'input_manifest.json')
    for d in source['arrays'].values():verify(d)
    save(OUT/'input_manifest.json',{'parent_manifest':desc(G1/'input_manifest.json'),
        'subject_ids':ctx.ids,'labels':ctx.labels,'folds':ctx.folds,'arrays':source['arrays'],
        'augmentation':False,'new_subjects':0,'new_preprocessing':False,'supervised_parent_weights_loaded':False},immutable=True)
    known=read(WORK/'output/analysis/fusion_g1_20260929/results.json')
    save(OUT/'prior_knowledge.json',{'G1_results_already_seen':True,'G1_registration_excluded_G4':True,
        'authorization':'User requested G4 after G0-G3 completion and disclosure of optional follow-up scope',
        'parent_results':desc(WORK/'output/analysis/fusion_g1_20260929/results.json'),
        'known_CCML_formal_test_results':[r for r in known['results'] if r['condition'] in CONDITIONS and r['endpoint']=='formal' and r['split']=='test'],
        'independent_validation_claim':False},immutable=True)
    parents=[]
    for c in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                parents += [desc(G1/f'runs/{c}/seed_{seed}/fold_{fold}/final.pt'),
                            desc(G1/f'predictions/{c}/seed_{seed}/fold_{fold}.json')]
    files=[PLAN,APPENDIX,OUT/'input_manifest.json',OUT/'historical_before.json',OUT/'prior_knowledge.json',
        G1/'registration.json',G1/'deliverable_hashes.json',G1/'input_manifest.json',
        WORK/'tridep_ccml/fusion_g1/common.py',WORK/'tridep_ccml/fusion_g1/model.py',WORK/'tridep_ccml/fusion_g1/evaluate.py',
        WORK/'tridep_ccml/ccml.py',WORK/'tridep_ccml/eeg_p3/methods.py',WORK/'tridep_ccml/eeg_p3/summarize.py',
        WORK/'tridep_ccml/eeg_replacement/models.py',WORK/'tridep_ccml/v2/common.py']
    files+=list(CODE.glob('*.py'))+sorted((OUT/'orders').rglob('*.json'))+sorted((G1/'initial').glob('*.pt'))
    save(OUT/'registration.json',{'registered_utc':now(),'stage':'G4_post_release_development',
        'G1_outer_results_already_seen':True,'G4_transformed_outer_results_seen':False,
        'conditions':CONDITIONS,'seeds':SEEDS,'folds':FOLDS,'outer_folds':ctx.folds,'inner_splits':splits,
        'inner_split_rule':'StratifiedKFold(3,shuffle=True,random_state=93000+outer_fold)',
        'new_inner_models':90,'new_outer_models':0,'reused_outer_models':30,'epochs':60,'batch_size':8,'updates':16200,
        'model_and_optimizer':'identical to G1 CCML, including actual V, lr groups, global clip and EEG max-norm',
        'initialization':'fresh G1 common initial states; never outer trained weights; reset base seed per inner fit',
        'fit_endpoint':'formal CCML fused p(MDD) only','variants':VARIANTS,'threshold_tie':'p>t, equality HC',
        'threshold_objective':'OOF macro-F1; candidates 0,0.5,1,adjacent unique-probability midpoints; tie closest to 0.5 then smaller',
        'sigmoid':'sigmoid(a*logit(clip(p,1e-7,1-1e-7))+b)',
        'sigmoid_objective':'mean binary NLL + 0.005*((a-1)^2+b^2)',
        'sigmoid_bounds':{'a':[.01,10.],'b':[-10.,10.]},'sigmoid_start':[1.,0.],
        'sigmoid_optimizer':{'method':'L-BFGS-B','ftol':1e-13,'gtol':1e-9,'maxiter':1000},
        'compose_threshold_and_sigmoid':False,'calibrate_dirichlet_evidence':False,
        'all_calibrators_locked_before_new_outer_transform':True,'raw_G1_results_unchanged':True,
        'classification_development_gate':{'delta_f1_at_least':.01,'strictly_improving_seeds_at_least':2,'delta_MDD_recall_at_least':-.05},
        'probability_development_gate':'both NLL and Brier decrease; reported separately',
        'primary_comparisons':'threshold vs raw and sigmoid vs raw within each modality; EST vs ST at matched postprocessing as secondary',
        'no_test_based_winner_selection':True,'independent_population_validation':False,
        'parents':parents,'frozen_files':[desc(p) for p in files]},immutable=True)
    save(OUT/'status.json',{'stage':'registered','completed_inner_models':0,'target_inner_models':90,'updated_utc':now()})
    print('G4 registered as post-release development: 90 inner fits, 16200 updates, no outer refit.',flush=True)

if __name__=='__main__':main()
