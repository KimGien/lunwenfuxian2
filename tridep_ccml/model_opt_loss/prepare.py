import platform
import sklearn
from .core import *
from tridep_ccml.records import RecordContext

def main():
    assert not (OUT/'registration.json').exists()
    assert read(OUT/'tests.json')['passed']
    ctx=RecordContext();controls=[];files=[]
    for seed in SEEDS:
        for fold in FOLDS:
            for mods in ['ST','EST']:
                path=stem('M2-sum-'+mods,seed,fold);d=read(path/'complete.json')
                assert d['historical_G1_exact'] and d['reload_exact'] and d['updates']==240
                assert d['fit_ids']==ctx.folds[f'fold_{fold}']['train']
                m=Model('M2-mean-'+mods,seed)
                assert state_hash(m.state_dict())==d['initial_state_sha256']
                assert sum(p.numel() for p in m.parameters())==d['parameter_count']
                controls.append({'condition':'M2-sum-'+mods,'seed':seed,'fold':fold,'checkpoint':desc(path/'final.pt'),
                    'record':desc(path/'complete.json'),'initial_state_sha256':d['initial_state_sha256']})
                files.extend([path/'final.pt',path/'complete.json',path/'rng_initial.pt',path/'history.json',
                    PARENT/f'predictions/M1-global-{mods}/seed_{seed}/fold_{fold}.json'])
    files+=list(CODE.glob('*.py'))
    files += [WORK/'tridep_ccml'/p for p in ['ccml.py','records.py','metrics.py','eeg_e2/common.py',
        'eeg_replacement/models.py','v2/common.py','fusion_g1/common.py','fusion_g1/model.py',
        'fusion_g1/train.py','fusion_g1/evaluate.py','model_opt_v1/core.py','model_opt_v1/evaluate.py','model_opt_v1/report.py']]
    files += [APPENDIX,OUT/'tests.json',ctx.cohort_path,ctx.folds_path,G1/'input_manifest.json',
        PARENT/'registration.json',PARENT/'deliverable_hashes.json',PARENT/'metrics.json',PARENT/'diagnostics.json']
    files += [WORK/d['path'] for d in read(G1/'input_manifest.json')['arrays'].values()]
    save(OUT/'registration.json',{'registered_utc':now(),'authorization':'user explicitly requested next stage after batch one',
        'data_defect_audit':'skipped_by_user','conditions':CONDITIONS,'new_conditions':NEW,'new_neural_fits':30,
        'reused_control_models':30,'new_updates':7200,'mechanism_probes':90,'probe_optimizer_updates':0,
        'seeds':SEEDS,'folds':FOLDS,'subject_ids':ctx.ids,'splits':ctx.folds,'epochs':60,'batch_size':8,
        'optimizer':'Adamax','encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':0,'global_clip':1,
        'eeg_spatial_max_norm':1,'threshold':.5,'checkpoint':'fixed last epoch','model_selection':False,
        'only_change':'single-view loss SUM -> MEAN; consistent and complementary terms unchanged',
        'primary_comparisons':[['M2-mean-ST','M2-sum-ST'],['M2-mean-EST','M2-sum-EST']],
        'classification_criteria':{'f1_delta':.01,'seeds_improving':2,'BA_delta_min':0,'MDD_recall_delta_min':-.05},
        'versions':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'controls':controls,'frozen_files':[desc(p) for p in sorted(set(files))]},immutable=True)
    save(OUT/'status.json',{'stage':'registered','new_models_completed':0,'new_updates_completed':0,
        'reused_controls':30,'new_outer_results_released':False,'updated_utc':now()})
    print('Registered loss-block mean: 30 new fits / 7200 updates; 30 immutable controls reused.',flush=True)

if __name__=='__main__':main()
