import platform
import scipy
import sklearn
from .core import *
from tridep_ccml.records import RecordContext

def main():
    assert not (OUT/'registration.json').exists(),'Already registered'
    ctx=RecordContext();OUT.mkdir(parents=True,exist_ok=True)
    p=WORK/'tridep_ccml/experiments/CCML-v2-AS-37-fulltext/speech_pool.npz'
    with np.load(p) as z:
        assert z['subject_ids'].tolist()==ctx.ids
        std=z['question_means'].std(1,ddof=0,dtype=np.float64).astype(np.float32)
    (OUT/'cache').mkdir(exist_ok=True);np.save(OUT/'cache/speech_std.npy',std)
    sources=[APPENDIX,WORK/'output/plans/模型与数据缺陷排查及性能改进后续计划_v1.md',p,
        G1/'input_manifest.json',E2/'input_manifest.json',E2/'linear/new/features.npy',
        ctx.cohort_path,ctx.folds_path,WORK/'tridep_ccml/ccml.py',WORK/'tridep_ccml/metrics.py',
        WORK/'tridep_ccml/records.py',WORK/'tridep_ccml/v2/common.py',
        WORK/'tridep_ccml/eeg_replacement/models.py',WORK/'tridep_ccml/eeg_e2/common.py']
    sources+=sorted(CODE.glob('*.py'))+sorted((WORK/'tridep_ccml/fusion_g1').glob('*.py'))
    sources+=[WORK/d['path'] for d in read(G1/'input_manifest.json')['arrays'].values()]
    sources+=[WORK/r['new_cache']['path'] for r in read(E2/'input_manifest.json')['records']]
    sources+=[OUT/'cache/speech_std.npy']
    save(OUT/'registration.json',{'created_utc':now(),'conditions':CONDITIONS,'seeds':SEEDS,'folds':FOLDS,
        'data_defect_audit':'skipped_by_user','data_without_defects':'working_assumption_not_audit_finding',
        'neural_fits':90,'linear_fits':10,'updates':21600,'epochs':60,'batch_size':8,
        'optimizer':'Adamax','encoder_lr':1e-4,'head_lr':1e-3,'weight_decay':0,
        'clip_global':1.,'clip_branch':'1/sqrt(V)','threshold':.5,'tie':'HC',
        'loss':'unmodified CCML','rho':1.25,'delta':1.,'gamma':1.,'anneal_epochs':50,
        'selection':'none; all fixed last epoch','fit_scope':'original train only',
        'linear_C':.1,'linear_solver':'lbfgs','linear_max_iter':2000,'linear_tol':1e-8,
        'covariance':'window LedoitWolf -> subject arithmetic mean -> log-Euclidean tangent',
        'versions':{'python':platform.python_version(),'torch':str(torch.__version__),'numpy':np.__version__,
                    'scipy':scipy.__version__,'sklearn':sklearn.__version__},
        'subject_ids':ctx.ids,'folds_detail':ctx.folds,
        'frozen_files':[desc(p) for p in sorted(set(sources))]},immutable=True)
    save(OUT/'status.json',{'stage':'registered','completed_neural':0,'completed_linear':0,'outer_results_released':False,'updated_utc':now()})
    print('Registered 90 neural fits, 10 deterministic linear fits; data audit skipped.',flush=True)

if __name__=='__main__':main()
