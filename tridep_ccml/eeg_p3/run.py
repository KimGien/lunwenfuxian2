from __future__ import annotations
import argparse
import unittest
from .common import *
from .engine import *

def register():
    OUT.mkdir(parents=True,exist_ok=True)
    context=RecordContext()
    splits={str(f):inner_splits(context.folds[f'fold_{f}']['train'],context.labels,f) for f in FOLDS}
    if (OUT/'configuration.json').exists():
        lock_check();return
    history=history_check()
    write_new(OUT/'configuration.json',{'priority':3,'sources':source_hashes(),'augmentation':False,
        'seeds':SEEDS,'folds':FOLDS,'inner_splits':splits,'epochs':60,'batch_size':8,
        'new_outer_models':60,'inner_calibration_models':90,'BN_derived_states':60,
        'data_manifest':descriptor(R1/'data_manifest.json'),'registered_utc':now(),
        'historical_artifacts_before':history,'preexisting_filter_boundary_issue':True})

def prepare():
    register();lock_check()
    from . import test_contracts
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(test_contracts))
    require(result.wasSuccessful(),'P3 contract tests failed')
    write_new(OUT/'contract_tests.json',{'passed':True,'tests':result.testsRun})
    if not (OUT/'preflight.json').exists():
        data=Data(1)
        probes=[small_fit(data,arm) for arm in ['G','S']]
        write_new(OUT/'preflight.json',{'small_fit':probes,'passed':all(p['passed'] for p in probes)})
    require(read_json(OUT/'preflight.json')['passed'],'Small fit engineering gate failed')

def run():
    prepare()
    config=lock_check()
    for fold in FOLDS:
        data=Data(fold)
        for seed in SEEDS:
            replace_json(OUT/'status.json',{'state':'training_and_train_only_fit','fold':fold,'seed':seed,
                'outer_scores_released':False,'updated_utc':now()})
            for c in ['U1','C1']:
                for arm in ['G','S']:
                    train_new(data,c,arm,seed,fold,data.train)
                derive_bn(data,c,'B',seed,fold)
                derive_bn(data,c,'S',seed,fold)
            token_attention(data,seed,fold)
            for c in ['C3','C4']: calibrate(data,c,seed,fold,config['inner_splits'][str(fold)])
        del data;torch.cuda.empty_cache()
    lock_check()
    trained=list((OUT/'runs').rglob('trained.json')); derived=list((OUT/'runs').rglob('derived.json'))
    require(len(trained)==150 and len(derived)==60,'Incomplete training or BN stage')
    require(len(list((OUT/'calibrators').rglob('fold_*.json')))==30,'Incomplete calibration')
    locked={p.relative_to(OUT).as_posix():digest(p) for root in ['runs','calibrators','attention'] for p in sorted((OUT/root).rglob('*')) if p.is_file()}
    write_new(OUT/'release_gate.json',{'passed':True,'trained_models':150,'derived_states':60,
        'fitted_artifacts':locked,'configuration':descriptor(OUT/'configuration.json')})
    print('RELEASE GATE PASSED: all fitting fixed before outer predictions.',flush=True)
    for fold in FOLDS:
        data=Data(fold)
        for seed in SEEDS:
            for c in ['U1','C1']:
                for arm in ['B','BR','G','S','SR']:evaluate_model(data,c,arm,seed,fold)
            for c in ['C3','C4']: evaluate_model(data,c,'B',seed,fold)
        del data;torch.cuda.empty_cache()
        print(f'EVALUATED fold={fold}',flush=True)
    replace_json(OUT/'status.json',{'state':'evaluation_complete_report_pending','updated_utc':now(),
        'outer_scores_released':True,'models_trained':150})
    print('P3 numeric study completed.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare_only else run()
