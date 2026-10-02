import platform
import transformers
from .core import *

def main():
    assert not (OUT/'registration.json').exists()
    assert read(OUT/'tests.json')['passed'] and read(OUT/'cache_replay.json')['passed'] and read(OUT/'backward_probe.json')['passed']
    ctx=RecordContext();files=list(CODE.glob('*.py'))+[APPENDIX,OUT/'tests.json',OUT/'cache_replay.json',OUT/'backward_probe.json',OUT/'cache_manifest.json']
    files += sorted((OUT/'cache').rglob('*.pt'))+sorted((OUT/'cache').rglob('*.json'))
    bm=read(WORK/'S00/manifests/backbones.json')['macbert']
    for name,d in bm['files'].items():
        assert digest(BACKBONE/name)==d['sha256'],name
        files.append(BACKBONE/name)
    files += [WORK/'S00/manifests/backbones.json',ctx.cohort_path,ctx.folds_path,G1/'input_manifest.json']
    files += [WORK/read(G1/'input_manifest.json')['arrays']['speech']['path']]
    for row in read(OUT/'cache_manifest.json')['rows']:files += [WORK/d['path'] for d in row['sources']]
    files += [WORK/'tridep_ccml'/p for p in ['model_opt_loss/core.py','model_opt_v1/core.py','model_opt_v1/evaluate.py',
        'model_opt_v1/report.py','fusion_g1/model.py','fusion_g1/common.py','fusion_g1/evaluate.py','ccml.py','metrics.py','records.py','v2/common.py','eeg_e2/common.py','eeg_replacement/models.py']]
    files += [WORK/'S00/.venv/Lib/site-packages/transformers/models/bert/modeling_bert.py',WORK/'S00/.venv/Lib/site-packages/torch/utils/checkpoint.py',
        WORK/'tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/status.json',WORK/'tridep_ccml/experiments/MODEL-OPT-v1-LOSS-37-fulltext/deliverable_hashes.json']
    counts={}
    for c in CONDITIONS:
        m=Model(c,11);counts[c]={'top_parameters':sum(p.numel() for p in m.top.parameters()),
            'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),'adapter_parameters':sum(p.numel() for p in m.adapter_parameters())}
    assert counts['M3-lora-ST']['trainable_parameters']-counts['M3-frozen-ST']['trainable_parameters']==12288
    save(OUT/'registration.json',{'registered_utc':now(),'client_date':'2026-10-01','authorization':'user requested continuing to next stage',
        'conditions':CONDITIONS,'new_fits':30,'new_updates':7200,'seeds':SEEDS,'folds':FOLDS,'subject_ids':ctx.ids,'splits':ctx.folds,
        'epochs':60,'batch_size':8,'question_chunk_size':32,'loss':'ST mean single-view + unchanged consistent/complementary',
        'optimizer':'Adamax','projection_lr':1e-4,'head_lr':1e-3,'lora_lr':1e-5,'weight_decay':0,'clip_norm':1,
        'lora':{'layer':11,'targets':['query','value'],'rank':4,'alpha':4,'dropout':0,'B_init':'zero','trainable_parameters':12288},
        'pretrained':bm,'checkpoint':'last epoch','threshold':.5,'selection':'none','data_defect_audit':'skipped_by_user',
        'parameter_counts':counts,'primary_comparison':['M3-lora-ST','M3-frozen-ST'],'development_subjects':37,
        'versions':{'python':platform.python_version(),'torch':str(torch.__version__),'transformers':transformers.__version__,'numpy':np.__version__},
        'frozen_files':[desc(p) for p in sorted(set(files))]},immutable=True)
    save(OUT/'status.json',{'stage':'registered','models_completed':0,'updates_completed':0,'outer_results_released':False,'updated_utc':now()})
    print('Registered 30 matched fits, 7200 updates; last-layer Q/V LoRA only.',flush=True)

if __name__=='__main__':main()
