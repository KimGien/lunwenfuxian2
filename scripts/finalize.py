"""Record current evidence and exact local dependencies without training."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from prepare import ROOT,save,digest

def read(relative):
    path=ROOT/relative
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}

def main():
    checks={}
    for key,env in [('main','.venv'),('asr','.asr-venv')]:
        exe=ROOT/env/'Scripts/python.exe'
        proc=subprocess.run([str(exe),'-m','pip','check'],capture_output=True,text=True,encoding='utf-8',errors='replace')
        checks[key]={'passed':proc.returncode==0,'output':proc.stdout+proc.stderr}
        freeze=subprocess.run([str(exe),'-m','pip','freeze'],capture_output=True,text=True,encoding='utf-8',check=True)
        (ROOT/f'requirements-{key}.lock.txt').write_text(freeze.stdout,encoding='utf-8')
    tests=subprocess.run([sys.executable,str(ROOT/'scripts/test_protocol.py')],capture_output=True,text=True,encoding='utf-8',errors='replace')
    (ROOT/'reports/protocol_tests.log').write_text(tests.stdout+tests.stderr,encoding='utf-8')
    modma=read('reports/modma_audit.json'); models=read('reports/model_smoke.json'); backbones=read('reports/backbone_smoke.json'); asr=read('reports/asr_smoke.json')
    report={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'stage':'S00_PREPARATION_BLOCKED_BEFORE_FULL_FEATURES_AND_TRAINING','s00_reproduced':False,'training_ready':False,'real_training_run':False,'feature_extraction_complete':False,'real_subject_probability_files_created':False,'official_folds_passed':read('reports/fold_audit.json').get('official_json',{}).get('passed',False),'data':modma,'gpu_smoke_passed':models.get('device')=='cuda' and len(models.get('models',{}))==3 and all(v['forward_backward_passed'] and v['repeat_equal'] for v in models.get('models',{}).values()),'backbone_smoke_passed':backbones.get('device')=='cuda' and all(not backbones.get(k,{}).get('loading_info',{}).get('missing_keys',['unknown']) for k in ('macbert','hubert_code')),'asr_smoke_passed':asr.get('passed',False),'package_checks':checks,'protocol_tests_passed':tests.returncode==0,'outstanding':read('configs/b0.json')['unresolved'],'python':sys.version}
    if (ROOT/'recovery/status.json').exists():
        report['data_recovery']=read('recovery/status.json')
    from cohort import validate_active_cohort
    lock=validate_active_cohort(require_data=True)
    report.update(experiment='B0-37',cohort_id=lock['cohort_id'],folds_sha256=lock['folds_sha256'],
                  active_folds_passed=read('reports/fold_audit.json').get('active_cohort',{}).get('passed',False),
                  data_ready=modma['passed'],stage='S00_B0_37_DATA_READY_PROTOCOL_AND_FEATURES_PENDING')
    save(ROOT/'reports/preflight.json',report)
    support={}
    for p in [ROOT/'.asr-venv/Lib/site-packages/whisperx/assets/pytorch_model.bin',ROOT/'tools/ffmpeg.exe']:
        if p.exists(): support[str(p.relative_to(ROOT))]={'sha256':digest(p),'bytes':p.stat().st_size}
    save(ROOT/'manifests/asr_support_assets.json',support)
    files=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'configs').glob('*.json'))+list(ROOT.glob('requirements*.txt'))+[ROOT/'README.md',ROOT/'reports/差异与阻塞.md',ROOT/'reports/37人统一实验协议.md']
    save(ROOT/'manifests/preparation_files_sha256.json',{str(p.relative_to(ROOT)):digest(p) for p in files})
    recovery=ROOT/'recovery'
    if recovery.exists():
        save(ROOT/'manifests/recovery_files_sha256.json',{str(p.relative_to(ROOT)):digest(p) for p in recovery.rglob('*') if p.is_file()})
    print(json.dumps({k:report[k] for k in ('stage','cohort_id','data_ready','active_folds_passed','training_ready','official_folds_passed','gpu_smoke_passed','backbone_smoke_passed','asr_smoke_passed','protocol_tests_passed')},indent=2))

if __name__=='__main__': main()
