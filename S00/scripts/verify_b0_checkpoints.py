"""Reload all final checkpoints and reproduce every branch test probability."""
import json
import sys
import torch
import numpy as np
from prepare import ROOT,save,digest
from run_b0 import check_inputs
from smoke_models import seed
sys.path.insert(0,str(ROOT/'upstream-dev'))
from lib.models import models

def main():
    cfg=json.loads((ROOT/'configs/b0.json').read_text())
    folds=json.loads((ROOT/'manifests/fold_assignments.json').read_text())
    device='cuda' if torch.cuda.is_available() else 'cpu';seed(11)
    report={'passed':True,'device':device,'checkpoints':[],'predictions_reproduced':0,'max_absolute_error':0.0}
    for value in (11,22,33):
        folder=ROOT/'runs/B0-37'/f'seed_{value}'
        rows=[json.loads(s) for s in (folder/'probabilities.jsonl').read_text().splitlines() if s]
        paths=check_inputs(ROOT/'features/B0-37',value,folds,cfg)
        for name,f in folds.items():
            fold=int(name[-1]);test=[r for r in rows if r['outer_fold']==fold and r['split']=='test']
            for modality,c in cfg['branches'].items():
                path=folder/f'{name}_{modality}.pt';saved=torch.load(path,map_location='cpu',weights_only=True)
                if saved['seed']!=value or saved['fold']!=fold or saved['train_ids']!=f['train'] or saved['configuration']!=c:
                    raise ValueError('Checkpoint training provenance mismatch')
                if saved['cohort_id']!=cfg['cohort_id'] or saved['folds_sha256']!=cfg['folds_sha256']:
                    raise ValueError('Checkpoint cohort mismatch')
                model=getattr(models,c['model_class'])(**c['model_kwargs'])
                model.load_state_dict(saved['state_dict'],strict=True);model.to(device).eval()
                with torch.no_grad():
                    for row in test:
                        x=torch.tensor(np.load(paths[(fold,row['subject_id'],modality)]),dtype=torch.float32,device=device).unsqueeze(0)
                        probability=float(torch.softmax(model(x),dim=-1)[0,1].cpu())
                        difference=abs(probability-row['p_'+modality])
                        if not np.isfinite(probability) or difference>1e-6:raise ValueError('Reloaded test probability mismatch')
                        report['max_absolute_error']=max(report['max_absolute_error'],difference)
                        report['predictions_reproduced']+=1
                report['checkpoints'].append({'path':str(path.relative_to(ROOT)),'sha256':digest(path)})
                del model,saved
    if len(report['checkpoints'])!=45 or report['predictions_reproduced']!=333:raise ValueError('Incomplete checkpoint verification')
    save(ROOT/'reports/b0_checkpoint_verification.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checkpoints'},indent=2))

if __name__=='__main__':main()
