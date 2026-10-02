from torch.nn import functional as F
from .common import *
from .model import Model
from tridep_ccml.ccml import dirichlet_from_evidence,fuse_evidence

def outputs(logits,family,modalities):
    evidential=family!='CE'
    evidence=F.softplus(logits) if evidential else None
    branch=dirichlet_from_evidence(evidence) if evidential else {'p':logits.softmax(-1),'u':None}
    endpoints={m:branch['p'][:,i,1] for i,m in enumerate(modalities)}
    endpoints['probability_mean']=branch['p'].mean(1)[:,1]
    if evidential:
        endpoints['ccml']=fuse_evidence(evidence,'ccml')['p'][:,1]
        if family=='EDL':endpoints['evidence_mean']=fuse_evidence(evidence,'mean_evidence')['p'][:,1]
    endpoints['formal']=endpoints['ccml' if family=='CCML' else 'probability_mean']
    if 'eeg' in modalities:
        ix=[modalities.index(m) for m in ['speech','text']]
        endpoints['fixed_ST']=(fuse_evidence(evidence[:,ix],'ccml')['p'][:,1] if family=='CCML'
                                else branch['p'][:,ix].mean(1)[:,1])
    return endpoints,evidence,branch

@torch.no_grad()
def infer(model,data):
    model.eval();ls=[];zs=[];ns=[]
    for start in range(0,len(data.ids),8):
        x,_=data.batch(data.ids[start:start+8]);l,z,n=model(x,features=True)
        ls.append(l.cpu());zs.append(z.cpu());ns.append(n.cpu())
    return torch.cat(ls),torch.cat(zs),torch.cat(ns)

def main():
    lock_check();gate=read(OUT/'release_gate.json');assert gate['passed'] and gate['completed_models']==90
    for d in read(OUT/'evaluation_lock.json')['files']:verify(d)
    for p,sha in gate['fitted_artifacts'].items():assert digest(OUT/p)==sha,p
    data=Data();set_seed(11);count=0
    for condition in CONDITIONS:
        for seed in SEEDS:
            for fold in FOLDS:
                target=OUT/f'predictions/{condition}/seed_{seed}/fold_{fold}.json'
                assert not target.exists(),'Predictions already exist; inspect before rerunning'
                stem=OUT/f'runs/{condition}/seed_{seed}/fold_{fold}'
                cp=torch.load(stem/'final.pt',weights_only=True,map_location='cpu')
                model=Model(condition,seed).cuda();model.load_state_dict(cp['model'],strict=True)
                assert state_hash(model.state_dict())==cp['final_state_sha256']
                logits,z,n=infer(model,data); ep,e,b=outputs(logits,model.family,model.modalities)
                split_ids=data.ctx.folds[f'fold_{fold}'];rows=[]
                for i,s in enumerate(data.ids):
                    split=next(k for k in ['train','val','test'] if s in split_ids[k])
                    rows.append({'subject_id':s,'label':data.ctx.labels[s],'split':split,
                        'logits':logits[i].tolist(),'evidence':None if e is None else e[i].tolist(),
                        'branch_strength':None if e is None else (e[i]+1).sum(-1).tolist(),
                        'branch_u':None if e is None else b['u'][i].tolist(),
                        'probabilities':{k:float(v[i]) for k,v in ep.items()}})
                features=target.with_suffix('.pt');features.parent.mkdir(parents=True,exist_ok=True)
                torch.save({'subject_ids':data.ids,'modalities':model.modalities,'logits':logits,
                    'features':z,'normalized_features':n},features)
                save(target,{'condition':condition,'family':model.family,'seed':seed,'fold':fold,
                    'modalities':model.modalities,'threshold':.5,'checkpoint':desc(stem/'final.pt'),
                    'feature_file':desc(features),'rows':rows},immutable=True)
                count+=1
        print(f'{condition}: predictions saved after release gate; {count}/90',flush=True)
    save(OUT/'evaluation_complete.json',{'completed_utc':now(),'models':count,'subject_model_predictions':count*37,
        'test_predictions':6*3*37,'unique_test_people':37,'outer_results_released':True,
        'artifacts':{p.relative_to(OUT).as_posix():digest(p) for p in sorted((OUT/'predictions').rglob('*')) if p.is_file()}},immutable=True)
    save(OUT/'status.json',{'stage':'evaluated','completed_models':90,'completed_updates':21600,'outer_results_released':True,'updated_utc':now()})

if __name__=='__main__':main()
