"""Fixed EEG-only training and inner fitting; outer results released only after all fits."""
from tridep_ccml.eeg_replacement.common import RecordContext, set_seed, state_digest, torch
from tridep_ccml.eeg_replacement.train import orders, step
from tridep_ccml.eeg_p3.methods import make_model, subject_standardize, inner_splits, fit_threshold
from tridep_ccml.eeg_p3.engine import health
from .common import *
import hashlib
import math
import time
import warnings
import joblib
from types import SimpleNamespace
from scipy.signal import welch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

SEEDS=[11,22,33]
FOLDS=[1,2,3,4,5]


def config_check():
    cfg=read(OUT/'configuration.json')
    assert all(digest(WORK/p)==sha for p,sha in cfg['sources'].items()),'Training sources changed'
    for key in ['real_gate','input_manifest']:
        assert digest(WORK/cfg[key]['path'])==cfg[key]['sha256']
    return cfg


def register():
    assert read(OUT/'real_gate.json')['passed']
    assert read(OUT/'preflight.json')['passed']
    context=RecordContext()
    sources=[CODE/'training.py',CODE/'common.py',APPENDIX,PLAN,
        WORK/'tridep_ccml/eeg_replacement/models.py',WORK/'tridep_ccml/eeg_replacement/train.py',
        WORK/'tridep_ccml/eeg_p3/engine.py',WORK/'tridep_ccml/eeg_p3/methods.py',
        WORK/'tridep_ccml/eeg_p3/common.py',WORK/'tridep_ccml/v2/common.py']
    cfg={'version':'EEG-E2v2-fixed-training','registered_utc':now(),'sources':{p.relative_to(WORK).as_posix():digest(p) for p in sources},
        'real_gate':desc(OUT/'real_gate.json'),'input_manifest':desc(OUT/'input_manifest.json'),
        'preflight':desc(OUT/'preflight.json'),
        'seeds':SEEDS,'folds':FOLDS,'epochs':60,'batch_size':8,'augmentation':False,
        'outer_neural_models':15,'inner_neural_models':45,'linear_fits':10,
        'inner_splits':{str(f):inner_splits(context.folds[f'fold_{f}']['train'],context.labels,f) for f in FOLDS},
        'optimizer':{'name':'Adamax','lr':.001,'weight_decay':.0001,'grad_clip':1},
        'linear':{'bands':[[1,4],[4,8],[8,13],[13,30],[30,40]],'C':.1,'solver':'lbfgs','max_iter':2000,'tol':1e-8},
        'primary_endpoint':'unweighted fold-mean macro-F1, then seed mean; p>0.5',
        'secondary_endpoint':'threshold fit on outer-train threefold OOF only',
        'outer_scores_before_release':False,'torch':torch.__version__}
    if (OUT/'configuration.json').exists():return config_check()
    save(OUT/'configuration.json',cfg,immutable=True);return cfg


class Data:
    def __init__(self):
        self.context=RecordContext();self.ids=self.context.ids;self.labels=self.context.labels
        self.index={s:i for i,s in enumerate(self.ids)}
        new={r['subject_id']:r['new_cache'] for r in read(OUT/'input_manifest.json')['records']}
        old={r['subject_id']:r['raw'] for r in read(R1/'data_manifest.json')['raw_and_tokens']}
        self.raw={};self.tensors={};self.stats={}
        for kind,mapping in [('old',old),('new',new)]:
            arrays=[]
            for sid in self.ids:
                d=mapping[sid];assert digest(WORK/d['path'])==d['sha256']
                arrays.append(np.load(WORK/d['path'],allow_pickle=False))
            raw=np.stack(arrays);normalized,mean,std=subject_standardize(raw)
            self.raw[kind]=raw;self.tensors[kind]=torch.from_numpy(normalized).cuda()
            self.stats[kind]={'subject_ids':self.ids,'mean':mean.tolist(),'std':std.tolist(),'ddof':0,
                'scope':'each complete single-person recording; no cohort or label fit'}
        save(OUT/'subject_standardizers.json',self.stats,immutable=True)
        self.select(1,'new')

    def select(self,fold,kind='new'):
        self.fold=fold;self.kind=kind;self.splits=self.context.folds[f'fold_{fold}'];self.train=self.splits['train']
        self.base=SimpleNamespace(index=self.index,splits=self.splits)

    def batch(self,ids,condition='U1',arm='S'):
        assert condition=='U1' and arm=='S'
        indices=[self.index[s] for s in ids]
        return {'eeg':self.tensors[self.kind][indices]},torch.tensor([self.labels[s] for s in ids],device='cuda')

    def fit_batch(self,ids,condition,arm,allowed):
        assert set(ids)<=set(allowed)<=set(self.train)
        return self.batch(ids,condition,arm)


def folder(seed,fold,inner=None):
    p=OUT/f'runs/E-N1/seed_{seed}/fold_{fold}'
    return p if inner is None else p/f'inner_{inner}'


def rng():
    return {k:hashlib.sha256(v.cpu().numpy().tobytes()).hexdigest() for k,v in
            [('cpu',torch.get_rng_state()),('cuda',torch.cuda.get_rng_state())]}


@torch.no_grad()
def predict(model,data,ids):
    model.eval();before=state_digest(model.state_dict());p=[];zs=[]
    for i in range(0,len(ids),8):
        x,_=data.batch(ids[i:i+8]);p.append(model(x).softmax(-1)[:,1].cpu().numpy())
        zs.append(model.representations(x).cpu().numpy())
    assert before==state_digest(model.state_dict())
    return np.concatenate(p),np.concatenate(zs)


def train(data,seed,fold,fit_ids,inner=None):
    path=folder(seed,fold,inner);target=path/'trained.json'
    if target.exists():
        d=read(target);assert digest(WORK/d['checkpoint']['path'])==d['checkpoint']['sha256']
        assert d['configuration']==desc(OUT/'configuration.json') and d['fit_ids']==fit_ids
        return
    path.mkdir(parents=True,exist_ok=True)
    assert not (path/'checkpoint.pt').exists(),'Unmarked checkpoint needs review'
    set_seed(seed);model=make_model('U1',seed,'S').cuda()
    initial=state_digest(model.state_dict());paired=make_model('U1',seed,'B')
    assert all(torch.equal(p.detach().cpu(),dict(paired.named_parameters())[n]) for n,p in model.named_parameters())
    del paired
    seeded_rng=rng()
    initial_health=health(model,data,'U1','S',fit_ids)
    assert initial_health['passed'] and rng()==seeded_rng
    epoch_orders=orders(fit_ids,seed)
    opt=model.optimizer();assert rng()==seeded_rng
    pairing={'initial_state':initial,'initial_training_rng':seeded_rng,'fit_ids':fit_ids,
             'deterministic':torch.are_deterministic_algorithms_enabled(),
             'cudnn_tf32':torch.backends.cudnn.allow_tf32,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32}
    if inner is None:
        oldpath=P3/f'runs/U1_S/seed_{seed}/fold_{fold}'
        old=read(oldpath/'trained.json');bo=read(oldpath/'batch_order.json')
        assert initial==old['initial_state_sha256'] and fit_ids==old['fit_ids'] and epoch_orders==bo['orders']
        # Reconstruct the frozen P3 setup with old data, without training or outer scoring.
        set_seed(seed);control=make_model('U1',seed,'S').cuda();control_pair=make_model('U1',seed,'B');del control_pair
        data.kind='old';control_health=health(control,data,'U1','S',fit_ids)
        orders(fit_ids,seed);control.optimizer()
        assert control_health['passed'] and rng()==seeded_rng and state_digest(control.state_dict())==initial
        data.kind='new';del control
        pairing.update({'parent':desc(oldpath/'trained.json'),'parent_batch_order':desc(oldpath/'batch_order.json'),
                        'historical_initialization_exact':True,'historical_batch_order_exact':True,
                        'historical_rng_recipe_reconstructed_exact':True,
                        'rng_limitation':'P3 has no saved pre-update RNG tensor; frozen setup is rerun and compared.'})
    save(path/'pairing.json',pairing,immutable=True)
    save(path/'initial_health.json',initial_health,immutable=True)
    save(path/'batch_order.json',{'fit_ids':fit_ids,'orders':epoch_orders,'batch_size':8},immutable=True)
    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path/'initial_state.pt')
    records=[];updates=0;start=time.perf_counter()
    for epoch,order in enumerate(epoch_orders,1):
        model.train();total=0.;clipped=0
        for i in range(0,len(order),8):
            ids=order[i:i+8];x,y=data.fit_batch(ids,'U1','S',fit_ids)
            loss,norm=step(model,opt,x,y,epoch);updates+=1;total+=loss*len(ids);clipped+=norm>1
        records.append({'epoch':epoch,'loss':total/len(fit_ids),'clipped_batches':clipped})
    final_rng=rng();model.eval();final_health=health(model,data,'U1','S',fit_ids)
    assert final_health['passed'] and rng()==final_rng
    assert updates==60*math.ceil(len(fit_ids)/8)
    cp={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'fit_ids':fit_ids,
        'seed':seed,'fold':fold,'inner':inner,'initial_state_sha256':initial,'augmentation':False,
        'configuration':desc(OUT/'configuration.json')}
    cp['state_sha256']=state_digest(cp['model']);torch.save(cp,path/'checkpoint.pt')
    save(path/'history.json',records,immutable=True);save(path/'final_health.json',final_health,immutable=True)
    save(target,{'checkpoint':desc(path/'checkpoint.pt'),'configuration':desc(OUT/'configuration.json'),
        'fit_ids':fit_ids,'seed':seed,'fold':fold,'inner':inner,'updates':updates,'initial_state':initial,
        'final_state':cp['state_sha256'],'final_rng':final_rng,'seconds':time.perf_counter()-start},immutable=True)
    print('TRAINED',seed,fold,inner,'updates',updates,'seconds',round(time.perf_counter()-start,1),flush=True)
    del model,opt,cp;torch.cuda.empty_cache()


def load(seed,fold,inner=None,old=False):
    p=P3/f'runs/U1_S/seed_{seed}/fold_{fold}/checkpoint.pt' if old else folder(seed,fold,inner)/'checkpoint.pt'
    cp=torch.load(p,map_location='cpu',weights_only=True)
    assert state_digest(cp['model'])==cp['state_sha256']
    model=make_model('U1',seed,'S').cuda().eval();model.load_state_dict(cp['model'],strict=True)
    return model


def threshold(data,seed,fold,splits):
    target=OUT/f'thresholds/seed_{seed}/fold_{fold}.json'
    if target.exists():return
    rows=[]
    for part in splits:
        train(data,seed,fold,part['fit'],part['inner'])
        model=load(seed,fold,part['inner']);p,_=predict(model,data,part['calibration'])
        for sid,value in zip(part['calibration'],p):
            rows.append({'subject_id':sid,'label':data.labels[sid],'p':float(value),'fit_ids':part['fit'],'inner':part['inner']})
        del model
    params=fit_threshold(rows,data.train)
    save(target,{'seed':seed,'fold':fold,'oof':rows,'parameters':params,'source':'outer-train-only threefold OOF'},immutable=True)


def linear_features(raw):
    fs,p=welch(raw,fs=200,window='hann',nperseg=400,noverlap=200,nfft=400,detrend='constant',axis=-1)
    denominator=p[..., (fs>=1)&(fs<=40)].sum(-1)*.5
    assert np.all(denominator>0) and np.isfinite(denominator).all()
    features=[]
    for a,b in [(1,4),(4,8),(8,13),(13,30),(30,40)]:
        mask=(fs>=a)&((fs<=b) if b==40 else (fs<b))
        ratio=p[...,mask].sum(-1)*.5/denominator
        features.append(np.log(np.maximum(ratio,1e-8)))
    return np.stack(features,axis=-1).mean(axis=1).reshape(len(raw),95).astype(float)


def linear_fit(data):
    for kind in ['old','new']:
        x=linear_features(data.raw[kind]);p=OUT/f'linear/{kind}/features.npy';p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():assert np.array_equal(np.load(p),x)
        else:np.save(p,x)
        for fold in FOLDS:
            target=p.parent/f'fold_{fold}.json'
            if target.exists():continue
            ids=data.context.folds[f'fold_{fold}']['train'];indices=[data.index[s] for s in ids]
            model=make_pipeline(StandardScaler(),LogisticRegression(C=.1,penalty='l2',solver='lbfgs',class_weight=None,max_iter=2000,tol=1e-8))
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning);model.fit(x[indices],[data.labels[s] for s in ids])
            artifact=p.parent/f'fold_{fold}.joblib';joblib.dump(model,artifact)
            save(target,{'fit_ids':ids,'checkpoint':desc(artifact),'features':desc(p),'fold':fold,'input':kind,
                         'iterations':model[-1].n_iter_.tolist()},immutable=True)


def run():
    cfg=register();data=Data();linear_fit(data)
    for fold in FOLDS:
        data.select(fold)
        for seed in SEEDS:
            save(OUT/'status.json',{'state':'training_no_outer_release','fold':fold,'seed':seed,
                'completed_neural':len(list((OUT/'runs').rglob('trained.json'))),'updated_utc':now()})
            train(data,seed,fold,data.train)
            threshold(data,seed,fold,cfg['inner_splits'][str(fold)])
    config_check()
    assert len(list((OUT/'runs').rglob('trained.json')))==60
    assert len(list((OUT/'thresholds').rglob('fold_*.json')))==15
    assert len(list((OUT/'linear').rglob('fold_*.joblib')))==10
    files={p.relative_to(OUT).as_posix():digest(p) for root in ['runs','thresholds','linear'] for p in sorted((OUT/root).rglob('*')) if p.is_file()}
    save(OUT/'release_gate.json',{'passed':True,'configuration':desc(OUT/'configuration.json'),
        'neural_fits':60,'linear_fits':10,'fitted_artifacts':files,'locked_utc':now()},immutable=True)
    save(OUT/'status.json',{'state':'all_fits_locked_ready_for_evaluation','new_neural':60,'new_linear':10,'updated_utc':now()})
    print('RELEASE_GATE: 60 neural + 10 linear fits locked; ready for outer evaluation',flush=True)


if __name__=='__main__':run()
