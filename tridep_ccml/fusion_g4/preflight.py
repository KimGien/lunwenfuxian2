from .common import *
from .train import predict

def main():
    cfg=config_check();ctx=RecordContext();checks=[]
    for f in FOLDS:
        train=ctx.folds[f'fold_{f}']['train'];parts=cfg['inner_splits'][str(f)]
        assert parts==inner_splits(train,ctx.labels,f)
        assert sorted(s for p in parts for s in p['calibration'])==sorted(train)
        for p in parts:
            assert set(p['fit']).isdisjoint(p['calibration']) and set(p['fit'])|set(p['calibration'])==set(train)
            assert {ctx.labels[s] for s in p['fit']}=={0,1} and {ctx.labels[s] for s in p['calibration']}=={0,1}
        checks.append(f'fold {f}: exact P3 partitions, disjoint fit/calibration, both classes, train-only coverage')
    set_seed(11)
    for s in SEEDS:
        base=torch.load(G1/f'initial/seed_{s}.pt',weights_only=True)
        st=Model(CONDITIONS[0],s);est=Model(CONDITIONS[1],s)
        assert all(torch.equal(v,base[k]) for model in [st,est] for k,v in model.state_dict().items())
        assert all(torch.equal(v,est.state_dict()[k]) for k,v in st.state_dict().items())
        checks.append(f'seed {s}: paired fresh initial states from G1; no supervised parent model')
    ids=['a','b','c','d','e','f'];values=[.1,.3,.5,.5,.7,.9];ys=[0,0,1,0,1,1]
    rows=[{'subject_id':s,'label':y,'p':p,'fit_ids':[t for t in ids if t!=s]} for s,y,p in zip(ids,ys,values)]
    threshold=fit_threshold(rows,ids);sigmoid=fit_sigmoid(rows,ids)
    assert .01<=sigmoid['a']<=10 and -10<=sigmoid['b']<=10
    out=apply_sigmoid(np.array(values),sigmoid);assert np.isfinite(out).all() and np.all(np.diff(out)>=0)
    for bad in [[dict(r,fit_ids=ids) for r in rows],rows[:-1]]:
        try:validate_oof(bad,ids)
        except ValueError:pass
        else:raise AssertionError('OOF leakage/coverage failure was accepted')
    tied=[{'subject_id':s,'label':i%2,'p':.5,'fit_ids':[t for t in ids if t!=s]} for i,s in enumerate(ids)]
    assert fit_threshold(tied,ids)['threshold']==.5
    checks.append('OOF calibration: positive monotone sigmoid, leakage and missing coverage rejected, exact tie favors 0.5')
    data=Data();part=cfg['inner_splits']['1'][0]
    for c in CONDITIONS:
        model=Model(c,11).cuda();probe=predict(model,data,part['fit'][:8])
        assert probe['logits'].shape==(8,2 if c.endswith('-ST') else 3,2)
        assert float(probe['features'].std(0).norm())>0
        model.train();x,y=data.batch(part['fit'][:8]);model.objective(model(x),y,1).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checks.append(c+': legal real inner-fit batch, features vary and all parameter gradients finite; zero optimizer updates')
    save(OUT/'preflight.json',{'passed':True,'created_utc':now(),'checks':checks,'extra_optimizer_updates':0,
        'test_calibration':{'threshold':threshold,'sigmoid':sigmoid},'outer_val_test_inference':False,
        'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},immutable=True)
    print(f'G4 preflight passed: {len(checks)} grouped checks.',flush=True)

if __name__=='__main__':main()
