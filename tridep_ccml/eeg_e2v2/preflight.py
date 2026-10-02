"""Meaningful input/partition/pairing contracts before registered training."""
from .training import *


def run():
    assert read(OUT/'synthetic_gate.json')['passed'] and read(OUT/'real_gate.json')['passed']
    context=RecordContext();seen=[]
    for fold in FOLDS:
        split=context.folds[f'fold_{fold}']
        assert len(set(sum(split.values(),[])))==37
        assert set(split['train']).isdisjoint(split['test']) and set(split['train']).isdisjoint(split['val'])
        parts=inner_splits(split['train'],context.labels,fold)
        assert sorted(s for p in parts for s in p['calibration'])==sorted(split['train'])
        for part in parts:
            assert set(part['fit']).isdisjoint(part['calibration'])
            assert set(part['fit'])|set(part['calibration'])==set(split['train'])
        seen+=split['test']
    assert sorted(seen)==sorted(context.ids)
    t=np.arange(1000)/200
    tone=np.broadcast_to(np.sin(2*np.pi*10*t),(2,60,19,1000)).copy()
    z=linear_features(tone).reshape(2,19,5)
    assert (z[:,:,2]>z[:,:,[0,1,3,4]].max(axis=2)).all()
    assert np.allclose(linear_features(tone*100),linear_features(tone),atol=1e-9,rtol=0)
    normalized,_,_=subject_standardize(tone)
    assert np.allclose(normalized.mean((1,3)),0,atol=1e-7)
    assert np.allclose(normalized.std((1,3)),1,atol=1e-6)
    try:linear_features(tone*0)
    except AssertionError:pass
    else:raise AssertionError('Zero-power input accepted')
    initial=[]
    for seed in SEEDS:
        set_seed(seed);model=make_model('U1',seed,'S')
        sha=state_digest(model.state_dict())
        for fold in FOLDS:
            p=P3/f'runs/U1_S/seed_{seed}/fold_{fold}'
            assert sha==read(p/'trained.json')['initial_state_sha256']
            assert orders(context.folds[f'fold_{fold}']['train'],seed)==read(p/'batch_order.json')['orders']
        initial.append({'seed':seed,'initial_state_sha256':sha})
    report={'passed':True,'outer_folds':5,'inner_partitions':15,'parent_initialization_and_orders':15,
        'ten_hz_band_assignment':True,'spectral_scale_invariance':True,'zero_power_rejected':True,
        'per_person_standardization':True,'initial_states':initial,'source':desc(__file__)}
    save(OUT/'preflight.json',report,immutable=True);print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':run()
