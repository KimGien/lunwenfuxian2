"""Same P3 fit, accepting a line-search stop only after strict KKT verification.

Registered after the original optimizer stopped and before G4 outer transforms.
The candidate parameters, objective, bounds and original solver options are unchanged.
"""
from scipy.optimize import minimize
from scipy.special import expit,logit
from .common import np,validate_oof

def fit_sigmoid_checked(rows,train_ids):
    validate_oof(rows,train_ids)
    y=np.array([r['label'] for r in rows],dtype=float)
    x=logit(np.clip([r['p'] for r in rows],1e-7,1-1e-7))
    def objective(v):
        a,b=v;z=a*x+b;error=expit(z)-y
        return np.mean(np.logaddexp(0,z)-y*z)+.005*((a-1)**2+b*b),np.array([(error*x).mean()+.01*(a-1),error.mean()+.01*b])
    result=minimize(objective,[1.,0.],jac=True,method='L-BFGS-B',bounds=[(.01,10),(-10,10)],
        options={'ftol':1e-13,'gtol':1e-9,'maxiter':1000})
    assert np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(result.jac).all()
    residuals=[]
    for value,g,lo,hi in zip(result.x,result.jac,[.01,-10.],[10.,10.]):
        assert lo<=value<=hi
        residuals.append(min(g,0) if value<=lo+1e-8 else max(g,0) if value>=hi-1e-8 else g)
    if not result.success:
        assert result.status==2 and str(result.message).startswith('ABNORMAL'), 'Unregistered optimizer failure'
        assert max(abs(g) for g in residuals)<=1e-7, 'Failed solver has no sufficiently stationary solution'
    return {'a':float(result.x[0]),'b':float(result.x[1]),'objective':float(result.fun),
        'fit_subject_ids':list(train_ids),'positive_slope':True,'iterations':int(result.nit)}
