import sys; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map
def kr(Ktr, ytr, Kte, lam):
    n=len(ytr); m=ytr.mean()
    a=np.linalg.solve(Ktr+lam*np.eye(n), ytr-m); return Kte@a+m
for s in sys.argv[1:]:
    d=load('gb1_'+s,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']
    tr,te=d['folds'][0]
    var=np.where((X!=X[0]).any(0))[0]; 
    F=feature_map(X,len(d['alphabet']),'sqrt'); F=F.toarray() if hasattr(F,'toarray') else F
    F=F.reshape(len(X),-1)
    K1=F@F.T
    itr,iva=train_test_split(tr,test_size=0.1,random_state=0)
    res={}
    for name,fn in [('lin',lambda K:K)]+[(f'poly{p}c{c}',(lambda p,c:(lambda K:(K+c)**p))(p,c)) for p in [2,3,4] for c in [0.5,1,2]]+[(f'rbf{g}',(lambda g:(lambda K:np.exp(-g*(np.diag(K1)[:,None]+np.diag(K1)[None,:]-2*K))))(g)) for g in [0.5,1,2,4]]:
        K=fn(K1); K=K/np.sqrt(np.outer(np.diag(K),np.diag(K)))
        best=max([1e-3,1e-2,1e-1,1,10],key=lambda l:spearmanr(y[iva],kr(K[np.ix_(itr,itr)],y[itr],K[np.ix_(iva,itr)],l)).statistic)
        p=kr(K[np.ix_(tr,tr)],y[tr],K[np.ix_(te,tr)],best); res[name]=spearmanr(y[te],p).statistic
    print(s,'varying pos',var, {k:round(v,3) for k,v in res.items()},flush=True)
