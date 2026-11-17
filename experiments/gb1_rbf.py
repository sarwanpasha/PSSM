import sys; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map
def kr(Ktr, ytr, Kte, lam):
    m=ytr.mean(); a=np.linalg.solve(Ktr+lam*np.eye(len(ytr)), ytr-m); return Kte@a+m
G=[1,2,4,8,16,32,64]; LAM=[1e-3,1e-2,1e-1,1]
for s in sys.argv[1:]:
    d=load('gb1_'+s,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']; tr,te=d['folds'][0]
    out={}
    for kind in ['sqrt','match']:
        F=feature_map(X,len(d['alphabet']),kind); F=(F.toarray() if hasattr(F,'toarray') else F).reshape(len(X),-1)
        sq=(F**2).sum(1); D2=np.maximum(sq[:,None]+sq[None,:]-2*F@F.T,0)
        itr,iva=train_test_split(tr,test_size=0.1,random_state=0)
        vals={}
        for g in G:
            K=np.exp(-g*D2)
            for l in LAM: vals[(g,l)]=spearmanr(y[iva],kr(K[np.ix_(itr,itr)],y[itr],K[np.ix_(iva,itr)],l)).statistic
        g,l=max(vals,key=vals.get); K=np.exp(-g*D2)
        out[kind]=(round(spearmanr(y[te],kr(K[np.ix_(tr,tr)],y[tr],K[np.ix_(te,tr)],l)).statistic,3),g,l)
        # oracle across g for reference
        out[kind+'_all']={gg:round(spearmanr(y[te],kr(np.exp(-gg*D2)[np.ix_(tr,tr)],y[tr],np.exp(-gg*D2)[np.ix_(te,tr)],l)).statistic,3) for gg in G}
    print(s,out,flush=True)
