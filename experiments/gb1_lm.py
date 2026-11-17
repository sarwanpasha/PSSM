import sys; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map
W=np.load('/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_sampled__wtlogp.npy')  # (L,20) logp
def kr(Ktr, ytr, Kte, lam, Mtr=None, Mte=None):
    # kernel ridge with intercept and optional linear covariates M (zero-shot score), via augmented solve
    if Mtr is None:
        m=ytr.mean(); a=np.linalg.solve(Ktr+lam*np.eye(len(ytr)), ytr-m); return Kte@a+m
    Z=np.c_[np.ones(len(ytr)),Mtr]; Ki=np.linalg.inv(Ktr+lam*np.eye(len(ytr)))
    b=np.linalg.solve(Z.T@Ki@Z+1e-8*np.eye(Z.shape[1]),Z.T@Ki@ytr); a=Ki@(ytr-Z@b)
    return Kte@a+np.c_[np.ones(len(Kte)),Mte]@b
def evaluate(s, variants):
    d=load('gb1_'+s,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']; tr,te=d['folds'][0]
    A=20; n,L=X.shape
    Fc=feature_map(X,A,'sqrt'); Fc=(Fc.toarray() if hasattr(Fc,'toarray') else Fc).reshape(n,L,A)
    oh=np.zeros((n,L,A)); oh[np.arange(n)[:,None],np.arange(L)[None,:],X]=1
    P=np.exp(W)[None]  # (1,L,A)
    zs=(oh*W[None]).sum((1,2)); zs=zs-zs.max()
    splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
    out={}
    for vname,(F,useZ) in variants(Fc,oh,P).items():
        F=F.reshape(n,-1); sq=(F**2).sum(1); D2=np.maximum(sq[:,None]+sq[None,:]-2*F@F.T,0)
        med=np.median(D2[np.ix_(tr[:500],tr[:500])][D2[np.ix_(tr[:500],tr[:500])]>0])
        M=zs[:,None] if useZ else None
        sc={}
        for g in [0.5,1,2]:
            K=np.exp(-g*D2/med)
            for l in [1e-2,1e-1]:
                Pv=np.zeros(n)
                for itr,iva in splits: Pv[iva]=kr(K[np.ix_(itr,itr)],y[itr],K[np.ix_(iva,itr)],l,None if M is None else M[itr],None if M is None else M[iva])
                vi=np.concatenate([b for _,b in splits]); sc[(g,l)]=spearmanr(y[vi],Pv[vi]).statistic
        g,l=max(sc,key=sc.get); K=np.exp(-g*D2/med)
        p=kr(K[np.ix_(tr,tr)],y[tr],K[np.ix_(te,tr)],l,None if M is None else M[tr],None if M is None else M[te])
        out[vname]=round(spearmanr(y[te],p).statistic,3)
    out['zeroshot']=round(spearmanr(y[te],zs[te]).statistic,3)
    print(s,out,flush=True)
V=lambda Fc,oh,P:{'comp':(Fc,False),'lmw':(oh*np.sqrt(P),False),}
for s in sys.argv[1:]: evaluate(s,V)
