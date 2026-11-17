import sys; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map
def kr(Ktr, ytr, Kte, lam):
    m=ytr.mean(); a=np.linalg.solve(Ktr+lam*np.eye(len(ytr)), ytr-m); return Kte@a+m
E='/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_%s.npy'
for s in sys.argv[1:]:
    d=load('gb1_'+s,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']; tr,te=d['folds'][0]
    F=feature_map(X,len(d['alphabet']),'sqrt'); F=(F.toarray() if hasattr(F,'toarray') else F).reshape(len(X),-1)
    sq=(F**2).sum(1); D2=np.maximum(sq[:,None]+sq[None,:]-2*F@F.T,0)
    Z=np.load(E%s); Z=(Z-Z[tr].mean(0))/(Z[tr].std(0)+1e-6); Ke=Z@Z.T/Z.shape[1]
    zs=(Z**2).sum(1)/Z.shape[1]; De=np.maximum(zs[:,None]+zs[None,:]-2*Ke,0)
    from sklearn.model_selection import KFold
    splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
    cands={}
    for g in [1,2,4,8]:
        Kw=np.exp(-g*D2)
        for ke_name,KE in [('lin',Ke),('rbf',np.exp(-De/ np.median(De[np.ix_(tr,tr)])))]:
            for a in [0,0.25,0.5,1,2,4,None]:
                K=KE if a is None else Kw+a*KE
                for l in [1e-3,1e-2,1e-1,1]:
                    if len(splits)>1:
                        P=np.zeros(len(y))
                        for itr,iva in splits: P[iva]=kr(K[np.ix_(itr,itr)],y[itr],K[np.ix_(iva,itr)],l)
                        cands[(g,ke_name,a,l)]=spearmanr(y[tr],P[tr]).statistic
                    else:
                        itr,iva=splits[0]; cands[(g,ke_name,a,l)]=spearmanr(y[iva],kr(K[np.ix_(itr,itr)],y[itr],K[np.ix_(iva,itr)],l)).statistic
    def test(k):
        g,kn,a,l=k; KE=Ke if kn=='lin' else np.exp(-De/np.median(De[np.ix_(tr,tr)])); K=KE if a is None else np.exp(-g*D2)+a*KE
        return round(spearmanr(y[te],kr(K[np.ix_(tr,tr)],y[tr],K[np.ix_(te,tr)],l)).statistic,3)
    best_h=max([k for k in cands if k[2] not in (None,)],key=cands.get)
    best_e=max([k for k in cands if k[2] is None],key=cands.get)
    best_w=max([k for k in cands if k[2]==0],key=cands.get)
    print(s,'hybrid',best_h,test(best_h),'| esm-only',best_e,test(best_e),'| gauss-wpsskm',best_w,test(best_w),flush=True)
