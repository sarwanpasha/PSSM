"""GB1: zero-shot score as covariate (Hsu-style augmented regression) on top of each kernel.
Models: zero-shot alone; one-hot ridge; one-hot ridge + zs; Gaussian unweighted (+zs); Gaussian LM-weighted (+zs).
Semiparametric KRR: min ||y - Z b - K a||^2 + lam a'K a, Z = [1, zs]."""
from common import *
import json
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
split=sys.argv[1]; fast=len(sys.argv)>2
W=np.load('/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_wtlogp_fixed.npy')
d=load('gb1_'+split,ROOT); y=d['y'].astype(float); tr,te=d['folds'][0]; X=encode(d['seqs'],PROTEIN,265)
var=[38,39,40,53]; n=len(y); oh=np.zeros((n,4,20))
for j,t in enumerate(var): oh[np.arange(n),j,X[:,t]]=1
Pw=np.exp(W[var]); zs=sum(W[t,X[:,t]] for t in var); zs=(zs-zs.mean())/zs.std()
FE={'unweighted':oh.reshape(n,-1),'lm':(oh*np.sqrt(Pw)[None]).reshape(n,-1)}
splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
GS=[0.5,1,2] if fast else [0.03,0.1,0.25,0.5,1,2,4]; LS=[1e-2,1e-1] if fast else [1e-3,1e-2,1e-1,1]
def fit(K,idx,yy,lam,Z=None):
    """returns (alpha,beta) for rows idx; Z is covariate matrix for idx or None"""
    Ki=np.linalg.inv(K+lam*np.eye(len(idx)))
    if Z is None:
        b=np.array([yy.mean()]); a=Ki@(yy-yy.mean()); return a,b
    b=np.linalg.solve(Z.T@Ki@Z+1e-10*np.eye(Z.shape[1]),Z.T@Ki@yy); a=Ki@(yy-Z@b); return a,b
def predict(Kte,a,b,Zte=None): return Kte@a + (b[0] if Zte is None else Zte@b)
Zall=np.c_[np.ones(n),zs]
def evaluate(Kfull,usez):
    def cv(l):
        P=np.zeros(n); vi=np.concatenate([bb for _,bb in splits])
        for a_,b_ in splits:
            al,be=fit(Kfull[np.ix_(a_,a_)],a_,y[a_],l,Zall[a_] if usez else None)
            P[b_]=predict(Kfull[np.ix_(b_,a_)],al,be,Zall[b_] if usez else None)
        return spearmanr(y[vi],P[vi]).statistic
    l=max(LS,key=cv); v=cv(l)
    al,be=fit(Kfull[np.ix_(tr,tr)],tr,y[tr],l,Zall[tr] if usez else None)
    return v,predict(Kfull[np.ix_(te,tr)],al,be,Zall[te] if usez else None),l
res={'split':split,'ntrain':len(tr),'zeroshot_alone':float(spearmanr(y[te],zs[te]).statistic)}; preds={}
for kind,F in FE.items():
    s=(F**2).sum(1); K1=F@F.T; D2=np.maximum(s[:,None]+s[None,:]-2*K1,0); sub=D2[np.ix_(tr[:400],tr[:400])]; med=np.median(sub[sub>0])
    if kind=='unweighted':
        for uz in [False,True]:
            v,p,l=evaluate(K1,uz); preds['onehot_ridge'+('+zs' if uz else '')]=p
    for uz in [False,True]:
        best=None
        for g in GS:
            v,p,l=evaluate(np.exp(-g*D2/med),uz)
            if best is None or v>best[0]: best=(v,p,g,l)
        preds[f'gauss_{kind}'+('+zs' if uz else '')]=best[1]; res[f'g_{kind}'+('_zs' if uz else '')]=best[2]
yt=y[te]; res.update({m:round(float(spearmanr(yt,p).statistic),4) for m,p in preds.items()})
res['top100']={m:len(set(np.argsort(-yt)[:100])&set(np.argsort(-p)[:100]))/100 for m,p in preds.items()}
rng=np.random.default_rng(0); keys=list(preds); B=[]
for _ in range(600):
    i=rng.integers(0,len(te),len(te)); B.append([spearmanr(yt[i],preds[m][i]).statistic for m in keys])
B=np.array(B); base=keys.index('gauss_lm')
res['ci']={m:np.percentile(B[:,j],[2.5,97.5]).round(3).tolist() for j,m in enumerate(keys)}
res['vs_gauss_lm']={m:np.percentile(B[:,j]-B[:,base],[2.5,97.5]).round(3).tolist() for j,m in enumerate(keys) if m!='gauss_lm'}
json.dump(res,open(f'zscov_{split}.json','w')); print(json.dumps(res),flush=True)
