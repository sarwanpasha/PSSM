"""LM-weighted vs unweighted Gaussian kernel on GB1, same protocol as ablate.py.
LM features: phi[t,a] = [x_t=a] * sqrt(p(a | WT masked at t)), p from ESM-2 650M, correct WT background."""
from common import *
import json
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
split=sys.argv[1]; fast=len(sys.argv)>2
W=np.load('/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_wtlogp_fixed.npy')
d=load('gb1_'+split,ROOT); y=d['y'].astype(float); tr,te=d['folds'][0]; X=encode(d['seqs'],PROTEIN,265)
var=[38,39,40,53]; n=len(y); oh=np.zeros((n,4,20)); 
for j,t in enumerate(var): oh[np.arange(n),j,X[:,t]]=1
Pw=np.exp(W[var])  # (4,20)
FE={'match':oh.reshape(n,-1),'lm':(oh*np.sqrt(Pw)[None]).reshape(n,-1)}
zs=sum(W[t,X[:,t]] for t in var)
splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
GS=[0.5,1,2,4] if fast else [0.03,0.1,0.25,0.5,1,2,4,8]; LS=[1e-2,1e-1] if fast else [1e-3,1e-2,1e-1,1]
out={'split':split,'zeroshot':float(spearmanr(y[te],zs[te]).statistic),'p_at_var_sites':{f'{t+1}':{PROTEIN[a]:round(float(Pw[j,a]),3) for a in np.argsort(-Pw[j])[:4]} for j,t in enumerate(var)}}; preds={}
for kind,F in FE.items():
    s=(F**2).sum(1); D2=np.maximum(s[:,None]+s[None,:]-2*F@F.T,0); sub=D2[np.ix_(tr[:400],tr[:400])]; med=np.median(sub[sub>0])
    def cv(K,l):
        P=np.zeros(n); vi=np.concatenate([b for _,b in splits])
        for a_,b_ in splits:
            al,b0=krr(K[np.ix_(a_,a_)],y[a_],l); P[b_]=K[np.ix_(b_,a_)]@al+b0
        return spearmanr(y[vi],P[vi]).statistic
    best=None; dial={}
    for g in GS:
        K=np.exp(-g*D2/med); l=max(LS,key=lambda l:cv(K,l)); v=cv(K,l); al,b0=krr(K[np.ix_(tr,tr)],y[tr],l); p=K[np.ix_(te,tr)]@al+b0
        dial[g]=(round(float(v),4),round(float(spearmanr(y[te],p).statistic),4))
        if best is None or v>best[0]: best=(v,g,l,p)
    preds[kind]=best[3]; tt=set(np.argsort(-y[te])[:100])
    out[kind]=dict(test=float(spearmanr(y[te],best[3]).statistic),g=best[1],lam=best[2],dial=dial,top100=len(tt&set(np.argsort(-best[3])[:100]))/100)
rng=np.random.default_rng(0); yt=y[te]; B=[]
for _ in range(1000):
    i=rng.integers(0,len(te),len(te)); B.append((spearmanr(yt[i],preds['lm'][i]).statistic,spearmanr(yt[i],preds['match'][i]).statistic))
B=np.array(B); out['ci_lm']=np.percentile(B[:,0],[2.5,97.5]).round(3).tolist(); out['ci_match']=np.percentile(B[:,1],[2.5,97.5]).round(3).tolist(); out['diff_ci']=np.percentile(B[:,0]-B[:,1],[2.5,97.5]).round(4).tolist()
json.dump(out,open(f'lmw_{split}.json','w')); print(json.dumps(out),flush=True)
