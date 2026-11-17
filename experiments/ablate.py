"""gamma ablation on GB1: Gaussian kernel on weighted (sqrt) vs unweighted (match = exponentiated Hamming) features.
Same selection protocol as gb1.py. Adds bootstrap CI, paired bootstrap difference, top-100 recall."""
from common import *
import json
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
split=sys.argv[1]; d=load('gb1_'+split,ROOT); y=d['y'].astype(float); tr,te=d['folds'][0]; X=encode(d['seqs'],PROTEIN,265)
splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
GS=[0.03,0.1,0.25,0.5,1,2,4,8]; LS=[1e-3,1e-2,1e-1,1]; out={'split':split}; preds={}
for kind in ['sqrt','match']:
    F=feature_map(X,20,kind).tocsr().astype(np.float32); s=sqn(F); K1=lin(F,F); D2=np.maximum(s[:,None]+s[None,:]-2*K1,0)
    sub=D2[np.ix_(tr[:400],tr[:400])]; med=np.median(sub[sub>0])
    def cv(K,l):
        P=np.zeros(len(y)); vi=np.concatenate([b for _,b in splits])
        for a_,b_ in splits:
            al,b0=krr(K[np.ix_(a_,a_)],y[a_],l); P[b_]=K[np.ix_(b_,a_)]@al+b0
        return spearmanr(y[vi],P[vi]).statistic
    best=None; dial={}
    for g in GS:
        K=np.exp(-g*D2/med); l=max(LS,key=lambda l:cv(K,l)); v=cv(K,l); al,b0=krr(K[np.ix_(tr,tr)],y[tr],l); p=K[np.ix_(te,tr)]@al+b0
        dial[g]=(round(float(v),4),round(float(spearmanr(y[te],p).statistic),4))
        if best is None or v>best[0]: best=(v,g,l,p)
    preds[kind]=best[3]; k=100; tt=set(np.argsort(-y[te])[:k]); 
    out[kind]=dict(test=float(spearmanr(y[te],best[3]).statistic),g=best[1],lam=best[2],dial=dial,top100=len(tt&set(np.argsort(-best[3])[:k]))/k)
rng=np.random.default_rng(0); yt=y[te]; B=[]
for _ in range(1000):
    i=rng.integers(0,len(te),len(te)); B.append((spearmanr(yt[i],preds['sqrt'][i]).statistic,spearmanr(yt[i],preds['match'][i]).statistic))
B=np.array(B); out['ci_sqrt']=np.percentile(B[:,0],[2.5,97.5]).round(3).tolist(); out['ci_match']=np.percentile(B[:,1],[2.5,97.5]).round(3).tolist(); out['diff_ci']=np.percentile(B[:,0]-B[:,1],[2.5,97.5]).round(4).tolist()
json.dump(out,open(f'ablate_{split}.json','w')); print(json.dumps(out),flush=True)
