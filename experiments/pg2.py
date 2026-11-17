"""ProteinGym multi-mutant assays: extrapolation to higher mutation order.
Splits: A) train on singles (+WT), test on doubles;  B) train on <=2 mutations, test on >=3 (if >=200 such variants).
Models: one-hot ridge (linear, unweighted), Gaussian unweighted (exp. Hamming), Gaussian composition-weighted."""
from common import *
import json, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
name=sys.argv[1]; NTR=int(sys.argv[2]) if len(sys.argv)>2 else 4000; NTE=5000
d=pd.read_csv(f'/mnt/user-data/uploads/PSSM/stage/pg/{name}.csv'); d['k']=d.mutant.str.count(':')+1
seqs=d.mutated_sequence.values; y=d.DMS_score.values.astype(float); k=d.k.values; L=len(seqs[0])
X=encode(list(seqs),PROTEIN,L); var=np.where((X!=X[0]).any(0))[0]; rng=np.random.default_rng(0)
WT=None
try:
    WT=np.exp(np.load(f'/mnt/user-data/uploads/PSSM/stage/pgw/esm2_t33_650M_UR50D__pg_{name}__wtlogp.npy'))
except Exception as e: print('no LM weights:',e,flush=True)
def run(tr,te,tag):
    tr=rng.choice(tr,min(NTR,len(tr)),replace=False); te=rng.choice(te,min(NTE,len(te)),replace=False); idx=np.concatenate([tr,te]); n1=len(tr)
    itr,iva=train_test_split(np.arange(n1),test_size=0.15,random_state=0); out={}; P={}
    kinds=['match','sqrt']+(['lm'] if WT is not None else [])
    for kind in kinds:
        if kind=='lm':
            Xi=X[idx]; m=len(idx); Fd=np.zeros((m,len(var),20),dtype=np.float32)
            for j,t in enumerate(var):
                ok=Xi[:,t]>=0; Fd[np.arange(m)[ok],j,Xi[ok,t]]=np.sqrt(WT[t,Xi[ok,t]])
            Fd=Fd.reshape(m,-1); s=(Fd**2).sum(1).astype(np.float64); K1=(Fd@Fd.T).astype(np.float64)
        else:
            F=feature_map(X[idx],20,kind).tocsr().astype(np.float32); s=sqn(F); K1=lin(F,F)
        D2=np.maximum(s[:,None]+s[None,:]-2*K1,0); sub=D2[:400,:400]; med=np.median(sub[sub>0])
        def fit(K,grid):
            best=None
            for l in grid:
                a,b=krr(K[np.ix_(itr,itr)],y[tr][itr],l); v=spearmanr(y[tr][iva],K[np.ix_(iva,itr)]@a+b).statistic
                if best is None or v>best[0]: best=(v,l)
            a,b=krr(K[:n1,:n1],y[tr],best[1]); return best[0],K[n1:,:n1]@a+b
        if kind=='match': v,p=fit(K1,[1e-2,1e-1,1,10,100]); P['linear']=p
        best=None
        for g in [0.1,0.25,0.5,1,2,4]:
            v,p=fit(np.exp(-g*D2/med),[1e-3,1e-2,1e-1,1])
            if best is None or v>best[0]: best=(v,p,g)
        P['gauss_'+kind]=best[1]; out['g_'+kind]=best[2]
    yt=y[te]; out.update({m:round(float(spearmanr(yt,p).statistic),4) for m,p in P.items()})
    keys=list(P); B=[]; 
    for _ in range(500):
        i=rng.integers(0,len(te),len(te)); B.append([spearmanr(yt[i],P[m][i]).statistic for m in ['linear','gauss_match','gauss_sqrt']])
    B=np.array(B); out['ci_gauss_minus_linear']=np.percentile(B[:,1]-B[:,0],[2.5,97.5]).round(3).tolist(); out['ci_weighted_minus_unweighted']=np.percentile(B[:,2]-B[:,1],[2.5,97.5]).round(3).tolist()
    out.update(split=tag,ntrain=n1,ntest=len(te)); return out
res=dict(assay=name,L=L,nvar=len(var),counts={int(a):int((k==a).sum()) for a in sorted(set(k))[:6]},runs=[])
if (k==2).sum()>=200: res['runs'].append(run(np.where(k==1)[0],np.where(k==2)[0],'1->2'))
if (k>=3).sum()>=200: res['runs'].append(run(np.where(k<=2)[0],np.where(k>=3)[0],'<=2->>=3'))
json.dump(res,open(f'pg_{name}.json','w')); print(json.dumps(res),flush=True)
