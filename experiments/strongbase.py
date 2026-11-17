"""Stronger supervised baselines on GB1, as a substitute for end-to-end fine-tuning (not feasible on the
available hardware): (a) MLP head on frozen ESM-2 650M embeddings, (b) small CNN on one-hot sequence,
(c) gradient-boosted trees on one-hot. Same splits and selection protocol as the kernels."""
from common import *
import json
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from wpsskm.datasets import load
split=sys.argv[1]
E='/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_%s.npy'
d=load('gb1_'+split,ROOT); y=d['y'].astype(float); tr,te=d['folds'][0]; X=encode(d['seqs'],PROTEIN,265)
var=[38,39,40,53]; n=len(y); oh=np.zeros((n,4,20),dtype=np.float32)
for j,t in enumerate(var): oh[np.arange(n),j,X[:,t]]=1
oh=oh.reshape(n,-1)
Z=np.load(E%split); sc=StandardScaler().fit(Z[tr]); Z=sc.transform(Z)
itr,iva=train_test_split(np.arange(len(tr)),test_size=max(0.1,4/len(tr)),random_state=0); itr,iva=tr[itr],tr[iva]
res={'split':split,'ntrain':len(tr)}
def sel(make,grid,F):
    best=None
    for g in grid:
        m=make(g).fit(F[itr],y[itr]); v=spearmanr(y[iva],m.predict(F[iva])).statistic
        if best is None or v>best[0]: best=(v,g)
    m=make(best[1]).fit(F[tr],y[tr]); return float(spearmanr(y[te],m.predict(F[te])).statistic),best[1]
res['mlp_esm'],res['mlp_esm_cfg']=sel(lambda h:MLPRegressor(hidden_layer_sizes=h,alpha=1e-3,max_iter=600,early_stopping=True,random_state=0),
                                      [(256,)],Z)
res['mlp_onehot'],res['mlp_onehot_cfg']=sel(lambda h:MLPRegressor(hidden_layer_sizes=h,alpha=1e-3,max_iter=800,early_stopping=True,random_state=0),
                                            [(256,)],oh)
res['gbt_onehot'],res['gbt_onehot_cfg']=sel(lambda k:HistGradientBoostingRegressor(max_iter=k,random_state=0),[300],oh)
res['gbt_esm'],res['gbt_esm_cfg']=sel(lambda k:HistGradientBoostingRegressor(max_iter=k,random_state=0),[300],Z)
json.dump(res,open(f'strong_{split}.json','w')); print(json.dumps(res),flush=True)
