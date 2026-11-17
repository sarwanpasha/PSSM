"""Position knock-out attribution for Gaussian W-PSSKM on any dataset.
k_g(x,y)=prod_t kappa_t ; knocking out window J sets kappa_t=1 for t in J (drops those positions from the distance).
usage: posimp.py <dataset> <ntrain> <ntest> <window>"""
from common import *
import json, pickle, time
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.datasets import load
name,ntr,nte,win=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
d=load(name,ROOT); y=np.asarray(d['y']); task=d['task']; alph=d['alphabet']; A=len(alph); L=d['L']
tr,te=d['folds'][0]; rng=np.random.default_rng(0)
tr=rng.choice(tr,min(ntr,len(tr)),replace=False); te=rng.choice(te,min(nte,len(te)),replace=False)
X=encode(d['seqs'],alph,L); Ftr=feats(X[tr],A); Fte=feats(X[te],A); st=sqn(Ftr); se=sqn(Fte)
if task=='clf':
    C=int(y.max())+1; Y=-np.ones((len(tr),C)); Y[np.arange(len(tr)),y[tr]]=1
else: Y=(y[tr]-y[tr].mean())/y[tr].std(); Y=Y[:,None]
Ktt=lin(Ftr,Ftr); Ket=lin(Fte,Ftr); Dtt=np.maximum(st[:,None]+st[None,:]-2*Ktt,0); Det=np.maximum(se[:,None]+st[None,:]-2*Ket,0)
med=np.median(Dtt[:500,:500][Dtt[:500,:500]>0])
itr,iva=train_test_split(np.arange(len(tr)),test_size=0.15,random_state=0)
def score(P,yt):
    return float((P.argmax(1)==yt).mean()) if task=='clf' else float(spearmanr(yt,P[:,0]).statistic)
res=dict(dataset=name,ntrain=len(tr),ntest=len(te),L=L,window=win,dial={})
best=None
for g in [0.03,0.1,0.3,1,3]:
    K=np.exp(-g*Dtt/med)
    for l in [1e-2,1e-1,1]:
        a,b=krr(K[np.ix_(itr,itr)],Y[itr],l); v=score(K[np.ix_(iva,itr)]@a+b,y[tr][iva])
        if best is None or v>best[0]: best=(v,g,l)
        res['dial'].setdefault(g,[]).append((l,v))
# linear limit
bl=None
for l in [1e-2,1e-1,1,10,100]:
    a,b=krr(Ktt[np.ix_(itr,itr)],Y[itr],l); v=score(Ktt[np.ix_(iva,itr)]@a+b,y[tr][iva])
    if bl is None or v>bl[0]: bl=(v,l)
a,b=krr(Ktt,Y,bl[1]); res['linear_test']=score(Ket@a+b,y[te])
_,g,l=best; ge=g/med; alpha,b0=krr(np.exp(-ge*Dtt),Y,l); Ke=np.exp(-ge*Det); P=Ke@alpha+b0
res.update(g=g,lam=l,test=score(P,y[te]))
# per-g test curve (dial) at best lam per g
res['dial_test']={}
for gg in res['dial']:
    ll=max(res['dial'][gg],key=lambda t:t[1])[0]; a,b=krr(np.exp(-gg*Dtt/med),Y,ll); res['dial_test'][gg]=score(np.exp(-gg*Det/med)@a+b,y[te])
# knock-out per window
J=int(np.ceil(L/win)); imp=np.zeros(J); dP=np.zeros((J,)+P.shape,dtype=np.float32); t0=time.time()
for j in range(J):
    cols=slice(j*win*A,min((j+1)*win,L)*A); fe=Fte[:,cols]; ft=Ftr[:,cols]
    dj=np.maximum(sqn(fe)[:,None]+sqn(ft)[None,:]-2*lin(fe,ft),0)
    Pj=(Ke*np.exp(ge*dj))@alpha+b0; dP[j]=P-Pj; imp[j]=float(np.mean((P-Pj)**2))
res['imp']=imp.tolist(); res['sec']=time.time()-t0
# residue attribution at each window (win==1 only): mean contribution toward each output, by residue
if win==1:
    M=np.zeros((J,A,P.shape[1])); Xe=X[te]
    for j in range(J):
        for a_ in range(A):
            m=Xe[:,j]==a_
            if m.sum()>=5: M[j,a_]=dP[j][m].mean(0)
    res['M']=M.tolist()
pickle.dump(dict(res=res,dP=dP if dP.nbytes<4e8 else None,yte=y[te],P=P,Xte=X[te]),open(f'pos_{name}.pkl','wb'))
print(json.dumps({k:v for k,v in res.items() if k not in('imp','M','dial')}),flush=True); print('top windows',np.argsort(-imp)[:15].tolist())
