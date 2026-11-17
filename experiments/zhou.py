"""Pretrained vs learned per-site decay on GB1.
Compares product kernels k(x,y) = prod_t kappa_t(x_t,y_t) that differ only in where the decay comes from:
  uniform        kappa_t = exp(-2g)             on mismatch   (1 free parameter)
  per-site       kappa_t = rho_t                on mismatch   (L free parameters, fitted by GP marginal likelihood)
                 -- the family used by Zhou et al. for fitness landscapes
  per-site-residue, learned: kappa_t = exp(-g(v_{t,a}+v_{t,b})) with v fitted (L*A free parameters)
  per-site-residue, pretrained (ours): v_{t,a} = p_ESM(a | WT masked at t), only g fitted
Also a distance-only variance-component kernel k(d) = sum_k lambda_k W_k(d) with lambda_k >= 0 fitted,
W_k the Krawtchouk polynomials for the L-site, A-letter Hamming graph (Zhou et al. 2022 family).
All share the protocol of ablate.py: selection on training data only, bootstrap CIs on the test set."""
from common import *
import json
from math import comb
from scipy.stats import spearmanr
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
split=sys.argv[1]; MAXFIT=int(sys.argv[2]) if len(sys.argv)>2 else 1200
W=np.load('/mnt/user-data/uploads/PSSM/stage/emb/esm2_t33_650M_UR50D__gb1_wtlogp_fixed.npy')
d=load('gb1_'+split,ROOT); y=d['y'].astype(float); tr,te=d['folds'][0]; X=encode(d['seqs'],PROTEIN,265)
var=[38,39,40,53]; L=4; A=20; n=len(y); Xv=X[:,var]
def mismatch_t(t, rows=None, cols=None):
    a=Xv[:,t] if rows is None else Xv[rows,t]; b=Xv[:,t] if cols is None else Xv[cols,t]
    return (a[:,None]!=b[None,:])
D=np.zeros((n,n),dtype=np.int8)
for t in range(L): D+=mismatch_t(t)
def logK_site(logr, rows=None, cols=None):
    m=len(Xv) if rows is None else len(rows); k=len(Xv) if cols is None else len(cols)
    out=np.zeros((m,k),dtype=np.float32)
    for t in range(L): out+=logr[t]*mismatch_t(t,rows,cols)
    return out
rng=np.random.default_rng(0)
splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
def krr(K,yy,lam):
    b=yy.mean(); a=np.linalg.solve(K+lam*np.eye(len(yy)),yy-b); return a,b
def cv_score(K,lam):
    P=np.zeros(n); vi=np.concatenate([b for _,b in splits])
    for a_,b_ in splits:
        al,b0=krr(K[np.ix_(a_,a_)],y[a_],lam); P[b_]=K[np.ix_(b_,a_)]@al+b0
    return spearmanr(y[vi],P[vi]).statistic
def select(Kfull,lams=(1e-3,1e-2,1e-1,1)):
    lam=max(lams,key=lambda l:cv_score(Kfull,l)); al,b0=krr(Kfull[np.ix_(tr,tr)],y[tr],lam)
    return cv_score(Kfull,lam),Kfull[np.ix_(te,tr)]@al+b0,lam
fit=rng.choice(tr,min(MAXFIT,len(tr)),replace=False)          # subsample for marginal-likelihood fitting
yf=(y[fit]-y[fit].mean())/y[fit].std(); Df=D[np.ix_(fit,fit)]
def nll(K,s2):
    Ks=K+s2*np.eye(len(K))
    try: Lc=np.linalg.cholesky(Ks)
    except np.linalg.LinAlgError: return 1e6
    a=np.linalg.solve(Lc.T,np.linalg.solve(Lc,yf)); return float(0.5*yf@a+np.log(np.diag(Lc)).sum())
preds={}; info={}
# --- uniform
best=None
for g in [0.03,0.1,0.25,0.5,1,2,4,8]:
    K=np.exp(-g*D.astype(float)); v,p,lam=select(K)
    if best is None or v>best[0]: best=(v,p,g,lam)
preds['uniform']=best[1]; info['uniform_g']=best[2]
# --- per-site decay, fitted by GP marginal likelihood (Zhou-style)
def obj_site(th):
    r=1/(1+np.exp(-th[:L]))                                   # rho_t in (0,1)
    K=np.exp(logK_site(np.log(r),fit,fit).astype(np.float64)); return nll(K,np.exp(th[L]))
r0=np.concatenate([np.zeros(L),[np.log(0.1)]])
opt=minimize(obj_site,r0,method='Nelder-Mead',options=dict(maxiter=300,fatol=1e-4,xatol=1e-3))
rho=1/(1+np.exp(-opt.x[:L])); info['rho_per_site']=rho.round(4).tolist()
Ksite=np.exp(logK_site(np.log(rho)).astype(np.float64))
_,preds['per_site_learned'],info['lam_site']=select(Ksite)
# --- per-site-residue, learned (L*A params) with ridge-like shrinkage toward uniform
oh=np.zeros((n,L,A)); 
for j in range(L): oh[np.arange(n),j,Xv[:,j]]=1
def Kfrom(v,g):
    F=(oh*np.sqrt(np.maximum(v,1e-8))[None]).reshape(n,-1); s=(F**2).sum(1)
    return np.exp(-g*np.maximum(s[:,None]+s[None,:]-2*F@F.T,0))
def obj_full(th):
    v=np.exp(th[:L*A].reshape(L,A)); g=np.exp(th[L*A])
    F=(oh[fit]*np.sqrt(v)[None]).reshape(len(fit),-1); s=(F**2).sum(1)
    K=np.exp(-g*np.maximum(s[:,None]+s[None,:]-2*F@F.T,0))
    return nll(K,np.exp(th[L*A+1]))+0.01*np.sum(th[:L*A]**2)
th0=np.concatenate([np.zeros(L*A),[0.0,np.log(0.1)]])
opt2=minimize(obj_full,th0,method='L-BFGS-B',options=dict(maxiter=25))
vlearn=np.exp(opt2.x[:L*A].reshape(L,A)); glearn=float(np.exp(opt2.x[L*A]))
_,preds['per_site_residue_learned'],info['lam_fullres']=select(Kfrom(vlearn,glearn)); info['g_fullres']=glearn
# --- per-site-residue, pretrained (ours)
P=np.exp(W[var]); best=None
for g in [0.03,0.1,0.25,0.5,1,2,4,8]:
    K=Kfrom(P,g); v,p,lam=select(K)
    if best is None or v>best[0]: best=(v,p,g,lam)
preds['per_site_residue_pretrained']=best[1]; info['g_lm']=best[2]
# --- variance-component kernel (Zhou et al. 2022 family): estimate how phenotypic correlation decays with
# mutational distance on the TRAINING data, solve for non-negative variance components, then GP regression.
def krawtchouk(k,dd,L=L,A=A): return sum((-1)**q*(A-1)**(k-q)*comb(dd,q)*comb(L-dd,k-q) for q in range(0,k+1))
Wmat=np.array([[krawtchouk(k,dd) for dd in range(L+1)] for k in range(L+1)],dtype=float)   # (k, d)
yc=y[tr]-y[tr].mean(); Dtr=D[np.ix_(tr,tr)]
cd=np.zeros(L+1); cnt=np.zeros(L+1)
for dd in range(L+1):
    m=(Dtr==dd)
    if m.sum(): cd[dd]=float((np.outer(yc,yc)[m]).mean()); cnt[dd]=m.sum()
from scipy.optimize import nnls
Wn=Wmat/np.abs(Wmat).max(1,keepdims=True)                       # scale rows for conditioning
wts=np.sqrt(np.maximum(cnt,1)); lam_k,_=nnls((Wn.T*wts[:,None]),cd*wts)
info['corr_vs_distance']=(cd/max(abs(cd[0]),1e-12)).round(4).tolist()
info['variance_components']=(lam_k/max(lam_k.sum(),1e-12)).round(4).tolist()
Kvcf=np.tensordot(lam_k,Wn,axes=(0,0))[D]
dg=np.diag(Kvcf).copy(); dg[dg<=0]=1.0; Kvcf=Kvcf/np.sqrt(np.outer(dg,dg))
_,preds['variance_component'],info['lam_vc']=select(Kvcf)
yt=y[te]; res={'split':split,'ntrain':len(tr),'info':info}
res.update({m:round(float(spearmanr(yt,p).statistic),4) for m,p in preds.items()})
keys=list(preds); B=[]
for _ in range(800):
    i=rng.integers(0,len(te),len(te)); B.append([spearmanr(yt[i],preds[m][i]).statistic for m in keys])
B=np.array(B); base=keys.index('per_site_residue_pretrained')
res['ci']={m:np.percentile(B[:,j],[2.5,97.5]).round(3).tolist() for j,m in enumerate(keys)}
res['vs_ours']={m:np.percentile(B[:,j]-B[:,base],[2.5,97.5]).round(3).tolist() for j,m in enumerate(keys) if m!='per_site_residue_pretrained'}
json.dump(res,open(f'zhou_{split}.json','w')); print(json.dumps(res),flush=True)
