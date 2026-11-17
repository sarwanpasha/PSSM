from common import *
import itertools, json, pickle
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.datasets import load
split=sys.argv[1]
d=load('gb1_'+split,ROOT); seqs=d['seqs']; y=d['y'].astype(float); tr,te=d['folds'][0]; A=20
X=encode(seqs,PROTEIN,265); F=feats(X,A); s=sqn(F); K1=lin(F,F)
D2=np.maximum(s[:,None]+s[None,:]-2*K1,0); med=np.median(D2[np.ix_(tr[:400],tr[:400])][D2[np.ix_(tr[:400],tr[:400])]>0])
splits=[(tr[a],tr[b]) for a,b in KFold(5,shuffle=True,random_state=0).split(tr)] if len(tr)<1000 else [tuple(train_test_split(tr,test_size=0.1,random_state=0))]
def cv(K,l):
    P=np.zeros(len(y)); vi=np.concatenate([b for _,b in splits])
    for a_,b_ in splits:
        al,b0=krr(K[np.ix_(a_,a_)],y[a_],l); P[b_]=K[np.ix_(b_,a_)]@al+b0
    return spearmanr(y[vi],P[vi]).statistic
GS=[0.03,0.1,0.25,0.5,1,2,4,8]; LS=[1e-3,1e-2,1e-1,1]
res={'split':split,'ntrain':len(tr),'med':float(med),'dial':{}}
best=None
for g in GS:
    K=np.exp(-g*D2/med); l=max(LS,key=lambda l:cv(K,l)); v=cv(K,l)
    al,b0=krr(K[np.ix_(tr,tr)],y[tr],l); t=spearmanr(y[te],K[np.ix_(te,tr)]@al+b0).statistic
    res['dial'][g]=dict(val=float(v),test=float(t),lam=l)
    if best is None or v>best[0]: best=(v,g,l)
l_lin=max(LS+[10,100],key=lambda l:cv(K1,l)); al,b0=krr(K1[np.ix_(tr,tr)],y[tr],l_lin)
res['linear']=dict(test=float(spearmanr(y[te],K1[np.ix_(te,tr)]@al+b0).statistic),lam=l_lin)
_,g,l=best; geff=g/med; res['g']=g; res['lam']=l
K=np.exp(-geff*D2); alpha,b0=krr(K[np.ix_(tr,tr)],y[tr],l); pred=K[np.ix_(te,tr)]@alpha+b0
res['test']=float(spearmanr(y[te],pred).statistic)
# degree truncation: Spearman of f_{<=M}
parts=degree_parts(K1[np.ix_(te,tr)],s[te],s[tr],geff,alpha,60); cum=np.cumsum(parts,0)
res['trunc']=[float(spearmanr(y[te],c).statistic) if np.std(c)>0 else 0.0 for c in cum[:40]]
res['trunc_maxerr']=float(np.abs(cum[-1]+b0-pred).max())
# full landscape 20^4
wt=seqs[0]; var=[38,39,40,53]; assert y[0]==1.0
Xw=X[0].copy(); grid=np.array(list(itertools.product(range(20),repeat=4)),dtype=np.int16)
f=np.zeros(len(grid)); fl=np.zeros(len(grid)); Ftr=F[tr]; al_l,b_l=krr(K1[np.ix_(tr,tr)],y[tr],l_lin)
for i in range(0,len(grid),8000):
    Xg=np.tile(Xw,(len(grid[i:i+8000]),1)); Xg[:,var]=grid[i:i+8000]; Fg=feats(Xg,A); Kg=lin(Fg,Ftr)
    f[i:i+8000]=gauss(Kg,sqn(Fg),s[tr],geff)@alpha+b0; fl[i:i+8000]=Kg@al_l+b_l
def anova(f):
    T=f.reshape(20,20,20,20); mu=T.mean(); comp={}; 
    for r in range(1,5):
        for S in itertools.combinations(range(4),r):
            m=T.mean(axis=tuple(i for i in range(4) if i not in S),keepdims=True)-mu
            for q in range(1,r):
                for S2 in itertools.combinations(S,q): m=m-comp[S2]
            comp[S]=m
    tot=((T-mu)**2).mean()
    return mu,comp,{S:float((c**2).mean()/tot) for S,c in comp.items()}
mu,comp,share=anova(f); _,compl,sharel=anova(fl)
res['anova_gauss']={str(k):v for k,v in share.items()}; res['anova_linear']={str(k):v for k,v in sharel.items()}
res['order_gauss']=[sum(v for k,v in share.items() if len(k)==r) for r in range(1,5)]
res['order_linear']=[sum(v for k,v in sharel.items() if len(k)==r) for r in range(1,5)]
# measured singles / doubles for validation
key={tuple(X[i,var]):i for i in range(len(seqs))}; w=tuple(X[0,var]); gidx={tuple(gv):i for i,gv in enumerate(grid)}
teset=set(te.tolist()); sing=[]; dbl=[]
for k,i in key.items():
    diff=[p for p in range(4) if k[p]!=w[p]]
    if len(diff)==1: sing.append((diff[0],k[diff[0]],y[i],f[gidx[k]],fl[gidx[k]],i in teset))
    if len(diff)==2:
        a=list(w); a[diff[0]]=k[diff[0]]; b=list(w); b[diff[1]]=k[diff[1]]
        if tuple(a) in key and tuple(b) in key:
            eps_d=y[i]-y[key[tuple(a)]]-y[key[tuple(b)]]+1.0
            eps_m=f[gidx[k]]-f[gidx[tuple(a)]]-f[gidx[tuple(b)]]+f[gidx[w]]
            eps_l=fl[gidx[k]]-fl[gidx[tuple(a)]]-fl[gidx[tuple(b)]]+fl[gidx[w]]
            dbl.append((diff[0],diff[1],eps_d,eps_m,eps_l,i in teset))
res['singles']=sing; res['doubles']=dbl
dt=np.array([r for r in dbl if r[5]]); 
if len(dt)>5: res['epi_heldout']=dict(n=len(dt),gauss=float(spearmanr(dt[:,2],dt[:,3]).statistic),linear=float(spearmanr(dt[:,2],dt[:,4]).statistic),pearson_gauss=float(np.corrcoef(dt[:,2],dt[:,3])[0,1]))
# top-k discovery on test
k=100; top_true=set(te[np.argsort(-y[te])[:k]]); top_pred=te[np.argsort(-pred)[:k]]
res['topk']=dict(k=k,recall=len(top_true&set(top_pred))/k,mean_pred_top=float(y[top_pred].mean()),mean_test=float(y[te].mean()),max_test=float(y[te].max()),best_found=float(y[top_pred].max()))
pickle.dump(dict(res=res,f=f.astype(np.float32),fl=fl.astype(np.float32),main={S:comp[S].squeeze() for S in comp if len(S)<=2},wt=w),open(f'gb1_{split}.pkl','wb'))
print(json.dumps({k:v for k,v in res.items() if k not in('singles','doubles','anova_gauss','anova_linear')},indent=None,default=float)); print({k:round(v,3) for k,v in share.items() if len(k)<=2})
