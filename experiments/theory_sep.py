import sys, math; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from wpsskm.kernels import encode, feature_map, PROTEIN
rng=np.random.default_rng(0)
n,L,A=400,8,20
seqs=[''.join(rng.choice(list(PROTEIN[:6]),L)) for _ in range(n)]
X=encode(seqs,PROTEIN,L); F=feature_map(X,A,'sqrt'); F=(F.toarray() if hasattr(F,'toarray') else F).reshape(n,-1)
sq=(F**2).sum(1); K1=F@F.T
print('diag(K1) range', round(sq.min(),3), round(sq.max(),3))
for g,M in [(0.5,40),(2.0,200)]:
    lhs=np.exp(-g*(sq[:,None]+sq[None,:]-2*K1))
    rhs=np.exp(-g*(sq[:,None]+sq[None,:]))*sum(np.exp(m*np.log(2*g)-math.lgamma(m+1))*K1**m for m in range(0,M))
    print(f'A2 g={g} terms={M}: max|lhs-rhs| = {np.abs(lhs-rhs).max():.2e}')

# --- separation: additive vs pairwise-epistatic target, TRAIN error of kernel ridge (expressive power)
def kr_fit(K,y,lam=1e-8):
    a=np.linalg.solve(K+lam*np.eye(len(y)),y-y.mean()); return K@a+y.mean()
def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()
coefA=rng.normal(size=(L,A)); t1,t2=2,5
coefP=rng.normal(size=(A,A))
y_add=np.array([sum(coefA[t,X[i,t]] for t in range(L)) for i in range(n)])
y_epi=y_add+3*np.array([coefP[X[i,t1],X[i,t2]] for i in range(n)])
D2=np.maximum(sq[:,None]+sq[None,:]-2*K1,0); med=np.median(D2[D2>0])
Krbf=np.exp(-D2/med)
for nm,y in [('additive target',y_add),('epistatic target',y_epi)]:
    print(f'{nm}: train R2  linear W-PSSKM {r2(y,kr_fit(K1,y)):.3f} | Gaussian W-PSSKM {r2(y,kr_fit(Krbf,y)):.3f}')

# --- rank / interpolation separation
print('n =',n,' L*A =',L*A)
print('rank(K_linear) =',np.linalg.matrix_rank(K1,tol=1e-8),'  rank(K_gaussian) =',np.linalg.matrix_rank(Krbf,tol=1e-8))
print('min eig K_gaussian =',f'{np.linalg.eigvalsh(Krbf).min():.3e}')
# interpolation of pure noise: possible iff Gram nonsingular
yn=rng.normal(size=n)
print('random target train R2: linear',round(r2(yn,kr_fit(K1,yn)),3),' gaussian',round(r2(yn,kr_fit(Krbf,yn)),3))
