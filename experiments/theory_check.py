import sys; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np, math
from itertools import combinations
from wpsskm.kernels import encode, feature_map, PROTEIN
rng=np.random.default_rng(0)

# ---- setup: short random sequences, explicit feature map phi_{1/2}
n,L,A=200,8,20
seqs=[''.join(rng.choice(list(PROTEIN[:6]),L)) for _ in range(n)]
X=encode(seqs,PROTEIN,L)
F=feature_map(X,A,'sqrt'); F=(F.toarray() if hasattr(F,'toarray') else F).reshape(n,-1)

# (a) injectivity of phi on the data: distinct sequences -> distinct features
import hashlib
h=lambda v: hashlib.sha1(np.round(v,9).tobytes()).hexdigest()
print('A1 injective:', len({h(f) for f in F})==len(set(seqs)), len(set(seqs)),'distinct seqs')

# (b) Taylor expansion of the Gaussian on phi: exp(-g||px-py||^2) = e^{-g(sx+sy)} sum_m (2g)^m/m! <px,py>^m
sq=(F**2).sum(1); K1=F@F.T
for g in [0.5,2.0]:
    lhs=np.exp(-g*(sq[:,None]+sq[None,:]-2*K1))
    rhs=np.exp(-g*(sq[:,None]+sq[None,:]))*sum((2*g)**m/math.factorial(m)*K1**m for m in range(0,25))
    print(f'A2 g={g} max|lhs-rhs| = {np.abs(lhs-rhs).max():.2e}')

# <px,py>^m expands over m-tuples of positions: verify m=2 term equals sum over ordered position pairs
P=F.reshape(n,L,A)
pos=np.einsum('ita,jta->ijt',P,P)              # per-position match*weights
print('A3 <px,py> == sum_t pos:', np.abs(K1-pos.sum(2)).max()<1e-9)
print('A3 <px,py>^2 == sum_{t,s} pos_t*pos_s:', np.abs(K1**2-np.einsum('ijt,ijs->ij',pos,pos)).max()<1e-9)
