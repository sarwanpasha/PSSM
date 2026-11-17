import sys, math; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np, scipy.sparse as sp
from wpsskm.kernels import encode, feature_map, PROTEIN, DNA
ROOT='/home/claude/root'
import os
def feats(X,A): return feature_map(X,A,os.environ.get('KIND','sqrt')).tocsr().astype(np.float32)
def lin(Fa,Fb): return np.asarray((Fa@Fb.T).todense(),dtype=np.float64)
def sqn(F): return np.asarray(F.multiply(F).sum(1)).ravel().astype(np.float64)
def gauss(K1,sa,sb,g): return np.exp(-g*np.maximum(sa[:,None]+sb[None,:]-2*K1,0))
def krr(K,y,lam):
    """kernel ridge with intercept; y (n,) or (n,c). returns alpha, bias"""
    b=y.mean(0); a=np.linalg.solve(K+lam*np.eye(len(K)),y-b); return a,b
def degree_parts(K1,sa,sb,g,alpha,M):
    """exact Taylor split of f(x)=sum_i alpha_i exp(-g||px-pi||^2): returns list f_m(x), m=0..M-1 (without bias)"""
    ca=np.exp(-g*sa)[:,None]; cb=np.exp(-g*sb)
    out=[]; P=np.ones_like(K1)
    for m in range(M):
        out.append(ca[:,0][:,None]*0+ (ca*(P*np.exp(m*np.log(2*g)-math.lgamma(m+1))))@(cb[:,None]*alpha) if alpha.ndim>1 else (ca[:,0]*((P*np.exp(m*np.log(2*g)-math.lgamma(m+1)))@(cb*alpha))))
        P=P*K1
    return out
