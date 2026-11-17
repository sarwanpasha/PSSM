import sys, time; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np, scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode
from wpsskm.cwpk import CWPK
from wpsskm.stringkernels import spectrum
name=sys.argv[1]; d=load(name,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']; tr,te=d['folds'][0]; A=4
itr,iva=train_test_split(tr,test_size=0.1,random_state=0,stratify=y[tr])
def run(F,tag):
    t=time.time(); best=max([0.01,0.1,1],key=lambda c:accuracy_score(y[iva],LinearSVC(C=c,max_iter=3000).fit(F[itr],y[itr]).predict(F[iva])))
    acc=accuracy_score(y[te],LinearSVC(C=best,max_iter=3000).fit(F[tr],y[tr]).predict(F[te])); print(tag,F.shape[1],round(acc,4),best,f'{time.time()-t:.0f}s',flush=True)
for cfg in eval(sys.argv[2]):
    m=CWPK(**cfg).fit(X[tr],A); run(m.transform(X),str(cfg))
if len(sys.argv)>3:
    for k in eval(sys.argv[3]): run(spectrum(X,A,k),f'spec{k}')
