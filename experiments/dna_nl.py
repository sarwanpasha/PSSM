import sys, time; sys.path.insert(0,'/mnt/user-data/outputs/code')
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode
from wpsskm.cwpk import CWPK
name=sys.argv[1]; d=load(name,'/home/claude/root'); X=encode(d['seqs'],d['alphabet'],d['L']); y=d['y']; tr,te=d['folds'][0]
itr,iva=train_test_split(tr,test_size=0.1,random_state=0,stratify=y[tr])
F=CWPK(orders=[4,5,6,7],b=600,gamma=0.5,eta=1,beta='u').fit(X[tr],4).transform(X)
for g in eval(sys.argv[2]):
    t=time.time(); ny=Nystroem(kernel='rbf',gamma=g,n_components=int(sys.argv[3]),random_state=0).fit(F[tr]); Z=ny.transform(F)
    best=max([0.01,0.1,1],key=lambda c:accuracy_score(y[iva],LinearSVC(C=c).fit(Z[itr],y[itr]).predict(Z[iva])))
    print('rbf',g,round(accuracy_score(y[te],LinearSVC(C=best).fit(Z[tr],y[tr]).predict(Z[te])),4),best,f'{time.time()-t:.0f}s',flush=True)
