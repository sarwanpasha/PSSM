"""Workstream A variants on one dataset. usage: python run_variants.py <dataset> <root>"""
import sys, os, json, time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode
from wpsskm.variants import binned, blosum_features
from wpsskm.datasets import load

name, root = sys.argv[1], sys.argv[2]
ds = load(name, root); A = len(ds["alphabet"]); y = ds["y"]; clf = ds["task"] == "clf"
X = encode(ds["seqs"], ds["alphabet"], ds["L"])
res = dict(dataset=name, runs=[])

def metrics(yt, yp):
    if clf: return dict(acc=accuracy_score(yt, yp), f1_weighted=f1_score(yt, yp, average="weighted"), f1_macro=f1_score(yt, yp, average="macro"))
    return dict(spearman=spearmanr(yt, yp).statistic, r2=r2_score(yt, yp))

kinds = ["bin5", "bin25", "bin100"] + (["blosum"] if A == 20 and len(y) * ds["L"] * A * 4 < 6e9 else [])
for kind in kinds:
    F = binned(X, A, int(kind[3:])) if kind.startswith("bin") else blosum_features(X, A)
    for fi, (tr, te) in enumerate(ds["folds"]):
        try: itr, iva = train_test_split(tr, test_size=0.1, random_state=0, stratify=y[tr] if clf else None)
        except ValueError: itr, iva = train_test_split(tr, test_size=0.1, random_state=0)
        mk = (lambda c: LinearSVC(C=c, max_iter=5000)) if clf else (lambda a: Ridge(alpha=a))
        grid = [0.001, 0.01, 0.1, 1.0, 10.0] if clf else [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        sc = lambda g: (accuracy_score if clf else (lambda a, b: spearmanr(a, b).statistic))(y[iva], mk(g).fit(F[itr], y[itr]).predict(F[iva]))
        t0 = time.time(); best = max(grid, key=sc)
        r = dict(fold=fi, kernel=kind, model="svm" if clf else "kridge", hyper=best, sec=time.time() - t0,
                 **metrics(y[te], mk(best).fit(F[tr], y[tr]).predict(F[te])))
        res["runs"].append(r); print(name, r, flush=True)
    del F
os.makedirs(os.path.join(root, "results"), exist_ok=True)
json.dump(res, open(os.path.join(root, "results", f"variants_{name}.json"), "w"), indent=1, default=float)
print(name, "done", flush=True)
