"""Corrected vs original W-PSSKM on one dataset.
usage: python run_kernels.py <dataset> <root> [--no-original]
Writes <root>/results/kernels_<dataset>.json and appends progress to stdout."""
import sys, os, json, time
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode, composition, feature_map, _onehot_values
from wpsskm.datasets import load

name, root = sys.argv[1], sys.argv[2]
do_original = "--no-original" not in sys.argv
CAP = 15000
out = os.path.join(root, "results"); os.makedirs(out, exist_ok=True)
ds = load(name, root)
A = len(ds["alphabet"]); X = encode(ds["seqs"], ds["alphabet"], ds["L"]); y = ds["y"]
clf = ds["task"] == "clf"
print(f"[{name}] n={len(y)} L={ds['L']} task={ds['task']} folds={len(ds['folds'])}", flush=True)
res = dict(dataset=name, n=int(len(y)), L=ds["L"], task=ds["task"], runs=[])


def metrics(yt, yp):
    if clf:
        return dict(acc=accuracy_score(yt, yp), f1_weighted=f1_score(yt, yp, average="weighted"),
                    f1_macro=f1_score(yt, yp, average="macro"))
    return dict(spearman=spearmanr(yt, yp).statistic, r2=r2_score(yt, yp))


def inner_split(idx):
    strat = y[idx] if clf and np.min(np.bincount(y[idx])[np.bincount(y[idx]) > 0]) >= 2 else None
    try:
        return train_test_split(idx, test_size=0.1, random_state=0, stratify=strat)
    except ValueError:
        return train_test_split(idx, test_size=0.1, random_state=0)


def tune_and_fit(make, grid, Ftr, ytr, Fva, yva):
    best, bs = None, -np.inf
    for g in grid:
        m = make(g).fit(Ftr, ytr); p = m.predict(Fva)
        s = accuracy_score(yva, p) if clf else spearmanr(yva, p).statistic
        if s > bs: bs, best = s, g
    return best


class CenteredKR:
    """Kernel ridge with an intercept (targets centred), so it is comparable to Ridge on phi."""
    def __init__(self, alpha): self.alpha = alpha
    def fit(self, K, yy):
        self.mu = yy.mean(); self.m = KernelRidge(alpha=self.alpha, kernel="precomputed").fit(K, yy - self.mu); return self
    def predict(self, K): return self.m.predict(K) + self.mu


def original_gram(Xa, Xb):
    Wa, Wb = composition(Xa, A), composition(Xb, A)
    return (_onehot_values(Xa, A, Wa) @ _onehot_values(Xb, A).T + _onehot_values(Xa, A) @ _onehot_values(Xb, A, Wb).T).toarray()


# 1. Is the original kernel PSD on real data? (random subset of 3000)
rng = np.random.default_rng(0)
sub = rng.choice(len(y), min(3000, len(y)), replace=False)
K = original_gram(X[sub], X[sub])
lo = eigsh(K, k=1, which="SA", return_eigenvectors=False)[0]; hi = eigsh(K, k=1, which="LA", return_eigenvectors=False)[0]
res["original_psd_check"] = dict(subset=int(len(sub)), lambda_min=float(lo), lambda_max=float(hi), ratio=float(lo / hi))
print(f"[{name}] original kernel on 3000-seq subset: lambda_min={lo:.4g}, lambda_max={hi:.4g}", flush=True)
del K

for fi, (tr, te) in enumerate(ds["folds"]):
    itr, iva = inner_split(tr)
    for kind in ["sqrt", "product", "match"]:
        P = feature_map(X, A, kind)
        t0 = time.time()
        if clf:
            C = tune_and_fit(lambda c: LinearSVC(C=c, max_iter=5000), [0.001, 0.01, 0.1, 1.0, 10.0], P[itr], y[itr], P[iva], y[iva])
            m = LinearSVC(C=C, max_iter=5000).fit(P[tr], y[tr])
        else:
            C = tune_and_fit(lambda a: Ridge(alpha=a), [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], P[itr], y[itr], P[iva], y[iva])
            m = Ridge(alpha=C).fit(P[tr], y[tr])
        r = dict(fold=fi, kernel=kind, model="svm" if clf else "kridge", hyper=C, sec=time.time() - t0, **metrics(y[te], m.predict(P[te])))
        res["runs"].append(r); print(r, flush=True)
        if clf and kind == "sqrt":   # paper-style: kernel PCA (uncentred, via SVD of phi) + classical classifiers
            t0 = time.time()
            Z = TruncatedSVD(100, random_state=0).fit(P[tr]).transform(P)
            for mname, mdl in [("kpca+lr", LogisticRegression(max_iter=2000)),
                               ("kpca+rf", RandomForestClassifier(300, n_jobs=-1, random_state=0)),
                               ("kpca+knn", KNeighborsClassifier(5))]:
                mdl.fit(Z[tr], y[tr])
                r = dict(fold=fi, kernel=kind, model=mname, sec=time.time() - t0, **metrics(y[te], mdl.predict(Z[te])))
                res["runs"].append(r); print(r, flush=True)
    if do_original:
        t0 = time.time(); tag = ""
        if len(tr) > CAP:   # dense n x n Gram: compare on a fixed training subsample, corrected kernel on the same one
            tr = np.sort(np.random.default_rng(fi).choice(tr, CAP, replace=False)); itr, iva = inner_split(tr); tag = f"@{CAP // 1000}k"
            P = feature_map(X, A, "sqrt")
            if clf:
                C = tune_and_fit(lambda c: LinearSVC(C=c, max_iter=5000), [0.001, 0.01, 0.1, 1.0, 10.0], P[itr], y[itr], P[iva], y[iva])
                mm = LinearSVC(C=C, max_iter=5000).fit(P[tr], y[tr])
            else:
                C = tune_and_fit(lambda a: Ridge(alpha=a), [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], P[itr], y[itr], P[iva], y[iva])
                mm = Ridge(alpha=C).fit(P[tr], y[tr])
            r = dict(fold=fi, kernel="sqrt", model=("svm" if clf else "kridge") + tag, hyper=C, **metrics(y[te], mm.predict(P[te])))
            res["runs"].append(r); print(r, flush=True)
        Ktr = original_gram(X[tr], X[tr]); Kte = original_gram(X[te], X[tr])
        pos = {v: i for i, v in enumerate(tr)}; a = np.array([pos[i] for i in itr]); b = np.array([pos[i] for i in iva])
        if clf:
            C = tune_and_fit(lambda c: SVC(C=c, kernel="precomputed"), [0.01, 0.1, 1.0, 10.0], Ktr[np.ix_(a, a)], y[itr], Ktr[np.ix_(b, a)], y[iva])
            m = SVC(C=C, kernel="precomputed").fit(Ktr, y[tr])
        else:
            C = tune_and_fit(lambda al: CenteredKR(al), [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], Ktr[np.ix_(a, a)], y[itr], Ktr[np.ix_(b, a)], y[iva])
            m = CenteredKR(C).fit(Ktr, y[tr])
        r = dict(fold=fi, kernel="original", model=("svm" if clf else "kridge") + tag, hyper=C, sec=time.time() - t0, **metrics(y[te], m.predict(Kte)))
        res["runs"].append(r); print(r, flush=True)
        del Ktr, Kte
    json.dump(res, open(os.path.join(out, f"kernels_{name}.json"), "w"), indent=1, default=float)
print(f"[{name}] done", flush=True)
