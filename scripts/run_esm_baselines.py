"""Linear probes on mean-pooled ESM-2 embeddings, same folds/tuning as run_kernels.py.
usage: python run_esm_baselines.py <root>   (evaluates every emb/<model>__<dataset>.npy present)"""
import sys, os, glob, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load

root = sys.argv[1]; out = os.path.join(root, "results"); os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(os.path.join(root, "emb", "*__*.npy"))):
    parts = os.path.basename(f)[:-4].split("__")
    if len(parts) != 2: continue            # skip derived files such as <model>__<dataset>__bin.npy
    model, dname = parts
    of = os.path.join(out, f"esm_{model}__{dname}.json")
    if os.path.exists(of): continue
    ds = load(dname, root); E = np.load(f); y = ds["y"]; clf = ds["task"] == "clf"
    assert len(E) == len(y)
    res = dict(dataset=dname, model=model, runs=[])
    for fi, (tr, te) in enumerate(ds["folds"]):
        sc = StandardScaler().fit(E[tr]); Z = sc.transform(E)
        try:
            itr, iva = train_test_split(tr, test_size=0.1, random_state=0, stratify=y[tr] if clf else None)
        except ValueError:
            itr, iva = train_test_split(tr, test_size=0.1, random_state=0)
        def score(m, a, b):
            p = m.fit(Z[a], y[a]).predict(Z[b])
            return accuracy_score(y[b], p) if clf else spearmanr(y[b], p).statistic
        if clf:
            grid = [0.001, 0.01, 0.1, 1.0]; mk = lambda c: LogisticRegression(C=c, max_iter=3000)
        else:
            grid = [0.1, 1.0, 10.0, 100.0, 1000.0]; mk = lambda a: Ridge(alpha=a)
        best = max(grid, key=lambda g: score(mk(g), itr, iva))
        t0 = time.time(); p = mk(best).fit(Z[tr], y[tr]).predict(Z[te])
        if clf:
            m = dict(acc=accuracy_score(y[te], p), f1_weighted=f1_score(y[te], p, average="weighted"), f1_macro=f1_score(y[te], p, average="macro"))
        else:
            m = dict(spearman=spearmanr(y[te], p).statistic, r2=r2_score(y[te], p))
        r = dict(fold=fi, kernel=model, model="lr" if clf else "ridge", hyper=best, sec=time.time() - t0, **m)
        res["runs"].append(r); print(dname, r, flush=True)
        if clf:   # the conference paper's best ESM-2 setting used RF
            p = RandomForestClassifier(300, n_jobs=-1, random_state=0).fit(E[tr], y[tr]).predict(E[te])
            r = dict(fold=fi, kernel=model, model="rf", acc=accuracy_score(y[te], p),
                     f1_weighted=f1_score(y[te], p, average="weighted"), f1_macro=f1_score(y[te], p, average="macro"))
            res["runs"].append(r); print(dname, r, flush=True)
    json.dump(res, open(of, "w"), indent=1, default=float)
print("done")
