"""Combine ESM-2 embeddings with the corrected position-specific kernel.
Feature blocks are L2-normalised per row; the combined map [alpha*esm, phi] gives the PSD kernel
k = alpha^2 k_esm + k_1/2 (a weighted sum of kernels), alpha tuned on validation.
Representations: kern (phi_1/2), esm (mean-pooled ESM-2), esm+kern, and, when available,
esmbin (composition-weighted, position-binned per-residue ESM-2) and esmbin+kern.
usage: python run_combo.py <dataset|tape_rh> <root> <esm_model> [<esmbin_model>]"""
import sys, os, json, time
import numpy as np, scipy.sparse as sp
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode, feature_map
from wpsskm.datasets import load

name, root, model = sys.argv[1], sys.argv[2], sys.argv[3]
binmodel = sys.argv[4] if len(sys.argv) > 4 else None
emb = lambda m, d, kind="": os.path.join(root, "emb", f"{m}__{d}{kind}.npy")

def assemble(m, kind=""):
    """Embeddings aligned with the dataset (or the combined TAPE layout)."""
    if name != "tape_rh":
        f = emb(m, name, kind); return np.load(f) if os.path.exists(f) else None
    out = []
    for i, s in enumerate(["fold", "superfamily", "family"]):
        f = emb(m, f"tape_rh_{s}", kind)
        if not os.path.exists(f): return None
        E = np.load(f); d = load(f"tape_rh_{s}", root); tr, te = d["folds"][0]
        if i == 0: out.append(E[tr])
        out.append(E[te])
    return np.concatenate(out)

if name == "tape_rh":
    parts = {s: load(f"tape_rh_{s}", root) for s in ["fold", "superfamily", "family"]}
    d0 = parts["fold"]; ntr = len(d0["folds"][0][0]); seqs = d0["seqs"][:ntr]; y = list(d0["y"][:ntr]); tests = {}
    for s, d in parts.items():
        te = d["folds"][0][1]; tests[s] = np.arange(len(seqs), len(seqs) + len(te)); seqs += [d["seqs"][i] for i in te]; y += list(d["y"][te])
    y = np.array(y); task, alphabet, L = "clf", d0["alphabet"], d0["L"]; folds = [(np.arange(ntr), tests)]
else:
    d = load(name, root); seqs, y, task, alphabet, L = d["seqs"], d["y"], d["task"], d["alphabet"], d["L"]
    folds = [(tr, {"test": te}) for tr, te in d["folds"]]
clf = task == "clf"
PHI = normalize(feature_map(encode(seqs, alphabet, L), len(alphabet), "sqrt"))
E = assemble(model); EB = assemble(binmodel, "__bin") if binmodel else None
out = os.path.join(root, "results", f"combo_{name}__{model}.json")
res = json.load(open(out)) if os.path.exists(out) else dict(dataset=name, esm=model, runs=[])
done = {(r["kernel"], r["fold"]) for r in res["runs"]}

def metrics(yt, yp):
    if clf: return dict(acc=accuracy_score(yt, yp), f1_weighted=f1_score(yt, yp, average="weighted"), f1_macro=f1_score(yt, yp, average="macro"))
    return dict(spearman=spearmanr(yt, yp).statistic, r2=r2_score(yt, yp))

for fi, (tr, tests) in enumerate(folds):
    Z = normalize(StandardScaler().fit(E[tr]).transform(E))
    ZB = normalize(StandardScaler().fit(EB[tr]).transform(EB)) if EB is not None else None
    reps = {"kern": (PHI, [None]), "esm": (Z, [None]), "esm+kern": (None, [0.5, 1.0, 2.0, 4.0])}
    if ZB is not None:
        reps.update({"esmbin": (ZB, [None]), "esmbin+kern": (None, [0.5, 1.0, 2.0, 4.0])})
    try: itr, iva = train_test_split(tr, test_size=0.1, random_state=0, stratify=y[tr] if clf else None)
    except ValueError: itr, iva = train_test_split(tr, test_size=0.1, random_state=0)
    mk = (lambda c: LinearSVC(C=c, max_iter=5000)) if clf else (lambda a: Ridge(alpha=a))
    grid = [0.01, 0.1, 1.0, 10.0] if clf else [0.01, 0.1, 1.0, 10.0, 100.0]
    for rep, (F0, alphas) in reps.items():
        key = f"{rep}" if rep in ("kern", "esm", "esmbin") else f"{rep}"
        if (key, fi) in done: continue
        best, bs = None, -np.inf
        for a in alphas:
            if F0 is None:
                base = Z if rep == "esm+kern" else ZB
                F = sp.hstack([sp.csr_matrix(base * a), PHI], format="csr")
            else:
                F = F0
            for g in grid:
                p = mk(g).fit(F[itr], y[itr]).predict(F[iva])
                s = accuracy_score(y[iva], p) if clf else spearmanr(y[iva], p).statistic
                if s > bs: bs, best = s, (a, g, F)
        a, g, F = best
        t0 = time.time(); m = mk(g).fit(F[tr], y[tr]); tf = time.time() - t0
        for tname, te in tests.items():
            r = dict(fold=fi, kernel=key, model="svm" if clf else "kridge", test=tname, alpha=a, hyper=g, fit_sec=tf,
                     esm=model, **metrics(y[te], m.predict(F[te])))
            res["runs"].append(r); print(name, r, flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)
print(f"[{name}] combo done", flush=True)
