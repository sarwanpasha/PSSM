"""Full-data Gaussian product kernel via Nystrom, for the benchmarks too large for an exact solve.
usage: python run_gauss_full.py <dataset> <root> [n_components]
Writes results/gaussfull_<dataset>.json. Uses uniform weights ('match' feature map); the composition
variant is reported separately in the ablation."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map

name, root = sys.argv[1], sys.argv[2]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
d = load(name, root); y = np.asarray(d["y"]); task = d["task"]; A = len(d["alphabet"])
X = encode(d["seqs"], d["alphabet"], d["L"])
F = feature_map(X, A, "match").tocsr().astype(np.float32)
sq = np.asarray(F.multiply(F).sum(1)).ravel().astype(np.float64)
rng = np.random.default_rng(0)
out = {"dataset": name, "n": len(y), "M": M, "folds": []}

def nystrom(idx_land, g):
    """Z such that Z Z^T ~ K, using landmarks idx_land."""
    Fl = F[idx_land]; sl = sq[idx_land]
    Kmm = np.exp(-g * np.maximum(sl[:, None] + sl[None, :] - 2 * np.asarray((Fl @ Fl.T).todense()), 0))
    w, V = np.linalg.eigh(Kmm); keep = w > 1e-8 * w.max()
    Winv = V[:, keep] / np.sqrt(w[keep])
    Z = np.empty((F.shape[0], keep.sum()), dtype=np.float32); B = 4000
    for i in range(0, F.shape[0], B):
        Kb = np.exp(-g * np.maximum(sq[i:i+B, None] + sl[None, :] - 2 * np.asarray((F[i:i+B] @ Fl.T).todense()), 0))
        Z[i:i+B] = (Kb @ Winv).astype(np.float32)
    return Z

for fi, (tr, te) in enumerate(d["folds"]):
    t0 = time.time()
    land = rng.choice(tr, min(M, len(tr)), replace=False)
    sub = np.maximum(sq[tr[:400], None] + sq[None, tr[:400]] - 2 * np.asarray((F[tr[:400]] @ F[tr[:400]].T).todense()), 0)
    med = np.median(sub[sub > 0])
    itr, iva = train_test_split(tr, test_size=0.15, random_state=0)
    best = None
    for gs in [0.1, 0.3, 1, 3]:
        Z = nystrom(land, gs / med)
        for C in ([0.01, 0.1, 1] if task == "clf" else [1e-2, 1e-1, 1, 10]):
            if task == "clf":
                m = LinearSVC(C=C, max_iter=3000).fit(Z[itr], y[itr]); v = accuracy_score(y[iva], m.predict(Z[iva]))
            else:
                m = Ridge(alpha=C).fit(Z[itr], y[itr]); v = spearmanr(y[iva], m.predict(Z[iva])).statistic
            if best is None or v > best[0]: best = (v, gs, C)
    _, gs, C = best
    Z = nystrom(land, gs / med)
    if task == "clf":
        m = LinearSVC(C=C, max_iter=3000).fit(Z[tr], y[tr]); s = accuracy_score(y[te], m.predict(Z[te]))
    else:
        m = Ridge(alpha=C).fit(Z[tr], y[tr]); s = spearmanr(y[te], m.predict(Z[te])).statistic
    out["folds"].append(dict(fold=fi, score=float(s), g=gs, C=C, sec=time.time() - t0))
    print(name, fi, round(float(s), 4), "g", gs, "C", C, f"{time.time()-t0:.0f}s", flush=True)
out["mean"] = float(np.mean([f["score"] for f in out["folds"]]))
json.dump(out, open(os.path.join(root, "results", f"gaussfull_{name}.json"), "w"))
print(name, "MEAN", round(out["mean"], 4), flush=True)
