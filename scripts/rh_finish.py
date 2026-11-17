"""Final refit of run_gauss_full.py's protocol once the grid points are known.
Picks the best (gs, C) from results/val_<dataset>_*.json with the same tie-breaking as
run_gauss_full.py (first strict improvement in grid order), refits on the full training
split and scores the test split. Writes results/gaussfull_<dataset>.json.
usage: python rh_finish.py <dataset> <root> <M>"""
import sys, os, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map

name, root, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
vals = {}
for p in glob.glob(os.path.join(root, "results", f"val_{name}_*.json")):
    r = json.load(open(p)); vals[(r["gs"], r["C"])] = r["v"]
grid = [(gs, C) for gs in [0.1, 0.3, 1, 3] for C in [0.01, 0.1, 1]]
missing = [k for k in grid if k not in vals]
if missing:
    sys.exit("missing grid points: %s" % missing)
best = None
for k in grid:
    if best is None or vals[k] > best[0]:
        best = (vals[k], k[0], k[1])
_, gs, C = best
print("selected gs", gs, "C", C, "val", round(best[0], 4), flush=True)

d = load(name, root); y = np.asarray(d["y"]); A = len(d["alphabet"])
X = encode(d["seqs"], d["alphabet"], d["L"])
F = feature_map(X, A, "match").tocsr().astype(np.float32)
sq = np.asarray(F.multiply(F).sum(1)).ravel().astype(np.float64)
rng = np.random.default_rng(0)
tr, te = d["folds"][0]
land = rng.choice(tr, min(M, len(tr)), replace=False)
sub = np.maximum(sq[tr[:400], None] + sq[None, tr[:400]] - 2 * np.asarray((F[tr[:400]] @ F[tr[:400]].T).todense()), 0)
med = np.median(sub[sub > 0])

Fl = F[land]; sl = sq[land]; g = gs / med
Kmm = np.exp(-g * np.maximum(sl[:, None] + sl[None, :] - 2 * np.asarray((Fl @ Fl.T).todense()), 0))
w, V = np.linalg.eigh(Kmm); keep = w > 1e-8 * w.max()
Winv = V[:, keep] / np.sqrt(w[keep])
Z = np.empty((F.shape[0], keep.sum()), dtype=np.float32); B = 4000
for i in range(0, F.shape[0], B):
    Kb = np.exp(-g * np.maximum(sq[i:i+B, None] + sl[None, :] - 2 * np.asarray((F[i:i+B] @ Fl.T).todense()), 0))
    Z[i:i+B] = (Kb @ Winv).astype(np.float32)

t0 = time.time()
m = LinearSVC(C=C, max_iter=3000).fit(Z[tr], y[tr])
s = accuracy_score(y[te], m.predict(Z[te]))
out = {"dataset": name, "n": len(y), "M": M,
       "folds": [dict(fold=0, score=float(s), g=gs, C=C, sec=time.time() - t0)],
       "mean": float(s)}
json.dump(out, open(os.path.join(root, "results", f"gaussfull_{name}.json"), "w"))
print(name, "MEAN", round(float(s), 4), flush=True)
