"""One (gamma-scale, C) validation point of run_gauss_full.py's grid, for parallel execution.
Reproduces run_gauss_full.py exactly: same landmarks (rng seed 0), same median heuristic,
same train/validation split (random_state 0), same estimator settings.
usage: python rh_worker.py <dataset> <root> <M> <gs> <C[,C...]>
Writes results/val_<dataset>_<gs>_<C>.json per point."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map

name, root, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
gs = float(sys.argv[4]); Cs = [float(c) for c in sys.argv[5].split(",")]

d = load(name, root); y = np.asarray(d["y"]); A = len(d["alphabet"])
X = encode(d["seqs"], d["alphabet"], d["L"])
F = feature_map(X, A, "match").tocsr().astype(np.float32)
sq = np.asarray(F.multiply(F).sum(1)).ravel().astype(np.float64)
rng = np.random.default_rng(0)
tr, te = d["folds"][0]
land = rng.choice(tr, min(M, len(tr)), replace=False)
sub = np.maximum(sq[tr[:400], None] + sq[None, tr[:400]] - 2 * np.asarray((F[tr[:400]] @ F[tr[:400]].T).todense()), 0)
med = np.median(sub[sub > 0])
itr, iva = train_test_split(tr, test_size=0.15, random_state=0)

def nystrom(idx_land, g):
    Fl = F[idx_land]; sl = sq[idx_land]
    Kmm = np.exp(-g * np.maximum(sl[:, None] + sl[None, :] - 2 * np.asarray((Fl @ Fl.T).todense()), 0))
    w, V = np.linalg.eigh(Kmm); keep = w > 1e-8 * w.max()
    Winv = V[:, keep] / np.sqrt(w[keep])
    Z = np.empty((F.shape[0], keep.sum()), dtype=np.float32); B = 4000
    for i in range(0, F.shape[0], B):
        Kb = np.exp(-g * np.maximum(sq[i:i+B, None] + sl[None, :] - 2 * np.asarray((F[i:i+B] @ Fl.T).todense()), 0))
        Z[i:i+B] = (Kb @ Winv).astype(np.float32)
    return Z

Z = nystrom(land, gs / med)
os.makedirs(os.path.join(root, "results"), exist_ok=True)
for C in Cs:
    t0 = time.time()
    m = LinearSVC(C=C, max_iter=3000).fit(Z[itr], y[itr])
    v = accuracy_score(y[iva], m.predict(Z[iva]))
    p = os.path.join(root, "results", f"val_{name}_{gs}_{C}.json")
    json.dump({"gs": gs, "C": C, "v": float(v), "sec": time.time() - t0}, open(p, "w"))
    print("val", gs, C, round(float(v), 4), f"{time.time()-t0:.0f}s", flush=True)
