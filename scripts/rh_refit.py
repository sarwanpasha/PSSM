"""Final refit at an explicit (gamma-scale, C), matching run_gauss_full.py's protocol.
Fits on the full training split and scores the test split.
usage: python rh_refit.py <dataset> <root> <M> <gs> <C>
Writes results/refit_<dataset>_<gs>_<C>.json"""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map

name, root, M = sys.argv[1], sys.argv[2], int(sys.argv[3])
gs = float(sys.argv[4]); C = float(sys.argv[5])

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
os.makedirs(os.path.join(root, "results"), exist_ok=True)
json.dump({"dataset": name, "n": int(len(y)), "M": M, "gs": gs, "C": C,
           "score": float(s), "sec": time.time() - t0},
          open(os.path.join(root, "results", f"refit_{name}_{gs}_{C}.json"), "w"))
print(name, "REFIT gs", gs, "C", C, "score", round(float(s), 4), flush=True)
