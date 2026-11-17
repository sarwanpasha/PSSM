"""Nystrom product-kernel scores at all three remote-homology holdout levels.
The three tape_rh_* datasets share one training set (train+valid) and one label space;
only the test holdout differs. The hyperparameter search of run_gauss_full.py uses a
validation split of that shared training set, so it selects the same (gamma-scale, C) at
every level, and a single fit scores all three holdouts.
usage: python rh_levels.py <root> <M> <gs> <C>
Writes results/gaussfull_tape_rh_<level>.json for level in fold, superfamily, family."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from wpsskm.datasets import load
from wpsskm.kernels import encode, feature_map

root, M = sys.argv[1], int(sys.argv[2])
gs = float(sys.argv[3]); C = float(sys.argv[4])

def build(name):
    d = load(name, root); y = np.asarray(d["y"]); A = len(d["alphabet"])
    X = encode(d["seqs"], d["alphabet"], d["L"])
    F = feature_map(X, A, "match").tocsr().astype(np.float32)
    sq = np.asarray(F.multiply(F).sum(1)).ravel().astype(np.float64)
    tr, te = d["folds"][0]
    return F, sq, y, tr, te

F, sq, y, tr, te = build("tape_rh_family")
rng = np.random.default_rng(0)
land = rng.choice(tr, min(M, len(tr)), replace=False)
sub = np.maximum(sq[tr[:400], None] + sq[None, tr[:400]] - 2 * np.asarray((F[tr[:400]] @ F[tr[:400]].T).todense()), 0)
med = np.median(sub[sub > 0]); g = gs / med
Fl = F[land]; sl = sq[land]
Kmm = np.exp(-g * np.maximum(sl[:, None] + sl[None, :] - 2 * np.asarray((Fl @ Fl.T).todense()), 0))
w, V = np.linalg.eigh(Kmm); keep = w > 1e-8 * w.max()
Winv = V[:, keep] / np.sqrt(w[keep])
print("landmarks", len(land), "rank", int(keep.sum()), "med", float(med), flush=True)

def embed(Fx, sqx, rows):
    Z = np.empty((len(rows), Winv.shape[1]), dtype=np.float32); B = 4000
    for i in range(0, len(rows), B):
        idx = rows[i:i+B]
        Kb = np.exp(-g * np.maximum(sqx[idx, None] + sl[None, :] - 2 * np.asarray((Fx[idx] @ Fl.T).todense()), 0))
        Z[i:i+B] = (Kb @ Winv).astype(np.float32)
    return Z

t0 = time.time()
m = LinearSVC(C=C, max_iter=3000).fit(embed(F, sq, tr), y[tr])
print("fit done", f"{time.time()-t0:.0f}s", flush=True)

ntr = len(tr)
for level in ["fold", "superfamily", "family"]:
    name = "tape_rh_" + level
    if level == "family":
        Fx, sqx, yx, trx, tex = F, sq, y, tr, te
    else:
        Fx, sqx, yx, trx, tex = build(name)
        assert len(trx) == ntr, "training split differs at level %s" % level
    s = accuracy_score(yx[tex], m.predict(embed(Fx, sqx, tex)))
    out = {"dataset": name, "n": int(len(yx)), "M": M,
           "folds": [dict(fold=0, score=float(s), g=gs, C=C, sec=time.time() - t0)],
           "mean": float(s)}
    json.dump(out, open(os.path.join(root, "results", f"gaussfull_{name}.json"), "w"))
    print(name, "MEAN", round(float(s), 4), "n_test", len(tex), flush=True)
