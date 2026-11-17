"""FLIP AAV splits: product kernel with uniform, composition and language-model weights, plus one-hot ridge.
Identical protocol and selection grids to run_aav.py; the language-model feature map multiplies each
one-hot channel by sqrt(p(a | WT masked at that slot)) from ESM-2 masked marginals of the AAV wild type.
The gap channel carries weight 1, since no residue probability is defined for it.
usage: python run_aav_lm.py <split> <root> [max_train]
Writes results/aavlm_<split>.json"""
import sys, os, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.kernels import PROTEIN

split, root = sys.argv[1], sys.argv[2]
NTR = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
NTE = 5000
df = pd.read_csv(os.path.join(root, "data", "flip_aav", "splits", f"{split}.csv"))
df = df[df.target.notna()]
tr_mask = df.set.values == "train"; te_mask = ~tr_mask
L = max(len(s) for s in df.sequence)
idx2 = {a: i for i, a in enumerate(PROTEIN)}
def enc(seqs):
    X = np.full((len(seqs), L), -1, dtype=np.int16)
    for i, s in enumerate(seqs):
        for t, c in enumerate(s[:L]): X[i, t] = idx2.get(c, -1)
    return X
X = enc(df.sequence.values); y = df.target.values.astype(float)
var = np.where((X != X[0]).any(0))[0]
Xv = X[:, var]; V = len(var)
W = np.load(os.path.join(root, "emb", "esm2_t33_650M_UR50D__aav_wtlogp.npy"))
assert W.shape[0] >= L, (W.shape, L)
Pw = np.exp(W[var])                                   # (V,20) probabilities at the varying slots
print(split, "L", L, "V", V, "W", W.shape, flush=True)
rng = np.random.default_rng(0)
tr = rng.choice(np.where(tr_mask)[0], min(NTR, tr_mask.sum()), replace=False)
te = rng.choice(np.where(te_mask)[0], min(NTE, te_mask.sum()), replace=False)
idx = np.concatenate([tr, te]); n1 = len(tr)
oh = np.zeros((len(idx), V, 21), dtype=np.float32)
Xi = Xv[idx]
for j in range(V):
    ok = Xi[:, j] >= 0
    oh[np.arange(len(idx))[ok], j, Xi[ok, j]] = 1
    oh[np.arange(len(idx))[~ok], j, 20] = 1
comp = np.zeros_like(oh)
for i in range(len(idx)):
    c = oh[i].sum(0) / V; comp[i] = oh[i] * np.sqrt(np.maximum(c, 1e-6))[None, :]
wlm = np.ones((V, 21), dtype=np.float32); wlm[:, :20] = np.sqrt(Pw)
lm = oh * wlm[None]

def krr(K, yy, lam):
    b = yy.mean(); a = np.linalg.solve(K + lam * np.eye(len(yy)), yy - b); return a, b
itr, iva = train_test_split(np.arange(n1), test_size=0.15, random_state=0)
res = {"split": split, "ntrain": n1, "ntest": len(te), "nvar": int(V)}
for kind, Fm in [("onehot", oh), ("composition", comp), ("lm", lm)]:
    t0 = time.time()
    Fm = Fm.reshape(len(idx), -1).astype(np.float64)
    sq = (Fm ** 2).sum(1); K1 = Fm @ Fm.T; D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * K1, 0)
    sub = D2[:400, :400]; med = np.median(sub[sub > 0])
    if kind == "onehot":
        best = None
        for lam in [1e-2, 1e-1, 1, 10, 100]:
            a, b = krr(K1[np.ix_(itr, itr)], y[tr][itr], lam); v = spearmanr(y[tr][iva], K1[np.ix_(iva, itr)] @ a + b).statistic
            if best is None or v > best[0]: best = (v, lam)
        a, b = krr(K1[:n1, :n1], y[tr], best[1]); res["linear_onehot"] = float(spearmanr(y[te], K1[n1:, :n1] @ a + b).statistic)
    best = None
    for g in [0.1, 0.25, 0.5, 1, 2, 4]:
        K = np.exp(-g * D2 / med)
        for lam in [1e-3, 1e-2, 1e-1, 1]:
            a, b = krr(K[np.ix_(itr, itr)], y[tr][itr], lam); v = spearmanr(y[tr][iva], K[np.ix_(iva, itr)] @ a + b).statistic
            if best is None or v > best[0]: best = (v, g, lam)
    _, g, lam = best; K = np.exp(-g * D2 / med)
    a, b = krr(K[:n1, :n1], y[tr], lam); res["gauss_" + kind] = float(spearmanr(y[te], K[n1:, :n1] @ a + b).statistic)
    res["g_" + kind] = g
    print(" ", kind, round(res["gauss_" + kind], 4), "g", g, f"{time.time()-t0:.0f}s", flush=True)
    del Fm, K1, D2, K
json.dump(res, open(os.path.join(root, "results", f"aavlm_{split}.json"), "w"))
print(json.dumps(res), flush=True)
