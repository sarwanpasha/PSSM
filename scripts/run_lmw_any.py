"""Zero-shot score and language-model-weighted product kernel on GB1, for any source of
per-position residue log-probabilities. Mirrors the protocol of the ESM-2 run exactly, so the
numbers are directly comparable: same four variable sites, same selection grids, same splits.
usage: python run_lmw_any.py <split> <root> <wtlogp_npy> <tag>
Writes results/lmwany_<tag>_<split>.json
"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold

split, root, wpath, tag = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
PROTEIN = "ACDEFGHIKLMNPQRSTVWY"
SITES = [38, 39, 40, 53]
df = pd.read_csv(os.path.join(root, "data", "flip_gb1", "splits", split + ".csv"))
df = df[df.target.notna()].reset_index(drop=True)
idx2 = {a: i for i, a in enumerate(PROTEIN)}
X = np.array([[idx2.get(s[i], 0) for i in SITES] for s in df.sequence.values])
y = df.target.values.astype(float)
tr = np.where(df.set.values == "train")[0]
te = np.where(df.set.values != "train")[0]
n = len(y)

W = np.load(wpath)
if W.shape[0] <= max(SITES):
    raise SystemExit("weights too short: %s" % (W.shape,))
Wv = W[SITES]                                  # (4,20) log probabilities
Pw = np.exp(Wv)
oh = np.zeros((n, len(SITES), 20), dtype=np.float32)
for j in range(len(SITES)):
    oh[np.arange(n), j, X[:, j]] = 1
FE = {"match": oh.reshape(n, -1), "lm": (oh * np.sqrt(Pw)[None]).reshape(n, -1)}
zs = sum(Wv[j, X[:, j]] for j in range(len(SITES)))

splits = ([(tr[a], tr[b]) for a, b in KFold(5, shuffle=True, random_state=0).split(tr)]
          if len(tr) < 1000 else [tuple(train_test_split(tr, test_size=0.1, random_state=0))])
GS = [0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8]
LS = [1e-3, 1e-2, 1e-1, 1]

def krr(K, yy, lam):
    b = yy.mean(); a = np.linalg.solve(K + lam * np.eye(len(yy)), yy - b); return a, b

def cv(K, lam):
    P = np.zeros(n); vi = np.concatenate([b for _, b in splits])
    for a_, b_ in splits:
        al, b0 = krr(K[np.ix_(a_, a_)], y[a_], lam); P[b_] = K[np.ix_(b_, a_)] @ al + b0
    return spearmanr(y[vi], P[vi]).statistic

out = {"split": split, "tag": tag, "ntrain": int(len(tr)),
       "zeroshot": float(spearmanr(y[te], zs[te]).statistic)}
for kind, F in FE.items():
    t0 = time.time()
    F = F.astype(np.float64); s = (F ** 2).sum(1)
    D2 = np.maximum(s[:, None] + s[None, :] - 2 * F @ F.T, 0)
    sub = D2[np.ix_(tr[:400], tr[:400])]; med = np.median(sub[sub > 0])
    best = None
    for g in GS:
        K = np.exp(-g * D2 / med)
        lam = max(LS, key=lambda l: cv(K, l)); v = cv(K, lam)
        if best is None or v > best[0]: best = (v, g, lam)
    _, g, lam = best; K = np.exp(-g * D2 / med)
    al, b0 = krr(K[np.ix_(tr, tr)], y[tr], lam)
    out[kind] = float(spearmanr(y[te], K[np.ix_(te, tr)] @ al + b0).statistic)
    out["g_" + kind] = g
    print(" ", kind, round(out[kind], 4), "g", g, "%.0fs" % (time.time() - t0), flush=True)
json.dump(out, open(os.path.join(root, "results", "lmwany_%s_%s.json" % (tag, split)), "w"))
print(json.dumps(out), flush=True)
