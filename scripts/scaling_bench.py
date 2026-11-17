"""Time/memory: conference kernel (dense n x n Gram + SVC precomputed) vs corrected feature map (+ LinearSVC)."""
import sys, os, time, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode, feature_map, composition, _onehot_values, PROTEIN
from sklearn.svm import SVC, LinearSVC
rng = np.random.default_rng(0); L = 500; A = 20
out = []
for n in [500, 1000, 2000, 4000, 8000, 16000, 32000]:
    # 4-class synthetic task: class-specific residue preferences at 25 random positions
    pos = rng.choice(L, 25, replace=False); y = rng.integers(0, 4, n)
    X = rng.integers(0, A, (n, L)).astype(np.int16)
    for c in range(4):
        m = y == c; X[np.ix_(m, pos)] = np.where(rng.random((m.sum(), 25)) < 0.5, (c * 5) % A, X[np.ix_(m, pos)])
    r = dict(n=n)
    t = time.time(); P = feature_map(X, A, "sqrt"); LinearSVC(C=0.1, max_iter=2000).fit(P, y); r["corrected_sec"] = time.time() - t
    r["corrected_mem_mb"] = (P.data.nbytes + P.indices.nbytes + P.indptr.nbytes) / 2**20
    if n <= 8000:
        t = time.time()
        W = composition(X, A); U = _onehot_values(X, A, W); O = _onehot_values(X, A)
        K = (U @ O.T + O @ U.T).toarray(); r["original_gram_sec"] = time.time() - t
        t = time.time(); SVC(C=1, kernel="precomputed").fit(K, y); r["original_svm_sec"] = time.time() - t
        r["original_mem_mb"] = K.nbytes / 2**20; del K
    out.append(r); print(r, flush=True)
json.dump(out, open("/mnt/user-data/outputs/scaling/scaling.json", "w"), indent=1)
