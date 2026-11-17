"""Which (position, residue) features drive the corrected-kernel SVM on DeepLoc? Saves weights for plotting."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sklearn.svm import LinearSVC
from wpsskm.kernels import encode, feature_map, PROTEIN
from wpsskm.datasets import load
root, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
ds = load("deeploc", root); L = 200                     # N-terminal 200 residues is where sorting signals live
X = encode(ds["seqs"], PROTEIN, L); P = feature_map(X, 20, "sqrt"); y = ds["y"]
tr, te = ds["folds"][0]
m = LinearSVC(C=0.1, max_iter=5000).fit(P[tr], y[tr])
acc = (m.predict(P[te]) == y[te]).mean(); print("fold-0 accuracy using only first 200 residues:", round(acc, 4))
W = m.coef_.reshape(len(ds["classes"]), L, 20)
np.savez(os.path.join(outdir, "deeploc_weights.npz"), W=W, classes=np.array(ds["classes"]), alphabet=np.array(list(PROTEIN)), acc=acc)
for ci, c in enumerate(ds["classes"]):
    w = W[ci]; imp = np.abs(w).sum(1)
    top = np.argsort(-w.ravel())[:8]
    print(f"{c:22s} importance in pos 1-30: {imp[:30].sum()/imp.sum():.2f} | top +features:",
          ", ".join(f"{PROTEIN[k % 20]}{k // 20 + 1}" for k in top))
