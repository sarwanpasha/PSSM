"""Head-to-head against the released implementation of the Zhou/McCandlish variance-component model.
Runs gpmap-tools' VCregression on the FLIP GB1 splits, using the same training and test partitions
as our own runs, and reports test Spearman rho so the number is directly comparable to Table 4.
GB1 varies at four sites (0-based 38, 39, 40, 53 of the 265-residue sequence).
Follows the calling convention of gpmap-tools' own vc_regression command line script.
usage: python gpmap_head2head.py <root> <split> [outdir]
Writes <outdir>/gpmap_<split>.json"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from gpmap.inference import VCregression

root, split = sys.argv[1], sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.join(root, "results")
SITES = [38, 39, 40, 53]

df = pd.read_csv(os.path.join(root, "data", "flip_gb1", "splits", split + ".csv"))
df = df[df.target.notna()].reset_index(drop=True)
seq4 = np.array(["".join(s[i] for i in SITES) for s in df.sequence.values])
y = df.target.values.astype(float)
is_tr = (df.set.values == "train")
Xtr_all, ytr_all = seq4[is_tr], y[is_tr]
Xte, yte = seq4[~is_tr], y[~is_tr]
_, keep = np.unique(Xtr_all, return_index=True)
keep = np.sort(keep)
Xtr, ytr = Xtr_all[keep], ytr_all[keep]
print(split, "train", len(Xtr), "test", len(Xte), flush=True)

t0 = time.time()
# the space is the complete 20^4 GB1 combinatorial space
m = VCregression(seq_length=4, alphabet_type="protein", cross_validation=True, progress=False)
Xtr_s = pd.Series(ytr, index=list(Xtr)).index.values
m.fit(X=Xtr_s, y=ytr)
lambdas = np.asarray(m.lambdas).ravel()
m.set_data(X=Xtr_s, y=ytr)
m.set_lambdas(lambdas)
res = m.predict(X_pred=list(Xte))
if hasattr(res, "columns"):
    col = "ypred" if "ypred" in res.columns else res.columns[0]
    pred = res[col].values
else:
    pred = np.asarray(res).ravel()
pred = np.asarray(pred, dtype=float).ravel()
rho = float(spearmanr(yte, pred).statistic)
out = {"split": split, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
       "spearman": rho, "lambdas": lambdas.tolist(), "sec": time.time() - t0}
os.makedirs(outdir, exist_ok=True)
json.dump(out, open(os.path.join(outdir, "gpmap_" + split + ".json"), "w"))
print(split, "SPEARMAN", round(rho, 4), "n_pred", len(pred), "%.0fs" % (time.time() - t0), flush=True)
