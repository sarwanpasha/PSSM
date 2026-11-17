"""Classical string-kernel baselines with the same splits/tuning as run_kernels.py.
usage: python run_stringkernels.py <dataset|tape_rh> <root> [kind ...]
'tape_rh' trains once on TAPE train+valid and evaluates fold/superfamily/family test sets.
Writes results/strings_<dataset>.json (incremental) with feature-build and fit times."""
import sys, os, json, time
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode
from wpsskm.datasets import load
from wpsskm.stringkernels import featurize, KINDS

name, root = sys.argv[1], sys.argv[2]
if name == "tape_rh":
    parts = {s: load(f"tape_rh_{s}", root) for s in ["fold", "superfamily", "family"]}
    d0 = parts["fold"]; ntr = len(d0["folds"][0][0])
    seqs = d0["seqs"][:ntr]; y = list(d0["y"][:ntr]); tests = {}
    for s, d in parts.items():
        te = d["folds"][0][1]; tests[s] = np.arange(len(seqs), len(seqs) + len(te))
        seqs += [d["seqs"][i] for i in te]; y += list(d["y"][te])
    y = np.array(y); task, alphabet, L = "clf", d0["alphabet"], d0["L"]
    folds = [(np.arange(ntr), tests)]
else:
    d = load(name, root); seqs, y, task, alphabet, L = d["seqs"], d["y"], d["task"], d["alphabet"], d["L"]
    folds = [(tr, {"test": te}) for tr, te in d["folds"]]
clf = task == "clf"; A = len(alphabet)
kinds = sys.argv[3:] or KINDS[A]
X = encode(seqs, alphabet, L)
out = os.path.join(root, "results", f"strings_{name}.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
res = json.load(open(out)) if os.path.exists(out) else dict(dataset=name, runs=[])
done = {(r["kernel"], r["fold"]) for r in res["runs"]}

def metrics(yt, yp):
    if clf: return dict(acc=accuracy_score(yt, yp), f1_weighted=f1_score(yt, yp, average="weighted"), f1_macro=f1_score(yt, yp, average="macro"))
    return dict(spearman=spearmanr(yt, yp).statistic, r2=r2_score(yt, yp))

for kind in kinds:
    if all((kind, fi) in done for fi in range(len(folds))): continue
    t0 = time.time(); F = featurize(kind, X, A); t_feat = time.time() - t0
    print(f"[{name}] {kind}: features {F.shape} in {t_feat:.1f}s", flush=True)
    for fi, (tr, tests) in enumerate(folds):
        if (kind, fi) in done: continue
        try: itr, iva = train_test_split(tr, test_size=0.1, random_state=0, stratify=y[tr] if clf else None)
        except ValueError: itr, iva = train_test_split(tr, test_size=0.1, random_state=0)
        mk = (lambda c: LinearSVC(C=c, max_iter=5000)) if clf else (lambda a: Ridge(alpha=a))
        grid = [0.01, 0.1, 1.0, 10.0] if clf else [0.01, 0.1, 1.0, 10.0, 100.0]
        score = lambda g: (accuracy_score(y[iva], mk(g).fit(F[itr], y[itr]).predict(F[iva])) if clf
                           else spearmanr(y[iva], mk(g).fit(F[itr], y[itr]).predict(F[iva])).statistic)
        best = max(grid, key=score)
        t0 = time.time(); m = mk(best).fit(F[tr], y[tr]); t_fit = time.time() - t0
        for tname, te in tests.items():
            r = dict(fold=fi, kernel=kind, model="svm" if clf else "kridge", test=tname, hyper=best,
                     feat_sec=t_feat, fit_sec=t_fit, dims=int(F.shape[1]), **metrics(y[te], m.predict(F[te])))
            res["runs"].append(r); print(name, r, flush=True)
        json.dump(res, open(out, "w"), indent=1, default=float)
    del F
print(f"[{name}] done", flush=True)
