"""Compute-cost table: corrected kernel end-to-end (1 CPU thread) vs ESM-2 embedding extraction (from logs).
usage: python cost_bench.py <root>   (run when the machine is otherwise idle)  -> results/cost.json"""
import os, sys, re, json, time
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import encode, feature_map
from wpsskm.datasets import load
root = sys.argv[1]; out = {"kernel": {}, "esm_extraction_min": {}, "notes": []}

for name in ["gb1_sampled", "gb_nontata", "gb_cohn", "deeploc", "tape_rh_fold"]:
    d = load(name, root); tr, te = d["folds"][0]; clf = d["task"] == "clf"
    t0 = time.perf_counter(); X = encode(d["seqs"], d["alphabet"], d["L"]); P = feature_map(X, len(d["alphabet"]), "sqrt"); t_feat = time.perf_counter() - t0
    m = LinearSVC(C=0.1, max_iter=5000) if clf else Ridge(alpha=1.0)
    t0 = time.perf_counter(); m.fit(P[tr], d["y"][tr]); t_fit = time.perf_counter() - t0
    t0 = time.perf_counter(); m.predict(P[te]); t_pred = time.perf_counter() - t0
    out["kernel"][name] = dict(n=len(d["seqs"]), n_train=int(len(tr)), feature_sec=t_feat, fit_sec=t_fit, predict_sec=t_pred,
                               total_sec=t_feat + t_fit + t_pred, nnz=int(P.nnz), mem_mb=(P.data.nbytes + P.indices.nbytes + P.indptr.nbytes) / 2**20)
    print(name, out["kernel"][name], flush=True)

# ESM extraction minutes: last progress line of each "<model> <dataset>: N seqs, M uncached" block before "saved"
log = open(os.path.join(root, "logs", "esm_queue.log"), encoding="utf-8", errors="ignore").read().splitlines()
cur, last = None, 0.0
for line in log:
    m = re.match(r"(esm2_\S+) (\S+): (\d+) seqs, (\d+) uncached", line.strip())
    if m: cur, last = (m.group(1), m.group(2), int(m.group(4))), 0.0; continue
    p = re.search(r"(\d+)/(\d+)\s+([\d.]+) min", line)
    if p and cur: last = float(p.group(3))
    if line.startswith("saved") and cur:
        k = f"{cur[0]}|{cur[1]}"
        if cur[2] > 0: out["esm_extraction_min"][k] = out["esm_extraction_min"].get(k, 0.0) + last
        cur = None
npu = os.path.join(root, "logs", "npu_650M_deeploc.log")
if os.path.exists(npu):
    mins = [float(x) for x in re.findall(r"done, ([\d.]+) min", open(npu, encoding="utf-8", errors="ignore").read())]
    if mins: out["notes"].append(f"ESM-2 650M DeepLoc: NPU additionally ran {max(mins):.0f} min in parallel (3,600 sequences) before a device error.")
out["notes"].append("Kernel: 1 CPU thread, Intel Core Ultra 5 235U. ESM-2: Intel integrated GPU (PyTorch XPU), batch of ~4k tokens; sequences truncated to 1,022 residues.")
json.dump(out, open(os.path.join(root, "results", "cost.json"), "w"), indent=1)
print(json.dumps(out["esm_extraction_min"], indent=1))
