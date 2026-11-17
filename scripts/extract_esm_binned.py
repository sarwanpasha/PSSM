"""Composition-weighted, position-binned per-residue ESM-2 features (PyTorch, Intel XPU).
phi(x)[j, :] = sum_{t in bin j} sqrt(w_x(x_t)) * e_t(x)   -- the W-PSSKM weighting applied to ESM residue embeddings.
Bin width b: 5 for GB1 (aligned, short), 50 otherwise (positions up to 1022 -> 21 bins).
usage: python extract_esm_binned.py <model> <dataset> <root>  -> emb/<model>__<dataset>__bin.npy (float16)"""
import sys, os, time, hashlib
import numpy as np, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load
from wpsskm.kernels import PROTEIN

mname, dname, root = sys.argv[1:4]
b = 5 if dname.startswith("gb1") else 50
MAXLEN = 1022; NB = (265 + b - 1) // b if dname.startswith("gb1") else (MAXLEN + b - 1) // b
dev = "xpu" if torch.xpu.is_available() else "cpu"
seqs = load(dname, root)["seqs"]
cache = os.path.join(root, "emb", f"cachebin_{mname}_b{b}"); os.makedirs(cache, exist_ok=True)
key = lambda s: hashlib.sha1(s.encode()).hexdigest()
is_done = lambda s: os.path.exists(os.path.join(cache, key(s) + ".npy"))
todo = sorted({s for s in seqs if not is_done(s)}, key=len)
print(f"{mname} {dname} bins b={b} nb={NB}: {len(todo)} uncached on {dev}", flush=True)
idx = {a: i for i, a in enumerate(PROTEIN)}
if todo:
    model, alphabet = getattr(esm.pretrained, mname)(); layer = model.num_layers
    model = model.eval().to(dev); conv = alphabet.get_batch_converter(); TOK = 4096; t0 = time.time(); i = n = 0
    with torch.inference_mode():
        while i < len(todo):
            Lm = min(len(todo[i]), MAXLEN); batch = todo[i:i + max(1, TOK // (Lm + 2))]
            while len(batch) > 1 and min(len(batch[-1]), MAXLEN) * len(batch) > TOK * 1.2: batch = batch[:-1]
            _, _, tk = conv([(str(j), s[:MAXLEN]) for j, s in enumerate(batch)])
            rep = model(tk.to(dev), repr_layers=[layer])["representations"][layer].float().cpu().numpy()
            for j, s in enumerate(batch):
                s2 = s[:MAXLEN]; ids = np.array([idx.get(c, -1) for c in s2])
                cnt = np.bincount(ids[ids >= 0], minlength=20).astype(float); w = cnt / max(cnt.sum(), 1)
                wt = np.where(ids >= 0, np.sqrt(w[np.maximum(ids, 0)]), 0.0)
                R = rep[j, 1:len(s2) + 1] * wt[:, None]
                out = np.zeros((NB, R.shape[1]), dtype=np.float32)
                np.add.at(out, np.minimum(np.arange(len(s2)) // b, NB - 1), R)
                np.save(os.path.join(cache, key(s) + ".npy"), out.astype(np.float16))
            i += len(batch); n += len(batch)
            if n % 1000 < len(batch): print(f"  {n}/{len(todo)} {(time.time()-t0)/60:.1f} min", flush=True)
E = np.stack([np.load(os.path.join(cache, key(s) + ".npy")).reshape(-1) for s in seqs])
np.save(os.path.join(root, "emb", f"{mname}__{dname}__bin.npy"), E)
print(f"saved {E.shape} -> emb/{mname}__{dname}__bin.npy", flush=True)
