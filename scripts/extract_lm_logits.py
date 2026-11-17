"""Per-residue ESM-2 probabilities used as W-PSSKM weights (LM-weighted W-PSSKM).
usage: python extract_lm_logits.py <model> <dataset> <root> [wt]
  wt   : (for mutational scans, e.g. gb1_*) masked marginals of the wild type at every position
         -> emb/<model>__<dataset>__wtlogp.npy  shape (L, 20), log p(a | WT with position t masked), PROTEIN order
  else : one unmasked pass per sequence, log p(x_t | x) for the observed residue
         -> emb/<model>__<dataset>__logp.npy   shape (n, L) float16, NaN beyond the sequence end."""
import sys, os, time
import numpy as np, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load
from wpsskm.kernels import PROTEIN

mname, dname, root = sys.argv[1], sys.argv[2], sys.argv[3]
mode = sys.argv[4] if len(sys.argv) > 4 else "seq"
dev = "xpu" if torch.xpu.is_available() else "cpu"
d = load(dname, root); seqs, L = d["seqs"], d["L"]
model, alphabet = getattr(esm.pretrained, mname)()
model = model.eval().to(dev); conv = alphabet.get_batch_converter()
aa_idx = torch.tensor([alphabet.get_idx(a) for a in PROTEIN])
t0 = time.time()
with torch.inference_mode():
    if mode == "wt":
        from collections import Counter
        X = np.array([list(s) for s in seqs])
        # The reference background must be the real wild type: in a combinatorial mutagenesis library the
        # per-position consensus is meaningless at the mutated sites (all 20 residues occur about equally).
        if os.path.exists(os.path.join(root, "data", "wt", dname.split("_")[0] + ".txt")):
            wt = open(os.path.join(root, "data", "wt", dname.split("_")[0] + ".txt")).read().split()[0]
            assert len(wt) == X.shape[1], f"wt length {len(wt)} != {X.shape[1]}"
        else:
            wt = "".join(Counter(X[:, t]).most_common(1)[0][0] for t in range(X.shape[1]))
            print("WARNING: no data/wt/<task>.txt; using per-position consensus as background", flush=True)
        _, _, tk = conv([("wt", wt)])
        out = np.zeros((len(wt), 20), dtype=np.float32)
        B = 16
        for s in range(0, len(wt), B):
            pos = list(range(s, min(s + B, len(wt))))
            batch = tk.repeat(len(pos), 1)
            for j, p in enumerate(pos): batch[j, p + 1] = alphabet.mask_idx
            lg = torch.log_softmax(model(batch.to(dev))["logits"].float().cpu()[:, :, aa_idx], -1)
            for j, p in enumerate(pos): out[p] = lg[j, p + 1].numpy()
        np.save(os.path.join(root, "emb", f"{mname}__{dname}__wtlogp.npy"), out)
        print("wt", wt[:60], "...", out.shape, f"{time.time()-t0:.0f}s", flush=True)
    else:
        MAXLEN, TOKENS = 1022, 4096
        order = np.argsort([len(s) for s in seqs]); out = np.full((len(seqs), L), np.nan, dtype=np.float16)
        i = 0
        while i < len(order):
            Lb = min(len(seqs[order[min(i + 1, len(order) - 1)]]), MAXLEN)
            bs = max(1, TOKENS // (min(len(seqs[order[min(i + TOKENS // (Lb + 2), len(order)) - 1]]), MAXLEN) + 2))
            idx = order[i:i + bs]
            _, _, tk = conv([(str(j), seqs[j][:MAXLEN]) for j in idx])
            lg = torch.log_softmax(model(tk.to(dev))["logits"].float(), -1).cpu()
            for b, j in enumerate(idx):
                n = min(len(seqs[j]), MAXLEN, L)
                lp = lg[b, torch.arange(1, n + 1), tk[b, 1:n + 1]].numpy()
                out[j, :n] = lp
            i += len(idx)
            if (i // bs) % 50 == 0: print(f"  {i}/{len(seqs)} {(time.time()-t0)/60:.1f} min", flush=True)
        np.save(os.path.join(root, "emb", f"{mname}__{dname}__logp.npy"), out)
        print("saved", out.shape, f"{(time.time()-t0)/60:.1f} min", flush=True)
