"""ESM-2 masked marginals of the FLIP AAV wild type at every position.
The FLIP AAV sequences vary in length (734-750) because the variable region carries indels, and
run_aav.py encodes them in left-padded slots. The reference used here is the consensus over the
full-length sequences, which outside the variable region is the AAV2 VP1 wild type.
usage: python extract_aav_wt.py <model> <root>
Writes <root>/emb/<model>__aav_wtlogp.npy of shape (L,20), log p(a | WT with position t masked)."""
import sys, os, time, glob
import numpy as np, pandas as pd, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import PROTEIN

mname, root = sys.argv[1], sys.argv[2]
out_f = os.path.join(root, "emb", f"{mname}__aav_wtlogp.npy")
seqs = []
for f in sorted(glob.glob(os.path.join(root, "data", "flip_aav", "splits", "*.csv"))):
    d = pd.read_csv(f, usecols=["sequence", "target"])
    seqs.extend(d.sequence[d.target.notna()].tolist())
L = max(len(s) for s in seqs)
full = [s for s in seqs if len(s) == L]
arr = np.array([list(s) for s in full])
wt = "".join(pd.Series(arr[:, j]).mode()[0] for j in range(L))
print("sequences", len(seqs), "full-length", len(full), "L", L, flush=True)
print("WT head", wt[:40], flush=True)
np.save(os.path.join(root, "emb", "aav_wt_seq.npy"), np.array([wt]))

dev = "xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu"
model, alphabet = getattr(esm.pretrained, mname)()
model = model.eval().to(dev); conv = alphabet.get_batch_converter()
aa_idx = torch.tensor([alphabet.get_idx(a) for a in PROTEIN])
t0 = time.time(); _, _, tk = conv([("wt", wt)])
out = np.zeros((len(wt), 20), dtype=np.float32); B = max(1, 4096 // (len(wt) + 2))
with torch.inference_mode():
    for s in range(0, len(wt), B):
        pos = list(range(s, min(s + B, len(wt)))); batch = tk.repeat(len(pos), 1)
        for j, p in enumerate(pos): batch[j, p + 1] = alphabet.mask_idx
        lg = torch.log_softmax(model(batch.to(dev))["logits"].float().cpu()[:, :, aa_idx], -1)
        for j, p in enumerate(pos): out[p] = lg[j, p + 1].numpy()
        if s % 100 == 0: print("pos", s, f"{time.time()-t0:.0f}s", flush=True)
np.save(out_f, out)
print("wrote", out_f, out.shape, f"{time.time()-t0:.0f}s", flush=True)
