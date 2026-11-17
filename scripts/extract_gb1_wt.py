"""Masked marginals of the GB1 wild type at every position, for any ESM model.
usage: python extract_gb1_wt.py <model> <root>
Writes <root>/emb/<model>__gb1_wtlogp.npy of shape (L,20), log p(a | WT with position t masked)."""
import sys, os, time
import numpy as np, pandas as pd, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import PROTEIN

mname, root = sys.argv[1], sys.argv[2]
out_f = os.path.join(root, "emb", f"{mname}__gb1_wtlogp.npy")
SITES = [38, 39, 40, 53]
df = pd.read_csv(os.path.join(root, "data", "flip_gb1", "splits", "sampled.csv"))
df = df[df.target.notna()]
# the wild type is the variant carrying VDGV at the four variable sites
wt = None
for s in df.sequence.values:
    if "".join(s[i] for i in SITES) == "VDGV":
        wt = s; break
assert wt is not None, "wild type not found"
print("WT length", len(wt), "sites", "".join(wt[i] for i in SITES), flush=True)

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
np.save(out_f, out)
print("wrote", out_f, out.shape, "%.0fs" % (time.time() - t0), flush=True)
