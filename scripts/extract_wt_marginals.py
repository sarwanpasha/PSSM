"""ESM-2 masked marginals of a wild-type sequence at every position, for ProteinGym assays.
usage: python extract_wt_marginals.py <model> <root> <assay> [<assay> ...]
Reads <root>/data/proteingym/DMS_substitutions.csv (column target_seq).
Writes <root>/emb/<model>__pg_<assay>__wtlogp.npy of shape (L,20), log p(a | WT with position t masked), PROTEIN order."""
import sys, os, time, csv
import numpy as np, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.kernels import PROTEIN

mname, root, assays = sys.argv[1], sys.argv[2], sys.argv[3:]
csv.field_size_limit(10**7)
ref = {r["DMS_id"]: r["target_seq"] for r in csv.DictReader(open(os.path.join(root, "data", "proteingym", "DMS_substitutions.csv"), encoding="utf8"))}
dev = "xpu" if torch.xpu.is_available() else "cpu"
model, alphabet = getattr(esm.pretrained, mname)()
model = model.eval().to(dev); conv = alphabet.get_batch_converter()
aa_idx = torch.tensor([alphabet.get_idx(a) for a in PROTEIN])
for a in assays:
    out_f = os.path.join(root, "emb", f"{mname}__pg_{a}__wtlogp.npy")
    if os.path.exists(out_f): print("skip", a, flush=True); continue
    if a not in ref: print("MISSING in reference csv:", a, flush=True); continue
    wt = ref[a]
    if len(wt) > 1022: print("too long for one pass:", a, len(wt), flush=True); continue
    t0 = time.time(); _, _, tk = conv([("wt", wt)])
    out = np.zeros((len(wt), 20), dtype=np.float32); B = max(1, 4096 // (len(wt) + 2))
    with torch.inference_mode():
        for s in range(0, len(wt), B):
            pos = list(range(s, min(s + B, len(wt)))); batch = tk.repeat(len(pos), 1)
            for j, p in enumerate(pos): batch[j, p + 1] = alphabet.mask_idx
            lg = torch.log_softmax(model(batch.to(dev))["logits"].float().cpu()[:, :, aa_idx], -1)
            for j, p in enumerate(pos): out[p] = lg[j, p + 1].numpy()
    np.save(out_f, out); print(a, out.shape, f"{time.time()-t0:.0f}s", flush=True)
print("ALL DONE", flush=True)
