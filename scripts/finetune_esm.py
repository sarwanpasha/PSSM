"""Fine-tune ESM-2 on a FLIP GB1 split as a baseline for the kernel methods.
usage: python finetune_esm.py <model> <split> <root> [epochs] [lr] [bs] [n_trainable_blocks]
n_trainable_blocks: number of final transformer blocks to update (0 = head only, -1 = all).
Regression head on the mean-pooled final representation. Model selection on a validation split of the
training data. Writes results/ft_<model>__<split>.json and prints throughput for the first steps."""
import sys, os, json, time, copy
import numpy as np, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from wpsskm.datasets import load

mname, split, root = sys.argv[1], sys.argv[2], sys.argv[3]
EP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
LR = float(sys.argv[5]) if len(sys.argv) > 5 else 3e-4
BS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
NB = int(sys.argv[7]) if len(sys.argv) > 7 else 2
dev = "xpu" if torch.xpu.is_available() else "cpu"
d = load("gb1_" + split, root); seqs = d["seqs"]; y = np.asarray(d["y"], dtype=np.float32); tr, te = d["folds"][0]
backbone, alphabet = getattr(esm.pretrained, mname)(); conv = alphabet.get_batch_converter()
nl = backbone.num_layers

class Reg(torch.nn.Module):
    def __init__(self, bb, dim):
        super().__init__(); self.bb = bb; self.head = torch.nn.Linear(dim, 1)
    def forward(self, tk):
        r = self.bb(tk, repr_layers=[nl])["representations"][nl]
        m = (tk != alphabet.padding_idx).unsqueeze(-1).float()
        return self.head((r * m).sum(1) / m.sum(1)).squeeze(-1)

model = Reg(backbone, backbone.embed_dim).to(dev)
if NB >= 0:
    for p in model.bb.parameters(): p.requires_grad = False
    for i in range(max(0, nl - NB), nl):
        for p in model.bb.layers[i].parameters(): p.requires_grad = True
    if hasattr(model.bb, "emb_layer_norm_after"):
        for p in model.bb.emb_layer_norm_after.parameters(): p.requires_grad = True
ntr_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"device {dev}, trainable params {ntr_p/1e6:.2f}M of {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)
itr, iva = train_test_split(np.arange(len(tr)), test_size=max(0.1, 4 / len(tr)), random_state=0)
itr, iva = tr[itr], tr[iva]
mu, sd = y[itr].mean(), y[itr].std() + 1e-8
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
def batches(idx, bs, sh):
    o = np.random.permutation(idx) if sh else idx
    for i in range(0, len(o), bs): yield o[i:i + bs]
def tok(ii): return conv([(str(j), seqs[j]) for j in ii])[2].to(dev)
@torch.no_grad()
def pred(idx):
    model.eval(); out = []
    for b in batches(idx, 32, False): out.append(model(tok(b)).float().cpu().numpy())
    return np.concatenate(out) * sd + mu
best = (-2, None); t0 = time.time(); step = 0
for ep in range(EP):
    model.train()
    for b in batches(itr, BS, True):
        ts = time.time(); opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(tok(b)), torch.tensor((y[b] - mu) / sd, device=dev))
        loss.backward(); opt.step(); step += 1
        if step <= 5 or step % 100 == 0: print(f"  step {step} {time.time()-ts:.1f}s loss {loss.item():.3f}", flush=True)
    v = spearmanr(y[iva], pred(iva)).statistic
    print(f"  epoch {ep} val {v:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
    if v > best[0]: best = (v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()})
model.load_state_dict(best[1])
t = float(spearmanr(y[te], pred(te)).statistic)
r = dict(model=mname, split=split, val=float(best[0]), test=t, epochs=EP, lr=LR, bs=BS, blocks=NB,
         trainable_M=ntr_p / 1e6, ntrain=len(itr), min=(time.time() - t0) / 60)
json.dump(r, open(os.path.join(root, "results", f"ft_{mname}__{split}.json"), "w"))
print(json.dumps(r), flush=True)
