"""Convolutional baseline on the FLIP GB1 splits, in the style of the CNN supplied with FLIP.
The network sees the same varying positions the kernels use, so the comparison isolates the model
rather than the featurization. Selection uses the same held-out protocol as the kernels: the
bandwidth grid is replaced by the number of training epochs, chosen on the validation split.
usage: python run_cnn.py <split> <root>
Writes results/cnn_<split>.json"""
import sys, os, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, KFold
from wpsskm.kernels import PROTEIN

split, root = sys.argv[1], sys.argv[2]
SITES = [38, 39, 40, 53]
dev = os.environ.get("PSSM_DEV") or ("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu")
df = pd.read_csv(os.path.join(root, "data", "flip_gb1", "splits", split + ".csv"))
df = df[df.target.notna()].reset_index(drop=True)
idx2 = {a: i for i, a in enumerate(PROTEIN)}
X = np.array([[idx2.get(s[i], 0) for i in SITES] for s in df.sequence.values])
y = df.target.values.astype(np.float32)
tr = np.where(df.set.values == "train")[0]; te = np.where(df.set.values != "train")[0]
oh = np.zeros((len(y), len(SITES), 20), dtype=np.float32)
for j in range(len(SITES)):
    oh[np.arange(len(y)), j, X[:, j]] = 1
mu, sd = y[tr].mean(), y[tr].std() + 1e-8

class CNN(nn.Module):
    def __init__(self, A=20, ch=256):
        super().__init__()
        self.conv = nn.Conv1d(A, ch, kernel_size=3, padding=1)
        self.head = nn.Sequential(nn.Linear(2 * ch, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))
    def forward(self, x):                      # x: (B, L, A)
        h = torch.relu(self.conv(x.transpose(1, 2)))
        h = torch.cat([h.mean(-1), h.max(-1).values], -1)
        return self.head(h).squeeze(-1)

def fit_eval(itr, iva, ieval, epochs_grid=(20, 50, 100, 200)):
    torch.manual_seed(0)
    m = CNN().to(dev); opt = torch.optim.Adam(m.parameters(), lr=1e-3); lossf = nn.MSELoss()
    Xtr = torch.tensor(oh[itr]).to(dev); ytr = torch.tensor((y[itr] - mu) / sd).to(dev)
    Xva = torch.tensor(oh[iva]).to(dev); Xev = torch.tensor(oh[ieval]).to(dev)
    best = (-2, 0, None); B = 128
    for ep in range(1, max(epochs_grid) + 1):
        m.train(); perm = torch.randperm(len(itr), device=dev)
        for s in range(0, len(itr), B):
            b = perm[s:s+B]; opt.zero_grad(); loss = lossf(m(Xtr[b]), ytr[b]); loss.backward(); opt.step()
        if ep in epochs_grid:
            m.eval()
            with torch.no_grad():
                v = spearmanr(y[iva], m(Xva).float().cpu().numpy()).statistic
                p = m(Xev).float().cpu().numpy()
            if v > best[0]: best = (v, ep, p)
    return best

t0 = time.time()
if len(tr) < 1000:
    folds = list(KFold(5, shuffle=True, random_state=0).split(tr))
    itr, iva = tr[folds[0][0]], tr[folds[0][1]]
else:
    itr, iva = train_test_split(tr, test_size=0.1, random_state=0)
v, ep, pred = fit_eval(itr, iva, te)
rho = float(spearmanr(y[te], pred).statistic)
out = {"split": split, "n_train": int(len(tr)), "n_test": int(len(te)),
       "spearman": rho, "epochs": int(ep), "val": float(v), "sec": time.time() - t0}
json.dump(out, open(os.path.join(root, "results", "cnn_" + split + ".json"), "w"))
print(split, "CNN SPEARMAN", round(rho, 4), "epochs", ep, "%.0fs" % (time.time() - t0), flush=True)
