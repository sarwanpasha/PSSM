"""Mean-pooled ESM-2 embeddings on Intel XPU (falls back to CPU).
usage: python extract_esm.py <model> <dataset> <root>
Saves <root>/emb/<model>__<dataset>.npy aligned with load(dataset)['seqs'] order.
Resumable: unique sequences are cached in <root>/emb/cache_<model>/<sha1>.npy."""
import sys, os, time, hashlib
import numpy as np, torch, esm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load

mname, dname, root = sys.argv[1], sys.argv[2], sys.argv[3]
dev = "xpu" if torch.xpu.is_available() else "cpu"
seqs = load(dname, root)["seqs"]
cache = os.path.join(root, "emb", "cache_" + mname); os.makedirs(cache, exist_ok=True)
key = lambda s: hashlib.sha1(s.encode()).hexdigest()
todo = sorted({s for s in seqs if not os.path.exists(os.path.join(cache, key(s) + ".npy"))}, key=len)
print(f"{mname} {dname}: {len(seqs)} seqs, {len(todo)} uncached, device={dev}", flush=True)

if todo:
    model, alphabet = getattr(esm.pretrained, mname)()
    layer = model.num_layers; model = model.eval().to(dev); conv = alphabet.get_batch_converter()
    MAXLEN, TOKENS = 1022, 4096                       # truncate to ESM context; ~4k tokens per batch
    t0, done, i = time.time(), 0, 0
    with torch.inference_mode():
        is_done = lambda s: os.path.exists(os.path.join(cache, key(s) + ".npy"))
        while i < len(todo):
            while i < len(todo) and is_done(todo[i]): i += 1   # another worker (NPU) may have done these
            if i >= len(todo): break
            L = min(len(todo[i]), MAXLEN); bs = max(1, TOKENS // (L + 2))
            # batch of similar length (sorted), bounded by token budget of the longest
            batch = [x for x in todo[i:i + bs] if not is_done(x)] or todo[i:i + 1]
            while batch and min(len(batch[-1]), MAXLEN) * len(batch) > TOKENS * 1.2 and len(batch) > 1:
                batch = batch[:-1]
            _, _, tk = conv([(str(j), s[:MAXLEN]) for j, s in enumerate(batch)])
            rep = model(tk.to(dev), repr_layers=[layer])["representations"][layer].float().cpu().numpy()
            for j, s in enumerate(batch):
                n = min(len(s), MAXLEN)
                np.save(os.path.join(cache, key(s) + ".npy"), rep[j, 1:n + 1].mean(0).astype(np.float32))
            i += len(batch); done += len(batch)
            if done % 500 < len(batch):
                el = time.time() - t0
                print(f"  {done}/{len(todo)}  {el/60:.1f} min  (~{el/done*(len(todo)-done)/60:.0f} min left, len~{L})", flush=True)

E = np.stack([np.load(os.path.join(cache, key(s) + ".npy")) for s in seqs])
os.makedirs(os.path.join(root, "emb"), exist_ok=True)
np.save(os.path.join(root, "emb", f"{mname}__{dname}.npy"), E)
print(f"saved {E.shape} -> emb/{mname}__{dname}.npy", flush=True)
