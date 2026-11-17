"""ESM-2 embeddings with OpenVINO on NPU (or GPU), sharing the per-sequence cache with extract_esm.py.
Processes sequences longest-first (the PyTorch GPU worker goes shortest-first), re-checks the cache before
every batch, so the two workers split the work without duplicating it.
usage: python ov_extract.py <model> <dataset> <root> <NPU|GPU>"""
import sys, os, time, hashlib
import numpy as np, torch, esm, openvino as ov
import esm.rotary_embedding as _rot
_orig = _rot.RotaryEmbedding._update_cos_sin_tables
def _nocache(self, x, seq_dimension=1): self._seq_len_cached = None; return _orig(self, x, seq_dimension)
_rot.RotaryEmbedding._update_cos_sin_tables = _nocache
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load

mname, dname, root, dev = sys.argv[1:5]
MAXLEN = 1022; BUCKETS = [66, 130, 258, 514, 1024]
NPU_MAX = int(sys.argv[5]) if len(sys.argv) > 5 else 1024     # longest token bucket this worker handles
TOKENS = int(sys.argv[6]) if len(sys.argv) > 6 else 2048       # tokens per inference call
BUCKETS = [b for b in BUCKETS if b <= NPU_MAX]
cache = os.path.join(root, "emb", "cache_" + mname); os.makedirs(cache, exist_ok=True)
irdir = os.path.join(root, "ov_models"); os.makedirs(irdir, exist_ok=True)
key = lambda s: hashlib.sha1(s.encode()).hexdigest()
done = lambda s: os.path.exists(os.path.join(cache, key(s) + ".npy"))
seqs = load(dname, root)["seqs"]
todo = sorted({s for s in seqs if not done(s) and min(len(s), MAXLEN) + 2 <= BUCKETS[-1]}, key=len, reverse=True)
print(f"[{dev}] {mname} {dname}: {len(todo)} uncached", flush=True)
if not todo: sys.exit(0)

model, alphabet = getattr(esm.pretrained, mname)(); model.eval(); L = model.num_layers
conv = alphabet.get_batch_converter(); PAD = alphabet.padding_idx
class Rep(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, t): return s.m(t, repr_layers=[L])["representations"][L]
core = ov.Core(); core.set_property({"CACHE_DIR": os.path.join(irdir, "cache")})
compiled = {}
def get(Lp):
    B = max(1, TOKENS // Lp)
    if Lp not in compiled:
        xml = os.path.join(irdir, f"{mname}_B{B}_L{Lp}.xml")
        if not os.path.exists(xml):
            ex = torch.full((B, Lp), PAD, dtype=torch.long); ex[:, 0] = alphabet.cls_idx; ex[:, 1:Lp // 2] = 5
            with torch.no_grad(): tr = torch.jit.trace(Rep(model), ex, check_trace=False, strict=False)
            ov.save_model(ov.convert_model(tr, example_input=ex, input=[(B, Lp)]), xml, compress_to_fp16=True)
        t0 = time.perf_counter(); compiled[Lp] = (core.compile_model(xml, dev).create_infer_request(), B)
        print(f"[{dev}] compiled L={Lp} B={B} in {time.perf_counter()-t0:.0f}s", flush=True)
    return compiled[Lp]

t0, n_done, i = time.time(), 0, 0
while i < len(todo):
    while i < len(todo) and done(todo[i]): i += 1
    if i >= len(todo): break
    Lp = next(b for b in BUCKETS if min(len(todo[i]), MAXLEN) + 2 <= b)
    req, B = get(Lp)
    batch = [s for s in todo[i:i + B] if not done(s)]; i += B
    if not batch: continue
    _, _, tk = conv([(str(j), s[:MAXLEN]) for j, s in enumerate(batch)])
    tokens = np.full((B, Lp), PAD, dtype=np.int64); tokens[:len(batch), :tk.shape[1]] = tk.numpy()
    try:
        rep = req.infer({0: tokens})[0]
    except RuntimeError as e:
        print(f"[{dev}] inference failed at L={Lp} B={B}: {str(e).splitlines()[-1][:150]} -- stopping; GPU will cover the rest", flush=True)
        sys.exit(2)
    for j, s in enumerate(batch):
        n = min(len(s), MAXLEN)
        np.save(os.path.join(cache, key(s) + ".npy"), rep[j, 1:n + 1].astype(np.float32).mean(0))
    n_done += len(batch)
    if n_done % 200 < len(batch):
        print(f"[{dev}] {n_done} done, {(time.time()-t0)/60:.1f} min, current len~{len(batch[0])}", flush=True)
print(f"[{dev}] finished {n_done} seqs in {(time.time()-t0)/60:.1f} min", flush=True)
