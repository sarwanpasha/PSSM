"""Benchmark ESM-2 inference on Intel NPU vs GPU (OpenVINO and PyTorch XPU); check embedding agreement.
usage: python npu_bench.py <model_name> <root>"""
import sys, os, time, random
import numpy as np, torch, esm, openvino as ov
import esm.rotary_embedding as _rot
_orig_tables = _rot.RotaryEmbedding._update_cos_sin_tables
def _no_cache(self, x, seq_dimension=1):      # recompute rotary tables every call (tracing-safe)
    self._seq_len_cached = None
    return _orig_tables(self, x, seq_dimension)
_rot.RotaryEmbedding._update_cos_sin_tables = _no_cache

mname, root = sys.argv[1], sys.argv[2]
irdir = os.path.join(root, "ov_models"); os.makedirs(irdir, exist_ok=True)
model, alphabet = getattr(esm.pretrained, mname)(); model.eval(); L = model.num_layers
conv = alphabet.get_batch_converter(); PAD = alphabet.padding_idx

class Rep(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, tokens):
        return self.m(tokens, repr_layers=[L])["representations"][L]

def tokens_for(seqs, Lp):
    _, _, tk = conv([(str(i), s) for i, s in enumerate(seqs)])
    out = torch.full((len(seqs), Lp), PAD, dtype=torch.long); out[:, :tk.shape[1]] = tk[:, :Lp]; return out

def pool(rep, seqs):
    return np.stack([rep[i, 1:len(s) + 1].mean(0) for i, s in enumerate(seqs)])

AA = "ACDEFGHIKLMNPQRSTVWY"; random.seed(0)
core = ov.Core(); devs = core.available_devices; print("OpenVINO devices:", devs, flush=True)
for Lp, B in [(130, 16), (258, 8), (514, 4)]:
    seqs = ["".join(random.choice(AA) for _ in range(random.randint(Lp // 2, Lp - 2))) for _ in range(B * 4)]
    batches = [seqs[i:i + B] for i in range(0, len(seqs), B)]
    with torch.no_grad():
        ref = np.concatenate([pool(Rep(model)(tokens_for(b, Lp)).numpy(), b) for b in batches])
    xml = os.path.join(irdir, f"{mname}_B{B}_L{Lp}.xml")
    if not os.path.exists(xml):
        for mod in model.modules():               # drop cached rotary tables before tracing
            if hasattr(mod, "_seq_len_cached"): mod._seq_len_cached = None; mod._cos_cached = None; mod._sin_cached = None
        ex = tokens_for(batches[0], Lp); ex[0, -5:] = PAD           # trace the padded branch
        with torch.no_grad():
            traced = torch.jit.trace(Rep(model).eval(), ex, check_trace=False, strict=False)
        ovm = ov.convert_model(traced, example_input=ex, input=[(B, Lp)])
        ov.save_model(ovm, xml, compress_to_fp16=True)
    for dev in [d for d in ["NPU", "GPU"] if d in devs]:
        try:
            t0 = time.perf_counter(); cm = core.compile_model(xml, dev); tc = time.perf_counter() - t0
            req = cm.create_infer_request(); req.infer({0: tokens_for(batches[0], Lp).numpy()})   # warm-up
            t0 = time.perf_counter(); embs = []
            for b in batches:
                embs.append(pool(req.infer({0: tokens_for(b, Lp).numpy()})[0], b))
            dt = (time.perf_counter() - t0) / len(seqs); E = np.concatenate(embs)
            cos = (E * ref).sum(1) / np.linalg.norm(E, axis=1) / np.linalg.norm(ref, axis=1)
            print(f"L={Lp:4d} B={B:2d} OV-{dev}: compile {tc:5.1f}s  {dt*1000:7.1f} ms/seq  cos vs fp32 min {cos.min():.5f}", flush=True)
        except Exception as e:
            print(f"L={Lp:4d} OV-{dev}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
    if torch.xpu.is_available():
        m2 = Rep(model).to("xpu")
        with torch.no_grad():
            m2(tokens_for(batches[0], Lp).to("xpu")); torch.xpu.synchronize(); t0 = time.perf_counter()
            for b in batches: m2(tokens_for(b, Lp).to("xpu")).cpu()
            torch.xpu.synchronize()
        print(f"L={Lp:4d} B={B:2d} PyTorch-XPU: {(time.perf_counter()-t0)/len(seqs)*1000:7.1f} ms/seq (GPU is shared with the running queue)", flush=True)
        model.cpu()
