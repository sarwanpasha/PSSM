"""Compile an existing ESM-2 IR on one OpenVINO device, time it, and compare with PyTorch fp32.
usage: python npu_probe.py <xml> <device> <model_name>"""
import sys, time, random, re
import numpy as np, torch, esm, openvino as ov
xml, dev, mname = sys.argv[1], sys.argv[2], sys.argv[3]
B, Lp = map(int, re.search(r"_B(\d+)_L(\d+)\.xml", xml).groups())
model, alphabet = getattr(esm.pretrained, mname)(); model.eval(); L = model.num_layers
conv = alphabet.get_batch_converter(); PAD = alphabet.padding_idx
random.seed(1); AA = "ACDEFGHIKLMNPQRSTVWY"
seqs = ["".join(random.choice(AA) for _ in range(random.randint(Lp // 2, Lp - 2))) for _ in range(B * 4)]
def tok(b):
    _, _, tk = conv([(str(i), s) for i, s in enumerate(b)]); out = torch.full((len(b), Lp), PAD, dtype=torch.long); out[:, :tk.shape[1]] = tk; return out
core = ov.Core(); core.set_property({"CACHE_DIR": xml.rsplit("\\", 1)[0] + "\\cache"})
t0 = time.perf_counter(); print(f"compiling on {dev} ...", flush=True)
cm = core.compile_model(xml, dev); print(f"{dev} compile {time.perf_counter()-t0:.1f}s", flush=True)
req = cm.create_infer_request(); req.infer({0: tok(seqs[:B]).numpy()})
t0 = time.perf_counter(); outs = [req.infer({0: tok(seqs[i:i+B]).numpy()})[0].copy() for i in range(0, len(seqs), B)]
dt = (time.perf_counter() - t0) / len(seqs)
with torch.no_grad():
    ref = [model(tok(seqs[i:i+B]), repr_layers=[L])["representations"][L].numpy() for i in range(0, len(seqs), B)]
cos = []
for o, r, i in zip(outs, ref, range(0, len(seqs), B)):
    for j, s in enumerate(seqs[i:i+B]):
        a, b = o[j, 1:len(s)+1].mean(0), r[j, 1:len(s)+1].mean(0); cos.append(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
print(f"RESULT {dev} B={B} L={Lp}: {dt*1000:.1f} ms/seq, cosine vs PyTorch fp32 min={min(cos):.5f}", flush=True)
