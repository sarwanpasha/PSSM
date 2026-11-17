"""Mean-pooled Nucleotide Transformer embeddings for DNA benchmarks (PyTorch, Intel XPU).
Model: InstaDeepAI/nucleotide-transformer-500m-human-ref (EsmForMaskedLM, supported natively by transformers).
usage: python extract_dna_lm.py <dataset> <root>  -> emb/nt500m__<dataset>.npy"""
import sys, os, time
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wpsskm.datasets import load

dname, root = sys.argv[1:3]
MID = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
dev = "xpu" if torch.xpu.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MID)
model = AutoModelForMaskedLM.from_pretrained(MID).eval().to(dev)
seqs = load(dname, root)["seqs"]; order = np.argsort([len(s) for s in seqs]); E = np.zeros((len(seqs), model.config.hidden_size), np.float32)
t0 = time.time(); B = 32
with torch.inference_mode():
    for k in range(0, len(seqs), B):
        ids = order[k:k + B]
        enc = tok([seqs[i] for i in ids], return_tensors="pt", padding=True, truncation=True, max_length=1000).to(dev)
        h = model(**enc, output_hidden_states=True).hidden_states[-1]
        m = enc["attention_mask"].clone(); m[:, 0] = 0                       # drop CLS from the mean
        E[ids] = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True)).float().cpu().numpy()
        if (k // B) % 100 == 0: print(f"  {k}/{len(seqs)} {(time.time()-t0)/60:.1f} min", flush=True)
np.save(os.path.join(root, "emb", f"nt500m__{dname}.npy"), E)
print(f"saved {E.shape} in {(time.time()-t0)/60:.1f} min -> emb/nt500m__{dname}.npy", flush=True)
